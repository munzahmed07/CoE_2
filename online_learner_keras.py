from src.online_learning_keras.rehearsal_buffer_keras import RehearsalBuffer
from src.core.data_loader import LyftTrajectoryDataset
from src.core.keras_predictor import create_predictor_with_pretrained_encoder
from tqdm import tqdm
import numpy as np
from tensorflow import keras
import tensorflow as tf
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, './'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
ENCODER_WEIGHTS_PATH = "models/encoder_weights.h5"
BASE_MODEL_PATH = "models/baseline_model.h5"
ONLINE_MODEL_SAVE_PATH = "models/online_model.h5"
HISTORY_LEN = 10
FUTURE_LEN = 20
FEATURES = 2
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT_RATE = 0.3
ENCODER_LSTM_UNITS = 128
ENCODER_EMBEDDING_DIM = 64

# Online Learning Specifics
# Online Learning Specifics
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MEMORY_CAPACITY = 5000
# --- CHANGE THESE VALUES ---
ONLINE_ITERATIONS = 500  # Process 500 new samples in the online stream
# Start from a reasonable index, ensuring it's within your 15070 total samples
STREAM_START_INDEX = 10000
# --- END CHANGE ---
# This line will automatically adjust
STREAM_END_INDEX = STREAM_START_INDEX + ONLINE_ITERATIONS

# ... rest of the code ...

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(BASE_MODEL_PATH):
        print(
            f"Error: Baseline model not found at {BASE_MODEL_PATH}. Please run train_keras.py first.")
        exit()

    # Load the entire model, including its architecture and trained weights.
    # We should use load_model if the baseline model was saved with model.save()
    # If it was saved with model.save_weights(), we need to recreate the model structure first.
    # Assuming baseline_model.h5 was saved with model.save()
    try:
        model = keras.models.load_model(BASE_MODEL_PATH)
    except Exception as e:
        print(
            f"Failed to load full Keras model from {BASE_MODEL_PATH} ({e}). Attempting to recreate and load weights.")
        # If model.save() wasn't used, we need to recreate the model structure
        model = create_predictor_with_pretrained_encoder(
            encoder_weights_path=ENCODER_WEIGHTS_PATH,
            seq_len=HISTORY_LEN,
            features=FEATURES,
            future_steps=FUTURE_LEN,
            encoder_lstm_units=ENCODER_LSTM_UNITS,
            encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
            lstm_units=LSTM_UNITS,
            dense_units=DENSE_UNITS,
            dropout_rate=DROPOUT_RATE
        )
        model.load_weights(BASE_MODEL_PATH)

    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE), loss='mean_squared_error')
    print(f"Loaded baseline model from {BASE_MODEL_PATH}")

    # --- FIX START ---
    # Initialize Rehearsal Buffer, passing MEMORY_CAPACITY as 'max_size'
    rehearsal_buffer = RehearsalBuffer(max_size=MEMORY_CAPACITY)
    # --- FIX END ---

    online_stream_dataset = LyftTrajectoryDataset(
        processed_dir=PROCESSED_DATA_DIR,
        shuffle=False,
        subset_indices=range(STREAM_START_INDEX, STREAM_END_INDEX)
    )

    if len(online_stream_dataset) == 0:
        print(
            f"Error: No online stream data found for indices {STREAM_START_INDEX}-{STREAM_END_INDEX}. Check data or indices.")
        exit()

    for i in tqdm(range(len(online_stream_dataset)), desc="Online Adaptation"):
        new_sample = online_stream_dataset[i]
        new_history = new_sample['focal_history']
        new_future = new_sample['focal_future']

        rehearsal_buffer.add((new_history, new_future))

        if rehearsal_buffer.size() < BATCH_SIZE:
            continue

        # Ensure that memory_batch_size is at least 0
        memory_batch_size = max(0, BATCH_SIZE - 1)

        if memory_batch_size > 0:
            memory_samples = rehearsal_buffer.sample(memory_batch_size)
            # Ensure memory_samples is not empty before attempting to zip
            if memory_samples:
                memory_histories, memory_futures = zip(*memory_samples)
                history_batch = np.vstack(
                    [np.expand_dims(new_history, axis=0), np.array(memory_histories)])
                future_batch = np.vstack(
                    [np.expand_dims(new_future, axis=0), np.array(memory_futures)])
            else:  # Fallback if for some reason memory_samples is empty
                history_batch = np.expand_dims(new_history, axis=0)
                future_batch = np.expand_dims(new_future, axis=0)
        else:  # If batch size is 1, only use the new sample
            history_batch = np.expand_dims(new_history, axis=0)
            future_batch = np.expand_dims(new_future, axis=0)

        # Ensure batch dimensions match what model.train_on_batch expects
        # (BATCH_SIZE, HISTORY_LEN, FEATURES) for input and (BATCH_SIZE, FUTURE_LEN, FEATURES) for output
        if history_batch.shape[0] != BATCH_SIZE:
            # This can happen if rehearsal buffer has fewer than BATCH_SIZE items initially
            # Or if memory_batch_size calculation results in fewer items
            continue  # Skip training if batch is not full

        loss = model.train_on_batch(history_batch, future_batch)
        # Optional: print loss periodically
        # if i % 100 == 0:
        #     print(f"Step {i}, Online Loss: {loss:.4f}")

    # After online training, save the fine-tuned model
    # Use model.save() to save the full model (architecture + weights + optimizer state)
    # or model.save_weights() if you only want weights.
    # Given model.load_model() above, model.save() is appropriate here.
    model.save(ONLINE_MODEL_SAVE_PATH)
    print(
        f"\nOnline adaptation complete. Final model saved to {ONLINE_MODEL_SAVE_PATH}")
