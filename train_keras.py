from src.core.data_loader import LyftTrajectoryDataset
from src.core.keras_predictor import create_predictor_with_pretrained_encoder
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
import tensorflow as tf
import sys
import os

# --- START of the robust path fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming script is in C:\CoE_2
project_root = os.path.abspath(os.path.join(current_dir, './'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END of the robust path fix ---


# These imports should now work correctly

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
ENCODER_WEIGHTS_PATH = "models/encoder_weights.h5"
MODEL_SAVE_PATH = "models/baseline_model.h5"
HISTORY_LEN = 10
FUTURE_LEN = 20
FEATURES = 2  # x, y coordinates
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT_RATE = 0.3  # Dropout rate for regularization
ENCODER_LSTM_UNITS = 128  # Must match units used during encoder pre-training
ENCODER_EMBEDDING_DIM = 64  # Must match dim used during encoder pre-training

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print("Initializing and splitting dataset indices...")
    # Get a list of all file paths to determine total samples for splitting
    all_file_paths = [os.path.join(PROCESSED_DATA_DIR, f) for f in os.listdir(
        PROCESSED_DATA_DIR) if f.endswith(".npz")]
    if not all_file_paths:
        print(
            f"Error: No processed data found in {PROCESSED_DATA_DIR}. Please run preprocessing.py first.")
        exit()

    indices = np.arange(len(all_file_paths))
    train_indices, val_indices = train_test_split(
        indices, test_size=VALIDATION_SPLIT, random_state=42)

    # --- START OF THE FIX ---
    # Create a dataset instance JUST for training, passing the train_indices to its constructor
    train_dataset = LyftTrajectoryDataset(
        processed_dir=PROCESSED_DATA_DIR, shuffle=False, subset_indices=train_indices)
    train_tf_dataset = train_dataset.create_tf_dataset(
        BATCH_SIZE, shuffle=True)

    # Create a separate dataset instance JUST for validation
    val_dataset = LyftTrajectoryDataset(
        processed_dir=PROCESSED_DATA_DIR, shuffle=False, subset_indices=val_indices)
    val_tf_dataset = val_dataset.create_tf_dataset(BATCH_SIZE, shuffle=False)
    # --- END OF THE FIX ---

    print(
        f"Training on {len(train_indices)} samples, validating on {len(val_indices)} samples.")

    # Create the full predictor model, loading the pre-trained encoder weights
    if not os.path.exists(ENCODER_WEIGHTS_PATH):
        print(
            f"Error: Encoder weights not found at {ENCODER_WEIGHTS_PATH}. Please run train_encoder_keras.py first.")
        exit()

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

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE), loss='mean_squared_error')
    model.summary()

    # Callbacks for saving the best model and early stopping
    checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print(f"Starting baseline model training for {EPOCHS} epochs...")
    model.fit(
        train_tf_dataset,
        epochs=EPOCHS,
        validation_data=val_tf_dataset,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    print(
        f"Baseline model training complete. Best model saved to {MODEL_SAVE_PATH}")
