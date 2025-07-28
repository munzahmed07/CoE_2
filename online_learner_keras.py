# src/online_learning_keras/online_learner_keras.py
import tensorflow as tf
import numpy as np
import os
import sys

# Add src directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_loader import LyftTrajectoryDataset
from online_learning_keras.rehearsal_buffer_keras import RehearsalBufferKeras
from torch.utils.data import Subset, DataLoader

# --- Configuration ---
MEMORY_CAPACITY = 500
UPDATE_STEPS = 200 # How many new samples to learn from
BATCH_SIZE = 32

INITIAL_MODEL_PATH = "models/best/keras_model.h5"
FINAL_MODEL_PATH = "models/best/online_keras_model.h5"

if __name__ == "__main__":
    print("🚀 Starting Keras Online Learning Simulation")

    # 1. Load the initially trained Keras model
    model = tf.keras.models.load_model(INITIAL_MODEL_PATH)
    print(f"✅ Initial Keras model loaded from {INITIAL_MODEL_PATH}")

    # 2. Initialize the rehearsal buffer
    rehearsal_buffer = RehearsalBufferKeras(capacity=MEMORY_CAPACITY)
    print("🧠 Rehearsal buffer is ready.")

    # 3. Prepare the "new" data stream from our PyTorch dataset
    full_dataset = LyftTrajectoryDataset(processed_dir="data/processed")
    stream_dataset = Subset(full_dataset, range(len(full_dataset) - UPDATE_STEPS, len(full_dataset)))
    print(f"🌊 Data stream created with {len(stream_dataset)} new samples.")

    # 4. The Online Learning Loop
    print("🔥 Starting online updates...")
    for i, new_sample in enumerate(stream_dataset):
        # Add the new sample (as NumPy arrays) to the buffer
        new_history = new_sample['focal_history'].numpy()
        new_future = new_sample['focal_future'].numpy()
        rehearsal_buffer.add(new_history, new_future)

        # We can only train when the buffer has enough samples for a full batch
        if len(rehearsal_buffer) < BATCH_SIZE:
            continue
        
        # Get a batch of old experiences from memory
        history_batch, future_batch = rehearsal_buffer.sample(BATCH_SIZE)
        
        # Perform one incremental learning step using train_on_batch
        loss = model.train_on_batch(history_batch, future_batch)

        if (i + 1) % 20 == 0:
            print(f"   Update Step {i+1}/{UPDATE_STEPS} | Loss: {loss:.4f}")

    # 5. Save the final, online-updated model
    model.save(FINAL_MODEL_PATH)
    print(f"✅ Online-updated Keras model saved to {FINAL_MODEL_PATH}")