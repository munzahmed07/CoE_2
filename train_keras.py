# src/baseline/train_keras.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.keras_predictor import create_predictor_model
from core.data_loader import LyftTrajectoryDataset # We'll reuse our PyTorch data loader

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "models/best/keras_model.h5"
EPOCHS = 20
BATCH_SIZE = 64

# --- Data Loading ---
# We load the data into memory for simplicity with Keras
print("Loading data...")
pytorch_dataset = LyftTrajectoryDataset(processed_dir=PROCESSED_DATA_DIR)
histories = np.array([item['focal_history'].numpy() for item in pytorch_dataset])
futures = np.array([item['focal_future'].numpy() for item in pytorch_dataset])
print(f"✅ Data loaded: {len(histories)} samples.")

# --- Model Training ---
# After
model = create_predictor_model()
# Create an Adam optimizer with gradient clipping (clipnorm=1.0 is a good default)
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0) 
model.compile(optimizer=optimizer, loss='mean_squared_error')

print("🔥 Starting Keras model training...")
model.fit(histories, futures, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

# --- Save Model ---
model.save(MODEL_SAVE_PATH)
print(f"✅ Keras model saved to {MODEL_SAVE_PATH}")