# src/evaluation/evaluate_keras.py
import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_loader import LyftTrajectoryDataset
from torch.utils.data import random_split

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
INITIAL_MODEL_PATH = "models/best/keras_model.h5"
ONLINE_MODEL_PATH = "models/best/online_keras_model.h5"
VALIDATION_SPLIT = 0.2

def compute_metrics(pred, gt):
    """Computes ADE and FDE in meters using NumPy."""
    error = np.linalg.norm(pred - gt, axis=-1)
    ade = np.mean(error)
    fde = np.mean(error[:, -1])
    return ade, fde

if __name__ == "__main__":
    # 1. Prepare Test Data
    full_dataset = LyftTrajectoryDataset(processed_dir=PROCESSED_DATA_DIR)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    _, test_set = random_split(full_dataset, [train_size, val_size])
    
    test_histories = np.array([item['focal_history'].numpy() for item in test_set])
    test_futures = np.array([item['focal_future'].numpy() for item in test_set])
    print(f"✅ Test data prepared with {len(test_histories)} samples.")

    # 2. Load Models
    initial_model = tf.keras.models.load_model(INITIAL_MODEL_PATH)
    online_model = tf.keras.models.load_model(ONLINE_MODEL_PATH)
    print("✅ Both models loaded.")

    # 3. Make Predictions
    print("🔮 Making predictions with both models...")
    initial_preds = initial_model.predict(test_histories)
    online_preds = online_model.predict(test_histories)

    # 4. Calculate Metrics
    ade_initial, fde_initial = compute_metrics(initial_preds, test_futures)
    ade_online, fde_online = compute_metrics(online_preds, test_futures)

    # 5. Print Comparison
    print("\n--- Final Performance Comparison ---")
    print(f"Initial Model:")
    print(f"   🔹 ADE: {ade_initial:.4f} meters")
    print(f"   🔹 FDE: {fde_online:.4f} meters")
    print(f"\nOnline-Adapted Model:")
    print(f"   🔹 ADE: {ade_online:.4f} meters")
    print(f"   🔹 FDE: {fde_online:.4f} meters")
    print("------------------------------------")