# src/evaluation/evaluate_keras.py
import tensorflow as tf
import numpy as np
import os
import sys

# Add src directory to path to import the data loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_loader import LyftTrajectoryDataset
from torch.utils.data import random_split

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
INITIAL_MODEL_PATH = "models/best/keras_model_enhanced.h5"
ONLINE_MODEL_PATH = "models/best/online_keras_model_final.h5" 
VALIDATION_SPLIT = 0.2

# --- Custom Metric Placeholders for Loading ---
def ade_metric(y_true, y_pred): return 0.0
def fde_metric(y_true, y_pred): return 0.0

def compute_metrics_np(pred, gt):
    """Computes ADE and FDE in meters using NumPy."""
    # Ensure shapes are (num_samples, 20, 2)
    pred_reshaped = pred.reshape(-1, 20, 2)
    gt_reshaped = gt.reshape(-1, 20, 2)
    
    error = np.linalg.norm(pred_reshaped - gt_reshaped, axis=-1) # Shape: [Num_Samples, 20]
    ade = np.mean(error)
    fde = np.mean(error[:, -1]) # Error at the last step
    return ade, fde

if __name__ == "__main__":
    # 1. Prepare Test Data
    print("🔄 Preparing test data...")
    full_dataset = LyftTrajectoryDataset(processed_dir=PROCESSED_DATA_DIR)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    _, test_set = random_split(full_dataset, [train_size, val_size])
    
    test_histories = np.array([item['focal_history'].numpy() for item in test_set])
    test_futures = np.array([item['focal_future'].numpy() for item in test_set])
    print(f"✅ Test data prepared with {len(test_histories)} samples.")

    # 2. Load Models
    custom_objects = {'ade_metric': ade_metric, 'fde_metric': fde_metric}
    initial_model = tf.keras.models.load_model(INITIAL_MODEL_PATH, custom_objects=custom_objects)
    online_model = tf.keras.models.load_model(ONLINE_MODEL_PATH)
    print("✅ Both models loaded.")

    # 3. Make Predictions
    print("🔮 Making predictions with both models...")
    initial_preds = initial_model.predict(test_histories)
    online_preds = online_model.predict(test_histories)

    # 4. Calculate Metrics
    ade_initial, fde_initial = compute_metrics_np(initial_preds, test_futures)
    ade_online, fde_online = compute_metrics_np(online_preds, test_futures)

    # 5. Print Comparison
    print("\n--- Final Performance Comparison ---")
    print(f"Initial Enhanced Model:")
    print(f"   🔹 ADE: {ade_initial:.4f} meters")
    print(f"   🔹 FDE: {fde_initial:.4f} meters")
    print(f"\nFinal Online-Adapted Model:")
    print(f"   🔹 ADE: {ade_online:.4f} meters")
    print(f"   🔹 FDE: {fde_online:.4f} meters")
    
    improvement = ((ade_initial - ade_online) / ade_initial) * 100
    print("\n------------------------------------")
    print(f"   Improvement in ADE: {improvement:+.1f}%")
    print("------------------------------------")