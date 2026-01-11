# src/core/visualize.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random

# Add src directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data_loader import LyftTrajectoryDataset

# --- Configuration ---
# Use the final, stable model filenames
INITIAL_MODEL_PATH = "models/best/keras_model_balanced.h5" 
ONLINE_MODEL_PATH = "models/best/online_keras_model_final.h5"
NUM_SAMPLES = 4 # How many random samples to visualize

# --- Custom Metric Placeholders for Loading ---
# These are needed to load the model if it was saved with custom metrics
def ade_metric(y_true, y_pred): return 0.0
def fde_metric(y_true, y_pred): return 0.0

if __name__ == "__main__":
    print(f"🚀 Creating comparison visualizations for {NUM_SAMPLES} random samples...")
    os.makedirs("results/plots", exist_ok=True)

    # 1. Load both Keras models once
    print("📦 Loading models...")
    # The initial model might have been saved with custom metrics
    custom_objects = {'ade_metric': ade_metric, 'fde_metric': fde_metric}
    initial_model = tf.keras.models.load_model(INITIAL_MODEL_PATH, custom_objects=custom_objects)
    # The online model was saved after a simple compile, so it might not need them
    online_model = tf.keras.models.load_model(ONLINE_MODEL_PATH)
    print("✅ Both Keras models loaded.")

    # 2. Load the dataset once
    dataset = LyftTrajectoryDataset(processed_dir="data/processed")
    
    # 3. Select random sample indices from the "new" data portion
    # We choose from the end of the dataset where the online learner was trained
    start_range = len(dataset) - 500 # Corresponds to UPDATE_STEPS in the online learner
    sample_indices = random.sample(range(start_range, len(dataset)), NUM_SAMPLES)
    print(f"✅ Selected random samples: {sample_indices}")
    
    # 4. Loop through each random sample index
    for sample_index in sample_indices:
        print(f"--- Processing Sample {sample_index} ---")
        
        # a. Get the data for the current sample
        sample = dataset[sample_index]
        history = sample["focal_history"].numpy()
        future_gt = sample["focal_future"].numpy()

        # b. Get predictions from both models
        initial_pred = initial_model.predict(np.expand_dims(history, axis=0), verbose=0)[0]
        online_pred = online_model.predict(np.expand_dims(history, axis=0), verbose=0)[0]

        # c. Plot everything
        plt.figure(figsize=(12, 8))
        plt.plot(history[:, 0], history[:, 1], 'b-o', label='History')
        plt.plot(future_gt[:, 0], future_gt[:, 1], 'g-s', label='Ground Truth Future')
        plt.plot(initial_pred[:, 0], initial_pred[:, 1], 'r--x', label='Initial Model Prediction')
        plt.plot(online_pred[:, 0], online_pred[:, 1], '--p', color='orange', label='Online-Adapted Prediction')
        
        plt.title(f'Keras Model Before vs. After Online Learning (Sample {sample_index})')
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # d. Save the plot with a unique name
        save_path = f"results/plots/final_comparison_{sample_index}.png"
        plt.savefig(save_path)
        plt.close() # Close the figure to free up memory
        print(f"✅ Visualization for sample {sample_index} saved to {save_path}")

    print("\n🎉 All random comparison visualizations created successfully.")
