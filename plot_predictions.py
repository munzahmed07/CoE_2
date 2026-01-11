from src.core.keras_predictor import create_predictor_with_pretrained_encoder
from src.core.data_loader import LyftTrajectoryDataset, load_all_data_as_numpy
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- START of the robust path fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, './'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END of the robust path fix ---

# --- Project-specific Imports ---

# --- Configuration ---
BASELINE_MODEL_PATH = "models/baseline_model.h5"
ONLINE_MODEL_PATH = "models/online_model.h5"
# Needed if loading weights only
ENCODER_WEIGHTS_PATH = "models/encoder_weights.h5"
PROCESSED_DATA_DIR = "data/processed"
PLOTS_SAVE_DIR = "plots/predictions_for_experts"  # New directory for expert plots
NUM_PLOTS_TO_GENERATE = 5  # Generate 5 plots on random samples
TEST_DATA_SPLIT = 0.1  # Percentage of data to use for testing
DPI = 300  # High resolution for saving plots

# Model parameters (must match what was used for training)
HISTORY_LEN = 10
FUTURE_LEN = 20
FEATURES = 2  # x, y coordinates
ENCODER_LSTM_UNITS = 128
ENCODER_EMBEDDING_DIM = 64
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

# --- Helper Function for Model Loading ---


def load_robust_model(model_path, is_encoder=False):
    model = None
    try:
        model = tf.keras.models.load_model(model_path)
        print(
            f"✅ Model loaded using tf.keras.models.load_model() from {model_path}.")
    except Exception as e:
        print(
            f"WARNING: Could not load full Keras model directly from {model_path} ({e}).")
        print("Attempting to recreate model architecture and then load weights.")

        if not os.path.exists(ENCODER_WEIGHTS_PATH):
            print(
                f"CRITICAL ERROR: Encoder weights {ENCODER_WEIGHTS_PATH} not found. Cannot recreate model structure.")
            return None  # Fatal error, cannot proceed

        if is_encoder:
            from src.core.keras_predictor import create_encoder
            model = create_encoder(
                input_shape=(HISTORY_LEN, FEATURES),
                embedding_dim=ENCODER_EMBEDDING_DIM,
                lstm_units=ENCODER_LSTM_UNITS,
                name="encoder_for_loading"
            )
        else:  # It's a predictor model
            model = create_predictor_with_pretrained_encoder(
                encoder_weights_path=ENCODER_WEIGHTS_PATH,
                seq_len=HISTORY_LEN, features=FEATURES, future_steps=FUTURE_LEN,
                encoder_lstm_units=ENCODER_LSTM_UNITS, encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
                lstm_units=LSTM_UNITS, dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE
            )
        model.load_weights(model_path)
        print(
            f"✅ Model architecture recreated and weights loaded successfully from {model_path}.")
    return model

# --- Metrics Functions (Copied from profiler.py for self-containment) ---


def calculate_ade_fde(predictions, targets):
    """
    Calculates Average Displacement Error (ADE) and Final Displacement Error (FDE).
    predictions: (FUTURE_LEN, FEATURES) for a single sample
    targets: (FUTURE_LEN, FEATURES) for a single sample
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")

    displacement_errors = np.linalg.norm(predictions - targets, axis=-1)
    ade = np.mean(displacement_errors)
    # FDE for a single sample is just the last error
    fde = displacement_errors[-1]

    return ade, fde


if __name__ == "__main__":
    os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

    # 1. Load Models
    print("Loading Baseline Model...")
    baseline_model = load_robust_model(BASELINE_MODEL_PATH)
    if baseline_model is None:
        sys.exit(1)

    print("\nLoading Online-Adapted Model...")
    online_model = load_robust_model(ONLINE_MODEL_PATH)
    if online_model is None:
        sys.exit(1)

    # 2. Load Data for Plotting
    print(f"\nLoading data from {PROCESSED_DATA_DIR} for plotting...")
    all_histories, all_futures = load_all_data_as_numpy(PROCESSED_DATA_DIR)

    if len(all_histories) == 0:
        print(
            f"ERROR: No processed data found in {PROCESSED_DATA_DIR}. Cannot generate plots.")
        sys.exit(1)

    # Split data to get a test set, then select a few random samples from it
    total_samples = len(all_histories)
    indices = np.arange(total_samples)
    _, test_indices = train_test_split(
        indices, test_size=TEST_DATA_SPLIT, random_state=42)

    if len(test_indices) < NUM_PLOTS_TO_GENERATE:
        print(
            f"WARNING: Only {len(test_indices)} test samples available, generating plots for all of them.")
        sample_indices = test_indices
    else:
        sample_indices = np.random.choice(
            test_indices, NUM_PLOTS_TO_GENERATE, replace=False)

    print(f"Generating {len(sample_indices)} prediction plots for experts...")

    # Calculate overall min/max for consistent axis limits if desired
    # For simplicity, we'll let matplotlib auto-scale per plot but keep aspect ratio equal.
    # If a fixed range is needed, calculate global min/max here.

    # 3. Generate Predictions and Plots
    for i, idx in enumerate(tqdm(sample_indices, desc="Generating Plots")):
        history_gt = all_histories[idx]
        future_gt = all_futures[idx]

        input_for_prediction = np.expand_dims(history_gt, axis=0)

        baseline_pred = baseline_model.predict(
            input_for_prediction, verbose=0).squeeze(axis=0)
        online_pred = online_model.predict(
            input_for_prediction, verbose=0).squeeze(axis=0)

        # Calculate ADE/FDE for this specific sample
        baseline_ade, baseline_fde = calculate_ade_fde(
            baseline_pred, future_gt)
        online_ade, online_fde = calculate_ade_fde(online_pred, future_gt)

        # Slightly larger figure for better detail
        plt.figure(figsize=(12, 10))

        # Plot history (Ground Truth)
        plt.plot(history_gt[:, 0], history_gt[:, 1], 'o-',
                 color='tab:blue', linewidth=2, markersize=6, label='History')

        # Plot future (Ground Truth)
        plt.plot(future_gt[:, 0], future_gt[:, 1], 's-', color='tab:green',
                 linewidth=2, markersize=6, label='Ground Truth Future')

        # Plot Initial Model (Baseline) Prediction
        plt.plot(baseline_pred[:, 0], baseline_pred[:, 1], 'x--', color='tab:orange',
                 linewidth=1.5, markersize=5, label='Baseline Prediction')

        # Plot Online-Adapted Prediction
        plt.plot(online_pred[:, 0], online_pred[:, 1], '^:', color='tab:red', linewidth=1.5,
                 markersize=5, label='Online-Adapted Prediction')  # Changed to triangle marker, dotted line

        plt.title(
            f'Trajectory Prediction Comparison (Sample {idx})', fontsize=16)
        plt.xlabel('X Position (meters)', fontsize=12)
        plt.ylabel('Y Position (meters)', fontsize=12)

        # Position legend for clarity
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)  # Finer grid lines
        plt.axis('equal')  # Crucial for undistorted trajectory shapes

        # Add text box for metrics
        metrics_text = (
            f"Baseline Model:\n"
            f"  ADE: {baseline_ade:.2f}m\n"
            f"  FDE: {baseline_fde:.2f}m\n"
            f"Online-Adapted:\n"
            f"  ADE: {online_ade:.2f}m\n"
            f"  FDE: {online_fde:.2f}m"
        )
        plt.figtext(0.15, 0.75, metrics_text,
                    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.8),
                    fontsize=10, verticalalignment='top', horizontalalignment='left')

        plot_filename = os.path.join(
            PLOTS_SAVE_DIR, f"prediction_sample_{idx}.png")
        # Save with high DPI and tight bounding box
        plt.savefig(plot_filename, dpi=DPI, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

    print(f"\n✅ All enhanced plots saved to {PLOTS_SAVE_DIR}")
    print("\n--- PLOTTING PIPELINE COMPLETE ---")
