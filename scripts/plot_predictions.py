import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random

# Assuming this is correctly set up in your src/utils
from src.utils.data_loader import LyftTrajectoryDataset

# --- Configuration ---
# Path to your final trained model.
# Use .h5 if you want to test the Keras model directly (e.g., before quantization)
# Use .tflite if you want to test the quantized TFLite model
MODEL_PATH = "models/online_adapted_model.h5"
# MODEL_PATH = "models/online_adapted_model.tflite" # Uncomment to test TFLite model

# Directory where preprocessed NPZ files are stored
PROCESSED_DATA_DIR = "data/processed"
PLOT_OUTPUT_DIR = "plots/"  # Directory to save the generated plots

HISTORY_LEN = 10  # Number of past frames (must match model's input)
# Number of future frames to predict (must match model's output)
FUTURE_LEN = 20
FEATURES = 2     # Number of features per point (x, y coordinates)

if __name__ == "__main__":
    # Create the output directory for plots if it doesn't exist
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # Initialize the dataset to load preprocessed data
    # We set shuffle=False here because we'll randomly pick an index later
    dataset = LyftTrajectoryDataset(
        processed_dir=PROCESSED_DATA_DIR, shuffle=False)

    if len(dataset) == 0:
        print(
            f"Error: No processed data found in {PROCESSED_DATA_DIR}. Please run preprocessing.py first.")
        exit()

    # Randomly select a sample to visualize
    sample_idx = random.randint(0, len(dataset) - 1)
    sample_data = dataset[sample_idx]

    # Extract history and ground truth future from the sample
    # Shape (HISTORY_LEN, FEATURES)
    history_data = sample_data['focal_history'].numpy()
    # Shape (FUTURE_LEN, FEATURES)
    ground_truth_future = sample_data['focal_future'].numpy()

    print(
        f"Generating prediction for sample index {sample_idx} using model: {MODEL_PATH}...")

    # Load model and make prediction
    predicted_future = None
    if MODEL_PATH.endswith('.h5'):
        # Load Keras model
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            # Make prediction. np.expand_dims adds batch dimension (1, HISTORY_LEN, FEATURES)
            predicted_future = model.predict(np.expand_dims(history_data, axis=0))[
                0]  # [0] to remove batch dim
        except Exception as e:
            print(f"Error loading or predicting with Keras model: {e}")
            print(
                f"Please ensure '{MODEL_PATH}' exists and is a valid Keras model.")
            exit()
    elif MODEL_PATH.endswith('.tflite'):
        # Load TFLite interpreter
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()

            # Get input and output tensor details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Prepare input data: TFLite usually expects float32
            input_tensor = np.expand_dims(
                history_data, axis=0).astype(np.float32)

            # Set the input tensor
            interpreter.set_tensor(input_details[0]['index'], input_tensor)

            # Invoke the interpreter (run inference)
            interpreter.invoke()

            # Get the output tensor
            predicted_future = interpreter.get_tensor(output_details[0]['index'])[
                0]  # [0] to remove batch dim
        except Exception as e:
            print(f"Error loading or predicting with TFLite model: {e}")
            print(
                f"Please ensure '{MODEL_PATH}' exists and is a valid TFLite model.")
            exit()
    else:
        print(f"Error: Unsupported model format for MODEL_PATH: {MODEL_PATH}")
        print("MODEL_PATH must end with .h5 (Keras) or .tflite (TensorFlow Lite).")
        exit()

    # --- Plotting ---
    plt.figure(figsize=(10, 8))

    # Plot history (past trajectory)
    plt.plot(history_data[:, 0], history_data[:, 1], 'bo-',
             label='Past Trajectory (History)', markersize=5, linewidth=2, alpha=0.8)

    # Plot ground truth future (actual future path)
    plt.plot(ground_truth_future[:, 0], ground_truth_future[:, 1], 'g--',
             label='Ground Truth Future', markersize=5, linewidth=2, alpha=0.8)

    # Plot predicted future (model's output)
    plt.plot(predicted_future[:, 0], predicted_future[:, 1], 'rx-',
             label='Predicted Future', markersize=6, linewidth=2, alpha=1.0)

    # Mark the current position (the exact point where history ends and future begins)
    plt.plot(history_data[-1, 0], history_data[-1, 1], 'ko',
             # Black circle marker
             markersize=10, label='Current Position (Time=0)')

    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.title(
        f"Trajectory Prediction for Sample {sample_idx}\nModel: {os.path.basename(MODEL_PATH)}")
    plt.legend()  # Display labels
    # Add a grid for better readability
    plt.grid(True, linestyle=':', alpha=0.7)
    # Ensure X and Y scales are equal for a true spatial representation
    plt.axis('equal')

    # Save the plot to the specified output directory
    plot_filename = os.path.join(
        PLOT_OUTPUT_DIR, f"prediction_sample_{sample_idx}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # Display the plot window
    plt.show()
