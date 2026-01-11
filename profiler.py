# For splitting data for testing
from sklearn.model_selection import train_test_split
# Needed if loading weights only
from src.core.keras_predictor import create_predictor_with_pretrained_encoder
from src.core.data_loader import LyftTrajectoryDataset, load_all_data_as_numpy
import sys
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# --- START of the robust path fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, './'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END of the robust path fix ---

# --- Project-specific Imports ---


# --- Configuration ---
# Use the final, online-adapted Keras model as the source for profiling
ORIGINAL_MODEL_PATH = "models/online_model.h5"
# Point to the final TFLite model to be profiled
TFLITE_MODEL_PATH = "models/edge_optimized/quantized_model_final.tflite"

# Directory where processed data is stored
PROCESSED_DATA_DIR = "data/processed"
TEST_DATA_SPLIT = 0.1  # Percentage of data to use for testing
NUM_INFERENCE_RUNS = 100  # Number of times to run inference for speed benchmarking

# Model parameters (must match what was used for training)
HISTORY_LEN = 10
FUTURE_LEN = 20
FEATURES = 2  # x, y coordinates
ENCODER_LSTM_UNITS = 128
ENCODER_EMBEDDING_DIM = 64
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT_RATE = 0.3

# Input shape for inference (batch_size=1, HISTORY_LEN, FEATURES)
STATIC_INPUT_SHAPE = (1, HISTORY_LEN, FEATURES)


# --- Metrics Functions ---
def calculate_ade_fde(predictions, targets):
    """
    Calculates Average Displacement Error (ADE) and Final Displacement Error (FDE).
    predictions: (N, FUTURE_LEN, FEATURES)
    targets: (N, FUTURE_LEN, FEATURES)
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")

    # Displacement Error at each time step: (N, FUTURE_LEN)
    displacement_errors = np.linalg.norm(predictions - targets, axis=-1)

    # ADE: Average over all time steps and all samples
    ade = np.mean(displacement_errors)

    # FDE: Displacement error at the final time step
    fde = np.mean(displacement_errors[:, -1])

    return ade, fde


# --- Main Profiling Logic ---
if __name__ == "__main__":
    print("🚀 Starting final model profiling and evaluation...")

    # Ensure models exist
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        print(
            f"ERROR: Original Keras model not found at {ORIGINAL_MODEL_PATH}. Please run online_learner_keras.py first.")
        sys.exit(1)
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(
            f"ERROR: Quantized TFLite model not found at {TFLITE_MODEL_PATH}. Please run quantize_keras.py first.")
        sys.exit(1)

    # 1. Load Data for Evaluation
    print(f"\nLoading data from {PROCESSED_DATA_DIR} for evaluation...")
    all_histories, all_futures = load_all_data_as_numpy(PROCESSED_DATA_DIR)

    if len(all_histories) == 0:
        print(
            f"ERROR: No processed data found in {PROCESSED_DATA_DIR}. Cannot perform evaluation.")
        sys.exit(1)

    # Split data into training and testing set (using test set for evaluation)
    _, test_indices = train_test_split(
        np.arange(len(all_histories)), test_size=TEST_DATA_SPLIT, random_state=42)
    test_histories = all_histories[test_indices]
    test_futures = all_futures[test_indices]

    print(f"Loaded {len(test_histories)} samples for testing.")

    # 2. Load Keras Model
    keras_model = None
    try:
        keras_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
        print("✅ Keras model loaded using tf.keras.models.load_model().")
    except Exception as e:
        print(
            f"WARNING: Could not load full Keras model directly from {ORIGINAL_MODEL_PATH} ({e}).")
        print("Attempting to recreate model architecture and then load weights.")
        DUMMY_ENCODER_WEIGHTS_PATH = "models/encoder_weights.h5"
        if not os.path.exists(DUMMY_ENCODER_WEIGHTS_PATH):
            print(
                f"CRITICAL ERROR: Dummy encoder weights {DUMMY_ENCODER_WEIGHTS_PATH} not found. Cannot recreate Keras model structure.")
            sys.exit(1)
        keras_model = create_predictor_with_pretrained_encoder(
            encoder_weights_path=DUMMY_ENCODER_WEIGHTS_PATH,
            seq_len=HISTORY_LEN, features=FEATURES, future_steps=FUTURE_LEN,
            encoder_lstm_units=ENCODER_LSTM_UNITS, encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
            lstm_units=LSTM_UNITS, dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE
        )
        keras_model.load_weights(ORIGINAL_MODEL_PATH)
        print("✅ Keras model architecture recreated and weights loaded successfully.")

    # 3. Load TFLite Model (Interpreter)
    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    tflite_interpreter.allocate_tensors()
    tflite_input_details = tflite_interpreter.get_input_details()
    tflite_output_details = tflite_interpreter.get_output_details()
    print("✅ TFLite interpreter loaded.")

    # 4. Model Size Comparison
    original_size = os.path.getsize(
        ORIGINAL_MODEL_PATH) / (1024 * 1024)  # in MB
    quantized_size = os.path.getsize(
        TFLITE_MODEL_PATH) / (1024 * 1024)  # in MB

    print("\n--- Model Size Comparison ---")
    print(f"   Final Keras Model: {original_size:.2f} MB")
    print(f"   Quantized TFLite Model: {quantized_size:.2f} MB")
    if original_size > 0:
        print(
            f"   Reduction: {100 * (1 - quantized_size / original_size):.1f}%")
    else:
        print("   Cannot calculate reduction (original model size is zero).")

    # 5. Inference Speed Comparison
    print("\n--- Inference Speed Comparison ---")
    keras_times = []
    tflite_times = []

    dummy_input = np.random.rand(1, HISTORY_LEN, FEATURES).astype(np.float32)

    # Keras Model Inference Speed
    print("Profiling Keras model inference speed...")
    for _ in tqdm(range(NUM_INFERENCE_RUNS), desc="Keras Speed"):
        start_time = time.perf_counter()
        _ = keras_model.predict(dummy_input, verbose=0)
        end_time = time.perf_counter()
        keras_times.append((end_time - start_time) * 1000)  # milliseconds
    avg_keras_time = np.mean(keras_times)
    print(f"   Average Keras Model Prediction Time: {avg_keras_time:.2f} ms")

    # TFLite Model Inference Speed
    print("Profiling TFLite model inference speed...")
    tflite_interpreter.resize_tensor_input(
        tflite_input_details[0]['index'], (1, HISTORY_LEN, FEATURES))  # Ensure input shape is correct
    tflite_interpreter.allocate_tensors()  # Reallocate after resize

    for _ in tqdm(range(NUM_INFERENCE_RUNS), desc="TFLite Speed"):
        start_time = time.perf_counter()
        tflite_interpreter.set_tensor(
            tflite_input_details[0]['index'], dummy_input)
        tflite_interpreter.invoke()
        _ = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])
        end_time = time.perf_counter()
        tflite_times.append((end_time - start_time) * 1000)  # milliseconds
    avg_tflite_time = np.mean(tflite_times)
    print(f"   Average TFLite Model Prediction Time: {avg_tflite_time:.2f} ms")
    if avg_keras_time > 0:
        print(
            f"   Speedup (Keras/TFLite): {avg_keras_time / avg_tflite_time:.2f}x")

    # 6. Prediction Accuracy (ADE/FDE)
    print("\n--- Prediction Accuracy (ADE/FDE) ---")
    keras_predictions = []
    tflite_predictions = []

    print("Evaluating Keras model accuracy...")
    for i in tqdm(range(len(test_histories)), desc="Keras Eval"):
        input_history = np.expand_dims(
            test_histories[i], axis=0)  # Add batch dimension
        keras_pred = keras_model.predict(input_history, verbose=0)
        keras_predictions.append(keras_pred.squeeze(
            axis=0))  # Remove batch dimension

    print("Evaluating TFLite model accuracy...")
    for i in tqdm(range(len(test_histories)), desc="TFLite Eval"):
        input_history = np.expand_dims(test_histories[i], axis=0).astype(
            np.float32)  # Ensure float32
        tflite_interpreter.set_tensor(
            tflite_input_details[0]['index'], input_history)
        tflite_interpreter.invoke()
        tflite_pred = tflite_interpreter.get_tensor(
            tflite_output_details[0]['index'])
        tflite_predictions.append(tflite_pred.squeeze(axis=0))

    keras_predictions = np.array(keras_predictions)
    tflite_predictions = np.array(tflite_predictions)

    # Calculate ADE and FDE for Keras model
    keras_ade, keras_fde = calculate_ade_fde(keras_predictions, test_futures)
    print(f"   Keras Model: ADE={keras_ade:.4f}, FDE={keras_fde:.4f}")

    # Calculate ADE and FDE for TFLite model
    tflite_ade, tflite_fde = calculate_ade_fde(
        tflite_predictions, test_futures)
    print(f"   TFLite Model: ADE={tflite_ade:.4f}, FDE={tflite_fde:.4f}")

    print("\n🎉 Profiling Complete!")
    print("\n--- PROFILING PIPELINE COMPLETE ---")
