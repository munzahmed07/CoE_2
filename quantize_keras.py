from src.core.keras_predictor import create_encoder, create_predictor_with_pretrained_encoder
import tensorflow as tf
import os
import sys

# --- START of the robust path fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, './'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END of the robust path fix ---

# Import necessary components for custom objects, if the model requires them
# This is crucial if create_predictor_with_pretrained_encoder uses custom layers

# --- Configuration ---
# Use the path where your online_learner_keras.py saved the model
KERAS_MODEL_PATH = "models/online_model.h5"
TFLITE_MODEL_PATH = "models/edge_optimized/quantized_model_final.tflite"

# Ensure these match the values used during model creation (train_keras.py and online_learner_keras.py)
# These are needed to recreate the model structure if model.save_weights() was used instead of model.save()
HISTORY_LEN = 10
FEATURES = 2
FUTURE_LEN = 20
ENCODER_LSTM_UNITS = 128
ENCODER_EMBEDDING_DIM = 64
LSTM_UNITS = 256
DENSE_UNITS = 128
DROPOUT_RATE = 0.3


if __name__ == "__main__":
    print(f"🚀 Starting quantization for model: {KERAS_MODEL_PATH}")
    os.makedirs("models/edge_optimized", exist_ok=True)

    # 1. Load the Keras model
    # We need to handle cases where the model was saved with model.save_weights() (encoder training)
    # vs model.save() (baseline and online training)
    model = None
    try:
        # Attempt to load the full model (architecture + weights + optimizer state)
        # Assuming the online_learner_keras.py uses model.save()
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        print("✅ Keras model loaded using tf.keras.models.load_model().")
    except Exception as e:
        print(
            f"WARNING: Could not load full Keras model directly from {KERAS_MODEL_PATH} ({e}).")
        print("Attempting to recreate model architecture and then load weights.")

        # If model.save() failed, it means the model was likely saved only as weights.
        # In this case, we need to explicitly build its architecture first.
        # This requires matching parameters from how it was originally built.
        # The online_model.h5 is a full predictor, so it uses the create_predictor_with_pretrained_encoder function.
        try:
            # We need to provide a dummy encoder_weights_path as it's part of the predictor's build process,
            # but the actual weights will be loaded by model.load_weights(KERAS_MODEL_PATH)
            # This path *must exist* even if its content isn't strictly used for the full model load
            # Assuming this path exists from step 2
            DUMMY_ENCODER_WEIGHTS_PATH = "models/encoder_weights.h5"
            if not os.path.exists(DUMMY_ENCODER_WEIGHTS_PATH):
                print(
                    f"CRITICAL ERROR: Dummy encoder weights {DUMMY_ENCODER_WEIGHTS_PATH} not found. Cannot recreate model structure.")
                sys.exit(1)

            model = create_predictor_with_pretrained_encoder(
                # Dummy, as full model weights will override
                encoder_weights_path=DUMMY_ENCODER_WEIGHTS_PATH,
                seq_len=HISTORY_LEN,
                features=FEATURES,
                future_steps=FUTURE_LEN,
                encoder_lstm_units=ENCODER_LSTM_UNITS,
                encoder_embedding_dim=ENCODER_EMBEDDING_DIM,
                lstm_units=LSTM_UNITS,
                dense_units=DENSE_UNITS,
                dropout_rate=DROPOUT_RATE
            )
            # Now load the actual weights from the online model
            model.load_weights(KERAS_MODEL_PATH)
            print("✅ Model architecture recreated and weights loaded successfully.")
        except Exception as build_e:
            print(
                f"FATAL ERROR: Failed to recreate model architecture and load weights: {build_e}")
            sys.exit(1)

    # 2. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Default quantization (float16 or int8 based on settings)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Add the necessary flags for LSTM compatibility
    # Ensure all required operations are supported. SELECT_TF_OPS includes custom TF ops.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        # Required for some operations not yet in TFLite Builtins, like certain LSTM variants
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # The experimental flag is generally not needed and can be removed
    # converter._experimental_lower_tensor_list_ops = False

    print("Converting Keras model to TFLite with default optimizations...")
    tflite_quant_model = converter.convert()
    print("✅ Conversion to TFLite complete.")

    # 3. Save the final, quantized model
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"✅ Quantized TFLite model saved to {TFLITE_MODEL_PATH}")
    print("\n--- QUANTIZATION PIPELINE COMPLETE ---")
