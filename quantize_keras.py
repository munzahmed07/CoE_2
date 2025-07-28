# src/edge/quantize_keras.py
import tensorflow as tf
import os

# --- Configuration ---
KERAS_MODEL_PATH = "models/best/keras_model.h5"
TFLITE_MODEL_PATH = "models/edge_optimized/quantized_model.tflite"

# --- Conversion ---
print(f"📦 Loading Keras model from {KERAS_MODEL_PATH}")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# --- NEW: Add the flags suggested by the error message ---
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TFLite builtin ops.
    tf.lite.OpsSet.SELECT_TF_OPS   # Enable select TensorFlow ops.
]
converter._experimental_lower_tensor_list_ops = False
# --- End of NEW ---

tflite_quant_model = converter.convert()

# --- Save Model ---
os.makedirs("models/edge_optimized", exist_ok=True)
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_quant_model)

print(f"✅ Quantized TFLite model saved to {TFLITE_MODEL_PATH}")