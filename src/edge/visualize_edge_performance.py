# src/edge/visualize_edge_performance.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- Configuration ---
KERAS_MODEL_PATH = "models/best/keras_model.h5"
TFLITE_MODEL_PATH = "models/edge_optimized/quantized_model.tflite"
NUM_TEST_RUNS = 100
STATIC_INPUT_SHAPE = (1, 10, 3)
SAVE_PATH = "results/plots/edge_performance_comparison.png"

def profile_keras_model(model_path):
    """Loads a Keras model and profiles its inference speed."""
    model = tf.keras.models.load_model(model_path)
    input_data = np.array(np.random.random_sample(STATIC_INPUT_SHAPE), dtype=np.float32)
    
    # Warm-up run
    model.predict(input_data, verbose=0)
    
    times = []
    for _ in range(NUM_TEST_RUNS):
        start_time = time.time()
        model.predict(input_data, verbose=0)
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # milliseconds
    return np.mean(times)

def profile_tflite_model(model_path):
    """Loads a TFLite model and profiles its inference speed."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.resize_tensor_input(0, STATIC_INPUT_SHAPE)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = np.array(np.random.random_sample(STATIC_INPUT_SHAPE), dtype=np.float32)

    # Warm-up run
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    times = []
    for _ in range(NUM_TEST_RUNS):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # milliseconds
    return np.mean(times)

if __name__ == "__main__":
    print("🚀 Creating edge performance comparison visualization...")
    os.makedirs("results/plots", exist_ok=True)

    # 1. Get file sizes
    keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model sizes: Keras={keras_size:.2f}MB, TFLite={tflite_size:.2f}MB")

    # 2. Get inference speeds
    keras_speed = profile_keras_model(KERAS_MODEL_PATH)
    tflite_speed = profile_tflite_model(TFLITE_MODEL_PATH)
    print(f"✅ Inference speeds: Keras={keras_speed:.2f}ms, TFLite={tflite_speed:.2f}ms")

    # 3. Create the bar chart
    labels = ['Model Size (MB)', 'Inference Speed (ms)']
    original_metrics = [keras_size, keras_speed]
    quantized_metrics = [tflite_size, tflite_speed]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, original_metrics, width, label='Original Keras Model')
    rects2 = ax.bar(x + width/2, quantized_metrics, width, label='Quantized TFLite Model')

    ax.set_ylabel('Value')
    ax.set_title('Model Performance Before vs. After Edge Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ Performance chart saved to {SAVE_PATH}")