# src/edge/profiler.py
import os
import time
import numpy as np
import tensorflow as tf

# --- Configuration ---
ORIGINAL_MODEL_PATH = "models/predictor/online_model.pth"
TFLITE_MODEL_PATH = "models/edge_optimized/quantized_model.tflite"
NUM_TEST_RUNS = 100
# NEW: Define the exact input shape we want to test with
STATIC_INPUT_SHAPE = (1, 10, 3)

if __name__ == "__main__":
    print("🚀 Starting Phase 3, Step 2: Model Profiling")

    # 1. Measure file sizes
    original_size = os.path.getsize(ORIGINAL_MODEL_PATH) / (1024 * 1024) # in MB
    quantized_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024) # in MB
    
    print("\n--- Model Size Comparison ---")
    print(f"   Original PyTorch Model: {original_size:.2f} MB")
    print(f"   Quantized TFLite Model: {quantized_size:.2f} MB")
    print(f"   Reduction: {100 * (1 - quantized_size / original_size):.1f}%")

    # 2. Measure inference speed
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    
    # --- MODIFIED: Resize the input tensor to our static shape ---
    input_details = interpreter.get_input_details()
    interpreter.resize_tensor_input(input_details[0]['index'], STATIC_INPUT_SHAPE)
    interpreter.allocate_tensors() # Must re-allocate after resizing
    # --- End of MODIFIED ---
    
    output_details = interpreter.get_output_details()
    
    # Use the static shape to create dummy input data
    input_data = np.array(np.random.random_sample(STATIC_INPUT_SHAPE), dtype=np.float32)

    # Run inference multiple times to get an average speed
    times = []
    for _ in range(NUM_TEST_RUNS):
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # in milliseconds
        
    avg_time = np.mean(times)
    
    print("\n--- Inference Speed Comparison ---")
    print(f"   Average prediction time on CPU: {avg_time:.2f} ms")
    print("\n🎉 Project Complete!")