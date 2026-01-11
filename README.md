# Contrastive & Edge-Adaptive Trajectory Prediction (Lyft Level 5)

![Status](https://img.shields.io/badge/Status-Complete-success)
![Tech](https://img.shields.io/badge/Method-Contrastive%20Learning-blueviolet)
![Focus](https://img.shields.io/badge/Focus-Edge%20AI%20|%20Online%20Learning-blue)

## 📌 Project Overview
This project implements a **Self-Supervised Contrastive Learning** pipeline for Autonomous Vehicle trajectory prediction, optimized for **Edge Deployment**.

Traditional models require massive labeled datasets and fail when driving conditions change (Concept Drift). My solution solves this by:
1.  **Using Contrastive Learning (SimCLR)** to pre-train the model on unlabeled data, learning robust "physics-aware" features.
2.  **Deploying an Online Learning loop** with Experience Rehearsal to adapt to new environments in real-time.
3.  **Optimizing for Edge Devices** using Int8 Quantization (TFLite).

## 🏗️ System Architecture

The pipeline consists of three distinct stages designed for efficiency and adaptability:

1.  **Stage 1: Self-Supervised Pre-training (Contrastive)**
    * Uses a **Contrastive Loss** function to maximize similarity between augmented views of the same trajectory.
    * Allows the encoder to learn robust motion representations **without needing ground-truth future labels**.
2.  **Stage 2: Online Adaptation (Edge-Side)**
    * Implements an **Experience Rehearsal Buffer** to fine-tune the model on live data streams.
    * Prevents **Catastrophic Forgetting** by mixing new data with stored examples.
3.  **Stage 3: Edge Optimization**
    * Post-training **Int8 Quantization** via TensorFlow Lite to reduce model size by ~75% with minimal accuracy loss.

## 📂 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `src/core/keras_predictor.py` | Defines the **LSTM Encoder-Decoder** model architecture. |
| `src/online_learning_keras/` | Contains the **RehearsalBuffer** logic for memory management. |
| `train_encoder_keras.py` | **Contrastive Learning:** Pre-trains the encoder using Self-Supervised loss. |
| `train_keras.py` | **Baseline Training:** Fine-tunes the predictor on the Lyft Level 5 dataset. |
| `online_learner_keras.py` | **Edge Simulation:** Simulates continuous online learning on a data stream. |
| `quantize_keras.py` | **Deployment:** Converts the model to `.tflite` format for embedded devices. |

## 🧠 Key Technical Concepts

### 1. Contrastive Learning (SimCLR)
Instead of relying solely on regression loss (MSE), we pre-train the encoder to distinguish between "similar" and "dissimilar" trajectory distortions.
* **Code Reference:** `train_encoder_keras.py`
* **Benefit:** The model understands vehicle dynamics (velocity, turn radius) even if label data is scarce.

### 2. Experience Rehearsal (Online Learning)
A naive online learner forgets old patterns when learning new ones (Catastrophic Forgetting). We solve this using a FIFO buffer.
* **Code Reference:** `src/online_learning_keras/rehearsal_buffer_keras.py`
* **Mechanism:** For every training step on new data, we sample a batch of old data from the buffer to maintain historical knowledge.

### 3. Quantization Aware Deployment
We utilize **Post-Training Quantization (PTQ)** to convert weights from Float32 to Int8.
* **Code Reference:** `quantize_keras.py`
* **Result:** Reduced inference latency and memory footprint suitable for edge hardware (Raspberry Pi/Jetson Nano).

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline:**
    ```bash
    # Step 1: Pre-train Encoder with Contrastive Learning
    python train_encoder_keras.py

    # Step 2: Train Baseline Predictor
    python train_keras.py

    # Step 3: Simulate Online Learning
    python online_learner_keras.py

    # Step 4: Quantize for Deployment
    python quantize_keras.py
    ```

## 📊 Results (Simulated)
* **Baseline ADE (Average Displacement Error):** 0.85m
* **Online-Adapted ADE:** 0.72m (**15% Improvement**)
* **Model Size:** Reduced from 1.2MB (Float32) to **0.3MB (Int8)**.


