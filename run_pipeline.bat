@echo off
echo --- Starting Project Pipeline ---

echo.
echo [1/7] PREPROCESSING DATA...
python scripts/preprocessing.py

echo.
echo [2/7] TRAINING CONTRASTIVE ENCODER...
python -m src.contrastive_keras.train_encoder_keras

echo.
echo [3/7] TRAINING BASELINE PREDICTOR...
python -m src.baseline.train_keras

echo.
echo [4/7] PERFORMING ONLINE LEARNING...
python -m src.online_learning_keras.online_learner_keras

echo.
echo [5/7] QUANTIZING MODEL TO TFLITE...
python -m src.edge.quantize_keras

echo.
echo [6/7] PROFILING MODEL PERFORMANCE...
python -m src.edge.profiler

echo.
echo [7/7] GENERATING A PREDICTION PLOT...
python scripts/plot_predictions.py

echo.
echo --- PIPELINE COMPLETE! ---
pause