# src/core/keras_predictor.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_predictor_model(seq_len=10, features=3, future_steps=20, lstm_units=128):
    """Creates a non-autoregressive LSTM model in Keras."""
    
    # Input layer
    inputs = keras.Input(shape=(seq_len, features), name="input_trajectory")
    
    # LSTM layers to process the sequence
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out = layers.LSTM(lstm_units, return_sequences=False)(lstm_out) # Get the final context
    
    # Repeat the final context vector for each future step we want to predict
    repeated_context = layers.RepeatVector(future_steps)(lstm_out)
    
    # A final LSTM layer to generate the output sequence
    output_seq = layers.LSTM(lstm_units, return_sequences=True)(repeated_context)
    
    # A dense layer to get the (x, y) coordinates for each step
    outputs = layers.TimeDistributed(layers.Dense(2))(output_seq)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    model = create_predictor_model()
    model.summary()