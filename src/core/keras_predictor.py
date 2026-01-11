import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_encoder(input_shape, embedding_dim, lstm_units, name="trajectory_encoder_internal"):
    """
    Creates the trajectory encoder model.
    This function is internal to this module, used by create_predictor_with_pretrained_encoder.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=False,
                    name=f"{name}_lstm")(inputs)
    outputs = layers.Dense(embedding_dim, name=f"{name}_embedding")(x)
    return keras.Model(inputs, outputs, name=name)


def create_predictor_with_pretrained_encoder(
    encoder_weights_path,
    seq_len,
    features,
    future_steps,
    encoder_lstm_units,     # <-- Added these arguments
    encoder_embedding_dim,  # <-- Added these arguments
    lstm_units,
    dense_units,
    dropout_rate=0.0
):
    """
    Creates a trajectory prediction model that uses a pre-trained encoder.

    Args:
        encoder_weights_path (str): Path to the H5 file containing the pre-trained encoder weights.
        seq_len (int): Length of the input history sequence.
        features (int): Number of features per timestep (e.g., 2 for x,y coordinates).
        future_steps (int): Number of future timesteps to predict.
        encoder_lstm_units (int): Number of LSTM units used in the pre-trained encoder.
        encoder_embedding_dim (int): Dimension of the embedding output from the pre-trained encoder.
        lstm_units (int): Number of LSTM units for the decoder.
        dense_units (int): Number of units for the intermediate dense layer.
        dropout_rate (float): Dropout rate after the intermediate dense layer.

    Returns:
        tf.keras.Model: Compiled Keras model for trajectory prediction.
    """
    # 1. Create the encoder and load weights
    encoder = create_encoder(
        input_shape=(seq_len, features),
        embedding_dim=encoder_embedding_dim,
        lstm_units=encoder_lstm_units,
        name="pretrained_trajectory_encoder"
    )
    encoder.load_weights(encoder_weights_path)
    encoder.trainable = False  # Freeze encoder weights

    # 2. Define the Decoder
    decoder_inputs = keras.Input(
        shape=(seq_len, features), name="decoder_input")

    # Pass inputs through the (frozen) encoder to get the context vector (embedding)
    context_vector = encoder(decoder_inputs)

    # Repeat the context vector for each future step and concatenate for decoder input
    # This prepares it for a seq2seq-like structure, if the decoder needs repeated context
    # However, a simpler approach is to feed the context vector once and let the decoder LSTM manage it.
    # We will use the context vector as the initial state or simply feed it.

    # For a simple decoder, we can feed the context vector into the decoder LSTM as initial state,
    # or just use it as an input to dense layers after a decoder LSTM processes some dummy input
    # or the context directly.

    # Common approach: use context_vector as input to a dense layer, then feed that into LSTM
    # Or, as a simple one-shot prediction:

    # Option 1: Direct prediction from context (less common for sequences)
    # x = layers.Dense(dense_units, activation="relu")(context_vector)
    # x = layers.Dropout(dropout_rate)(x)
    # outputs = layers.Dense(future_steps * features)(x)
    # outputs = layers.Reshape((future_steps, features), name="future_prediction")(outputs)

    # Option 2: Using LSTM for sequence generation (more appropriate for trajectory prediction)
    # Initialize the decoder LSTM with the context vector
    # We need a dummy sequence input for the decoder if it's stateless,
    # or we can pass the context as initial_state.

    # A simple way for prediction:
    # Use context_vector (from encoder) as input to a new LSTM layer which outputs the sequence.
    # This implies the decoder LSTM directly learns to unroll the sequence from the context.

    # The context vector is effectively an encoding of the history.
    # We want the decoder to take this context and output `future_steps` coordinates.

    # Common practice: Add a dense layer to transform context, then repeat and feed to LSTM,
    # or directly feed context to LSTM (if it expects single timestep input to produce sequence)

    # Let's simplify and use the context vector to drive the prediction
    # We'll use a dense layer to expand the context to match the LSTM units
    # and then an LSTM to sequence the predictions.

    # Expand the context vector
    x = layers.Dense(lstm_units, activation="relu",
                     name="decoder_dense_context")(context_vector)

    # Create a "dummy" sequence of the length of future_steps,
    # then feed the context through it. A more advanced seq2seq would
    # feed the actual last observed point or predicted point.

    # For simplicity, we can use the context as the initial state of the decoder LSTM
    # and then feed it a placeholder sequence (e.g., zeros) or just a single input
    # to "unroll" the sequence.

    # Let's use a simpler approach that's effective:
    # The context vector is directly transformed into the future sequence.

    # Intermediate dense layers
    x = layers.Dense(dense_units, activation="relu",
                     name="decoder_intermediate_dense_1")(context_vector)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu",
                     name="decoder_intermediate_dense_2")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    # Output layer to predict the future trajectory points
    # Reshape to (future_steps, features)
    outputs = layers.Dense(future_steps * features,
                           name="future_prediction_flat")(x)
    outputs = layers.Reshape((future_steps, features),
                             name="future_prediction")(outputs)

    model = keras.Model(inputs=decoder_inputs, outputs=outputs,
                        name="trajectory_predictor_with_encoder")
    return model
