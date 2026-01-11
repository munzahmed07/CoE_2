from src.core.data_loader import LyftTrajectoryDataset, load_all_data_as_numpy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import sys
import os

# --- START of the robust path fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming script is in C:\CoE_2
project_root = os.path.abspath(os.path.join(current_dir, './'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END of the robust path fix ---


# --- CORRECTED IMPORT LINE ---
# Now importing from src.core where data_loader.py is located

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed"
ENCODER_SAVE_PATH = "models/encoder_weights.h5"
HISTORY_LEN = 10
FEATURES = 2  # x, y coordinates
EMBEDDING_DIM = 64
LSTM_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4
TEMPERATURE = 0.07  # Hyperparameter for InfoNCE loss
NUM_NEGATIVES = 15  # Number of negative samples per anchor
VALIDATION_SPLIT = 0.2

# --- Model Definition ---


def create_encoder(input_shape=(HISTORY_LEN, FEATURES), embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=False,
                    name="encoder_lstm_contrastive")(inputs)
    outputs = layers.Dense(
        embedding_dim, name="embedding_output_contrastive")(x)
    return keras.Model(inputs, outputs, name="trajectory_encoder")

# --- InfoNCE Loss Function ---


@tf.function
def info_nce_loss(anchor_emb, all_embs, temperature=TEMPERATURE):
    anchor_emb = tf.math.l2_normalize(anchor_emb, axis=1)
    all_embs = tf.math.l2_normalize(all_embs, axis=2)
    similarity_scores = tf.matmul(tf.expand_dims(
        anchor_emb, 1), all_embs, transpose_b=True)
    similarity_scores = tf.squeeze(similarity_scores, axis=1) / temperature
    labels = tf.zeros(tf.shape(similarity_scores)[0], dtype=tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, similarity_scores, from_logits=True)
    return tf.reduce_mean(loss)


# --- Data Generator for Contrastive Learning ---
def contrastive_data_generator(histories, batch_size, num_negatives=NUM_NEGATIVES):
    dataset_size = len(histories)
    indices = np.arange(dataset_size)
    while True:
        batch_indices = np.random.choice(indices, batch_size, replace=False)
        anchor_histories = histories[batch_indices]

        positive_histories = anchor_histories

        negative_histories_list = []
        for _ in range(num_negatives):
            neg_indices = np.random.choice(indices, batch_size, replace=False)
            negative_histories_list.append(histories[neg_indices])

        negative_histories = np.stack(negative_histories_list, axis=0)
        negative_histories = np.transpose(negative_histories, (1, 0, 2, 3))

        yield anchor_histories, positive_histories, negative_histories


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    print(f"Loading all history data from {PROCESSED_DATA_DIR}...")
    all_histories, _ = load_all_data_as_numpy(PROCESSED_DATA_DIR)

    if len(all_histories) < BATCH_SIZE * 2:
        print("Warning: Not enough data for meaningful train/validation split. Consider increasing dataset size or reducing BATCH_SIZE.")
        if len(all_histories) < BATCH_SIZE:
            print("Error: Dataset size is smaller than batch size. Exiting.")
            exit()

    train_histories, val_histories = train_test_split(
        all_histories, test_size=VALIDATION_SPLIT, random_state=42)

    print(f"Total history samples: {len(all_histories)}")
    print(f"Training history samples: {len(train_histories)}")
    print(f"Validation history samples: {len(val_histories)}")

    encoder = create_encoder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    print(f"Starting contrastive training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_generator = contrastive_data_generator(
            train_histories, BATCH_SIZE)
        val_generator = contrastive_data_generator(val_histories, BATCH_SIZE)

        total_train_loss = 0
        num_train_batches = len(train_histories) // BATCH_SIZE

        for i in tqdm(range(num_train_batches), desc=f"Epoch {epoch+1}/{EPOCHS} Training"):
            anchor_hist, positive_hist, negative_hists = next(train_generator)

            with tf.GradientTape() as tape:
                anchor_emb = encoder(anchor_hist, training=True)
                positive_emb = encoder(positive_hist, training=True)

                reshaped_neg_hists = tf.reshape(
                    negative_hists, (-1, HISTORY_LEN, FEATURES))
                negative_embs_flat = encoder(reshaped_neg_hists, training=True)

                negative_embs = tf.reshape(
                    negative_embs_flat, (BATCH_SIZE, NUM_NEGATIVES, EMBEDDING_DIM))

                all_embs = tf.concat(
                    [tf.expand_dims(positive_emb, 1), negative_embs], axis=1)

                loss = info_nce_loss(anchor_emb, all_embs,
                                     temperature=TEMPERATURE)

            gradients = tape.gradient(loss, encoder.trainable_weights)
            optimizer.apply_gradients(
                zip(gradients, encoder.trainable_weights))
            total_train_loss += loss.numpy()

        avg_train_loss = total_train_loss / num_train_batches

        total_val_loss = 0
        num_val_batches = len(val_histories) // BATCH_SIZE
        if num_val_batches > 0:
            for i in range(num_val_batches):
                anchor_hist_val, positive_hist_val, negative_hists_val = next(
                    val_generator)
                anchor_emb_val = encoder(anchor_hist_val, training=False)
                positive_emb_val = encoder(positive_hist_val, training=False)
                reshaped_neg_hists_val = tf.reshape(
                    negative_hists_val, (-1, HISTORY_LEN, FEATURES))
                negative_embs_flat_val = encoder(
                    reshaped_neg_hists_val, training=False)
                negative_embs_val = tf.reshape(
                    negative_embs_flat_val, (BATCH_SIZE, NUM_NEGATIVES, EMBEDDING_DIM))
                all_embs_val = tf.concat(
                    [tf.expand_dims(positive_emb_val, 1), negative_embs_val], axis=1)
                val_loss = info_nce_loss(
                    anchor_emb_val, all_embs_val, temperature=TEMPERATURE)
                total_val_loss += val_loss.numpy()
            avg_val_loss = total_val_loss / num_val_batches
            print(
                f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

    encoder.save_weights(ENCODER_SAVE_PATH)
    print(f"Pre-trained encoder weights saved to {ENCODER_SAVE_PATH}")
