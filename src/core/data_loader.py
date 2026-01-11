import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LyftTrajectoryDataset:
    """
    A custom class to handle loading of preprocessed Lyft trajectory data.
    """

    def __init__(self, processed_dir="data/processed", shuffle=True, subset_indices=None):
        self.processed_dir = processed_dir
        self.file_paths = []
        for root, _, files in os.walk(processed_dir):
            for file in files:
                if file.endswith(".npz"):
                    self.file_paths.append(os.path.join(root, file))

        if subset_indices is not None:
            # Filter file paths based on the provided indices
            self.file_paths = [self.file_paths[i]
                               for i in subset_indices if i < len(self.file_paths)]

        if shuffle:
            np.random.shuffle(self.file_paths)

        print(
            f"Initialized dataset with {len(self.file_paths)} samples from {processed_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with np.load(file_path) as data:
            history = data['focal_history'].astype(np.float32)
            future = data['focal_future'].astype(np.float32)
        return {'focal_history': history, 'focal_future': future}

    def create_tf_dataset(self, batch_size, prefetch_buffer=tf.data.AUTOTUNE, shuffle=True):
        """Creates an efficient TensorFlow tf.data.Dataset for training."""

        # Use a generator that can be re-shuffled each epoch if needed
        def generator():
            indices = np.arange(len(self.file_paths))
            if shuffle:
                np.random.shuffle(indices)
            for i in indices:
                sample = self[i]
                yield sample['focal_history'], sample['focal_future']

        sample_data = self[0]
        output_types = (tf.float32, tf.float32)
        output_shapes = (
            tf.TensorShape(sample_data['focal_history'].shape),
            tf.TensorShape(sample_data['focal_future'].shape)
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes
        )

        dataset = dataset.batch(batch_size).prefetch(prefetch_buffer)

        return dataset

# --- THIS IS THE MISSING FUNCTION ---


def load_all_data_as_numpy(dataset_path, limit=None):
    """
    Helper function to load all processed .npz files from a directory
    into two large NumPy arrays. Useful for scripts that need all data in memory.
    """
    histories, futures = [], []
    file_paths = [os.path.join(dataset_path, f)
                  for f in os.listdir(dataset_path) if f.endswith(".npz")]

    if limit:
        file_paths = file_paths[:limit]

    for file_path in tqdm(file_paths, desc="Loading all data into memory"):
        with np.load(file_path) as data:
            histories.append(data['focal_history'])
            futures.append(data['focal_future'])

    return np.array(histories, dtype=np.float32), np.array(futures, dtype=np.float32)
