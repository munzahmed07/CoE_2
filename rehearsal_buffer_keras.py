# src/online_learning_keras/rehearsal_buffer_keras.py
import random
from collections import deque
import numpy as np

class RehearsalBufferKeras:
    """A buffer to store and sample past experiences for Keras models."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, history, future):
        """Add a new experience (history and future arrays) to the buffer."""
        self.buffer.append((history, future))

    def sample(self, batch_size):
        """Sample a batch of experiences and return them as NumPy arrays."""
        samples = random.sample(self.buffer, batch_size)
        histories, futures = zip(*samples)
        return np.array(histories), np.array(futures)

    def __len__(self):
        return len(self.buffer)