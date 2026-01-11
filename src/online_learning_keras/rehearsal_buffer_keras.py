import random


class RehearsalBuffer:
    """
    A simple rehearsal buffer to store past experiences (data samples)
    for continual learning.
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.ptr = 0

    def add(self, experience):
        """
        Adds a new experience to the buffer.
        If the buffer is full, it replaces the oldest experience.
        """
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience

        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        """
        Samples a random batch of experiences from the buffer.
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def size(self):
        """
        Returns the current number of experiences in the buffer.
        """
        return len(self.buffer)
