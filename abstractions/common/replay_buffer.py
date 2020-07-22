import numpy as np
import random
from collections import namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def batchify(experiences):
    """Convert a list of experiences to a tuple of np.arrays"""
    return tuple(map(np.asarray, list(zip(*experiences))))

# Adapted from OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def append(self, experience):
        if self._next_idx >= len(self._storage):
            self._storage.append(experience)
        else:
            self._storage[self._next_idx] = experience
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size, wrap=True):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        wrap: bool
            Whether to wrap the batch in an Experience namedtuple

        Returns
        -------
        batch: (Experience or tuple)
            If wrap==True, an Experience namedtuple whose elements are np.arrays,
            otherwise, a primitive tuple whose elements are np.arrays
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        experiences = [self._storage[i] for i in idxes]
        batch = batchify(experiences)
        if wrap:
            return Experience(*batch)
        else:
            return batch
