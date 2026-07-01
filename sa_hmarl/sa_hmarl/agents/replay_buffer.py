"""Minimal replay buffer for DQN."""
import numpy as np
from typing import Optional, Tuple, List


class ReplayBuffer:
    """Fixed-capacity replay buffer for off-policy RL.

    Stores transitions of the form:
        (obs_features, mask, action_idx, reward,
         next_obs_features, next_mask, done)

    Each obs_features is (num_actions, input_dim) and may vary in num_actions
    across transitions.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.pos = 0

    def push(self,
             obs_features: np.ndarray,
             mask: np.ndarray,
             action_idx: int,
             reward: float,
             next_obs_features: np.ndarray,
             next_mask: np.ndarray,
             done: bool):
        """Store a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (
            obs_features, mask, action_idx, reward,
            next_obs_features, next_mask, done,
        )
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Optional[Tuple]:
        """Sample a batch of transitions.

        Returns None if buffer has fewer than batch_size entries.
        """
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return tuple(zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)
