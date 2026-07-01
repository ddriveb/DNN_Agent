"""Joint replay buffer for cooperative Agent-C / Agent-R training.

This buffer is the first shared interface for true hierarchical co-training:
one transition stores both the high-level split/server decision and the
low-level RMSA decision produced for the same request event.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class JointTransition:
    """One event-level transition containing both agents' learning views."""

    obs_c_features: np.ndarray
    mask_c: np.ndarray
    action_c_idx: int
    obs_r_features: np.ndarray
    mask_r: np.ndarray
    action_r_idx: int
    reward_global: float
    reward_c: float
    reward_r: float
    next_obs_c_features: np.ndarray
    next_mask_c: np.ndarray
    next_obs_r_features: np.ndarray
    next_mask_r: np.ndarray
    done: bool
    info: Dict[str, Any]


class JointReplayBuffer:
    """Fixed-capacity replay buffer for hierarchical dual-agent DQN.

    The buffer keeps the complete joint transition, then exposes agent-specific
    sampled batches matching the existing AgentC/AgentR.optimize() interface:

        (obs_features, mask, action_idx, reward,
         next_obs_features, next_mask, done)
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer: List[Optional[JointTransition]] = []
        self.pos = 0
        self.rng = np.random.RandomState(seed)

    def push(self,
             obs_c_features: np.ndarray,
             mask_c: np.ndarray,
             action_c_idx: int,
             obs_r_features: np.ndarray,
             mask_r: np.ndarray,
             action_r_idx: int,
             reward_global: float,
             reward_c: float,
             reward_r: float,
             next_obs_c_features: np.ndarray,
             next_mask_c: np.ndarray,
             next_obs_r_features: np.ndarray,
             next_mask_r: np.ndarray,
             done: bool,
             info: Optional[Dict[str, Any]] = None):
        """Store one complete joint transition.

        Arrays are copied on insertion so later observation mutations do not
        silently corrupt replay memory.
        """
        transition = JointTransition(
            obs_c_features=np.array(obs_c_features, dtype=np.float32, copy=True),
            mask_c=np.array(mask_c, dtype=bool, copy=True),
            action_c_idx=int(action_c_idx),
            obs_r_features=np.array(obs_r_features, dtype=np.float32, copy=True),
            mask_r=np.array(mask_r, dtype=bool, copy=True),
            action_r_idx=int(action_r_idx),
            reward_global=float(reward_global),
            reward_c=float(reward_c),
            reward_r=float(reward_r),
            next_obs_c_features=np.array(next_obs_c_features, dtype=np.float32, copy=True),
            next_mask_c=np.array(next_mask_c, dtype=bool, copy=True),
            next_obs_r_features=np.array(next_obs_r_features, dtype=np.float32, copy=True),
            next_mask_r=np.array(next_mask_r, dtype=bool, copy=True),
            done=bool(done),
            info=dict(info or {}),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Optional[List[JointTransition]]:
        """Sample complete joint transitions.

        Returns None if the buffer has fewer than batch_size entries.
        """
        valid = [t for t in self.buffer if t is not None]
        if len(valid) < batch_size:
            return None
        indices = self.rng.choice(len(valid), batch_size, replace=False)
        return [valid[int(i)] for i in indices]

    def sample_agent_c(self, batch_size: int) -> Optional[Tuple]:
        """Sample Agent-C view of joint transitions."""
        batch = self.sample(batch_size)
        if batch is None:
            return None
        return (
            tuple(t.obs_c_features for t in batch),
            tuple(t.mask_c for t in batch),
            tuple(t.action_c_idx for t in batch),
            tuple(t.reward_c for t in batch),
            tuple(t.next_obs_c_features for t in batch),
            tuple(t.next_mask_c for t in batch),
            tuple(t.done for t in batch),
        )

    def sample_agent_r(self, batch_size: int) -> Optional[Tuple]:
        """Sample Agent-R view of joint transitions."""
        batch = self.sample(batch_size)
        if batch is None:
            return None
        return (
            tuple(t.obs_r_features for t in batch),
            tuple(t.mask_r for t in batch),
            tuple(t.action_r_idx for t in batch),
            tuple(t.reward_r for t in batch),
            tuple(t.next_obs_r_features for t in batch),
            tuple(t.next_mask_r for t in batch),
            tuple(t.done for t in batch),
        )

    def __len__(self) -> int:
        return len(self.buffer)
