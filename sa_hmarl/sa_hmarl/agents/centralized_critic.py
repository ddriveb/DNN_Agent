"""Centralized critic for CTDE-style joint Agent-C / Agent-R training.

The critic is used only during training. It receives compact summaries of both
agents' local action spaces plus the selected high-level and low-level action
features, and estimates a joint Q value for the event-level transition.

Execution remains decentralized/hierarchical:
    Agent-C selects (split, server), then Agent-R selects (path, mod, block).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sa_hmarl.agents.joint_replay_buffer import JointTransition


class CentralizedCriticNetwork(nn.Module):
    """MLP mapping compact joint state-action features to scalar Q."""

    def __init__(self, input_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CentralizedCritic:
    """Joint critic trained with global event reward.

    Input vector layout:
        mean_valid_c_features || selected_c_features ||
        mean_valid_r_features || selected_r_features

    This keeps the critic independent of variable action-space sizes while
    still exposing it to both agents' local context and chosen actions.
    """

    def __init__(
        self,
        c_dim: int = 17,
        r_dim: int = 11,
        hidden_dims=(256, 128),
        gamma: float = 0.95,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.c_dim = c_dim
        self.r_dim = r_dim
        self.input_dim = 2 * c_dim + 2 * r_dim
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = gamma
        self.device = device
        self.step_count = 0

        self.q_net = CentralizedCriticNetwork(self.input_dim, self.hidden_dims).to(device)
        self.target_net = CentralizedCriticNetwork(self.input_dim, self.hidden_dims).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    @staticmethod
    def _valid_mean(features: np.ndarray, mask: np.ndarray, feature_dim: int) -> np.ndarray:
        if features.size == 0:
            return np.zeros((feature_dim,), dtype=np.float32)
        if len(mask) == len(features) and np.any(mask):
            return features[mask].mean(axis=0).astype(np.float32)
        return features.mean(axis=0).astype(np.float32)

    @staticmethod
    def _selected(features: np.ndarray, action_idx: int, feature_dim: int) -> np.ndarray:
        if features.size == 0:
            return np.zeros((feature_dim,), dtype=np.float32)
        if action_idx < 0 or action_idx >= len(features):
            action_idx = 0
        return features[action_idx].astype(np.float32)

    @staticmethod
    def _greedy_action(agent, features: np.ndarray, mask: np.ndarray) -> Optional[int]:
        if len(mask) == 0 or not np.any(mask) or features.size == 0:
            return None
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=agent.device).unsqueeze(0)
            q = agent.target_net(x).squeeze(0).cpu().numpy()
            q[~mask] = -np.inf
            return int(np.argmax(q))

    def _encode(
        self,
        c_features: np.ndarray,
        c_mask: np.ndarray,
        c_action: int,
        r_features: np.ndarray,
        r_mask: np.ndarray,
        r_action: int,
    ) -> np.ndarray:
        c_mean = self._valid_mean(c_features, c_mask, self.c_dim)
        r_mean = self._valid_mean(r_features, r_mask, self.r_dim)
        c_sel = self._selected(c_features, c_action, self.c_dim)
        r_sel = self._selected(r_features, r_action, self.r_dim)
        return np.concatenate([c_mean, c_sel, r_mean, r_sel]).astype(np.float32)

    def _encode_current_batch(self, batch: List[JointTransition]) -> np.ndarray:
        return np.stack([
            self._encode(
                t.obs_c_features, t.mask_c, t.action_c_idx,
                t.obs_r_features, t.mask_r, t.action_r_idx,
            )
            for t in batch
        ])

    def _encode_next_batch(self, batch: List[JointTransition], agent_c, agent_r) -> np.ndarray:
        encoded = []
        for t in batch:
            if t.done:
                encoded.append(np.zeros(self.input_dim, dtype=np.float32))
                continue
            next_c_action = self._greedy_action(agent_c, t.next_obs_c_features, t.next_mask_c)
            next_r_action = self._greedy_action(agent_r, t.next_obs_r_features, t.next_mask_r)
            if next_c_action is None or next_r_action is None:
                encoded.append(np.zeros(self.input_dim, dtype=np.float32))
                continue
            encoded.append(
                self._encode(
                    t.next_obs_c_features, t.next_mask_c, next_c_action,
                    t.next_obs_r_features, t.next_mask_r, next_r_action,
                )
            )
        return np.stack(encoded)

    def optimize(self, batch: Optional[List[JointTransition]], agent_c, agent_r) -> Optional[float]:
        """Update centralized critic using global reward TD target."""
        if batch is None:
            return None

        current_np = self._encode_current_batch(batch)
        next_np = self._encode_next_batch(batch, agent_c, agent_r)
        rewards = np.array([t.reward_global for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        current_t = torch.tensor(current_np, dtype=torch.float32, device=self.device)
        next_t = torch.tensor(next_np, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.q_net(current_t)
        with torch.no_grad():
            next_q = self.target_net(next_t)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        return float(loss.item())

    def joint_q(self, batch: List[JointTransition]) -> torch.Tensor:
        """Return detached centralized Q for current transitions."""
        current_np = self._encode_current_batch(batch)
        current_t = torch.tensor(current_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.q_net(current_t).detach()

    def distill_agents(
        self,
        batch: Optional[List[JointTransition]],
        agent_c,
        agent_r,
        coef: float,
    ) -> Optional[float]:
        """Regularize decentralized Q estimates toward centralized joint Q."""
        if batch is None or coef <= 0:
            return None

        target_joint_q = self.joint_q(batch)

        loss_c = self._distill_one_agent(
            agent_c,
            tuple(t.obs_c_features for t in batch),
            tuple(t.mask_c for t in batch),
            tuple(t.action_c_idx for t in batch),
            target_joint_q,
        )
        loss_r = self._distill_one_agent(
            agent_r,
            tuple(t.obs_r_features for t in batch),
            tuple(t.mask_r for t in batch),
            tuple(t.action_r_idx for t in batch),
            target_joint_q,
        )

        agent_c.optimizer.zero_grad()
        (coef * loss_c).backward()
        agent_c.optimizer.step()

        agent_r.optimizer.zero_grad()
        (coef * loss_r).backward()
        agent_r.optimizer.step()

        return float((loss_c + loss_r).item())

    @staticmethod
    def _distill_one_agent(agent, features_list, masks_list, actions, target_q):
        max_actions = max(len(m) for m in masks_list)
        padded = []
        for features, mask in zip(features_list, masks_list):
            pad_len = max_actions - len(mask)
            if pad_len > 0:
                features = np.concatenate([
                    features,
                    np.zeros((pad_len, features.shape[1]), dtype=np.float32),
                ], axis=0)
            padded.append(features)

        obs_t = torch.tensor(np.stack(padded), dtype=torch.float32, device=agent.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=agent.device)
        q_all = agent.q_net(obs_t)
        q_selected = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        return F.smooth_l1_loss(q_selected, target_q.to(agent.device))

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
