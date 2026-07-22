"""Topology-matched adapted DeepRMSA K=50 hops for SA-HMARL fixed-C / all-OD.

This agent keeps DeepRMSA's 5-layer 128-unit ELU MLP backbone and A2C training
algorithm, but adapts the output head to the unified RMSA action space of
K_path * |M| * max_blocks = 50 * 4 * 10 = 2000 actions.  It applies the same
physical R mask as PPO-R and evaluates with masked argmax.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from sa_hmarl.agents.deep_rmsa_agent import ACNetwork, DeepRMSAAgent
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.network.modulation import ModulationRegistry


class DeepRMSAAdaptedK50Agent(DeepRMSAAgent):
    """Adapted DeepRMSA with 2000-dim output and PPO-R-compatible R mask.

    Changes from the original DeepRMSAAgent:
    - ``n_actions`` = k_path * num_modulations * m_blocks (2000 for COST239 K=50).
    - Policy head output dimension is ``n_actions``.
    - Action mask is the SA-HMARL ``agent_r_mask`` over all 2000 flat actions.
    - State encoding uses the original DeepRMSA style but with k_path=50 and
      m_blocks=10 per best-modulation path.
    - Action decoding returns the SA-HMARL flat action index directly.
    """

    def __init__(self,
                 num_nodes: int,
                 num_slots: int,
                 k_path: int = 50,
                 num_modulations: int = 4,
                 m_blocks: int = 10,
                 mod_registry: ModulationRegistry = None,
                 gamma: float = 0.95,
                 lr: float = 1e-5,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 40.0,
                 num_layers: int = 5,
                 layer_size: int = 128,
                 device: str = 'cpu'):
        # Store the extra param before calling super, which will set n_actions.
        self.num_modulations = num_modulations
        self.mod_registry = mod_registry
        # Note: we do NOT call super().__init__ because its n_actions and
        # state_dim formula are for the original (k_path * m_blocks) action
        # space.  Instead we replicate the initialization below with the new
        # action space.
        self.num_nodes = num_nodes
        self.num_slots = num_slots
        self.k_path = k_path
        self.m_blocks = m_blocks
        self.n_actions = k_path * num_modulations * m_blocks
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.step_count = 0
        self.training = True
        self.episode_buffer = []

        # State dimension matches the original DeepRMSA feature layout but
        # scaled to k_path paths and m_blocks blocks per path.
        per_path_dim = 1 + m_blocks * 2 + 2
        self.state_dim = num_nodes * 2 + k_path * per_path_dim

        self.policy_net = ACNetwork(self.state_dim, self.n_actions,
                                    num_layers, layer_size).to(device)
        self.value_net = ACNetwork(self.state_dim, 1,
                                   num_layers, layer_size).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr,
        )

    def _best_mod_for_path(self, obs: Dict[str, Any], path_idx: int) -> Optional[int]:
        """Return highest-SE feasible modulation index for a path."""
        if self.mod_registry is None:
            return None
        num_mods = len(obs["mod_names"])
        best_mod_idx = None
        best_se = -1.0
        for m_idx in range(num_mods):
            if obs["feasible_mask_per_path_mod"][path_idx][m_idx]:
                if m_idx < self.mod_registry.num_formats:
                    se = self.mod_registry[m_idx].spectral_efficiency
                else:
                    se = 1.0
                if se > best_se:
                    best_se = se
                    best_mod_idx = m_idx
        return best_mod_idx

    def encode_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Convert SA-HMARL Agent-R observation to DeepRMSA-style state vector.

        Layout: src/dst one-hot + per-path features for up to k_path paths.
        Per-path features (for the best feasible modulation on that path):
          - required FS (normalized)
          - first m_blocks (start, size) pairs (normalized)
          - total available FS (normalized)
          - mean block size (normalized)
        """
        src = int(obs.get("src_node", 0))
        dst = int(obs.get("dst_node", 0))
        num_paths = len(obs["candidate_paths"])

        src_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        dst_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        if 0 <= src < self.num_nodes:
            src_onehot[src] = 1.0
        if 0 <= dst < self.num_nodes:
            dst_onehot[dst] = 1.0

        features: list = src_onehot.tolist() + dst_onehot.tolist()
        per_path_dim = 1 + self.m_blocks * 2 + 2

        for p_idx in range(min(num_paths, self.k_path)):
            best_mod_idx = self._best_mod_for_path(obs, p_idx)
            if best_mod_idx is None:
                features.extend([-1.0] * per_path_dim)
                continue

            req_fs = obs["required_fs_per_path_mod"][p_idx][best_mod_idx]
            if req_fs is None:
                req_fs = 0
            blocks = obs["candidate_blocks_per_path_mod"][p_idx][best_mod_idx]

            # Required FS count (normalized, approximate range 1-16)
            features.append((float(req_fs) - 8.5) / 7.5)

            # First m_blocks (start, size) pairs
            for b_idx in range(self.m_blocks):
                if b_idx < len(blocks):
                    start_idx, size = blocks[b_idx]
                    features.append(2.0 * (float(start_idx) - 0.5 * self.num_slots) / self.num_slots)
                    features.append((float(size) - self.num_slots / 2) / (self.num_slots / 2))
                else:
                    features.extend([-1.0, -1.0])

            # Total available FS among candidate blocks
            total_avail = sum(float(size) for _, size in blocks)
            features.append(2.0 * (total_avail - 0.5 * self.num_slots) / self.num_slots)

            # Mean block size
            avg_size = np.mean([float(size) for _, size in blocks]) if blocks else 0.0
            features.append((avg_size - self.num_slots / 4) / (self.num_slots / 4))

        # Pad to fixed state_dim if fewer paths than k_path
        expected_paths = self.k_path * per_path_dim
        actual_paths = len(features) - self.num_nodes * 2
        if actual_paths < expected_paths:
            features.extend([-1.0] * (expected_paths - actual_paths))

        return np.asarray(features, dtype=np.float32)

    def _build_action_mask(self, obs: Dict[str, Any]) -> np.ndarray:
        """Return the SA-HMARL physical R mask over all 2000 flat actions."""
        mask = np.asarray(obs["agent_r_mask"], dtype=bool)
        if mask.size != self.n_actions:
            # If the environment produced a different size mask, fail closed.
            return np.zeros(self.n_actions, dtype=bool)
        return mask

    def select_action(self, obs: Dict[str, Any]) -> Optional[int]:
        """Select a masked-greedy action and return SA-HMARL flat index."""
        state = self.encode_state(obs)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy_net(state_t)
            value = self.value_net(state_t).cpu().item()

        action_mask = self._build_action_mask(obs)
        if not np.any(action_mask):
            self._last_valid = False
            return None

        logits_np = logits.cpu().numpy().flatten()
        masked_logits = np.full_like(logits_np, -np.inf)
        masked_logits[action_mask] = logits_np[action_mask]

        max_logit = np.max(masked_logits)
        exp_logits = np.exp(masked_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        if self.training:
            action_id = int(np.random.choice(self.n_actions, p=probs))
        else:
            action_id = int(np.argmax(probs))

        self._last_valid = True
        self._last_state = state
        self._last_action_id = action_id
        self._last_value = value
        self._last_action_mask = action_mask

        # action_id is already the SA-HMARL flat index
        return int(action_id)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.value_net.parameters() if p.requires_grad)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["num_modulations"] = self.num_modulations
        state["implementation"] = "topology_matched_adapted_deeprmsa_k50"
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        self.num_modulations = state_dict.get("num_modulations", self.num_modulations)
