"""Source-semantic PyTorch port of the upstream DeepRMSA policy.

This class deliberately restores the upstream action semantics: every policy
output remains selectable, and selecting an unavailable path/block produces a
blocked request.  It is separate from :mod:`deep_rmsa_agent`, whose legal-action
mask is an optimization not present in the upstream implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent


class DeepRMSASourceSemanticAgent(DeepRMSAAgent):
    """DeepRMSA state/action semantics with a local PyTorch A2C trainer.

    The topology and candidate-path count may be changed by the evaluation
    protocol, but the upstream M-block action abstraction, feature scaling,
    unmasked policy, and invalid-action blocking behavior are retained.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_like_upstream()

    def _initialize_like_upstream(self) -> None:
        """Match the initializers in upstream ``AC_Net.py``."""
        for network, head_std in ((self.policy_net, 0.01), (self.value_net, 1.0)):
            for module in network.backbone:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.3)
                    nn.init.constant_(module.bias, 0.1)
            with torch.no_grad():
                weight = torch.randn_like(network.head.weight)
                norms = torch.sqrt(torch.sum(weight.square(), dim=1, keepdim=True))
                network.head.weight.copy_(weight * (head_std / norms.clamp_min(1e-12)))
                network.head.bias.zero_()

    def encode_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Encode the state using the normalization in upstream DeepRMSA."""
        src = int(obs.get("src_node", 0))
        dst = int(obs.get("dst_node", 0))
        num_paths = len(obs["candidate_paths"])

        src_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        dst_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        if 0 <= src < self.num_nodes:
            src_onehot[src] = 1.0
        if 0 <= dst < self.num_nodes:
            dst_onehot[dst] = 1.0

        features: List[float] = src_onehot.tolist() + dst_onehot.tolist()
        per_path_dim = 1 + self.m_blocks * 2 + 2
        for path_idx in range(num_paths):
            mod_idx = self._best_mod_for_path(obs, path_idx)
            if mod_idx is None:
                features.extend([-1.0] * per_path_dim)
                continue

            req_fs = obs["required_fs_per_path_mod"][path_idx][mod_idx]
            blocks = obs["candidate_blocks_per_path_mod"][path_idx][mod_idx]
            # Upstream marks the entire path feature segment unavailable when
            # it has no contiguous block large enough for the request.
            if req_fs is None or not blocks:
                features.extend([-1.0] * per_path_dim)
                continue

            features.append((float(req_fs) - 5.5) / 3.5)
            for block_idx in range(self.m_blocks):
                if block_idx < len(blocks):
                    start, size = blocks[block_idx]
                    features.append(2.0 * (float(start) - 0.5 * self.num_slots) / self.num_slots)
                    features.append((float(size) - 8.0) / 8.0)
                else:
                    features.extend([-1.0, -1.0])
            sizes = [float(size) for _, size in blocks]
            features.append(2.0 * (sum(sizes) - 0.5 * self.num_slots) / self.num_slots)
            features.append((float(np.mean(sizes)) - 4.0) / 4.0)

        if num_paths < self.k_path:
            features.extend([-1.0] * ((self.k_path - num_paths) * per_path_dim))
        state = np.asarray(features, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"DeepRMSA state shape {state.shape} != {(self.state_dim,)}")
        return state

    def select_action(self, obs: Dict[str, Any]) -> Optional[int]:
        """Sample/argmax over all outputs, without a legal-action mask."""
        state = self.encode_state(obs)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_t).squeeze(0)
            value = float(self.value_net(state_t).cpu().item())
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        if self.training:
            action_id = int(np.random.choice(self.n_actions, p=probs))
        else:
            action_id = int(np.argmax(probs))

        # A failed decode is still a policy decision. The caller must count it
        # as blocking and store reward -1 during training.
        self._last_valid = True
        self._last_state = state
        self._last_action_id = action_id
        self._last_value = value
        self._last_action_mask = np.ones(self.n_actions, dtype=bool)
        return self._decode_to_sahmarl(action_id, obs)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["action_semantics"] = "upstream_unmasked_invalid_action_blocks"
        state["feature_semantics"] = "upstream_deeprmsa_model1"
        return state
