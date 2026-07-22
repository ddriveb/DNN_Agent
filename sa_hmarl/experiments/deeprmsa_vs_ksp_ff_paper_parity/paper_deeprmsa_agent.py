"""Upstream-exact DeepRMSA agent adapter for the paper-parity pipeline.

``PaperDeepRMSAAgent`` subclasses the audited source-semantic port and restores
one upstream state-encoding detail: the "total available FS" and "mean FS-block
size" features are computed over **all** contiguous free blocks on the path
(upstream ``slotscontinue`` from ``mark_vector``), while the first-M block
features use only blocks with ``size >= required_FS``.  Unmasked action
semantics, initializers, and the optimize() path are inherited unchanged.

The observation dict is built by :func:`build_agent_obs` from the paper core's
request view, so no SA-HMARL environment machinery is involved.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sa_hmarl"))

from sa_hmarl.agents.deep_rmsa_source_semantic_agent import DeepRMSASourceSemanticAgent
from sa_hmarl.network.modulation import ModulationRegistry

from paper_rmsa_core import required_fs


DEFAULT_MAX_BLOCKS = 10


class PaperDeepRMSAAgent(DeepRMSASourceSemanticAgent):
    """Source-semantic DeepRMSA with exact upstream block aggregation.

    ``optimize`` is overridden to optionally add per-window advantage
    normalization (``normalize_advantages=True``) — a standard A2C
    variance-reduction step (disclosed deviation D7).  With raw ±1 rewards and
    γ=0.95 the window returns span roughly ±8, which at lr=1e-4 produces
    seed-dependent divergence/stall.  The value loss always uses raw returns;
    all other upstream training semantics are unchanged.
    """

    def __init__(self, *args, normalize_advantages: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_advantages = normalize_advantages

    def optimize(self, bootstrap_value: Optional[float] = None) -> Optional[Dict[str, float]]:
        if len(self.episode_buffer) == 0:
            return None

        states = np.stack([t[0] for t in self.episode_buffer])
        actions = np.array([t[1] for t in self.episode_buffer], dtype=np.int64)
        rewards = np.array([t[2] for t in self.episode_buffer], dtype=np.float32)
        values = np.array([t[3] for t in self.episode_buffer], dtype=np.float32)
        action_masks = np.stack([t[4] for t in self.episode_buffer])
        dones = np.array([t[5] for t in self.episode_buffer], dtype=np.float32)

        if bootstrap_value is None:
            if dones[-1] > 0.5:
                bootstrap_value = 0.0
            else:
                last_state = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    bootstrap_value = self.value_net(last_state).cpu().item()

        returns = self._compute_returns_with_dones(rewards, dones, bootstrap_value)
        advantages = returns - values
        if self.normalize_advantages:
            # Per-window advantage normalization (deviation D7, variance reduction).
            adv_std = float(advantages.std())
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        action_masks_t = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

        logits = self.policy_net(states_t)
        masked_logits = logits.masked_fill(~action_masks_t, -1e9)

        log_probs = F.log_softmax(masked_logits, dim=-1)
        log_probs_actions = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        probs = F.softmax(masked_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        policy_loss = -(log_probs_actions * advantages_t).mean()
        policy_loss = policy_loss - self.entropy_coef * entropy

        values_pred = self.value_net(states_t).squeeze(-1)
        value_loss = F.mse_loss(values_pred, returns_t)

        loss = policy_loss + self.value_loss_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()
        self.step_count += 1

        self.clear_buffer()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def encode_state(self, obs: Dict[str, Any]) -> np.ndarray:
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
            # Upstream: path segment unavailable when no contiguous block is
            # large enough for the request (flag == 0 or max < num_FS).
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
            # Upstream aggregates use ALL contiguous free blocks on the path.
            all_blocks = obs["all_free_blocks_per_path"][path_idx]
            sizes = [float(size) for _, size in all_blocks]
            features.append(2.0 * (sum(sizes) - 0.5 * self.num_slots) / self.num_slots)
            features.append((float(np.mean(sizes)) - 4.0) / 4.0)

        if num_paths < self.k_path:
            features.extend([-1.0] * ((self.k_path - num_paths) * per_path_dim))
        state = np.asarray(features, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"DeepRMSA state shape {state.shape} != {(self.state_dim,)}")
        return state

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["feature_semantics"] = "upstream_deeprmsa_model1_exact_block_aggregation"
        return state


def build_agent_obs(
    view: Dict[str, Any],
    mod_reg: ModulationRegistry,
    num_slots: int = 100,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    slot_bw_hz: float = 12.5e9,
    guard_band_fs: int = 1,
) -> Dict[str, Any]:
    """Convert a paper-core request view to the agent's observation dict."""
    num_paths = len(view["paths"])
    num_mods = mod_reg.num_formats
    bitrate = int(view["bitrate_gbps"])

    candidate_paths: List[List[int]] = []
    feasible_mask: List[List[bool]] = []
    required_fs_pm: List[List[Optional[int]]] = []
    blocks_pm: List[List[List[Tuple[int, int]]]] = []
    all_free_per_path: List[List[Tuple[int, int]]] = []

    for path_info in view["paths"]:
        candidate_paths.append(list(path_info["path"]))
        path_len = float(path_info["path_len_km"])
        all_blocks = list(path_info["all_free_blocks"])
        all_free_per_path.append(all_blocks)
        feas_row: List[bool] = []
        fs_row: List[Optional[int]] = []
        blk_row: List[List[Tuple[int, int]]] = []
        for m_idx in range(num_mods):
            mod = mod_reg[m_idx]
            if path_len <= mod.reach_km:
                fs = required_fs(bitrate, mod.spectral_efficiency, slot_bw_hz, guard_band_fs)
                elig = [b for b in all_blocks if b[1] >= fs][:max_blocks]
                feas_row.append(True)
                fs_row.append(fs)
                blk_row.append(elig)
            else:
                feas_row.append(False)
                fs_row.append(None)
                blk_row.append([])
        feasible_mask.append(feas_row)
        required_fs_pm.append(fs_row)
        blocks_pm.append(blk_row)

    mask: List[bool] = []
    for p_idx in range(num_paths):
        for m_idx in range(num_mods):
            for b_idx in range(max_blocks):
                mask.append(
                    feasible_mask[p_idx][m_idx] and b_idx < len(blocks_pm[p_idx][m_idx])
                )

    return {
        "src_node": int(view["src_node"]),
        "dst_node": int(view["dst_node"]),
        "num_slots": int(num_slots),
        "candidate_paths": candidate_paths,
        "mod_names": list(mod_reg.names),
        "feasible_mask_per_path_mod": feasible_mask,
        "required_fs_per_path_mod": required_fs_pm,
        "candidate_blocks_per_path_mod": blocks_pm,
        "all_free_blocks_per_path": all_free_per_path,
        "agent_r_mask": np.asarray(mask, dtype=bool),
        "request_id": int(view["req_id"]),
    }


def decode_flat_action(
    flat_action: int, num_mods: int, max_blocks: int = DEFAULT_MAX_BLOCKS
) -> Tuple[int, int, int]:
    """Decode flat action to (path_idx, mod_idx, block_idx)."""
    path_idx = flat_action // (num_mods * max_blocks)
    rem = flat_action % (num_mods * max_blocks)
    return path_idx, rem // max_blocks, rem % max_blocks


def make_deeprmsa_decide(agent: PaperDeepRMSAAgent, mod_reg: ModulationRegistry,
                         max_blocks: int = DEFAULT_MAX_BLOCKS):
    """Decision closure for paper_rmsa_core.run_simulation (evaluation)."""
    num_mods = mod_reg.num_formats

    def decide(view, env, req):
        obs = build_agent_obs(
            view, mod_reg, num_slots=env.num_slots, max_blocks=max_blocks,
            slot_bw_hz=env.slot_bw_hz, guard_band_fs=env.guard_band_fs,
        )
        flat = agent.select_action(obs)
        if flat is None:
            return None
        path_idx, _mod_idx, _block_idx = decode_flat_action(flat, num_mods, max_blocks)
        if path_idx >= len(view["paths"]):
            return None
        return path_idx, view["paths"][path_idx]["first_fit_start"]

    return decide
