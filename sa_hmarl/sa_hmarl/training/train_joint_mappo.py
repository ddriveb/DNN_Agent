"""MAPPO-style hierarchical multi-agent training.

This is the first true PPO/actor-critic training path for SA-HMARL:

* Agent-C and Agent-R are masked categorical PPO actors.
* A centralized value critic is used only during training.
* Execution remains hierarchical and decentralized.

The implementation is intentionally conservative and side-by-side with the
existing DQN/CTDE code so prior results remain reproducible.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sa_hmarl.agents.ppo_agents import (
    GatedMaskedPPOActorNetwork,
    MaskedPPOActorNetwork,
    PPOAgentC,
    PPOAgentR,
)
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_joint_alternating import (
    _build_eval_set,
    _get_reference_mod_name,
    _load_frozen_reference,
)
from sa_hmarl.training.utils import (
    compute_agent_c_reward,
    compute_reward,
    generate_requests,
    make_env,
)
from sa_hmarl.utils.checkpoint import load_checkpoint


@dataclass
class PPOJointTransition:
    c_features: np.ndarray
    c_mask: np.ndarray
    c_action: int
    c_log_prob: float
    c_valid: bool
    r_features: np.ndarray
    r_mask: np.ndarray
    r_action: int
    r_log_prob: float
    r_valid: bool
    value_features: np.ndarray
    value: float
    reward: float
    done: bool
    info: Dict


class PPORolloutBuffer:
    """Episode rollout storage with GAE support.

    The project has variable-length masked action spaces, so we keep per-step
    feature arrays as Python objects instead of forcing a fixed tensor upfront.
    """

    def __init__(self):
        self.transitions: List[PPOJointTransition] = []

    def add(self, transition: PPOJointTransition):
        self.transitions.append(transition)

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_gae(self, gamma: float, gae_lambda: float, normalize: bool = True):
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
        values = np.array([t.value for t in self.transitions], dtype=np.float32)
        dones = np.array([t.done for t in self.transitions], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(self.transitions))):
            if t == len(self.transitions) - 1:
                next_value = 0.0
                next_nonterminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        if normalize and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.astype(np.float32), returns.astype(np.float32), values


class CentralizedValueNetwork(nn.Module):
    """Centralized value function V(s) for MAPPO-style training."""

    def __init__(self, input_dim: int = 28, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CentralizedValueCritic:
    """Centralized critic over compact C/R action-space and global summaries."""

    def __init__(
        self,
        c_dim: int = 17,
        r_dim: int = 11,
        global_dim: int = 14,
        context_dim: int = 0,
        hidden_dims=(256, 128),
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.c_dim = c_dim
        self.r_dim = r_dim
        self.global_dim = global_dim
        self.context_dim = context_dim
        self.input_dim = c_dim + r_dim + global_dim + context_dim
        self.hidden_dims = tuple(hidden_dims)
        self.device = device
        self.value_net = CentralizedValueNetwork(self.input_dim, self.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

    @staticmethod
    def _valid_mean(features: np.ndarray, mask: np.ndarray, feature_dim: int) -> np.ndarray:
        if features.size == 0:
            return np.zeros((feature_dim,), dtype=np.float32)
        if len(mask) == len(features) and np.any(mask):
            return features[mask].mean(axis=0).astype(np.float32)
        return features.mean(axis=0).astype(np.float32)

    @staticmethod
    def _global_summary(env, c_mask: np.ndarray, r_mask: np.ndarray) -> np.ndarray:
        spectrum = env.net.get_global_spectrum_stats()
        utils = np.array(env.mec.server_utilizations(), dtype=np.float32)
        if utils.size == 0:
            utils = np.zeros(1, dtype=np.float32)
        avail_ratios = np.array(
            [
                s.available_compute / max(s.compute_capacity, 1e-8)
                for s in env.mec.servers
            ],
            dtype=np.float32,
        )
        if avail_ratios.size == 0:
            avail_ratios = np.ones(1, dtype=np.float32)

        c_total = max(len(c_mask), 1)
        r_total = max(len(r_mask), 1)
        c_valid = float(np.sum(c_mask)) if len(c_mask) else 0.0
        r_valid = float(np.sum(r_mask)) if len(r_mask) else 0.0

        return np.array(
            [
                spectrum.get("spectrum_utilization", 0.0),
                spectrum.get("avg_frag_index", 0.0),
                spectrum.get("largest_free_block_ratio", 0.0),
                spectrum.get("avg_free_block_count", 0.0) / max(env.net.num_slots, 1),
                float(np.mean(utils)),
                float(np.max(utils)),
                float(np.min(utils)),
                float(np.std(utils)),
                float(np.mean(avail_ratios)),
                float(np.min(avail_ratios)),
                c_valid / c_total,
                r_valid / r_total,
                1.0 if c_valid > 0 else 0.0,
                1.0 if r_valid > 0 else 0.0,
            ],
            dtype=np.float32,
        )

    def encode(
        self,
        c_features: np.ndarray,
        c_mask: np.ndarray,
        r_features: np.ndarray,
        r_mask: np.ndarray,
        env=None,
        extra_context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        c_mean = self._valid_mean(c_features, c_mask, self.c_dim)
        r_mean = self._valid_mean(r_features, r_mask, self.r_dim)
        if env is None:
            global_summary = np.zeros((self.global_dim,), dtype=np.float32)
        else:
            global_summary = self._global_summary(env, c_mask, r_mask)
        if self.context_dim == 0:
            context = np.zeros((0,), dtype=np.float32)
        elif extra_context is None:
            context = np.zeros((self.context_dim,), dtype=np.float32)
        else:
            context = np.asarray(extra_context, dtype=np.float32).reshape(-1)
            if len(context) != self.context_dim:
                raise ValueError(
                    f"Expected critic context_dim={self.context_dim}, got {len(context)}"
                )
        return np.concatenate([c_mean, r_mean, global_summary, context]).astype(np.float32)

    def value(self, value_features: np.ndarray) -> float:
        x = torch.tensor(value_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.value_net(x).item())

    def optimize(
        self,
        value_features,
        returns,
        old_values=None,
        epochs: int = 4,
        clip_coef: Optional[float] = None,
    ) -> float:
        features_t = torch.tensor(
            np.stack(value_features), dtype=torch.float32, device=self.device
        )
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_values_t = None
        if old_values is not None:
            old_values_t = torch.tensor(old_values, dtype=torch.float32, device=self.device)
        last_loss = 0.0
        for _ in range(epochs):
            values = self.value_net(features_t)
            if clip_coef is not None and old_values_t is not None:
                values_clipped = old_values_t + torch.clamp(
                    values - old_values_t, -clip_coef, clip_coef
                )
                loss_unclipped = (values - returns_t).pow(2)
                loss_clipped = (values_clipped - returns_t).pow(2)
                loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
            else:
                loss = F.mse_loss(values, returns_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())
        return last_loss


def _entropy_coef_for_episode(args, episode: int) -> float:
    if args.entropy_final_coef is None or args.entropy_decay_episodes <= 0:
        return args.entropy_coef
    progress = min(max(episode / max(args.entropy_decay_episodes, 1), 0.0), 1.0)
    return float(
        args.entropy_coef
        + progress * (args.entropy_final_coef - args.entropy_coef)
    )


def _warm_start_ppo_actor(agent, ckpt_path: str, device: str, label: str):
    path = Path(ckpt_path)
    if not path.exists():
        print(f"WARNING: {label} warm-start checkpoint not found: {path}")
        return
    ckpt = load_checkpoint(path, map_location=device)
    agent.policy_net.load_state_dict(ckpt["model_state"])
    print(f"Warm-started {label} from {path}")


def _load_reference_ppo_policy(
    ckpt_path: str,
    device: str,
) -> nn.Module:
    """Load a frozen policy network for KL reference regularization."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    input_dim = ckpt.get("input_dim", 11)
    hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64)))
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode",
        ckpt.get("agent_c_feature_mode", "default"),
    )
    num_servers = ckpt_args.get("num_servers", ckpt.get("num_servers"))

    if feature_mode == "gated_typed_mean_field":
        if num_servers is None:
            raise ValueError(
                "gated_typed_mean_field reference policy requires num_servers"
            )
        policy = GatedMaskedPPOActorNetwork(
            base_dim=17,
            mf_dim=3 * (num_servers + 2),
            hidden_dims=hidden_dims,
        ).to(device)
    else:
        policy = MaskedPPOActorNetwork(input_dim, hidden_dims).to(device)
    policy.load_state_dict(ckpt["model_state"])
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy


def _filter_actor_batch(transitions, which: str):
    if which == "c":
        indexed = [(i, t) for i, t in enumerate(transitions) if t.c_valid]
        valid = [t for _, t in indexed]
        return (
            [t.c_features for t in valid],
            [t.c_mask for t in valid],
            [t.c_action for t in valid],
            [t.c_log_prob for t in valid],
            [i for i, _ in indexed],
        )
    indexed = [(i, t) for i, t in enumerate(transitions) if t.r_valid]
    valid = [t for _, t in indexed]
    return (
        [t.r_features for t in valid],
        [t.r_mask for t in valid],
        [t.r_action for t in valid],
        [t.r_log_prob for t in valid],
        [i for i, _ in indexed],
    )


def _evaluate_on_val_set(env, agent_c, agent_r, episodes_list, args=None):
    blocked = 0
    total = 0
    total_success = 0
    total_reward = 0.0
    mod_counter = Counter()
    reason_counter = Counter()
    pm_bpsk_count = 0
    for requests in episodes_list:
        env.reset(requests)
        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            action_idx_c = agent_c.select_action(obs_c, deterministic=True)
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, len(env.mec.servers)
            )
            split_id, server_id = action_c
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = agent_r.select_action(obs_r, deterministic=True)
            action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )
            _, _, _, info = env.step(action_c, action_r)
            total += 1
            server = env.mec.servers[server_id]
            waste_coef = args.waste_coef if args is not None else 0.8
            pm_penalty = args.pm_bpsk_penalty if args is not None else 0.0
            reward = compute_agent_c_reward(
                info, req.deadline_ms, waste_coef, server.utilization
            ) + compute_reward(
                info,
                waste_coef,
                mod_name=info.get("modulation", None),
                pm_bpsk_penalty=pm_penalty,
            )
            total_reward += reward
            if info.get("success", False):
                total_success += 1
                mod_name = info.get("modulation", "unknown")
                mod_counter[mod_name] += 1
                if mod_name == "PM-BPSK":
                    pm_bpsk_count += 1
            else:
                blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1
    return {
        "blocking_rate": blocked / max(total, 1),
        "success_rate": total_success / max(total, 1),
        "avg_reward": total_reward / max(total, 1),
        "failure_reasons": dict(reason_counter),
        "mod_counts": dict(mod_counter),
        "pm_bpsk_share": pm_bpsk_count / max(total_success, 1),
    }


def train(args):
    rng = np.random.RandomState(args.seed)
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if not topologies:
        raise ValueError("--topologies must contain at least one topology")

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    # Auto-detect C input_dim and feature_mode from warm-start checkpoint
    c_input_dim = 17
    c_feature_mode = args.agent_c_feature_mode
    if args.warm_start_c:
        wc_path = Path(args.warm_start_c)
        if wc_path.exists():
            wc_ckpt = load_checkpoint(wc_path, map_location=args.device)
            c_input_dim = wc_ckpt.get("input_dim", 17)
            c_feature_mode = wc_ckpt.get("agent_c_feature_mode",
                                          wc_ckpt.get("args", {}).get("agent_c_feature_mode", "default"))
            print(f"Auto-detected from warm-start C: input_dim={c_input_dim} feature_mode={c_feature_mode}")

    agent_c = PPOAgentC(
        input_dim=c_input_dim,
        hidden_dims=(128, 64),
        lr=args.lr_actor,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        feature_mode=c_feature_mode,
        num_servers=args.num_servers,
    )
    lr_r = args.lr_r_actor if args.lr_r_actor is not None else args.lr_actor
    agent_r = PPOAgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        lr=lr_r,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )
    if args.warm_start_c:
        _warm_start_ppo_actor(agent_c, args.warm_start_c, args.device, "PPO Agent-C")
    if args.warm_start_r:
        _warm_start_ppo_actor(agent_r, args.warm_start_r, args.device, "PPO Agent-R")
    critic = CentralizedValueCritic(
        c_dim=agent_c.input_dim,
        r_dim=agent_r.input_dim,
        lr=args.lr_critic,
        device=args.device,
    )

    package_root = Path(__file__).resolve().parents[2]
    ckpt_dir = package_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ref_agent_r = None
    if args.r_bc_coef > 0 and args.reference_r_checkpoint:
        ref_agent_r = _load_frozen_reference(
            args.reference_r_checkpoint, mod_reg, args.device
        )

    # KL reference policy for Agent-C regularization
    ref_c_policy: Optional[MaskedPPOActorNetwork] = None
    if args.c_ref_kl_coef > 0 and args.reference_c_checkpoint:
        ref_c_policy = _load_reference_ppo_policy(
            args.reference_c_checkpoint, args.device
        )
        print(f"Loaded C reference policy for KL reg from {args.reference_c_checkpoint}")

    # KL reference policy for Agent-R regularization
    ref_r_policy: Optional[MaskedPPOActorNetwork] = None
    if args.r_ref_kl_coef > 0 and args.reference_r_checkpoint:
        ref_r_policy = _load_reference_ppo_policy(
            args.reference_r_checkpoint, args.device
        )
        print(f"Loaded R reference policy for KL reg from {args.reference_r_checkpoint}")

    eval_envs = {
        topo: make_env(
            topology=topo,
            num_servers=args.num_servers,
            seed=args.seed + 17,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            k=args.k_paths,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )
        for topo in topologies
    }
    eval_sets = {
        topo: _build_eval_set(env, np.random.RandomState(args.seed + 1000 + i), args)
        for i, (topo, env) in enumerate(eval_envs.items())
    }

    best_blocking = float("inf")
    best_episode = -1
    metrics = {
        "episode_blocking": [],
        "losses_c": [],
        "losses_r": [],
        "losses_critic": [],
        "kl_c": [],
        "kl_r": [],
        "c_ref_kl": [],
        "r_ref_kl": [],
        "entropy_coef": [],
        "invalid_c": [],
        "invalid_r": [],
        "eval_blocking": [],
        "eval_reward": [],
        "eval_pm_bpsk_share": [],
    }

    print("=" * 72)
    print("MAPPO-style Joint Training")
    print(f"Topologies: {topologies}")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"PPO epochs: {args.ppo_epochs}, clip={args.clip_coef}, entropy={args.entropy_coef}")
    if args.entropy_final_coef is not None and args.entropy_decay_episodes > 0:
        print(
            f"Entropy decay: {args.entropy_coef} -> {args.entropy_final_coef} "
            f"over {args.entropy_decay_episodes} episodes"
        )
    print(f"Value clip: {args.value_clip_coef}, target KL: {args.target_kl}")
    print(f"Max grad norm: {args.max_grad_norm}")
    if args.pm_bpsk_penalty > 0:
        print(f"PM-BPSK penalty: {args.pm_bpsk_penalty}")
    if args.r_bc_coef > 0:
        print(f"BC regularization coef: {args.r_bc_coef} "
              f"(ref={'frozen' if ref_agent_r else 'N/A'})")
    if args.c_valid_action_pressure_coef > 0 or args.c_spectrum_block_penalty > 0:
        print(
            f"Agent-C shaping: valid_pressure={args.c_valid_action_pressure_coef}, "
            f"no_suitable_block={args.c_spectrum_block_penalty}"
        )
    if args.freeze_r_episodes > 0:
        print(f"Freeze Agent-R: first {args.freeze_r_episodes} episodes")
    if args.r_ref_kl_coef > 0:
        print(f"R ref KL coef: {args.r_ref_kl_coef}  "
              f"lr_r_actor: {lr_r}")
    if ref_r_policy is not None:
        print(f"R KL reference: {args.reference_r_checkpoint}")
    print(f"Critic input dim: {critic.input_dim}")
    print(f"Checkpoint prefix: {args.ckpt_prefix}")
    print("=" * 72)

    for episode in range(args.episodes):
        topo = str(rng.choice(topologies))
        env = make_env(
            topology=topo,
            num_servers=args.num_servers,
            seed=args.seed + episode,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            k=args.k_paths,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env,
            rng,
            src,
            args.requests_per_episode,
            arrival_interval=args.arrival_interval,
            holding_min=args.holding_min,
            holding_max=args.holding_max,
            deadline_min=args.deadline_min,
            deadline_max=args.deadline_max,
            size_min_mb=args.size_min_mb,
            size_max_mb=args.size_max_mb,
            edge_cost_min=args.edge_cost_min,
            edge_cost_max=args.edge_cost_max,
            num_splits=args.num_splits,
            split_profile=args.split_profile,
        )
        env.reset(requests)

        rollout = PPORolloutBuffer()
        blocked = 0
        reasons = Counter()
        invalid_c = 0
        invalid_r = 0
        current_entropy_coef = _entropy_coef_for_episode(args, episode)
        agent_c.entropy_coef = current_entropy_coef
        agent_r.entropy_coef = current_entropy_coef

        for t, req in enumerate(requests):
            obs_c = build_agent_c_observation(env, req)
            c_features, c_mask = agent_c.build_action_features(obs_c)
            action_idx_c, logp_c, _ = agent_c.select_from_features(
                c_features, c_mask, deterministic=False
            )
            c_valid = action_idx_c is not None
            if not c_valid:
                invalid_c += 1
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, len(env.mec.servers)
            )
            split_id, server_id = action_c

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_features, r_mask = agent_r.build_action_features(obs_r)
            action_idx_r, logp_r, _ = agent_r.select_from_features(
                r_features, r_mask, deterministic=False
            )
            r_valid = action_idx_r is not None
            if not r_valid:
                invalid_r += 1
            action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )
            selected_mod_name = None
            if action_idx_r is not None:
                _, mod_idx, _ = action_r
                if 0 <= mod_idx < len(obs_r["mod_names"]):
                    selected_mod_name = obs_r["mod_names"][mod_idx]

            value_features = critic.encode(c_features, c_mask, r_features, r_mask, env=env)
            value = critic.value(value_features)

            _, _, _, info = env.step(action_c, action_r)
            server = env.mec.servers[server_id]
            reward_c = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            r_valid_ratio = (
                float(np.sum(r_mask)) / max(len(r_mask), 1)
                if len(r_mask) > 0 else 0.0
            )
            valid_action_pressure = 1.0 - r_valid_ratio
            reward_c -= args.c_valid_action_pressure_coef * valid_action_pressure
            if (not info.get("success", False)
                    and info.get("reason") == "no_suitable_block"):
                reward_c -= args.c_spectrum_block_penalty
            mod_name = info.get("modulation", selected_mod_name)
            reward_r = compute_reward(
                info,
                args.waste_coef,
                mod_name=mod_name,
                pm_bpsk_penalty=args.pm_bpsk_penalty,
            )
            if info.get("success", False) and args.fs_penalty_coef > 0:
                reward_r -= args.fs_penalty_coef * info.get("num_slots", 0)
            reward_r += args.team_coef if info.get("success", False) else -args.team_coef
            if (args.r_bc_coef > 0 and ref_agent_r is not None
                    and info.get("success", False)
                    and mod_name == "PM-BPSK"):
                ref_mod = _get_reference_mod_name(
                    ref_agent_r, obs_r, obs_r["mod_names"]
                )
                if ref_mod is not None and ref_mod in ("BPSK", "QPSK", "8QAM"):
                    reward_r -= args.r_bc_coef
            reward_global = reward_c + reward_r
            if not info.get("success", False):
                blocked += 1
                reasons[info.get("reason", "unknown")] += 1

            rollout.add(
                PPOJointTransition(
                    c_features=c_features,
                    c_mask=c_mask,
                    c_action=action_idx_c if action_idx_c is not None else 0,
                    c_log_prob=logp_c,
                    c_valid=c_valid,
                    r_features=r_features,
                    r_mask=r_mask,
                    r_action=action_idx_r if action_idx_r is not None else 0,
                    r_log_prob=logp_r,
                    r_valid=r_valid,
                    value_features=value_features,
                    value=value,
                    reward=reward_global,
                    done=(t == len(requests) - 1),
                    info=info,
                )
            )

        transitions = rollout.transitions
        advantages, returns, old_values = rollout.compute_gae(
            args.gamma, args.gae_lambda, normalize=not args.no_advantage_norm
        )
        value_loss = critic.optimize(
            [t.value_features for t in transitions],
            returns,
            old_values=old_values,
            epochs=args.ppo_epochs,
            clip_coef=args.value_clip_coef,
        )
        metrics["losses_critic"].append(value_loss)
        metrics["entropy_coef"].append(current_entropy_coef)

        c_features, c_masks, c_actions, c_old_logp, c_indices = _filter_actor_batch(
            transitions, "c"
        )
        c_frozen = args.freeze_c_episodes > 0 and episode < args.freeze_c_episodes
        if c_frozen:
            c_loss = 0.0; c_kl = 0.0; c_ref_kl = 0.0
        elif c_indices:
            c_loss, _, c_kl, c_ref_kl = agent_c.optimize_ppo(
                c_features, c_masks, c_actions, c_old_logp,
                advantages[c_indices], args.clip_coef,
                epochs=args.ppo_epochs, target_kl=args.target_kl,
                reference_policy=ref_c_policy,
                ref_kl_coef=args.c_ref_kl_coef,
            )
        else:
            c_loss = 0.0; c_kl = 0.0; c_ref_kl = 0.0
        metrics["losses_c"].append(c_loss)
        metrics["kl_c"].append(c_kl)
        metrics.setdefault("c_ref_kl", []).append(c_ref_kl)

        r_features, r_masks, r_actions, r_old_logp, r_indices = _filter_actor_batch(
            transitions, "r"
        )
        r_frozen = args.freeze_r_episodes > 0 and episode < args.freeze_r_episodes
        if r_frozen:
            r_loss = 0.0
            r_kl = 0.0
            r_ref_kl = 0.0
        elif r_indices:
            r_loss, _, r_kl, r_ref_kl = agent_r.optimize_ppo(
                r_features,
                r_masks,
                r_actions,
                r_old_logp,
                advantages[r_indices],
                args.clip_coef,
                epochs=args.ppo_epochs,
                target_kl=args.target_kl,
                reference_policy=ref_r_policy,
                ref_kl_coef=args.r_ref_kl_coef,
            )
        else:
            r_loss = 0.0
            r_kl = 0.0
            r_ref_kl = 0.0
        metrics["losses_r"].append(r_loss)
        metrics["kl_r"].append(r_kl)
        metrics["r_ref_kl"].append(r_ref_kl)

        ep_blocking = blocked / max(len(requests), 1)
        metrics["episode_blocking"].append(ep_blocking)
        metrics["invalid_c"].append(invalid_c)
        metrics["invalid_r"].append(invalid_r)

        if (episode > 0 and episode % args.eval_freq == 0) or episode == args.episodes - 1:
            eval_results = {
                topo_name: _evaluate_on_val_set(
                    eval_env, agent_c, agent_r, eval_sets[topo_name], args
                )
                for topo_name, eval_env in eval_envs.items()
            }
            avg_blocking = float(np.mean([r["blocking_rate"] for r in eval_results.values()]))
            avg_reward = float(np.mean([r["avg_reward"] for r in eval_results.values()]))
            avg_pm_bpsk = float(np.mean([r["pm_bpsk_share"] for r in eval_results.values()]))
            metrics["eval_blocking"].append(avg_blocking)
            metrics["eval_reward"].append(avg_reward)
            metrics["eval_pm_bpsk_share"].append(avg_pm_bpsk)
            if avg_blocking < best_blocking:
                best_blocking = avg_blocking
                best_episode = episode
                _save_checkpoints(ckpt_dir, args.ckpt_prefix, "best", agent_c, agent_r, critic, args, metrics)
                per_topo = ", ".join(
                    f"{topo_name}=blk{res['blocking_rate']:.3f}/r{res['avg_reward']:+.3f}"
                    for topo_name, res in eval_results.items()
                )
                # Compute aggregate modulation shares for logging
                all_mods_best = Counter()
                for res in eval_results.values():
                    all_mods_best.update(res["mod_counts"])
                mod_total_best = max(sum(all_mods_best.values()), 1)
                _16qam = all_mods_best.get("16QAM", 0) / mod_total_best
                _qpsk = all_mods_best.get("QPSK", 0) / mod_total_best
                _bpsk = all_mods_best.get("BPSK", 0) / mod_total_best
                print(
                    f"  >> Best MAPPO @ ep {episode}: avg_blk={avg_blocking:.3f} "
                    f"avg_r={avg_reward:+.3f} "
                    f"16QAM={_16qam:.1%} QPSK={_qpsk:.1%} BPSK={_bpsk:.1%} "
                    f"| {per_topo}"
                )
            if args.print_eval_diagnostics:
                all_reasons = Counter()
                all_mods = Counter()
                for res in eval_results.values():
                    all_reasons.update(res["failure_reasons"])
                    all_mods.update(res["mod_counts"])
                reason_str = ", ".join(
                    f"{k}={v}" for k, v in all_reasons.most_common(6)
                ) or "none"
                mod_total = max(sum(all_mods.values()), 1)
                mod_str = ", ".join(
                    f"{k}={v / mod_total:.1%}" for k, v in all_mods.most_common()
                ) or "none"
                print(f"         Eval reasons: {reason_str}")
                print(f"         Eval mods: {mod_str}")

        if episode % args.log_interval == 0 or episode == args.episodes - 1:
            reason_str = ", ".join(f"{k}={v}" for k, v in reasons.most_common(4))
            c_kl = metrics["kl_c"][-1] if metrics["kl_c"] else 0.0
            r_kl = metrics["kl_r"][-1] if metrics["kl_r"] else 0.0
            c_ref_kl = metrics["c_ref_kl"][-1] if metrics.get("c_ref_kl") else 0.0
            r_ref_kl = metrics["r_ref_kl"][-1] if metrics["r_ref_kl"] else 0.0
            ref_str = ""
            if args.c_ref_kl_coef > 0:
                ref_str += f" cref_kl={c_ref_kl:.4f}"
            if args.r_ref_kl_coef > 0:
                ref_str += f" rref_kl={r_ref_kl:.4f}"
            print(
                f"Ep {episode:4d} [{topo}] blk={ep_blocking:.2f} "
                f"ent={current_entropy_coef:.4f} kl_c={c_kl:.4f} kl_r={r_kl:.4f}{ref_str} "
                f"invalid=({invalid_c},{invalid_r})"
            )
            if reason_str:
                print(f"         Reasons: {reason_str}")

    _save_checkpoints(ckpt_dir, args.ckpt_prefix, "last", agent_c, agent_r, critic, args, metrics)
    print(f"Best MAPPO checkpoint: avg_blk={best_blocking:.3f} @ ep {best_episode}")
    return agent_c, agent_r, critic, metrics


def _save_checkpoints(ckpt_dir, prefix, suffix, agent_c, agent_r, critic, args, metrics):
    torch.save(
        {
            "model_state": agent_c.policy_net.state_dict(),
            "input_dim": agent_c.input_dim,
            "hidden_dims": agent_c.hidden_dims,
            "args": vars(args),
            "training_metrics": metrics,
        },
        str(ckpt_dir / f"{prefix}_c_{suffix}.pt"),
    )
    torch.save(
        {
            "model_state": agent_r.policy_net.state_dict(),
            "input_dim": agent_r.input_dim,
            "hidden_dims": agent_r.hidden_dims,
            "args": vars(args),
            "training_metrics": metrics,
        },
        str(ckpt_dir / f"{prefix}_r_{suffix}.pt"),
    )
    torch.save(
        {
            "model_state": critic.value_net.state_dict(),
            "input_dim": critic.input_dim,
            "hidden_dims": critic.hidden_dims,
            "args": vars(args),
            "training_metrics": metrics,
        },
        str(ckpt_dir / f"{prefix}_critic_{suffix}.pt"),
    )


def main():
    parser = argparse.ArgumentParser(description="MAPPO-style joint PPO training")
    parser.add_argument("--topologies", type=str, default="net1,net2,net3")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--warm_start_c", type=str, default=None,
                        help="Optional PPO Agent-C checkpoint for warm-start")
    parser.add_argument("--warm_start_r", type=str, default=None,
                        help="Optional PPO Agent-R checkpoint for warm-start")
    parser.add_argument("--freeze_r_episodes", type=int, default=0,
                        help="Freeze Agent-R for the first N episodes (0 = disabled)")
    parser.add_argument("--freeze_c_episodes", type=int, default=0,
                        help="Freeze Agent-C for the first N episodes (0 = disabled)")
    parser.add_argument("--c_ref_kl_coef", type=float, default=0.0,
                        help="KL reference regularization coefficient for Agent-C")
    parser.add_argument("--r_ref_kl_coef", type=float, default=0.0,
                        help="KL reference regularization coefficient for Agent-R")
    parser.add_argument("--reference_c_checkpoint", type=str, default=None,
                        help="Optional Agent-C checkpoint for KL reference policy")
    parser.add_argument("--agent_c_feature_mode", type=str, default="default",
                        choices=[
                            "default",
                            "enhanced",
                            "pressure_aware",
                            "cross_pressure",
                            "r_feasibility",
                            "mean_field",
                            "typed_mean_field",
                            "gated_typed_mean_field",
                        ],
                        help="Agent-C feature mode (auto-detected from checkpoint if warm_start_c set)")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--requests_per_episode", type=int, default=30)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--team_coef", type=float, default=0.02)
    parser.add_argument("--fs_penalty_coef", type=float, default=0.0,
                        help="Extra success reward penalty per allocated FS to encourage high-SE choices")
    parser.add_argument("--c_valid_action_pressure_coef", type=float, default=0.0,
                        help="Agent-C penalty coefficient for low downstream Agent-R valid-action ratio")
    parser.add_argument("--c_spectrum_block_penalty", type=float, default=0.0,
                        help="Extra Agent-C penalty when failure reason is no_suitable_block")
    parser.add_argument("--pm_bpsk_penalty", type=float, default=0.0,
                        help="Penalty when successful Agent-R action uses PM-BPSK")
    parser.add_argument("--r_bc_coef", type=float, default=0.0,
                        help="Penalty when current R selects PM-BPSK but frozen reference selects BPSK/QPSK/8QAM")
    parser.add_argument("--reference_r_checkpoint", type=str, default=None,
                        help="Frozen DQN Agent-R checkpoint used for r_bc_coef regularization")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--entropy_final_coef", type=float, default=None,
                        help="If set, linearly decay entropy_coef to this value")
    parser.add_argument("--entropy_decay_episodes", type=int, default=0,
                        help="Number of episodes for entropy coefficient decay")
    parser.add_argument("--value_clip_coef", type=float, default=0.2,
                        help="PPO-style value clipping coefficient; set negative to disable")
    parser.add_argument("--target_kl", type=float, default=0.03,
                        help="Early-stop actor PPO epochs when approximate KL exceeds this; set <=0 to disable")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Actor gradient clipping norm; set <=0 to disable")
    parser.add_argument("--no_advantage_norm", action="store_true",
                        help="Disable advantage normalization")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_r_actor", type=float, default=None,
                        help="Separate Agent-R learning rate (default: same as --lr_actor)")
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", type=str, default="default3",
                        choices=["default3", "complex5", "complex5_v2", "complex5_v2_lite"])
    parser.add_argument("--k_paths", type=int, default=3,
                        help="Number of shortest paths for R action space")
    parser.add_argument("--max_blocks", type=int, default=5)
    parser.add_argument("--block_sort_strategy", type=str, default="size_desc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--print_eval_diagnostics", action="store_true",
                        help="Print validation failure reasons and modulation distribution")
    parser.add_argument("--ckpt_prefix", type=str, default="joint_mappo")
    args = parser.parse_args()
    if args.value_clip_coef is not None and args.value_clip_coef < 0:
        args.value_clip_coef = None
    if args.target_kl is not None and args.target_kl <= 0:
        args.target_kl = None
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        args.max_grad_norm = None
    train(args)


if __name__ == "__main__":
    main()
