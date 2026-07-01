"""Train Agent-C against a frozen learned Agent-R.

This stage fills the curriculum gap between separate pretraining and joint
MAPPO fine-tuning:

1. Load a frozen Agent-R checkpoint.
2. Train only PPO Agent-C.
3. Execute real RMSA with the frozen R for every selected split/server.
4. Save an Agent-C checkpoint that can warm-start MAPPO.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.env.request_demand_mean_field import RequestDemandMeanField
from sa_hmarl.env.c_action_risk import (
    apply_agent_c_risk_mask,
    compute_agent_c_action_risk_penalty,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_joint_alternating import _build_eval_set
from sa_hmarl.training.train_joint_mappo import (
    CentralizedValueCritic,
    PPOJointTransition,
    PPORolloutBuffer,
    _entropy_coef_for_episode,
    _filter_actor_batch,
)
from sa_hmarl.training.utils import (
    compute_agent_c_reward,
    compute_agent_c_reward_delay_aware,
    compute_agent_c_reward_pressure_aware,
    compute_agent_c_reward_state_aware,
    generate_requests,
    make_env,
)
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_frozen_ppo_r(
    ckpt_path: str,
    mod_reg: ModulationRegistry,
    device: str,
) -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    for p in agent.policy_net.parameters():
        p.requires_grad = False
    agent.policy_net.eval()
    return agent


def _load_frozen_dqn_r(
    ckpt_path: str,
    mod_reg: ModulationRegistry,
    device: str,
) -> AgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = AgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        gamma=ckpt.get("gamma", 0.95),
        epsilon=0.0,
        device=device,
    )
    agent.q_net.load_state_dict(ckpt["model_state"])
    if "target_state" in ckpt:
        agent.target_net.load_state_dict(ckpt["target_state"])
    else:
        agent.target_net.load_state_dict(ckpt["model_state"])
    for p in agent.q_net.parameters():
        p.requires_grad = False
    for p in agent.target_net.parameters():
        p.requires_grad = False
    agent.q_net.eval()
    agent.target_net.eval()
    return agent


def _load_frozen_r(args, mod_reg):
    if args.frozen_r_type == "ppo":
        return _load_frozen_ppo_r(args.frozen_r_checkpoint, mod_reg, args.device)
    return _load_frozen_dqn_r(args.frozen_r_checkpoint, mod_reg, args.device)


def _select_frozen_r_action(agent_r, obs_r, r_type: str):
    if r_type == "ppo":
        return agent_r.select_action(obs_r, deterministic=True)
    return agent_r.select_action(obs_r, epsilon=0.0)


def _risk_mask_enabled(args) -> bool:
    return (
        getattr(args, "c_max_spectrum_pressure", 0.0) > 0
        or getattr(args, "c_max_path_km", 0.0) > 0
        or getattr(args, "c_max_safe_fs_ratio", 0.0) > 0
    )


def _apply_c_risk_mask(obs_c, mask, env, args):
    if not _risk_mask_enabled(args):
        return mask
    return apply_agent_c_risk_mask(
        obs_c,
        mask,
        num_slots_total=env.net.num_slots,
        path_norm_km=args.c_path_norm_km,
        path_metric=args.c_path_metric,
        max_spectrum_pressure=args.c_max_spectrum_pressure,
        max_path_km=args.c_max_path_km,
        max_safe_fs_ratio=args.c_max_safe_fs_ratio,
        min_valid_after_mask=args.c_min_valid_after_risk_mask,
    )


def _c_risk_penalty(obs_c, action_idx_c, env, args) -> float:
    return compute_agent_c_action_risk_penalty(
        obs_c,
        action_idx_c,
        num_slots_total=env.net.num_slots,
        path_norm_km=args.c_path_norm_km,
        path_metric=args.c_path_metric,
        fs_coef=args.c_fs_request_penalty_coef,
        safe_fs_coef=args.c_safe_fs_penalty_coef,
        spectrum_pressure_coef=args.c_spectrum_pressure_penalty_coef,
        path_coef=args.c_path_penalty_coef,
        delay_risk_coef=args.c_delay_risk_penalty_coef,
    )


def _load_ppo_c_weights(agent_c: PPOAgentC, ckpt_path: str, device: str):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    old_state = ckpt["model_state"]
    new_state = agent_c.policy_net.state_dict()

    old_input_dim = None
    new_input_dim = None

    # Find the first linear layer to check input dimension mismatch
    for key in old_state:
        if "net.0.weight" in key:
            old_input_dim = old_state[key].shape[1]
            new_input_dim = new_state[key].shape[1]
            break

    if old_input_dim is not None and new_input_dim is not None and old_input_dim != new_input_dim:
        print(f"Input dimension mismatch: old={old_input_dim}, new={new_input_dim}. "
              f"Performing partial warm-start...")
        with torch.no_grad():
            for key in new_state:
                if key not in old_state:
                    continue
                old_param = old_state[key]
                new_param = new_state[key]
                if "net.0.weight" in key:
                    # First layer: preserve old behavior and let fine-tuning
                    # gradually learn any newly appended feature dimensions.
                    n_copy = min(old_param.shape[1], new_param.shape[1])
                    new_param[:, :n_copy] = old_param[:, :n_copy]
                    if new_param.shape[1] > old_param.shape[1]:
                        new_param[:, old_param.shape[1]:] = 0.0
                elif "net.0.bias" in key:
                    new_param.copy_(old_param)
                elif old_param.shape == new_param.shape:
                    new_param.copy_(old_param)
                else:
                    print(f"  Skipping {key}: shape mismatch {old_param.shape} vs {new_param.shape}")
        print(f"Warm-started PPO Agent-C from {ckpt_path} (partial, {old_input_dim}->{new_input_dim})")
    else:
        agent_c.policy_net.load_state_dict(old_state)
        print(f"Warm-started PPO Agent-C from {ckpt_path}")


def _compute_objective(blocking_rate: float, avg_delay_ms: float) -> float:
    return blocking_rate + 0.5 * (avg_delay_ms / 65.0)


def _selected_c_pressure(transition: PPOJointTransition) -> float:
    """Estimate selected C-action pressure from the stable 17-dim prefix.

    The first 17 Agent-C features are kept stable across default/enhanced/
    pressure-aware modes:
      [size, local_ms, edge_ms, util, best_fs, safe_fs, feasible_count,
       lfb_max, lfb_mean, lfb_min, lfb_p25, frag_mean, ...]
    """
    if transition.c_features.size == 0:
        return 1.0
    action = int(np.clip(transition.c_action, 0, len(transition.c_features) - 1))
    feat = transition.c_features[action]
    if len(feat) < 12:
        return 1.0

    def finite(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(value):
            return default
        return value

    best_fs = finite(feat[4], 0.0)
    safe_fs = finite(feat[5], best_fs)
    feasible_count = max(finite(feat[6], 0.0), 0.0)
    lfb_max = max(finite(feat[7], 1.0), 1.0)
    lfb_p25 = max(finite(feat[10], lfb_max), 1.0)
    frag_mean = float(np.clip(finite(feat[11], 0.0), 0.0, 1.0))

    best_pressure = min(best_fs / lfb_max, 10.0)
    safe_pressure = min(safe_fs / lfb_p25, 10.0)
    scarcity = 1.0 / (1.0 + feasible_count)
    return 0.45 * best_pressure + 0.45 * safe_pressure + 0.6 * scarcity + 0.25 * frag_mean


def _episode_credit_weights(transitions, mode: str) -> np.ndarray:
    n = len(transitions)
    if n == 0 or mode == "off":
        return np.zeros((n,), dtype=np.float32)
    if mode == "uniform":
        return np.ones((n,), dtype=np.float32) / n

    weights = np.zeros((n,), dtype=np.float32)
    if mode in ("failure", "mixed"):
        for i, t in enumerate(transitions):
            if not t.info.get("success", False):
                weights[i] += 2.0 if t.info.get("reason") == "no_suitable_block" else 1.0
            else:
                weights[i] += 0.15

    if mode in ("pressure", "mixed"):
        pressures = np.array([_selected_c_pressure(t) for t in transitions], dtype=np.float32)
        if np.all(np.isfinite(pressures)):
            pressures = np.clip(pressures, 0.0, 10.0)
            if float(pressures.max() - pressures.min()) > 1e-6:
                pressures = (pressures - pressures.min()) / (pressures.max() - pressures.min())
            weights += pressures + 0.05

    total = float(weights.sum())
    if total <= 1e-8:
        return np.ones((n,), dtype=np.float32) / n
    return weights / total


def _apply_episode_level_credit(rollout: PPORolloutBuffer, args) -> Dict[str, float]:
    """Add episode-level blocking/delay credit to per-step C rewards.

    This is the key difference from single-request shaping: the whole episode's
    realized blocking and latency are assigned back to the C decisions before
    GAE is computed.
    """
    transitions = rollout.transitions
    n = len(transitions)
    if n == 0 or args.episode_credit_mode == "off":
        return {
            "episode_credit_penalty": 0.0,
            "episode_credit_blocking": 0.0,
            "episode_credit_delay_norm": 0.0,
            "episode_credit_no_block": 0.0,
        }

    blocked = sum(1 for t in transitions if not t.info.get("success", False))
    no_block = sum(
        1
        for t in transitions
        if (not t.info.get("success", False)) and t.info.get("reason") == "no_suitable_block"
    )
    delays = [float(t.info.get("delay_ms", 0.0)) for t in transitions if t.info.get("success", False)]
    blocking_rate = blocked / max(n, 1)
    no_block_rate = no_block / max(n, 1)
    avg_delay = float(np.mean(delays)) if delays else float(args.deadline_max)
    delay_norm = avg_delay / max(float(args.deadline_max), 1.0)

    penalty = (
        args.episode_blocking_coef * blocking_rate
        + args.episode_no_block_coef * no_block_rate
        + args.episode_delay_coef * delay_norm
    )
    weights = _episode_credit_weights(transitions, args.episode_credit_mode)
    # Keep the total episode penalty proportional to episode length so that
    # each decision receives a meaningful long-horizon signal.
    for transition, weight in zip(transitions, weights):
        transition.reward -= float(args.episode_credit_scale * penalty * n * weight)

    return {
        "episode_credit_penalty": float(penalty),
        "episode_credit_blocking": float(blocking_rate),
        "episode_credit_delay_norm": float(delay_norm),
        "episode_credit_no_block": float(no_block_rate),
    }


def _validation_seeds_for_topology(args, topo_index: int):
    """Return deterministic validation seeds for one topology.

    Priority:
      1. ``--validation_seeds`` overrides everything (same seeds for all topologies).
      2. ``--num_validation_seeds`` together with the old per-topology offset.
      3. Historical fallback: ``seed + 1000 + topo_index * 10000``.
    """
    if getattr(args, "validation_seeds", ""):
        seeds = [int(s.strip()) for s in args.validation_seeds.split(",") if s.strip()]
        if seeds:
            return seeds
    num_seeds = max(1, int(getattr(args, "num_validation_seeds", 1)))
    base = args.seed + 1000 + topo_index * 10000
    return [base + i for i in range(num_seeds)]


def _evaluate(env, agent_c, frozen_r, frozen_r_type: str, episodes_list, args) -> Dict:
    total = 0
    blocked = 0
    raw_mask_empty = 0
    total_reward = 0.0
    successes = 0
    total_fs = 0.0
    total_path = 0.0
    total_delay = 0.0
    deadline_met = 0
    reasons = Counter()
    mods = Counter()
    splits = Counter()
    servers = Counter()

    for requests in episodes_list:
        env.reset(requests)
        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            c_features, raw_c_mask = agent_c.build_action_features(obs_c)
            if int(np.sum(raw_c_mask)) == 0:
                raw_mask_empty += 1
            c_mask = _apply_c_risk_mask(obs_c, raw_c_mask, env, args)
            action_idx_c, _, _ = agent_c.select_from_features(
                c_features, c_mask, deterministic=True
            )
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, len(env.mec.servers)
            )
            split_id, server_id = action_c
            splits[f"split{split_id}"] += 1
            servers[f"s{server_id}"] += 1
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = _select_frozen_r_action(frozen_r, obs_r, frozen_r_type)
            action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )
            _, _, _, info = env.step(action_c, action_r)
            server = env.mec.servers[server_id]
            if args.objective == "blocking_delay":
                reward_c = compute_agent_c_reward_delay_aware(
                    info,
                    deadline_ms=req.deadline_ms,
                    num_slots_total=env.net.num_slots,
                    success_reward=args.success_reward,
                    block_penalty=args.block_penalty,
                    delay_coef=args.delay_coef,
                    deadline_penalty=args.deadline_penalty,
                    server_overload_penalty=args.server_overload_penalty,
                    no_block_penalty=args.no_block_penalty,
                    fs_penalty_coef=args.fs_penalty_coef,
                    server_util_coef=args.server_util_coef,
                    server_utilization_after=server.utilization,
                    spectrum_fail_extra=args.spectrum_fail_extra,
                    server_fail_extra=args.server_fail_extra,
                )
            elif args.objective == "pressure_aware_blocking_delay":
                reward_c, _ = compute_agent_c_reward_pressure_aware(
                    info,
                    obs_c,
                    action_idx_c,
                    len(env.mec.servers),
                    deadline_ms=req.deadline_ms,
                    num_slots_total=env.net.num_slots,
                    success_reward=args.success_reward,
                    base_block_penalty=args.base_block_penalty,
                    base_delay_coef=args.base_delay_coef,
                    deadline_penalty=args.deadline_penalty,
                    server_overload_penalty=args.server_overload_penalty,
                    base_no_block_penalty=args.base_no_block_penalty,
                    pressure_penalty_coef=args.pressure_penalty_coef,
                    pressure_source=args.pressure_source,
                    pressure_clip_min=args.pressure_clip_min,
                    pressure_clip_max=args.pressure_clip_max,
                )
            elif args.objective == "state_aware_blocking_delay":
                reward_c, _, _, _, _ = compute_agent_c_reward_state_aware(
                    info,
                    obs_c,
                    action_idx_c,
                    len(env.mec.servers),
                    deadline_ms=req.deadline_ms,
                    num_slots_total=env.net.num_slots,
                    success_reward=args.success_reward,
                    base_block_penalty=args.base_block_penalty,
                    base_delay_coef=args.base_delay_coef,
                    deadline_penalty=args.deadline_penalty,
                    server_overload_penalty=args.server_overload_penalty,
                    base_no_block_penalty=args.base_no_block_penalty,
                    pressure_source=args.pressure_source,
                    pressure_clip_min=args.pressure_clip_min,
                    pressure_clip_max=args.pressure_clip_max,
                )
            else:
                reward_c = compute_agent_c_reward(
                    info, req.deadline_ms, args.waste_coef, server.utilization
                )
            reward_c -= _c_risk_penalty(obs_c, action_idx_c, env, args)
            total += 1
            total_reward += reward_c
            if info.get("success", False):
                successes += 1
                delay = info.get("delay_ms", 0.0)
                total_delay += delay
                if delay <= req.deadline_ms:
                    deadline_met += 1
                total_fs += info.get("num_slots", 0)
                total_path += info.get("path_dist_km", 0.0)
                mods[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reasons[info.get("reason", "unknown")] += 1

    s = max(successes, 1)
    blocking_rate = blocked / max(total, 1)
    avg_delay_ms = total_delay / s
    objective = _compute_objective(blocking_rate, avg_delay_ms)
    total_cnt = sum(splits.values())
    srv_cnt = sum(servers.values())
    return {
        "blocking_rate": blocking_rate,
        "raw_mask_empty_rate": raw_mask_empty / max(total, 1),
        "success_rate": successes / max(total, 1),
        "avg_reward": total_reward / max(total, 1),
        "avg_delay_ms": avg_delay_ms,
        "deadline_sat_rate": deadline_met / max(total, 1),
        "avg_fs": total_fs / s,
        "avg_path_len_km": total_path / s,
        "reasons": reasons,
        "mods": mods,
        "splits": dict(splits),
        "servers": dict(servers),
        "split_distribution": {k: v / total_cnt for k, v in splits.items()} if total_cnt else {},
        "server_distribution": {k: v / srv_cnt for k, v in servers.items()} if srv_cnt else {},
        "objective": objective,
    }


def _fmt_counter(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in counter.most_common())


def _evaluate_validation(env, agent_c, frozen_r, frozen_r_type: str, eval_sets, args):
    """Run validation and return per-seed metrics plus aggregates.

    ``eval_sets`` is a list of (seed, episodes_list) pairs so that statistics
    can be reported per validation seed independently.
    """
    per_seed = {}
    all_results = []
    for seed, episodes_list in eval_sets:
        res = _evaluate(env, agent_c, frozen_r, frozen_r_type, episodes_list, args)
        per_seed[str(seed)] = {
            "blocking_rate": res["blocking_rate"],
            "raw_mask_empty_rate": res.get("raw_mask_empty_rate", 0.0),
            "server_overload_rate": _server_overload_rate_from_reasons(res.get("reasons", {})),
            "avg_delay_ms": res["avg_delay_ms"],
            "avg_fs": res["avg_fs"],
            "avg_reward": res["avg_reward"],
            "objective": res["objective"],
        }
        all_results.append(per_seed[str(seed)])

    aggregates = {
        "mean_blocking": float(np.mean([r["blocking_rate"] for r in all_results])),
        "std_blocking": float(np.std([r["blocking_rate"] for r in all_results])),
        "mean_raw_mask_empty": float(np.mean([r["raw_mask_empty_rate"] for r in all_results])),
        "mean_server_overload": float(np.mean([r["server_overload_rate"] for r in all_results])),
        "mean_delay_ms": float(np.mean([r["avg_delay_ms"] for r in all_results])),
        "mean_fs": float(np.mean([r["avg_fs"] for r in all_results])),
        "mean_reward": float(np.mean([r["avg_reward"] for r in all_results])),
        "mean_objective": float(np.mean([r["objective"] for r in all_results])),
    }
    return {"per_seed": per_seed, "aggregate": aggregates}


def _server_overload_rate_from_reasons(reasons: Dict[str, int]) -> float:
    total = sum(reasons.values())
    if total == 0:
        return 0.0
    return reasons.get("server_overload", 0) / total


def _compute_base_reward_c(info, obs_c, action_idx_c, server, env, req, args):
    """Compute the unshaped per-step reward for Agent-C.

    Returns:
        (reward_c, aux) where aux is a dict with optional pressure/logging info.
    """
    if args.objective == "blocking_delay":
        reward_c = compute_agent_c_reward_delay_aware(
            info,
            deadline_ms=req.deadline_ms,
            num_slots_total=env.net.num_slots,
            success_reward=args.success_reward,
            block_penalty=args.block_penalty,
            delay_coef=args.delay_coef,
            deadline_penalty=args.deadline_penalty,
            server_overload_penalty=args.server_overload_penalty,
            no_block_penalty=args.no_block_penalty,
            fs_penalty_coef=args.fs_penalty_coef,
            server_util_coef=args.server_util_coef,
            server_utilization_after=server.utilization,
            spectrum_fail_extra=args.spectrum_fail_extra,
            server_fail_extra=args.server_fail_extra,
        )
        return reward_c, {}
    if args.objective == "pressure_aware_blocking_delay":
        reward_c, pressure = compute_agent_c_reward_pressure_aware(
            info,
            obs_c,
            action_idx_c,
            len(env.mec.servers),
            deadline_ms=req.deadline_ms,
            num_slots_total=env.net.num_slots,
            success_reward=args.success_reward,
            base_block_penalty=args.base_block_penalty,
            base_delay_coef=args.base_delay_coef,
            deadline_penalty=args.deadline_penalty,
            server_overload_penalty=args.server_overload_penalty,
            base_no_block_penalty=args.base_no_block_penalty,
            pressure_penalty_coef=args.pressure_penalty_coef,
            pressure_source=args.pressure_source,
            pressure_clip_min=args.pressure_clip_min,
            pressure_clip_max=args.pressure_clip_max,
        )
        return reward_c, {"pressure": pressure}
    if args.objective == "state_aware_blocking_delay":
        reward_c, pressure, dw, bw, nbw = compute_agent_c_reward_state_aware(
            info,
            obs_c,
            action_idx_c,
            len(env.mec.servers),
            deadline_ms=req.deadline_ms,
            num_slots_total=env.net.num_slots,
            success_reward=args.success_reward,
            base_block_penalty=args.base_block_penalty,
            base_delay_coef=args.base_delay_coef,
            deadline_penalty=args.deadline_penalty,
            server_overload_penalty=args.server_overload_penalty,
            base_no_block_penalty=args.base_no_block_penalty,
            pressure_source=args.pressure_source,
            pressure_clip_min=args.pressure_clip_min,
            pressure_clip_max=args.pressure_clip_max,
        )
        return reward_c, {"pressure": pressure, "delay_weight": dw, "block_weight": bw, "no_block_weight": nbw}
    reward_c = compute_agent_c_reward(info, req.deadline_ms, args.waste_coef, server.utilization)
    return reward_c, {}


def _run_training_episode(env, requests, agent_c, frozen_r, frozen_r_type, critic, args):
    """Run one training episode and return transitions + episode statistics.

    Implements pending-transition spectrum-collapse penalty: when the current
    decision state's raw C-mask is empty, the previous transition is charged
    ``args.spectrum_collapse_penalty_coef``.
    """
    rollout = PPORolloutBuffer()
    blocked = 0
    raw_mask_empty = 0
    server_overload = 0
    reasons = Counter()

    ep_base_rewards = []
    ep_shaped_rewards = []
    ep_collapse_penalties = []
    ep_k_c_valid_next = []
    ep_collapse_trigger_count = 0

    ep_pressures = []
    ep_delay_weights = []
    ep_block_weights = []
    ep_no_block_weights = []
    ep_delays = []
    ep_splits = Counter()

    previous_transition = None
    request_mean_field = RequestDemandMeanField.from_config(
        window_size=getattr(args, "request_mean_field_window", 20),
        size_min_mb=args.size_min_mb,
        size_max_mb=args.size_max_mb,
        deadline_min=args.deadline_min,
        deadline_max=args.deadline_max,
        split_profile=args.split_profile,
    )

    for t, req in enumerate(requests):
        obs_c = build_agent_c_observation(env, req)
        c_features, raw_c_mask = agent_c.build_action_features(obs_c)

        k_c_valid_current = int(np.sum(raw_c_mask))
        collapse_current = int(k_c_valid_current == 0)

        # --- Pending-transition collapse penalty attribution ---
        if previous_transition is not None:
            penalty = args.spectrum_collapse_penalty_coef if collapse_current else 0.0
            previous_transition.reward -= penalty
            previous_transition.info["collapse_next"] = bool(collapse_current)
            previous_transition.info["k_c_valid_next"] = k_c_valid_current
            previous_transition.info["collapse_penalty"] = -penalty
            ep_collapse_penalties.append(-penalty)
            if collapse_current:
                ep_collapse_trigger_count += 1

        ep_k_c_valid_next.append(k_c_valid_current)
        if collapse_current:
            raw_mask_empty += 1

        c_mask = _apply_c_risk_mask(obs_c, raw_c_mask, env, args)
        action_idx_c, logp_c, _ = agent_c.select_from_features(
            c_features, c_mask, deterministic=False
        )
        c_valid = action_idx_c is not None
        action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
            action_idx_c, len(env.mec.servers)
        )
        split_id, server_id = action_c

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = frozen_r.build_action_features(obs_r)
        action_idx_r = _select_frozen_r_action(frozen_r, obs_r, frozen_r_type)
        action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
            action_idx_r, len(obs_r["mod_names"]), env.max_blocks
        )

        critic_context = (
            request_mean_field.vector()
            if getattr(args, "critic_request_mean_field", False)
            else None
        )
        value_features = critic.encode(
            c_features,
            c_mask,
            r_features,
            r_mask,
            env=env,
            extra_context=critic_context,
        )
        value = critic.value(value_features)

        _, _, _, info = env.step(action_c, action_r)
        server = env.mec.servers[server_id]

        base_reward_c, reward_aux = _compute_base_reward_c(info, obs_c, action_idx_c, server, env, req, args)
        if "pressure" in reward_aux:
            ep_pressures.append(reward_aux["pressure"])
        if "delay_weight" in reward_aux:
            ep_delay_weights.append(reward_aux["delay_weight"])
        if "block_weight" in reward_aux:
            ep_block_weights.append(reward_aux["block_weight"])
        if "no_block_weight" in reward_aux:
            ep_no_block_weights.append(reward_aux["no_block_weight"])
        if action_idx_c is not None:
            split_id_log, _ = decode_agent_c_action(action_idx_c, len(env.mec.servers))
            ep_splits[f"split{split_id_log}"] += 1
        if info.get("success", False):
            ep_delays.append(info.get("delay_ms", 0.0))

        r_valid_ratio = (
            float(np.sum(r_mask)) / max(len(r_mask), 1)
            if len(r_mask) > 0 else 0.0
        )
        reward_c = base_reward_c
        reward_c -= args.c_valid_action_pressure_coef * (1.0 - r_valid_ratio)
        if (not info.get("success", False)
                and info.get("reason") == "no_suitable_block"):
            reward_c -= args.c_spectrum_block_penalty
        reward_c -= _c_risk_penalty(obs_c, action_idx_c, env, args)

        ep_base_rewards.append(base_reward_c)
        ep_shaped_rewards.append(reward_c)

        if not info.get("success", False):
            blocked += 1
            reasons[info.get("reason", "unknown")] += 1
            if info.get("reason") == "server_overload":
                server_overload += 1

        transition = PPOJointTransition(
            c_features=c_features,
            c_mask=c_mask,
            c_action=action_idx_c if action_idx_c is not None else 0,
            c_log_prob=logp_c,
            c_valid=c_valid,
            r_features=r_features,
            r_mask=r_mask,
            r_action=action_idx_r if action_idx_r is not None else 0,
            r_log_prob=0.0,
            r_valid=False,
            value_features=value_features,
            value=value,
            reward=reward_c,
            done=(t == len(requests) - 1),
            info=info,
        )
        rollout.add(transition)
        previous_transition = transition
        # Update only after encoding state t, so the context contains requests
        # strictly before the current decision and cannot leak future demand.
        request_mean_field.observe(req)

    episode_stats = {
        "blocked": blocked,
        "raw_mask_empty": raw_mask_empty,
        "server_overload": server_overload,
        "base_reward_mean": float(np.mean(ep_base_rewards)) if ep_base_rewards else 0.0,
        "shaped_reward_mean": float(np.mean(ep_shaped_rewards)) if ep_shaped_rewards else 0.0,
        "collapse_penalty_mean": float(np.mean(ep_collapse_penalties)) if ep_collapse_penalties else 0.0,
        "collapse_penalty_trigger_count": ep_collapse_trigger_count,
        "collapse_penalty_trigger_rate": ep_collapse_trigger_count / max(len(requests) - 1, 1),
        "avg_k_c_valid_next": float(np.mean(ep_k_c_valid_next)) if ep_k_c_valid_next else 0.0,
    }
    aux = {
        "pressures": ep_pressures,
        "delay_weights": ep_delay_weights,
        "block_weights": ep_block_weights,
        "no_block_weights": ep_no_block_weights,
        "delays": ep_delays,
        "splits": ep_splits,
    }
    return rollout.transitions, reasons, episode_stats, aux


def _save_c_checkpoint(path: Path, agent_c, critic, args, metrics, best_validation=None):
    ckpt = {
        "model_state": agent_c.policy_net.state_dict(),
        "input_dim": agent_c.input_dim,
        "hidden_dims": agent_c.hidden_dims,
        "args": vars(args),
        "training_metrics": metrics,
        "critic_input_dim": critic.input_dim,
        "critic_state": critic.value_net.state_dict(),
        "metadata": {
            "train_seed": args.seed,
            "validation_seeds": _validation_seeds_for_topology(args, 0),
            "validation_episodes": args.validation_episodes,
            "checkpoint_metric": args.checkpoint_metric,
            "feature_mode": args.agent_c_feature_mode,
            "spectrum_collapse_penalty_coef": args.spectrum_collapse_penalty_coef,
            "critic_request_mean_field": getattr(args, "critic_request_mean_field", False),
            "request_mean_field_window": getattr(args, "request_mean_field_window", 20),
            "torch_seed": args.seed,
            "policy_sampling_seed": args.seed + 100_000,
        },
    }
    if best_validation is not None:
        ckpt["metadata"]["best_validation_mean"] = best_validation
    torch.save(ckpt, str(path))


def train(args):
    # Make --seed control model initialization as well as request generation.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.RandomState(args.seed)
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if not topologies:
        raise ValueError("--topologies must contain at least one topology")

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    frozen_r = _load_frozen_r(args, mod_reg)
    print(f"Loaded frozen {args.frozen_r_type.upper()} Agent-R from {args.frozen_r_checkpoint}")

    agent_c_input_dim = args.agent_c_input_dim
    if agent_c_input_dim is None:
        if args.agent_c_feature_mode == "enhanced":
            agent_c_input_dim = 24
        elif args.agent_c_feature_mode == "pressure_aware":
            # PPOAgentC auto-adds 8 dims on top of base 17
            agent_c_input_dim = 17
        elif args.agent_c_feature_mode == "r_feasibility":
            # PPOAgentC auto-adds 10 direct R-feasibility dims on top of base 17
            agent_c_input_dim = 17
        elif args.agent_c_feature_mode == "r_feasibility_safe":
            # PPOAgentC auto-adds 10 direct R-feasibility + 2 server-margin dims
            agent_c_input_dim = 17
        elif args.agent_c_feature_mode == "r_feasibility_spectrum_impact":
            # PPOAgentC auto-adds 10 r_feasibility + 5 spectrum-impact dims
            agent_c_input_dim = 17
        elif args.agent_c_feature_mode == "r_feasibility_mr_spec_compute":
            # PPOAgentC auto-adds 10 r_feasibility + 6 spec + 5 compute dims
            agent_c_input_dim = 17
        elif args.agent_c_feature_mode == "mr_feasibility_rule":
            # PPOAgentC auto-adds 10 r_feasibility + 6 MR-rule dims
            agent_c_input_dim = 17
        elif args.agent_c_feature_mode in ("mean_field", "typed_mean_field", "gated_typed_mean_field", "fixed_blend_typed_mean_field", "candidate_mean_field", "candidate_mean_field_count_only"):
            # PPOAgentC needs num_servers to compute the appended mean-field dims.
            agent_c_input_dim = 17
        elif args.agent_c_feature_mode == "cross_pressure":
            agent_c_input_dim = 29  # cross_pressure replaces the entire 17-dim base
        else:
            agent_c_input_dim = 17

    agent_c = PPOAgentC(
        input_dim=agent_c_input_dim,
        hidden_dims=(128, 64),
        lr=args.lr_actor,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        feature_mode=args.agent_c_feature_mode,
        activation=args.agent_c_activation,
        num_servers=args.num_servers,
        gate_reg_coef=args.gate_reg_coef,
        fixed_blend_alpha=args.fixed_blend_alpha,
    )
    if args.warm_start_c:
        _load_ppo_c_weights(agent_c, args.warm_start_c, args.device)

    critic = CentralizedValueCritic(
        c_dim=agent_c.input_dim,
        r_dim=11,
        context_dim=(
            RequestDemandMeanField.DIM
            if getattr(args, "critic_request_mean_field", False)
            else 0
        ),
        lr=args.lr_critic,
        device=args.device,
    )
    # Different critic input sizes consume different initialization draws.
    # Reset the categorical-policy sampling stream after network construction
    # so critic-only ablations begin with a paired actor sampling sequence.
    policy_sampling_seed = args.seed + 100_000
    torch.manual_seed(policy_sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_sampling_seed)

    package_root = Path(__file__).resolve().parents[2]
    ckpt_dir = package_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{args.ckpt_prefix}_best.pt"
    last_path = ckpt_dir / f"{args.ckpt_prefix}_last.pt"

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
    eval_seed_map = {
        topo: _validation_seeds_for_topology(args, i)
        for i, topo in enumerate(eval_envs.keys())
    }
    eval_sets = {}
    for topo, env in eval_envs.items():
        # Group validation episodes by validation seed so that per-seed stats are available.
        # Use args.validation_episodes for the validation set size.
        topo_sets = []
        for eval_seed in eval_seed_map[topo]:
            build_args = SimpleNamespace(**vars(args))
            build_args.eval_episodes = args.validation_episodes
            episodes_list = _build_eval_set(env, np.random.RandomState(eval_seed), build_args)
            topo_sets.append((eval_seed, episodes_list))
        eval_sets[topo] = topo_sets

    metrics = {
        "episode_blocking": [],
        "raw_mask_empty_rate": [],
        "server_overload_rate": [],
        "base_reward_mean": [],
        "shaped_reward_mean": [],
        "collapse_penalty_mean": [],
        "collapse_penalty_trigger_count": [],
        "collapse_penalty_trigger_rate": [],
        "avg_k_c_valid_next": [],
        "losses_c": [],
        "losses_critic": [],
        "kl_c": [],
        "gate_loss": [],
        "entropy_coef": [],
        "eval_blocking": [],
        "eval_reward": [],
        "eval_mean_blocking": [],
        "eval_std_blocking": [],
        "eval_mean_raw_mask_empty": [],
        "eval_mean_server_overload": [],
        "episode_credit_penalty": [],
        "episode_credit_blocking": [],
        "episode_credit_delay_norm": [],
        "episode_credit_no_block": [],
    }
    best_metric = float("inf")
    best_episode = -1
    last_validation_agg = None

    print("=" * 76)
    print("Train PPO Agent-C with Frozen Agent-R")
    print(f"Topologies: {topologies}")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"Traffic: mode={getattr(args, 'traffic_mode', 'iid')}, "
          f"regime_stay_prob={getattr(args, 'regime_stay_prob', 0.9)}")
    print(f"Frozen R type: {args.frozen_r_type}")
    print(f"Agent-C features: mode={args.agent_c_feature_mode}, input_dim={agent_c.input_dim}")
    if args.agent_c_feature_mode == "gated_typed_mean_field":
        print(f"Gated TMF: gate_reg_coef={args.gate_reg_coef}, gate_init_bias=-2.0")
    if args.agent_c_feature_mode == "fixed_blend_typed_mean_field":
        print(f"Fixed blend TMF: alpha={args.fixed_blend_alpha}")
    print(f"Agent-C shaping: valid_pressure={args.c_valid_action_pressure_coef}, "
          f"no_suitable_block={args.c_spectrum_block_penalty}")
    print(f"Agent-C risk: fs={args.c_fs_request_penalty_coef} safe_fs={args.c_safe_fs_penalty_coef} "
          f"pressure={args.c_spectrum_pressure_penalty_coef} path={args.c_path_penalty_coef} "
          f"delay_risk={args.c_delay_risk_penalty_coef} mask_pressure={args.c_max_spectrum_pressure} "
          f"mask_path={args.c_max_path_km} path_metric={args.c_path_metric} "
          f"mask_safe_fs={args.c_max_safe_fs_ratio}")
    print(f"Objective: {args.objective}")
    print(f"Episode-level credit: mode={args.episode_credit_mode} "
          f"scale={args.episode_credit_scale} block={args.episode_blocking_coef} "
          f"no_block={args.episode_no_block_coef} delay={args.episode_delay_coef}")
    if args.objective == "blocking_delay":
        print(f"  delay_coef={args.delay_coef} block_penalty={args.block_penalty} "
              f"success_reward={args.success_reward} deadline_penalty={args.deadline_penalty} "
              f"server_overload_penalty={args.server_overload_penalty} no_block_penalty={args.no_block_penalty} "
              f"fs_penalty_coef={args.fs_penalty_coef}")
    elif args.objective == "pressure_aware_blocking_delay":
        print(f"  base_delay_coef={args.base_delay_coef} base_block_penalty={args.base_block_penalty} "
              f"base_no_block_penalty={args.base_no_block_penalty} pressure_penalty_coef={args.pressure_penalty_coef} "
              f"pressure_source={args.pressure_source} pressure_clip=[{args.pressure_clip_min},{args.pressure_clip_max}]")
    elif args.objective == "state_aware_blocking_delay":
        print(f"  base_delay_coef={args.base_delay_coef} base_block_penalty={args.base_block_penalty} "
              f"base_no_block_penalty={args.base_no_block_penalty} pressure_source={args.pressure_source} "
              f"pressure_clip=[{args.pressure_clip_min},{args.pressure_clip_max}]")
    print(f"Agent-C activation: {args.agent_c_activation}")
    print(f"Spectrum collapse penalty: coef={args.spectrum_collapse_penalty_coef}")
    print(f"Validation seeds: {eval_seed_map} (episodes_per_seed={args.validation_episodes})")
    print(f"Checkpoint metric: {args.checkpoint_metric}")
    print(f"Env config: slots={args.num_slots} k_paths={args.k_paths} "
          f"max_blocks={args.max_blocks} block_sort={args.block_sort_strategy}")
    print(f"Critic input dim: {critic.input_dim}")
    print(f"Critic request mean field: "
          f"enabled={getattr(args, 'critic_request_mean_field', False)}, "
          f"window={getattr(args, 'request_mean_field_window', 20)}")
    print(f"Checkpoint prefix: {args.ckpt_prefix}")
    print("=" * 76)

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
            traffic_mode=getattr(args, "traffic_mode", "iid"),
            regime_stay_prob=getattr(args, "regime_stay_prob", 0.9),
        )
        env.reset(requests)

        entropy_coef = _entropy_coef_for_episode(args, episode)
        agent_c.entropy_coef = entropy_coef

        # State-aware tracking
        ep_pressures = []
        ep_delay_weights = []
        ep_block_weights = []
        ep_no_block_weights = []
        ep_delays = []
        ep_splits = Counter()

        transitions, reasons, ep_stats, ep_aux = _run_training_episode(
            env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
        )
        blocked = ep_stats["blocked"]
        ep_pressures = ep_aux["pressures"]
        ep_delay_weights = ep_aux["delay_weights"]
        ep_block_weights = ep_aux["block_weights"]
        ep_no_block_weights = ep_aux["no_block_weights"]
        ep_delays = ep_aux["delays"]
        ep_splits = ep_aux["splits"]
        rollout = PPORolloutBuffer()
        for transition in transitions:
            rollout.add(transition)
        episode_credit = _apply_episode_level_credit(rollout, args)
        for key, value in episode_credit.items():
            metrics.setdefault(key, []).append(value)

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
        metrics["entropy_coef"].append(entropy_coef)

        c_features, c_masks, c_actions, c_old_logp, c_indices = _filter_actor_batch(
            transitions, "c"
        )
        if c_indices:
            c_loss, _, c_kl, _ = agent_c.optimize_ppo(
                c_features,
                c_masks,
                c_actions,
                c_old_logp,
                advantages[c_indices],
                args.clip_coef,
                epochs=args.ppo_epochs,
                target_kl=args.target_kl,
            )
            metrics["losses_c"].append(c_loss)
            metrics["kl_c"].append(c_kl)
            metrics["gate_loss"].append(agent_c.last_gate_loss)

        ep_blocking = blocked / max(len(requests), 1)
        metrics["episode_blocking"].append(ep_blocking)
        metrics["raw_mask_empty_rate"].append(ep_stats["raw_mask_empty"] / max(len(requests), 1))
        metrics["server_overload_rate"].append(ep_stats["server_overload"] / max(len(requests), 1))
        metrics["base_reward_mean"].append(ep_stats["base_reward_mean"])
        metrics["shaped_reward_mean"].append(ep_stats["shaped_reward_mean"])
        metrics["collapse_penalty_mean"].append(ep_stats["collapse_penalty_mean"])
        metrics["collapse_penalty_trigger_count"].append(ep_stats["collapse_penalty_trigger_count"])
        metrics["collapse_penalty_trigger_rate"].append(ep_stats["collapse_penalty_trigger_rate"])
        metrics["avg_k_c_valid_next"].append(ep_stats["avg_k_c_valid_next"])

        if (episode > 0 and episode % args.eval_freq == 0) or episode == args.episodes - 1:
            eval_results = {
                topo_name: _evaluate_validation(
                    eval_env,
                    agent_c,
                    frozen_r,
                    args.frozen_r_type,
                    eval_sets[topo_name],
                    args,
                )
                for topo_name, eval_env in eval_envs.items()
            }
            # Aggregate across topologies if multiple; with one topology this is a no-op.
            agg = {
                "mean_blocking": float(np.mean([r["aggregate"]["mean_blocking"] for r in eval_results.values()])),
                "std_blocking": float(np.mean([r["aggregate"]["std_blocking"] for r in eval_results.values()])),
                "mean_raw_mask_empty": float(np.mean([r["aggregate"]["mean_raw_mask_empty"] for r in eval_results.values()])),
                "mean_server_overload": float(np.mean([r["aggregate"]["mean_server_overload"] for r in eval_results.values()])),
                "mean_delay_ms": float(np.mean([r["aggregate"]["mean_delay_ms"] for r in eval_results.values()])),
                "mean_fs": float(np.mean([r["aggregate"]["mean_fs"] for r in eval_results.values()])),
                "mean_reward": float(np.mean([r["aggregate"]["mean_reward"] for r in eval_results.values()])),
                "mean_objective": float(np.mean([r["aggregate"]["mean_objective"] for r in eval_results.values()])),
            }
            metrics["eval_blocking"].append(agg["mean_blocking"])
            metrics["eval_reward"].append(agg["mean_reward"])
            metrics["eval_mean_blocking"].append(agg["mean_blocking"])
            metrics["eval_std_blocking"].append(agg["std_blocking"])
            metrics["eval_mean_raw_mask_empty"].append(agg["mean_raw_mask_empty"])
            metrics["eval_mean_server_overload"].append(agg["mean_server_overload"])
            metrics.setdefault("eval_delay", []).append(agg["mean_delay_ms"])
            metrics.setdefault("eval_objective", []).append(agg["mean_objective"])
            last_validation_agg = agg

            metric_value = agg["mean_objective"] if args.checkpoint_metric == "mean_objective" else agg["mean_blocking"]
            if metric_value < best_metric:
                best_metric = metric_value
                best_episode = episode
                _save_c_checkpoint(best_path, agent_c, critic, args, metrics, best_validation=agg)
                per_topo = ", ".join(
                    f"{topo}=blk{r['aggregate']['mean_blocking']:.3f}/d{r['aggregate']['mean_delay_ms']:.1f}ms/obj{r['aggregate']['mean_objective']:.4f}"
                    for topo, r in eval_results.items()
                )
                print(f"  >> Best frozen-R C @ ep {episode}: "
                      f"mean_blk={agg['mean_blocking']:.3f} std_blk={agg['std_blocking']:.3f} "
                      f"mean_rme={agg['mean_raw_mask_empty']:.3f} mean_so={agg['mean_server_overload']:.3f} "
                      f"d={agg['mean_delay_ms']:.1f}ms obj={agg['mean_objective']:.4f} | {per_topo}")
            if args.print_eval_diagnostics:
                all_reasons = Counter()
                all_mods = Counter()
                all_splits = Counter()
                all_servers = Counter()
                for res in eval_results.values():
                    all_reasons.update(res["reasons"])
                    all_mods.update(res["mods"])
                    all_splits.update(res["splits"])
                    all_servers.update(res["servers"])
                print(f"         Eval reasons: {_fmt_counter(all_reasons)}")
                print(f"         Eval mods: {_fmt_counter(all_mods)}")
                print(f"         Eval splits: {_fmt_counter(all_splits)}")
                print(f"         Eval servers: {_fmt_counter(all_servers)}")

        if episode % args.log_interval == 0 or episode == args.episodes - 1:
            reason_str = ", ".join(f"{k}={v}" for k, v in reasons.most_common(4))
            c_kl = metrics["kl_c"][-1] if metrics["kl_c"] else 0.0
            gate_info = ""
            if args.agent_c_feature_mode == "gated_typed_mean_field" and metrics["gate_loss"]:
                gate_info = f" gate_loss={metrics['gate_loss'][-1]:.4f}"
            collapse_info = ""
            if args.spectrum_collapse_penalty_coef != 0.0:
                collapse_info = (
                    f" collapse_trigger={ep_stats['collapse_penalty_trigger_count']}"
                    f"({ep_stats['collapse_penalty_trigger_rate']:.2%})"
                    f" avg_knext={ep_stats['avg_k_c_valid_next']:.1f}"
                )
            print(f"Ep {episode:4d} [{topo}] blk={ep_blocking:.2f} "
                  f"rme={ep_stats['raw_mask_empty']/max(len(requests),1):.2f} "
                  f"so={ep_stats['server_overload']/max(len(requests),1):.2f} "
                  f"base_r={ep_stats['base_reward_mean']:+.3f} "
                  f"shape_r={ep_stats['shaped_reward_mean']:+.3f} "
                  f"ent={entropy_coef:.4f} kl_c={c_kl:.4f}{gate_info}{collapse_info}")
            if args.episode_credit_mode != "off":
                print(
                    f"         ep_credit penalty={episode_credit['episode_credit_penalty']:.3f} "
                    f"blk={episode_credit['episode_credit_blocking']:.3f} "
                    f"no_block={episode_credit['episode_credit_no_block']:.3f} "
                    f"d_norm={episode_credit['episode_credit_delay_norm']:.3f}"
                )
            if args.objective == "pressure_aware_blocking_delay" and ep_pressures:
                avg_p = float(np.mean(ep_pressures))
                avg_d = float(np.mean(ep_delays)) if ep_delays else 0.0
                total_s = sum(ep_splits.values())
                split_str = ", ".join(f"{k}={v/total_s:.1%}" for k, v in ep_splits.most_common()) if total_s > 0 else "none"
                print(f"         pressure={avg_p:.3f} avg_delay={avg_d:.1f}ms | {split_str}")
            elif args.objective == "state_aware_blocking_delay" and ep_pressures:
                avg_p = float(np.mean(ep_pressures))
                avg_dw = float(np.mean(ep_delay_weights))
                avg_bw = float(np.mean(ep_block_weights))
                avg_nbw = float(np.mean(ep_no_block_weights))
                avg_d = float(np.mean(ep_delays)) if ep_delays else 0.0
                total_s = sum(ep_splits.values())
                split_str = ", ".join(f"{k}={v/total_s:.1%}" for k, v in ep_splits.most_common()) if total_s > 0 else "none"
                print(f"         pressure={avg_p:.3f} dw={avg_dw:.3f} bw={avg_bw:.3f} nbw={avg_nbw:.3f} "
                      f"avg_delay={avg_d:.1f}ms | {split_str}")
            if reason_str:
                print(f"         Reasons: {reason_str}")

    _save_c_checkpoint(last_path, agent_c, critic, args, metrics, best_validation=last_validation_agg)
    print(f"Best frozen-R Agent-C checkpoint: {best_path}")
    print(f"Best validation {args.checkpoint_metric}: {best_metric:.4f} @ ep {best_episode}")
    return agent_c, critic, metrics


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent-C with frozen Agent-R")
    parser.add_argument("--topologies", type=str, default="metro24_bottleneck")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--frozen_r_checkpoint", type=str, default=None)
    parser.add_argument("--frozen_r_type", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--warm_start_c", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--requests_per_episode", type=int, default=60)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--traffic_mode", type=str, default="iid",
                        choices=["iid", "markov_regime"],
                        help="IID traffic or temporally persistent demand regimes.")
    parser.add_argument("--regime_stay_prob", type=float, default=0.9,
                        help="Probability that markov_regime keeps its current demand regime.")
    parser.add_argument("--critic_request_mean_field", action="store_true",
                        help="Append rolling request demand distribution to critic only.")
    parser.add_argument("--request_mean_field_window", type=int, default=20,
                        help="Number of prior requests used by critic request mean field.")
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--c_valid_action_pressure_coef", type=float, default=0.2)
    parser.add_argument("--c_spectrum_block_penalty", type=float, default=0.3)
    parser.add_argument("--c_fs_request_penalty_coef", type=float, default=0.0)
    parser.add_argument("--c_safe_fs_penalty_coef", type=float, default=0.0)
    parser.add_argument("--c_spectrum_pressure_penalty_coef", type=float, default=0.0)
    parser.add_argument("--c_path_penalty_coef", type=float, default=0.0)
    parser.add_argument("--c_delay_risk_penalty_coef", type=float, default=0.0)
    parser.add_argument("--c_path_norm_km", type=float, default=600.0)
    parser.add_argument("--c_path_metric", type=str, default="min", choices=["min", "mean"])
    parser.add_argument("--c_max_spectrum_pressure", type=float, default=0.0)
    parser.add_argument("--c_max_path_km", type=float, default=0.0)
    parser.add_argument("--c_max_safe_fs_ratio", type=float, default=0.0)
    parser.add_argument("--c_min_valid_after_risk_mask", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--entropy_final_coef", type=float, default=0.001)
    parser.add_argument("--entropy_decay_episodes", type=int, default=1200)
    parser.add_argument("--value_clip_coef", type=float, default=0.2)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--no_advantage_norm", action="store_true")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", type=str, default="default3",
                        choices=["default3", "complex5", "complex5_v2", "complex5_v2_lite"])
    parser.add_argument("--agent_c_feature_mode", type=str, default="default",
                        choices=[
                            "default",
                            "enhanced",
                            "pressure_aware",
                            "cross_pressure",
                            "r_feasibility",
                            "r_feasibility_safe",
                            "r_feasibility_spectrum_impact",
                            "r_feasibility_mr_spec_compute",
                            "mr_feasibility_rule",
                            "mean_field",
                            "typed_mean_field",
                            "gated_typed_mean_field",
                            "fixed_blend_typed_mean_field",
                            "candidate_mean_field",
                            "candidate_mean_field_count_only",
                        ])
    parser.add_argument("--fixed_blend_alpha", type=float, default=0.5,
                        help="Fixed blend weight for typed mean-field features "
                             "(only used with fixed_blend_typed_mean_field).")
    parser.add_argument("--gate_reg_coef", type=float, default=0.0,
                        help="Coefficient for gated policy binary gate regularization "
                             "(gate*(1-gate)). Only used with gated_typed_mean_field.")
    parser.add_argument("--agent_c_activation", type=str, default="tanh",
                        choices=["tanh", "relu", "silu", "gelu", "leaky_relu"],
                        help="Activation function for Agent-C policy network.")
    parser.add_argument("--k_paths", type=int, default=3,
                        help="Number of shortest paths for R action space.")
    parser.add_argument("--max_blocks", type=int, default=5,
                        help="Number of candidate spectrum blocks per path/mod.")
    parser.add_argument("--block_sort_strategy", type=str, default="size_desc",
                        choices=["size_desc", "waste_asc", "start_asc", "center_asc", "mixed"],
                        help="Candidate block ranking strategy.")
    parser.add_argument("--agent_c_input_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--num_validation_seeds", type=int, default=1,
                        help="Number of deterministic validation seeds per topology.")
    parser.add_argument("--validation_seeds", type=str, default="1001,1002,1003",
                        help="Comma-separated validation seeds. Same seeds are used for every train seed.")
    parser.add_argument("--validation_episodes", type=int, default=5,
                        help="Number of validation episodes per validation seed.")
    parser.add_argument("--checkpoint_metric", type=str, default="mean_blocking",
                        choices=["mean_blocking", "mean_objective"],
                        help="Metric used to select the best checkpoint.")
    parser.add_argument("--spectrum_collapse_penalty_coef", type=float, default=0.0,
                        help="Penalty applied to the previous transition when the next decision state's raw C-mask is empty.")
    parser.add_argument("--print_eval_diagnostics", action="store_true")
    parser.add_argument("--ckpt_prefix", type=str, default="agent_c_frozen_r")
    parser.add_argument("--objective", type=str, default="default",
                        choices=["default", "blocking_delay", "state_aware_blocking_delay", "pressure_aware_blocking_delay"])
    parser.add_argument("--delay_coef", type=float, default=0.5)
    parser.add_argument("--block_penalty", type=float, default=2.0)
    parser.add_argument("--success_reward", type=float, default=1.0)
    parser.add_argument("--deadline_penalty", type=float, default=1.0)
    parser.add_argument("--server_overload_penalty", type=float, default=1.0)
    parser.add_argument("--no_block_penalty", type=float, default=1.2)
    parser.add_argument("--fs_penalty_coef", type=float, default=0.0)
    parser.add_argument("--base_delay_coef", type=float, default=0.8)
    parser.add_argument("--base_block_penalty", type=float, default=2.0)
    parser.add_argument("--base_no_block_penalty", type=float, default=1.2)
    parser.add_argument("--pressure_source", type=str, default="auto",
                        choices=["auto", "feasible_count", "pressure", "required_over_lfb"])
    parser.add_argument("--pressure_clip_min", type=float, default=0.0)
    parser.add_argument("--pressure_clip_max", type=float, default=1.0)
    parser.add_argument("--pressure_penalty_coef", type=float, default=0.5)
    parser.add_argument("--server_util_coef", type=float, default=0.0)
    parser.add_argument("--spectrum_fail_extra", type=float, default=0.0)
    parser.add_argument("--server_fail_extra", type=float, default=0.0)
    parser.add_argument("--best_metric", type=str, default="blocking",
                        choices=["blocking", "objective"])
    parser.add_argument("--episode_credit_mode", type=str, default="off",
                        choices=["off", "uniform", "failure", "pressure", "mixed"],
                        help="Assign episode-level blocking/delay outcome back to C decisions before GAE.")
    parser.add_argument("--episode_credit_scale", type=float, default=1.0)
    parser.add_argument("--episode_blocking_coef", type=float, default=1.0)
    parser.add_argument("--episode_no_block_coef", type=float, default=1.0)
    parser.add_argument("--episode_delay_coef", type=float, default=0.5)
    args = parser.parse_args()

    if args.frozen_r_checkpoint is None:
        package_root = Path(__file__).resolve().parents[2]
        args.frozen_r_checkpoint = str(
            package_root / "checkpoints" / "joint_mappo_v2_cshape_metro24_fs002_r_best.pt"
        )
    if args.value_clip_coef is not None and args.value_clip_coef < 0:
        args.value_clip_coef = None
    if args.target_kl is not None and args.target_kl <= 0:
        args.target_kl = None
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        args.max_grad_norm = None
    train(args)


if __name__ == "__main__":
    main()
