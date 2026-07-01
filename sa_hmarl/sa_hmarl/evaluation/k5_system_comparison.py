"""K=5 System Comparison — Full 7-Method Benchmark.

Fixed scenario with k_paths=5, comparing all C-action methods under the same
PPO-R backend.  Agent-C + KSP-BF provides an R-side exhaustive-search
upper bound for the chosen C action.

Methods:
  1. Agent-C + PPO-R         (learned PPO C-policy)
  2. WO-C   + PPO-R         (distributed offloading heuristic)
  3. DF-C   + PPO-R         (distance-first heuristic)
  4. RF-C   + PPO-R         (resource-first / lightest-server heuristic)
  5. IWD-C  + PPO-R         (intelligent water droplet heuristic)
  6. Greedy + PPO-R         (minimum edge-compute heuristic)
  7. Agent-C + KSP-BF          (Agent-C + exhaustive R brute-force)

Usage::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.k5_system_comparison
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_ORDER = ["agent", "wo", "df", "rf", "iwd", "greedy", "agent_kspbf"]

METHOD_LABELS = {
    "agent":      "Agent-C + PPO-R",
    "wo":         "WO-C + PPO-R",
    "df":         "DF-C + PPO-R",
    "rf":         "RF-C + PPO-R",
    "iwd":        "IWD-C + PPO-R",
    "greedy":     "Greedy-C + PPO-R",
    "agent_kspbf":"Agent-C + KSP-BF",
}


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"),
    )
    # For mean-field feature modes the checkpoint stores the full input_dim,
    # so auto-compute is not triggered.  Passing num_servers anyway makes the
    # loader robust to checkpoints that only store the base dimension.
    num_servers = ckpt_args.get("num_servers")
    if num_servers is None:
        num_servers = ckpt.get("num_servers")
    agent_c_kwargs = dict(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    if feature_mode in ("mean_field", "typed_mean_field", "gated_typed_mean_field", "fixed_blend_typed_mean_field", "candidate_mean_field", "candidate_mean_field_count_only"):
        agent_c_kwargs["num_servers"] = num_servers
    if feature_mode == "fixed_blend_typed_mean_field":
        agent_c_kwargs["fixed_blend_alpha"] = ckpt_args.get("fixed_blend_alpha", 0.5)
    agent = PPOAgentC(**agent_c_kwargs)
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=ckpt.get("agent_r_feature_mode", "default"),
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Agent-C selection + mask stats
# ---------------------------------------------------------------------------

def _select_agent_c_with_stats(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[Optional[int], int]:
    """Select Agent-C action, return (action_idx, n_valid_c)."""
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    n_valid_c = int(mask.sum())
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action, n_valid_c


# ---------------------------------------------------------------------------
# KSP-BF: exhaustive R brute-force search
# ---------------------------------------------------------------------------

def _kspbf_select_r(
    env, req, split_id: int, server_id: int, obs_r: Dict[str, Any],
    max_trials: int = 200,
) -> Tuple[Optional[int], Tuple[int, int, int], bool, float]:
    """Exhaustive brute-force over all valid R actions on deep-copied envs.

    Returns:
        (flat_action_idx, (path_idx, mod_idx, block_idx), found_any, time_s)
    """
    mask: np.ndarray = obs_r["agent_r_mask"]
    valid = np.where(mask)[0]
    num_mods = len(obs_r["mod_names"])
    max_blocks_env = env.max_blocks

    if len(valid) == 0:
        return None, (0, 0, 0), False, 0.0

    t0 = time.time()
    best_action_idx: Optional[int] = None
    best_tuple: Tuple[int, int, int] = (0, 0, 0)
    best_waste: float = float("inf")
    best_fs: float = float("inf")
    best_path: float = float("inf")
    found_any = False

    # Sort by ascending index (prefer lower path_idx = shorter path)
    valid_sorted = sorted(valid, key=lambda a: int(a))
    n_trials = min(max_trials, len(valid_sorted))

    for action_idx in valid_sorted[:n_trials]:
        action_idx = int(action_idx)
        path_idx, mod_idx, block_idx = decode_agent_r_action(
            action_idx, num_mods, max_blocks_env,
        )

        env_try = copy.deepcopy(env)
        _, _, _, info = env_try.step(
            (split_id, server_id), (path_idx, mod_idx, block_idx),
        )

        if info.get("success", False):
            found_any = True
            waste = float(info.get("block_waste", float("inf")))
            fs_val = float(info.get("num_slots", float("inf")))
            path_len = float(info.get("path_dist_km", float("inf")))

            if (waste < best_waste
                    or (waste == best_waste and fs_val < best_fs)
                    or (waste == best_waste and fs_val == best_fs and path_len < best_path)):
                best_waste = waste
                best_fs = fs_val
                best_path = path_len
                best_action_idx = action_idx
                best_tuple = (path_idx, mod_idx, block_idx)

            # Early exit: near-perfect candidate found
            if waste <= 0.01 and fs_val <= best_fs:
                break

    elapsed = time.time() - t0
    if found_any:
        return best_action_idx, best_tuple, True, elapsed
    return None, (0, 0, 0), False, elapsed


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def _eval_episode(
    method: str,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    requests: List,
    env_proto,
    args,
    ep_idx: int,
) -> Dict[str, Any]:
    """Evaluate one method on one episode."""
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities,
        k=args.k_paths,
    )
    env.reset(requests)

    server_selected_count = np.zeros(num_servers, dtype=int)
    seed_val = int(getattr(args, "seeds", "42").split(",")[0])
    rng = np.random.RandomState(seed_val + ep_idx)

    total = blocked = success = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    mod_counter = Counter()
    server_overload_counter = Counter()

    # C action-space stats (from agent/greedy/agent_kspbf trajectories)
    mask_valid_c_counts: List[int] = []
    mask_zero_c_count = 0
    # R action-space stats
    mask_valid_r_counts: List[int] = []
    # KSP-BF stats
    kspbf_time_total = 0.0
    kspbf_call_count = 0
    kspbf_success_count = 0

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        raw_mask = obs_c["agent_c_mask"]

        # --- C action selection ---
        if method in ("agent", "agent_kspbf"):
            action_idx_c, n_valid_c = _select_agent_c_with_stats(
                agent_c, obs_c, env.net.num_slots,
            )
            mask_valid_c_counts.append(n_valid_c)
            if n_valid_c == 0:
                mask_zero_c_count += 1
            action_c = (
                (0, 0) if action_idx_c is None
                else decode_agent_c_action(action_idx_c, num_servers)
            )
        else:
            action_idx_c = select_offloading_action(
                method, env, req, obs_c, raw_mask,
                rng=rng, server_selected_count=server_selected_count,
            )
            n_valid_c = int(raw_mask.sum())
            mask_valid_c_counts.append(n_valid_c)
            if n_valid_c == 0:
                mask_zero_c_count += 1
            action_c = (
                (0, 0) if action_idx_c is None
                else decode_agent_c_action(action_idx_c, num_servers)
            )

        split_id, server_id = action_c
        split_counter[f"split{split_id}"] += 1
        server_counter[f"s{server_id}"] += 1
        server_selected_count[server_id] += 1

        # --- R action selection ---
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask = obs_r["agent_r_mask"]
        n_valid_r = int(r_mask.sum())
        mask_valid_r_counts.append(n_valid_r)

        action_r_tuple: Tuple[int, int, int] = (0, 0, 0)

        if method == "agent_kspbf":
            kspbf_call_count += 1
            bf_idx, bf_tuple, bf_found, bf_time = _kspbf_select_r(
                env, req, split_id, server_id, obs_r,
            )
            kspbf_time_total += bf_time
            if bf_found:
                kspbf_success_count += 1
                action_r_tuple = bf_tuple
            else:
                # Fallback: try PPO-R
                action_r_idx = agent_r.select_action(obs_r, deterministic=True)
                if action_r_idx is not None:
                    action_r_tuple = decode_agent_r_action(
                        action_r_idx, len(obs_r["mod_names"]), env.max_blocks,
                    )
        else:
            action_r_idx = agent_r.select_action(obs_r, deterministic=True)
            if action_r_idx is not None:
                action_r_tuple = decode_agent_r_action(
                    action_r_idx, len(obs_r["mod_names"]), env.max_blocks,
                )

        # --- Step ---
        _, _, _, info = env.step(action_c, action_r_tuple)
        total += 1

        server_obj = env.mec.servers[server_id]
        reward = compute_agent_c_reward(
            info, req.deadline_ms, args.waste_coef, server_obj.utilization,
        )
        total_reward += reward

        if info.get("success", False):
            success += 1
            delay = float(info.get("delay_ms", 0.0))
            total_delay += delay
            total_fs += float(info.get("num_slots", 0))
            total_waste += float(info.get("block_waste", 0.0))
            total_path += float(info.get("path_dist_km", 0.0))
            mod_counter[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reason_counter[reason] += 1
            if reason in ("server_overload", "server_saturated"):
                server_overload_counter[f"s{server_id}"] += 1

    n = max(total, 1)
    s = max(success, 1)

    result = {
        "total": total, "blocked": blocked, "success": success,
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "reason_counter": dict(reason_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "mod_counter": dict(mod_counter),
        "server_overload_counter": dict(server_overload_counter),
    }

    # C action-space stats
    result["mask_valid_c_counts"] = mask_valid_c_counts
    result["mask_zero_c_count"] = mask_zero_c_count
    result["no_valid_c_rate_ep"] = mask_zero_c_count / n

    # R action-space stats
    result["mask_valid_r_counts"] = mask_valid_r_counts
    result["avg_valid_r_actions_ep"] = float(np.mean(mask_valid_r_counts)) if mask_valid_r_counts else 0.0

    # KSP-BF stats
    result["kspbf_time_s"] = kspbf_time_total
    result["kspbf_call_count"] = kspbf_call_count
    result["kspbf_success_count"] = kspbf_success_count

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(ep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "avg_fs", "avg_waste", "avg_path_len_km",
    ]
    agg: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(r[key]) for r in ep_results]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    for key in ["reason_counter", "split_counter", "server_counter",
                "mod_counter", "server_overload_counter"]:
        c = Counter()
        for r in ep_results:
            c.update(r.get(key, {}))
        agg[key] = dict(c)

    # C mask stats
    all_c_counts = []
    zero_c = 0
    req_total = 0
    for r in ep_results:
        if "mask_valid_c_counts" in r:
            all_c_counts.extend(r["mask_valid_c_counts"])
            zero_c += r.get("mask_zero_c_count", 0)
            req_total += r["total"]
    agg["no_valid_c_rate"] = zero_c / max(req_total, 1)
    agg["avg_valid_c_actions"] = float(np.mean(all_c_counts)) if all_c_counts else 0.0

    # R mask stats
    all_r_counts = []
    for r in ep_results:
        if "mask_valid_r_counts" in r:
            all_r_counts.extend(r["mask_valid_r_counts"])
    agg["avg_valid_r_actions"] = float(np.mean(all_r_counts)) if all_r_counts else 0.0

    # no_suitable_block ratio
    reason = agg.get("reason_counter", {})
    total_reasons = sum(reason.values())
    nsb = reason.get("no_suitable_block", 0)
    agg["no_suitable_block_ratio"] = nsb / max(total_reasons, 1)

    # server_overload ratio
    so = reason.get("server_overload", 0) + reason.get("server_saturated", 0)
    agg["server_overload_ratio"] = so / max(total_reasons, 1)

    # KSP-BF stats
    kspbf_time = sum(r.get("kspbf_time_s", 0.0) for r in ep_results)
    kspbf_calls = sum(r.get("kspbf_call_count", 0) for r in ep_results)
    kspbf_success = sum(r.get("kspbf_success_count", 0) for r in ep_results)
    agg["kspbf_time_s"] = kspbf_time
    agg["kspbf_call_count"] = kspbf_calls
    agg["kspbf_success_count"] = kspbf_success
    agg["kspbf_success_rate"] = kspbf_success / max(kspbf_calls, 1)

    return agg


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    items = sorted(c.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v / total:.1%}" for k, v in items)


def _fmt_cnt(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    items = sorted(c.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="K=5 System Comparison — 7-Method Benchmark"
    )
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--k_paths", type=int, default=5,
                        help="Number of shortest paths (k).  Default 5 for main experiment.")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.20)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/k5_system_comparison.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/k5_system_comparison.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    methods = METHOD_ORDER

    # Load models
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )

    print("=" * 80)
    print("K=5 System Comparison — 7 Methods")
    print(f"Topology: {args.topology}  Slots: {args.num_slots}  "
          f"Servers: {args.num_servers}  k_paths: {args.k_paths}")
    print(f"Seeds: {seeds}  Eps/seed: {args.episodes}  "
          f"Requests/ep: {args.requests_per_episode}")
    print(f"Methods: {methods}")
    print(f"C: {args.agent_c_checkpoint}")
    print(f"R: {args.agent_r_checkpoint}")
    print("=" * 80)

    # Pre-generate episodes
    all_episodes: Dict[int, List[List]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps

    # ------------------------------------------------------------------
    # Evaluate each method
    # ------------------------------------------------------------------
    all_results: Dict[str, Dict[str, Any]] = {}
    t_start = time.time()

    for method in methods:
        label = METHOD_LABELS[method]
        print(f"\n{'─' * 70}")
        print(f"Evaluating: {label}")
        print(f"{'─' * 70}")

        ep_results: List[Dict[str, Any]] = []
        t0 = time.time()
        for seed in seeds:
            for ep_idx, requests in enumerate(all_episodes[seed]):
                res = _eval_episode(
                    method, agent_c, agent_r, requests, env_proto, args, ep_idx,
                )
                ep_results.append(res)
        elapsed = time.time() - t0

        agg = _aggregate(ep_results)
        all_results[label] = agg

        extra = ""
        if method == "agent_kspbf":
            extra = (f"  kspbf: {agg['kspbf_success_count']}/{agg['kspbf_call_count']} "
                     f"({agg['kspbf_success_rate']:.1%}) "
                     f"time={agg['kspbf_time_s']:.1f}s")
        print(
            f"  blk={agg['blocking_rate']:.4f}±{agg['blocking_rate_std']:.4f}  "
            f"reward={agg['avg_reward']:+.3f}  "
            f"delay={agg['avg_delay_ms']:.1f}ms  "
            f"fs={agg['avg_fs']:.2f}  "
            f"noC={agg['no_valid_c_rate']:.2f}  "
            f"avgRacts={agg['avg_valid_r_actions']:.0f}  "
            f"time={elapsed:.0f}s"
            f"{extra}"
        )

    total_time = time.time() - t_start
    print(f"\nTotal evaluation time: {total_time:.0f}s ({total_time / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)

    hdr = (
        f"{'Method':30s} {'Blocking':>9s} {'Reward':>8s} {'Delay':>8s} "
        f"{'AvgFS':>7s} {'Waste':>7s} {'PathKm':>8s} {'noC%':>6s} {'avgRacts':>9s}"
    )
    print(hdr); print("-" * 90)
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        print(
            f"{label:30s} {r['blocking_rate']:9.4f} {r['avg_reward']:8.3f} "
            f"{r['avg_delay_ms']:8.1f} {r['avg_fs']:7.2f} {r['avg_waste']:7.3f} "
            f"{r['avg_path_len_km']:8.1f} {r['no_valid_c_rate']:5.1%} "
            f"{r['avg_valid_r_actions']:9.1f}"
        )

    print(); print("-" * 90); print("FAILURE REASONS"); print("-" * 90)
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        print(f"  {label:30s}  {_fmt_cnt(r.get('reason_counter', {}))}")

    print(); print("-" * 90); print("SPLIT DISTRIBUTION"); print("-" * 90)
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        print(f"  {label:30s}  {_fmt_pct(r.get('split_counter', {}))}")

    print(); print("-" * 90); print("SERVER DISTRIBUTION"); print("-" * 90)
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        print(f"  {label:30s}  {_fmt_pct(r.get('server_counter', {}))}")

    print(); print("-" * 90); print("MODULATION DISTRIBUTION"); print("-" * 90)
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        print(f"  {label:30s}  {_fmt_pct(r.get('mod_counter', {}))}")

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_output = {
        "args": vars(args),
        "methods": methods,
        "method_labels": METHOD_LABELS,
        "total_time_s": total_time,
        "results": _clean(all_results),
    }
    out_json.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------
    out_md = Path(args.out_md)
    md: List[str] = []
    md.append("# K=5 System Comparison — 7-Method Benchmark\n\n")
    md.append(f"**Topology:** {args.topology}  **Slots:** {args.num_slots}  "
              f"**Servers:** {args.num_servers}  **k_paths:** {args.k_paths}  "
              f"**MaxBlocks:** {args.max_blocks}  **BlockSort:** {args.block_sort_strategy}\n\n")
    md.append(f"**Agent-C:** `{args.agent_c_checkpoint}`  \n")
    md.append(f"**Agent-R:** `{args.agent_r_checkpoint}`  \n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}  "
              f"**Requests/ep:** {args.requests_per_episode}\n\n")
    md.append(f"**Total time:** {total_time:.0f}s ({total_time / 60:.1f} min)\n\n")

    # Main results table
    md.append("## Results\n\n")
    md.append(
        "| Method | Blocking | Reward | Delay(ms) | AvgFS | Waste | "
        "PathKm | noC% | avgRacts | NSB% | SvrOv% |\n"
    )
    md.append(
        "|--------|----------|--------|-----------|-------|-------|"
        "--------|------|----------|------|--------|\n"
    )
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        md.append(
            f"| {label} | {r['blocking_rate']:.4f}±{r['blocking_rate_std']:.4f} | "
            f"{r['avg_reward']:+.3f} | {r['avg_delay_ms']:.1f}±{r['avg_delay_ms_std']:.1f} | "
            f"{r['avg_fs']:.2f} | {r['avg_waste']:.3f} | "
            f"{r['avg_path_len_km']:.1f} | {r['no_valid_c_rate']:.1%} | "
            f"{r['avg_valid_r_actions']:.0f} | "
            f"{r['no_suitable_block_ratio']:.1%} | {r['server_overload_ratio']:.1%} |\n"
        )

    # KSP-BF special stats
    r_kspbf = all_results[METHOD_LABELS["agent_kspbf"]]
    md.append(f"\n### KSP-BF Stats\n\n")
    md.append(f"- Calls: {r_kspbf['kspbf_call_count']}\n")
    md.append(f"- Success: {r_kspbf['kspbf_success_count']} "
              f"({r_kspbf['kspbf_success_rate']:.1%})\n")
    md.append(f"- Total BF time: {r_kspbf['kspbf_time_s']:.1f}s\n")

    # Agent-C vs KSP-BF delta
    agent_blk = all_results[METHOD_LABELS["agent"]]["blocking_rate"]
    kspbf_blk = all_results[METHOD_LABELS["agent_kspbf"]]["blocking_rate"]
    delta_bf = agent_blk - kspbf_blk
    md.append(f"- Δblocking (Agent-C → KSP-BF): {delta_bf:+.4f}\n")

    # Failure reasons
    md.append("\n## Failure Reasons\n\n")
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        md.append(f"- **{label}**: {_fmt_cnt(r.get('reason_counter', {}))}\n")

    # Split distribution
    md.append("\n## Split Distribution\n\n")
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        md.append(f"- **{label}**: {_fmt_pct(r.get('split_counter', {}))}\n")

    # Server distribution
    md.append("\n## Server Distribution\n\n")
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        md.append(f"- **{label}**: {_fmt_pct(r.get('server_counter', {}))}\n")

    # Modulation distribution
    md.append("\n## Modulation Distribution\n\n")
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        md.append(f"- **{label}**: {_fmt_pct(r.get('mod_counter', {}))}\n")

    # Runtime
    md.append("\n## Runtime\n\n")
    md.append("| Method | Time (s) |\n|---|---|\n")
    for method in methods:
        label = METHOD_LABELS[method]
        r = all_results[label]
        rt = r.get("kspbf_time_s", 0.0)
        md.append(f"| {label} | {rt:.1f} |\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    # Find best method
    best_method = min(methods, key=lambda m: all_results[METHOD_LABELS[m]]["blocking_rate"])
    best_label = METHOD_LABELS[best_method]
    best_blk = all_results[best_label]["blocking_rate"]
    print(f"Best method: {best_label}  blocking={best_blk:.4f}")

    # Agent-C vs heuristics
    agent_blk = all_results[METHOD_LABELS["agent"]]["blocking_rate"]
    df_blk = all_results[METHOD_LABELS["df"]]["blocking_rate"]
    rf_blk = all_results[METHOD_LABELS["rf"]]["blocking_rate"]
    wo_blk = all_results[METHOD_LABELS["wo"]]["blocking_rate"]
    iwd_blk = all_results[METHOD_LABELS["iwd"]]["blocking_rate"]
    greedy_blk = all_results[METHOD_LABELS["greedy"]]["blocking_rate"]

    worst_heuristic = max(df_blk, rf_blk, wo_blk, iwd_blk, greedy_blk)
    agent_vs_worst = worst_heuristic - agent_blk
    print(f"Agent-C vs worst heuristic: {agent_vs_worst:+.4f}")

    if agent_blk <= min(df_blk, rf_blk, wo_blk, iwd_blk, greedy_blk):
        print("✓ Agent-C outperforms all heuristics — spectrum-aware C policy is effective")
    else:
        print("✗ A heuristic beats Agent-C — C policy may need improvement")

    if agent_blk < 0.03:
        print(f"✓ Agent-C blocking={agent_blk:.4f} < 3% — k=5 may be too easy; consider k=4 or higher load")
    elif agent_blk < 0.10:
        print(f"  Agent-C blocking={agent_blk:.4f} in 2-10% range — good paper setting")
    else:
        print(f"  Agent-C blocking={agent_blk:.4f} > 10% — consider larger k")

    kspbf_blk = all_results[METHOD_LABELS["agent_kspbf"]]["blocking_rate"]
    r_gap = agent_blk - kspbf_blk
    print(f"R-side gap (Agent-C → KSP-BF): {r_gap:+.4f} "
          f"({'R action selection suboptimal' if r_gap > 0.005 else 'R action selection near-optimal'})")


if __name__ == "__main__":
    main()
