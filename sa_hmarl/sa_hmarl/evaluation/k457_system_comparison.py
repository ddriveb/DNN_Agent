"""K=4/5/7 System Comparison with Per-Seed Breakdown.

Sweeps k_paths = [4, 5, 7] with 5 diverse seeds, tracking per-seed
statistics to diagnose source-node difficulty effects.

Methods per k:
  - Agent-C + PPO-R
  - IWD-C  + PPO-R
  - DF-C   + PPO-R
  - Agent-C + KSP-BF

Output includes per-seed blocking, noC%, avg valid C/R actions,
failure reasons, and source-node info.

Usage::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.k457_system_comparison
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

K_VALUES = [4, 5, 7]
METHOD_ORDER = ["agent", "iwd", "df", "agent_kspbf"]

METHOD_LABELS = {
    "agent":       "Agent-C + PPO-R",
    "iwd":         "IWD-C + PPO-R",
    "df":          "DF-C + PPO-R",
    "agent_kspbf": "Agent-C + KSP-BF",
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
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=feature_mode,
    )
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
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask, num_slots_total=num_slots,
            min_valid_after_mask=min_valid, **risk_kwargs,
        )
    n_valid_c = int(mask.sum())
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action, n_valid_c


# ---------------------------------------------------------------------------
# KSP-BF
# ---------------------------------------------------------------------------

def _kspbf_select_r(
    env, req, split_id: int, server_id: int, obs_r: Dict[str, Any],
    max_trials: int = 100,
) -> Tuple[Optional[int], Tuple[int, int, int], bool, float]:
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
                best_waste = waste; best_fs = fs_val; best_path = path_len
                best_action_idx = action_idx
                best_tuple = (path_idx, mod_idx, block_idx)
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
    method: str, agent_c: PPOAgentC, agent_r: PPOAgentR,
    requests: List, env_proto, args, ep_idx: int, seed: int,
) -> Dict[str, Any]:
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities, k=args.k_paths,
    )
    env.reset(requests)

    server_selected_count = np.zeros(num_servers, dtype=int)
    rng = np.random.RandomState(seed + ep_idx)

    total = blocked = success = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    mod_counter = Counter()

    mask_valid_c_counts: List[int] = []
    mask_zero_c_count = 0
    mask_valid_r_counts: List[int] = []

    # Source node (same for all requests in this episode)
    source_node = requests[0].src_node if requests else -1

    kspbf_time_total = 0.0
    kspbf_call_count = 0
    kspbf_success_count = 0

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        raw_mask = obs_c["agent_c_mask"]

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
            reason_counter[info.get("reason", "unknown")] += 1

    n = max(total, 1)
    s = max(success, 1)

    return {
        "seed": seed, "ep_idx": ep_idx,
        "source_node": source_node,
        "total": total, "blocked": blocked, "success": success,
        "blocking_rate": blocked / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "reason_counter": dict(reason_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "mod_counter": dict(mod_counter),
        "mask_valid_c_counts": mask_valid_c_counts,
        "mask_valid_r_counts": mask_valid_r_counts,
        "no_valid_c_rate_ep": mask_zero_c_count / n,
        "kspbf_time_s": kspbf_time_total,
        "kspbf_call_count": kspbf_call_count,
        "kspbf_success_count": kspbf_success_count,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_per_seed(ep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate episodes for a single seed."""
    scalar_keys = [
        "blocking_rate", "avg_reward", "avg_delay_ms",
        "avg_fs", "avg_waste", "avg_path_len_km",
    ]
    agg: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(r[key]) for r in ep_results]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    for key in ["reason_counter", "split_counter", "server_counter", "mod_counter"]:
        c = Counter()
        for r in ep_results:
            c.update(r.get(key, {}))
        agg[key] = dict(c)

    all_c = []; zero_c = 0; req_c = 0
    all_r = []
    for r in ep_results:
        if "mask_valid_c_counts" in r:
            all_c.extend(r["mask_valid_c_counts"])
            zero_c += r.get("no_valid_c_rate_ep", 0) * r["total"]
            req_c += r["total"]
        if "mask_valid_r_counts" in r:
            all_r.extend(r["mask_valid_r_counts"])
    agg["no_valid_c_rate"] = zero_c / max(req_c, 1)
    agg["avg_valid_c_actions"] = float(np.mean(all_c)) if all_c else 0.0
    agg["avg_valid_r_actions"] = float(np.mean(all_r)) if all_r else 0.0

    total_reasons = sum(agg.get("reason_counter", {}).values())
    nsb = agg.get("reason_counter", {}).get("no_suitable_block", 0)
    agg["no_suitable_block_ratio"] = nsb / max(total_reasons, 1)

    # source_node (first episode's src)
    agg["source_node"] = ep_results[0].get("source_node", -1)

    # KSP-BF
    kspbf_time = sum(r.get("kspbf_time_s", 0.0) for r in ep_results)
    kspbf_calls = sum(r.get("kspbf_call_count", 0) for r in ep_results)
    kspbf_success = sum(r.get("kspbf_success_count", 0) for r in ep_results)
    agg["kspbf_time_s"] = kspbf_time
    agg["kspbf_success_rate"] = kspbf_success / max(kspbf_calls, 1)

    return agg


def _agg_all_seeds(per_seed_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate across seeds."""
    scalar_keys = [
        "blocking_rate", "avg_delay_ms",
        "avg_fs", "avg_waste", "avg_path_len_km",
        "no_valid_c_rate", "avg_valid_c_actions", "avg_valid_r_actions",
        "no_suitable_block_ratio",
    ]
    agg: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(s[key]) for s in per_seed_list]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    for key in ["reason_counter", "split_counter", "server_counter", "mod_counter"]:
        c = Counter()
        for s in per_seed_list:
            c.update(s.get(key, {}))
        agg[key] = dict(c)

    agg["avg_reward"] = float(np.mean([s["avg_reward"] for s in per_seed_list]))
    agg["kspbf_time_s"] = float(sum(s.get("kspbf_time_s", 0.0) for s in per_seed_list))
    agg["kspbf_success_rate"] = float(np.mean([
        s.get("kspbf_success_rate", 0.0) for s in per_seed_list
    ]))
    agg["source_nodes"] = [s.get("source_node", -1) for s in per_seed_list]

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
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
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/k457_system_comparison.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/k457_system_comparison.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    # Pre-generate all episodes (one set per seed, shared across k values)
    env_proto_base = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=3,
    )
    all_episodes: Dict[int, List[List]] = {}
    source_nodes: Dict[int, int] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto_base.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto_base, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps
        source_nodes[seed] = eps[0][0].src_node if eps and eps[0] else -1

    print("=" * 90)
    print("K=4/5/7 System Comparison with Per-Seed Breakdown")
    print(f"Topology: {args.topology}  Slots: {args.num_slots}  Servers: {args.num_servers}")
    print(f"Seeds: {seeds}  Eps/seed: {args.episodes}  Req/ep: {args.requests_per_episode}")
    print(f"K values: {K_VALUES}  Methods: {METHOD_ORDER}")
    print(f"Source nodes: {source_nodes}")
    print("=" * 90)

    t_start = time.time()

    # Results structure: {k_value: {method_label: {aggregate, per_seed: {seed: {...}}}}}
    all_data: Dict[int, Dict[str, Any]] = {}

    for k_val in K_VALUES:
        print(f"\n{'#' * 70}")
        print(f"# K = {k_val}")
        print(f"{'#' * 70}")

        k_args = argparse.Namespace(**{**vars(args), "k_paths": k_val})
        env_proto = make_env(
            topology=args.topology, num_slots=args.num_slots,
            num_servers=args.num_servers, seed=42,
            slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
            k=k_val,
        )

        k_data: Dict[str, Any] = {}

        for method in METHOD_ORDER:
            label = METHOD_LABELS[method]
            t0 = time.time()

            # Per-seed results
            per_seed: Dict[int, List[Dict[str, Any]]] = {s: [] for s in seeds}
            for seed in seeds:
                for ep_idx, requests in enumerate(all_episodes[seed]):
                    res = _eval_episode(
                        method, agent_c, agent_r, requests, env_proto, k_args, ep_idx, seed,
                    )
                    res["source_node"] = source_nodes[seed]  # ensure correct
                    per_seed[seed].append(res)

            # Aggregate per seed
            per_seed_agg: Dict[int, Dict[str, Any]] = {}
            for seed in seeds:
                per_seed_agg[seed] = _agg_per_seed(per_seed[seed])

            # Aggregate across seeds
            agg = _agg_all_seeds(list(per_seed_agg.values()))

            elapsed = time.time() - t0
            extra = ""
            if method == "agent_kspbf":
                extra = f"  kspbf_sr={agg['kspbf_success_rate']:.1%} t={agg['kspbf_time_s']:.0f}s"
            print(
                f"  {label:25s} blk={agg['blocking_rate']:.4f}±{agg['blocking_rate_std']:.4f}  "
                f"noC={agg['no_valid_c_rate']:.2f}  avgCa={agg['avg_valid_c_actions']:.1f}  "
                f"avgRa={agg['avg_valid_r_actions']:.1f}  "
                f"delay={agg['avg_delay_ms']:.1f}ms  fs={agg['avg_fs']:.2f}  "
                f"t={elapsed:.0f}s{extra}"
            )

            k_data[label] = {"aggregate": agg, "per_seed": per_seed_agg}

        all_data[k_val] = k_data

    total_elapsed = time.time() - t_start
    print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Per-seed diagnosis (k=5 focus)
    # ------------------------------------------------------------------
    k5_data = all_data[5]
    print(f"\n{'=' * 90}")
    print("PER-SEED DIAGNOSIS (K=5)")
    print(f"{'=' * 90}")

    for method in METHOD_ORDER:
        label = METHOD_LABELS[method]
        print(f"\n  {label}:")
        print(f"  {'Seed':<6s} {'Src':>4s} {'Blocking':>9s} {'noC%':>6s} "
              f"{'avgCa':>6s} {'avgRa':>6s} {'NSB%':>6s} {'Delay':>7s} {'FS':>5s}")
        print(f"  {'-' * 55}")
        for seed in seeds:
            s = k5_data[label]["per_seed"][seed]
            print(f"  {seed:<6d} {s['source_node']:>4d} "
                  f"{s['blocking_rate']:9.4f} {s['no_valid_c_rate']:5.1%} "
                  f"{s['avg_valid_c_actions']:5.1f} {s['avg_valid_r_actions']:5.1f} "
                  f"{s['no_suitable_block_ratio']:5.1%} "
                  f"{s['avg_delay_ms']:6.1f} {s['avg_fs']:5.2f}")

    # ------------------------------------------------------------------
    # Aggregate table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("AGGREGATE: K=4/5/7 COMPARISON")
    print(f"{'=' * 90}")
    hdr = (f"{'K':<3s} {'Method':25s} {'Blocking':>10s} {'noC%':>6s} "
           f"{'avgCa':>6s} {'avgRa':>6s} {'Delay':>7s} {'FS':>5s} "
           f"{'NSB%':>6s} {'KSPBFsr':>8s}")
    print(hdr); print("-" * 90)
    for k_val in K_VALUES:
        for method in METHOD_ORDER:
            label = METHOD_LABELS[method]
            agg = all_data[k_val][label]["aggregate"]
            kspbf = f"{agg['kspbf_success_rate']:.1%}" if method == "agent_kspbf" else "—"
            print(f"{k_val:<3d} {label:25s} {agg['blocking_rate']:9.4f}±{agg['blocking_rate_std']:.2f} "
                  f"{agg['no_valid_c_rate']:5.1%} {agg['avg_valid_c_actions']:5.1f} "
                  f"{agg['avg_valid_r_actions']:5.1f} {agg['avg_delay_ms']:6.1f} "
                  f"{agg['avg_fs']:5.2f} {agg['no_suitable_block_ratio']:5.1%} "
                  f"{kspbf:>8s}")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    json_output = {
        "args": vars(args), "k_values": K_VALUES, "methods": METHOD_ORDER,
        "source_nodes": source_nodes,
        "total_time_s": total_elapsed,
        "results": {str(k): {label: {
            "aggregate": _clean(v["aggregate"]),
            "per_seed": {str(s): _clean(v["per_seed"][s]) for s in seeds},
        } for label, v in kd.items()} for k, kd in all_data.items()},
    }
    out_json.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# K=4/5/7 System Comparison with Per-Seed Breakdown\n\n")
    md.append(f"**Topology:** {args.topology}  **Slots:** {args.num_slots}  "
              f"**Servers:** {args.num_servers}  **Req/ep:** {args.requests_per_episode}\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}\n\n")
    md.append(f"**K values:** {K_VALUES}  **Methods:** {[METHOD_LABELS[m] for m in METHOD_ORDER]}\n\n")
    md.append(f"**Source nodes per seed:** {source_nodes}\n\n")
    md.append(f"**Total time:** {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)\n\n")

    # Per-seed K=5 diagnosis
    md.append("## Per-Seed Diagnosis (K=5)\n\n")
    md.append("Each seed uses a fixed source node for all 10 episodes (80 req/ep = 800 req/seed).\n\n")
    for method in METHOD_ORDER:
        label = METHOD_LABELS[method]
        md.append(f"### {label}\n\n")
        md.append("| Seed | SrcNode | Blocking | noC% | avgCacts | avgRacts | NSB% | Delay | FS |\n")
        md.append("|------|---------|----------|------|----------|----------|------|-------|----|\n")
        for seed in seeds:
            s = k5_data[label]["per_seed"][seed]
            md.append(
                f"| {seed} | {s['source_node']} "
                f"| {s['blocking_rate']:.4f} | {s['no_valid_c_rate']:.1%} "
                f"| {s['avg_valid_c_actions']:.1f} | {s['avg_valid_r_actions']:.1f} "
                f"| {s['no_suitable_block_ratio']:.1%} "
                f"| {s['avg_delay_ms']:.1f} | {s['avg_fs']:.2f} |\n"
            )
        md.append("\n")

    # Per-seed for k=4 and k=7 (Agent-C only)
    for k_val in [4, 7]:
        md.append(f"## Per-Seed: Agent-C + PPO-R (K={k_val})\n\n")
        md.append("| Seed | SrcNode | Blocking | noC% | avgCacts | avgRacts | NSB% | Delay | FS |\n")
        md.append("|------|---------|----------|------|----------|----------|------|-------|----|\n")
        for seed in seeds:
            s = all_data[k_val][METHOD_LABELS["agent"]]["per_seed"][seed]
            md.append(
                f"| {seed} | {s['source_node']} "
                f"| {s['blocking_rate']:.4f} | {s['no_valid_c_rate']:.1%} "
                f"| {s['avg_valid_c_actions']:.1f} | {s['avg_valid_r_actions']:.1f} "
                f"| {s['no_suitable_block_ratio']:.1%} "
                f"| {s['avg_delay_ms']:.1f} | {s['avg_fs']:.2f} |\n"
            )
        md.append("\n")

    # Aggregate comparison
    md.append("## Aggregate: K=4/5/7 Comparison\n\n")
    md.append("| K | Method | Blocking | noC% | avgCacts | avgRacts | Delay | FS | NSB% | KSPBFsr |\n")
    md.append("|---|--------|----------|------|----------|----------|-------|----|------|--------|\n")
    for k_val in K_VALUES:
        for method in METHOD_ORDER:
            label = METHOD_LABELS[method]
            agg = all_data[k_val][label]["aggregate"]
            kspbf_str = f"{agg['kspbf_success_rate']:.1%}" if method == "agent_kspbf" else "—"
            md.append(
                f"| {k_val} | {label} "
                f"| {agg['blocking_rate']:.4f}±{agg['blocking_rate_std']:.2f} "
                f"| {agg['no_valid_c_rate']:.1%} "
                f"| {agg['avg_valid_c_actions']:.1f} "
                f"| {agg['avg_valid_r_actions']:.1f} "
                f"| {agg['avg_delay_ms']:.1f} | {agg['avg_fs']:.2f} "
                f"| {agg['no_suitable_block_ratio']:.1%} "
                f"| {kspbf_str} |\n"
            )

    # Analysis
    md.append("\n## Analysis\n\n")

    # Q1: Does blocking decrease monotonically with k?
    md.append("### 1. Blocking vs K (Agent-C)\n\n")
    md.append("| K | Blocking | ±Std | noC% | avgRacts |\n")
    md.append("|---|---|---|---|---|\n")
    for k_val in K_VALUES:
        agg = all_data[k_val][METHOD_LABELS["agent"]]["aggregate"]
        md.append(f"| {k_val} | {agg['blocking_rate']:.4f} | ±{agg['blocking_rate_std']:.2f} "
                  f"| {agg['no_valid_c_rate']:.1%} | {agg['avg_valid_r_actions']:.1f} |\n")

    # Q2: Does k=7 approach 0?
    k7_blk = all_data[7][METHOD_LABELS["agent"]]["aggregate"]["blocking_rate"]
    md.append(f"\n### 2. K=7 Residual Blocking\n\n")
    md.append(f"Agent-C + PPO-R blocking at k=7: **{k7_blk:.4f}** "
              f"({'near zero' if k7_blk < 0.05 else 'significant residual' if k7_blk > 0.15 else 'moderate'})\n\n")

    # Q3: Which k lands in 10-30%?
    md.append("### 3. Paper-Suitable K\n\n")
    for k_val in K_VALUES:
        agg = all_data[k_val][METHOD_LABELS["agent"]]["aggregate"]
        blk = agg["blocking_rate"]
        in_range = "✓ IDEAL" if 0.10 <= blk <= 0.30 else ("too easy" if blk < 0.10 else "too hard")
        md.append(f"- k={k_val}: blocking={blk:.4f} → **{in_range}**\n")

    # Q4: Agent-C vs heuristics per k
    md.append("\n### 4. Agent-C vs Heuristics\n\n")
    md.append("| K | AgentBlk | DFBlk | IWDBlk | AgentAdvantage |\n")
    md.append("|---|---|---|---|---|\n")
    for k_val in K_VALUES:
        a_blk = all_data[k_val][METHOD_LABELS["agent"]]["aggregate"]["blocking_rate"]
        df_blk = all_data[k_val][METHOD_LABELS["df"]]["aggregate"]["blocking_rate"]
        iwd_blk = all_data[k_val][METHOD_LABELS["iwd"]]["aggregate"]["blocking_rate"]
        adv = min(df_blk, iwd_blk) - a_blk
        md.append(f"| {k_val} | {a_blk:.4f} | {df_blk:.4f} | {iwd_blk:.4f} | {adv:+.4f} |\n")

    # Q5-7: Trends
    md.append("\n### 5-7. Trends with K (Agent-C)\n\n")
    md.append("| K | noC% | avgRacts | ±Std |\n")
    md.append("|---|---|---|---|\n")
    for k_val in K_VALUES:
        agg = all_data[k_val][METHOD_LABELS["agent"]]["aggregate"]
        md.append(f"| {k_val} | {agg['no_valid_c_rate']:.1%} "
                  f"| {agg['avg_valid_r_actions']:.1f} "
                  f"| ±{agg['blocking_rate_std']:.2f} |\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    for k_val in K_VALUES:
        agg = all_data[k_val][METHOD_LABELS["agent"]]["aggregate"]
        a_blk = agg["blocking_rate"]
        df_blk = all_data[k_val][METHOD_LABELS["df"]]["aggregate"]["blocking_rate"]
        iwd_blk = all_data[k_val][METHOD_LABELS["iwd"]]["aggregate"]["blocking_rate"]
        adv = min(df_blk, iwd_blk) - a_blk
        tag = "✓" if adv > 0 else "✗"
        print(f"  k={k_val}: Agent blk={a_blk:.4f}±{agg['blocking_rate_std']:.2f}  "
              f"vs DF={df_blk:.4f} IWD={iwd_blk:.4f}  {tag} adv={adv:+.4f}  "
              f"noC={agg['no_valid_c_rate']:.1%}  avgRa={agg['avg_valid_r_actions']:.1f}")

    # Recommendation
    best_k = min(K_VALUES, key=lambda kv: abs(all_data[kv][METHOD_LABELS["agent"]]["aggregate"]["blocking_rate"] - 0.20))
    best_blk_val = all_data[best_k][METHOD_LABELS["agent"]]["aggregate"]["blocking_rate"]
    print(f"\nRecommended main-experiment k: {best_k} (closest to 20% center, blk={best_blk_val:.4f})")


if __name__ == "__main__":
    main()
