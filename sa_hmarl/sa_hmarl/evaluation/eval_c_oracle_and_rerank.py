"""C Oracle upper-bound and rerank evaluation.

Compares 7 C-action selection methods, all paired with the SAME fixed
BC-PPO-R backend:

1. Agent-C (learned PPO policy)
2. TopK-rerank (policy top-K reranked by spectrum pressure via c_topk_rerank)
3. Oracle-C (one-step myopic: enumerate all valid C actions, simulate each
   with R on a deep-copied env, pick best immediate outcome.  NOT a global
   upper bound — does not consider future request blocking.)
4. WO-C  (distributed offloading heuristic)
5. DF-C  (distance-first heuristic)
6. RF-C  (resource-first / least-loaded heuristic)
7. IWD-C (intelligent water droplet heuristic)

Usage
-----
Smoke test::

    PYTHONPATH=. .venv/bin/python -m sa_hmarl.evaluation.eval_c_oracle_and_rerank \\
        --seeds 42 --episodes 1 --requests_per_episode 5

Full run::

    PYTHONPATH=. .venv/bin/python -m sa_hmarl.evaluation.eval_c_oracle_and_rerank
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.c_candidate_pressure import compute_c_candidate_pressure
from sa_hmarl.evaluation.c_topk_rerank import select_c_topk_rerank
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode",
        ckpt.get("agent_c_feature_mode", "default"),
    )
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry,
                device: str = "cpu") -> PPOAgentR:
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
# Helpers
# ---------------------------------------------------------------------------

def _c_features_and_mask(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    return features, mask


def _select_r(
    agent_r: PPOAgentR, env, req, split_id: int, server_id: int
) -> Tuple[Tuple[int, int, int], Optional[int]]:
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    action_idx_r = agent_r.select_action(obs_r, deterministic=True)
    if action_idx_r is None:
        return (0, 0, 0), None
    return (
        decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), env.max_blocks),
        action_idx_r,
    )


def _select_agent_c(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int
) -> Optional[int]:
    features, mask = _c_features_and_mask(agent_c, obs_c, num_slots)
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


# ---------------------------------------------------------------------------
# Oracle-C scoring
# ---------------------------------------------------------------------------

def _oracle_score_tuple(
    info: Dict[str, Any],
    req_deadline_ms: float,
    pressure: Dict[str, Any],
) -> Tuple:
    """Oracle priority tuple — lexicographic comparison, lower = better.

    1. success (0 = success, 1 = failure)
    2. deadline satisfied (0 = ok, 1 = violated)
    3. delay_ms (lower better)
    4. fs_lfb_ratio (lower better)
    5. frag_pressure (lower better)
    6. server_utilization (lower better)
    """
    if not info.get("success", False):
        return (1, 0, 1e9, 1e9, 1e9, 1e9)

    delay = float(info.get("delay_ms", 0.0))
    deadline_ok = 0 if delay <= req_deadline_ms else 1

    ratio = pressure.get("fs_lfb_ratio", 1.0)
    frag = pressure.get("frag_pressure", 1.0)
    util = pressure.get("server_utilization", 0.0)

    # Sanitise non-finite values
    if not np.isfinite(ratio):
        ratio = 1e9
    if not np.isfinite(frag):
        frag = 1e9

    return (0, deadline_ok, delay, ratio, frag, util)


def _select_oracle_c(
    agent_r: PPOAgentR,
    env,
    req,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int, int], Dict[str, Any]]:
    """Oracle-C: try every valid (split, server), simulate, pick best."""
    valid = np.where(mask)[0]
    num_servers = len(env.mec.servers)

    if len(valid) == 0:
        return (0, 0), (0, 0, 0), {"success": False, "reason": "no_valid_c_oracle"}

    best_score: Optional[Tuple] = None
    best_action_c: Tuple[int, int] = (0, 0)
    best_action_r: Tuple[int, int, int] = (0, 0, 0)
    best_info: Dict[str, Any] = {"success": False, "reason": "oracle_no_trials"}
    best_pressure: Dict[str, Any] = {}

    for action_idx_c in valid:
        action_idx_c = int(action_idx_c)
        split_id = action_idx_c // num_servers
        server_id = action_idx_c % num_servers

        # Compute pressure from the CURRENT env (pre-allocation)
        pressure = compute_c_candidate_pressure(env, req, split_id, server_id)

        # Simulate on a deep copy
        action_r_tuple, _ = _select_r(agent_r, env, req, split_id, server_id)
        env_try = copy.deepcopy(env)
        _, _, _, info_try = env_try.step(
            (split_id, server_id), action_r_tuple,
        )

        score = _oracle_score_tuple(info_try, req.deadline_ms, pressure)
        if best_score is None or score < best_score:
            best_score = score
            best_action_c = (split_id, server_id)
            best_action_r = action_r_tuple
            best_info = info_try
            best_pressure = pressure

    return best_action_c, best_action_r, best_info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_method(
    method: str,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    episodes_data: List[List],
    env_proto,
    args,
) -> Dict[str, Any]:
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    total = blocked = success = deadline_met = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    split_counter = Counter()
    server_counter = Counter()
    server_success_counter = Counter()
    server_overload_counter = Counter()
    reason_counter = Counter()
    mod_counter = Counter()
    total_oracle_time = 0.0

    for ep_idx, requests in enumerate(episodes_data):
        env = make_env(
            topology=args.topology,
            num_slots=args.num_slots,
            num_servers=args.num_servers,
            seed=42,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
            server_nodes=server_nodes,
            capacities=capacities,
            k=args.k_paths,
        )
        env.reset(requests)
        server_selected_count = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)

            if method == "agent":
                action_idx_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
                action_c = (
                    (0, 0) if action_idx_c is None
                    else decode_agent_c_action(action_idx_c, num_servers)
                )
                action_r, _ = _select_r(agent_r, env, req, action_c[0], action_c[1])

            elif method == "topk_rerank":
                # Apply the same risk mask that Agent-C uses, so the
                # reranker sees the same valid-action set.
                features_c, mask_c = _c_features_and_mask(
                    agent_c, obs_c, env.net.num_slots,
                )
                obs_c_masked = dict(obs_c)  # shallow copy
                obs_c_masked["agent_c_mask"] = mask_c
                action_idx_c = select_c_topk_rerank(
                    agent_c, obs_c_masked, env, req, top_k=args.top_k,
                )
                action_c = (
                    (0, 0) if action_idx_c is None
                    else decode_agent_c_action(action_idx_c, num_servers)
                )
                action_r, _ = _select_r(agent_r, env, req, action_c[0], action_c[1])

            elif method == "oracle":
                # One-step myopic oracle: enumerates every valid (split, server)
                # from the ORIGINAL C mask (no risk clipping), simulates each
                # with BC-PPO-R on a deep-copied env, and picks the one with
                # the best immediate outcome.  This is NOT a global upper bound
                # — it does not look ahead to future requests.
                mask = obs_c["agent_c_mask"]
                t0 = time.time()
                action_c, action_r, _ = _select_oracle_c(
                    agent_r, env, req, obs_c, mask,
                )
                total_oracle_time += time.time() - t0

            else:
                # WO, DF, RF, IWD via offloading_baselines
                action_idx_c = select_offloading_action(
                    method,
                    env, req, obs_c, obs_c["agent_c_mask"],
                    rng=np.random.RandomState(args.seed + ep_idx),
                    server_selected_count=server_selected_count,
                )
                action_c = (
                    (0, 0) if action_idx_c is None
                    else decode_agent_c_action(action_idx_c, num_servers)
                )
                action_r, _ = _select_r(agent_r, env, req, action_c[0], action_c[1])

            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count[server_id] += 1

            _, _, _, info = env.step(action_c, action_r)
            total += 1
            server_obj = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server_obj.utilization,
            )
            total_reward += reward

            if info.get("success", False):
                success += 1
                server_success_counter[f"s{server_id}"] += 1
                delay = float(info.get("delay_ms", 0.0))
                total_delay += delay
                total_fs += float(info.get("num_slots", 0))
                total_waste += float(info.get("block_waste", 0.0))
                total_path += float(info.get("path_dist_km", 0.0))
                if delay <= req.deadline_ms:
                    deadline_met += 1
                mod_counter[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1
                if reason in ("server_overload", "server_saturated"):
                    server_overload_counter[f"s{server_id}"] += 1

    n = max(total, 1)
    s = max(success, 1)
    avg_delay = total_delay / s
    blocking = blocked / n

    return {
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocking,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": avg_delay,
        "deadline_sat_rate": deadline_met / n,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "server_success_counter": dict(server_success_counter),
        "server_overload_counter": dict(server_overload_counter),
        "reason_counter": dict(reason_counter),
        "mod_counter": dict(mod_counter),
        "oracle_time_s": total_oracle_time,
    }


# ---------------------------------------------------------------------------
# Aggregation & formatting
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
    ]
    out: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(r[key]) for r in results]
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))

    for key in ["split_counter", "server_counter", "server_success_counter",
                "server_overload_counter", "reason_counter", "mod_counter"]:
        c = Counter()
        for r in results:
            c.update(r.get(key, {}))
        out[key] = dict(c)

    # Per-server ratios
    server_total = Counter()
    for r in results:
        server_total.update(r.get("server_counter", {}))
    server_success = Counter()
    for r in results:
        server_success.update(r.get("server_success_counter", {}))
    server_overload = Counter()
    for r in results:
        server_overload.update(r.get("server_overload_counter", {}))

    out["per_server_selected_ratio"] = {}
    for k, v in server_total.items():
        out["per_server_selected_ratio"][k] = v / max(sum(server_total.values()), 1)

    out["per_server_overload_ratio"] = {}
    for k in server_total:
        ov = server_overload.get(k, 0)
        out["per_server_overload_ratio"][k] = (
            ov / max(server_total.get(k, 1), 1)
        )

    # Oracle timing
    times = [r.get("oracle_time_s", 0.0) for r in results]
    out["total_oracle_time_s"] = float(sum(times))

    return out


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
        description="C Oracle upper-bound and rerank evaluation"
    )
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed",
                        choices=["size_desc", "waste_asc", "start_asc", "center_asc", "mixed"])
    parser.add_argument("--k_paths", type=int, default=3,
                        help="Number of shortest paths for R action space (k).")
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
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--methods", type=str,
                        default="agent,topk_rerank,oracle,wo,df,rf,iwd")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/c_oracle_rerank_eval.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/c_oracle_rerank_eval.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    METHOD_LABELS = {
        "agent": "Agent-C + BC-PPO-R",
        "topk_rerank": "TopK-rerank + BC-PPO-R",
        "oracle": "Oracle-C (1-step myopic) + BC-PPO-R",
        "wo": "WO-C + BC-PPO-R",
        "df": "DF-C + BC-PPO-R",
        "rf": "RF-C + BC-PPO-R",
        "iwd": "IWD-C + BC-PPO-R",
        "greedy": "Greedy-C + BC-PPO-R",
    }

    # Load models once
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )

    print("=" * 90)
    print("C One-Step Myopic Oracle & Rerank Evaluation")
    print(f"Topology: {args.topology}  Slots: {args.num_slots}  "
          f"Servers: {args.num_servers}  MaxBlocks: {args.max_blocks}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}  Episodes/seed: {args.episodes}  "
          f"Requests/ep: {args.requests_per_episode}")
    print(f"C: {args.agent_c_checkpoint}")
    print(f"R: {args.agent_r_checkpoint}")
    print("=" * 90)

    # Pre-generate episodes per seed (same episodes for all methods)
    all_episodes: Dict[int, List[List]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(
                generate_requests(
                    env_proto, rng, src,
                    num_requests=args.requests_per_episode,
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
            )
        all_episodes[seed] = eps

    # Evaluate each method
    all_results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        print(f"\n{'─' * 70}")
        print(f"Evaluating: {label}")
        print(f"{'─' * 70}")

        seed_results: List[Dict[str, Any]] = []
        for seed in seeds:
            res = evaluate_method(
                method, agent_c, agent_r, all_episodes[seed], env_proto, args,
            )
            seed_results.append(res)
            print(
                f"  seed={seed:4d}: blk={res['blocking_rate']:.4f} "
                f"reward={res['avg_reward']:+.3f} "
                f"delay={res['avg_delay_ms']:.1f}ms "
                f"fs={res['avg_fs']:.2f} waste={res['avg_waste']:.3f}"
            )

        agg = aggregate(seed_results)
        all_results[label] = agg
        print(
            f"  => AVG: blk={agg['blocking_rate']:.4f}±{agg['blocking_rate_std']:.4f} "
            f"delay={agg['avg_delay_ms']:.1f}ms "
            f"fs={agg['avg_fs']:.2f} "
            f"oracle_time={agg.get('total_oracle_time_s', 0):.0f}s"
        )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)

    header = (
        f"{'Method':30s} {'Blocking':>9s} {'Reward':>8s} {'Delay':>8s} "
        f"{'AvgFS':>7s} {'Waste':>7s} {'PathKm':>7s} {'DeadSat':>8s}"
    )
    print(header)
    print("-" * 90)

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        print(
            f"{label:30s} "
            f"{agg['blocking_rate']:9.4f} "
            f"{agg['avg_reward']:8.3f} "
            f"{agg['avg_delay_ms']:8.1f} "
            f"{agg['avg_fs']:7.2f} "
            f"{agg['avg_waste']:7.3f} "
            f"{agg['avg_path_len_km']:7.1f} "
            f"{agg['deadline_sat_rate']:8.3f}"
        )

    print()
    print("-" * 90)
    print("FAILURE REASONS")
    print("-" * 90)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        print(f"  {label:30s}  {_fmt_cnt(agg.get('reason_counter', {}))}")

    print()
    print("-" * 90)
    print("SPLIT DISTRIBUTION")
    print("-" * 90)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        print(f"  {label:30s}  {_fmt_pct(agg.get('split_counter', {}))}")

    print()
    print("-" * 90)
    print("SERVER DISTRIBUTION")
    print("-" * 90)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        print(f"  {label:30s}  {_fmt_pct(agg.get('server_counter', {}))}")

    print()
    print("-" * 90)
    print("PER-SERVER OVERLOAD RATIO")
    print("-" * 90)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        ratios = agg.get("per_server_overload_ratio", {})
        ratio_str = ", ".join(
            f"{k}={v:.1%}" for k, v in sorted(ratios.items())
        ) if ratios else "none"
        print(f"  {label:30s}  {ratio_str}")

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for label, agg in all_results.items():
        clean[label] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in agg.items()
        }
    out_json.write_text(
        json.dumps({"args": vars(args), "results": clean}, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------
    out_md = Path(args.out_md)
    md_lines = [
        "# C One-Step Myopic Oracle & Rerank Evaluation\n\n",
        f"**Topology:** {args.topology}  ",
        f"**Servers:** {args.num_servers}  **Slots:** {args.num_slots}  ",
        f"**MaxBlocks:** {args.max_blocks}  **BlockSort:** {args.block_sort_strategy}\n\n",
        f"**Agent-C:** `{args.agent_c_checkpoint}`  \n",
        f"**Agent-R:** `{args.agent_r_checkpoint}`  \n",
        f"**Seeds:** {args.seeds}  ",
        f"**Episodes/seed:** {args.episodes}  ",
        f"**Requests/ep:** {args.requests_per_episode}\n\n",
    ]

    md_lines.append("## Results\n\n")
    md_lines.append(
        "| Method | Blocking | Reward | Delay(ms) | AvgFS | Waste | PathKm | DeadSat |\n"
    )
    md_lines.append(
        "|--------|----------|--------|-----------|-------|-------|--------|--------|\n"
    )
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        md_lines.append(
            f"| {label} | {agg['blocking_rate']:.4f} | {agg['avg_reward']:+.3f} | "
            f"{agg['avg_delay_ms']:.1f} | {agg['avg_fs']:.2f} | "
            f"{agg['avg_waste']:.3f} | {agg['avg_path_len_km']:.1f} | "
            f"{agg['deadline_sat_rate']:.3f} |\n"
        )

    md_lines.append("\n## Failure Reasons\n\n")
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        md_lines.append(
            f"- **{label}**: {_fmt_cnt(agg.get('reason_counter', {}))}\n"
        )

    md_lines.append("\n## Split Distribution\n\n")
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        md_lines.append(
            f"- **{label}**: {_fmt_pct(agg.get('split_counter', {}))}\n"
        )

    md_lines.append("\n## Server Distribution\n\n")
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        md_lines.append(
            f"- **{label}**: {_fmt_pct(agg.get('server_counter', {}))}\n"
        )

    md_lines.append("\n## Per-Server Overload Ratio\n\n")
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        agg = all_results[label]
        ratios = agg.get("per_server_overload_ratio", {})
        ratio_str = ", ".join(
            f"{k}={v:.1%}" for k, v in sorted(ratios.items())
        ) if ratios else "none"
        md_lines.append(f"- **{label}**: {ratio_str}\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md_lines), encoding="utf-8")
    print(f"Markdown → {out_md}")


if __name__ == "__main__":
    main()
