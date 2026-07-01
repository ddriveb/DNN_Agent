"""Yin-style offloading baseline sweep.

This script is intentionally different from the SA-HMARL masked evaluation.
It approximates the paper-style DNN offloading baselines:

  - WO: no optimization / random distributed offloading.
  - DF: distance first, choose nearest server.
  - RF: resource first, choose lightest-loaded server.
  - IWD: lightweight population-style online search over split/server actions.

The C-side selectors do not use Agent-C feasibility masks. Infeasibility is
counted as execution blocking, which is closer to traditional baseline
semantics than "choose among already valid actions".

RMSA backend is fixed to shortest path + highest feasible modulation +
first-fit spectrum, so this still runs inside the SA-HMARL optical environment.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
)
from sa_hmarl.evaluation.eval_sp_hm_ff_offloading import sp_hm_ff_action
from sa_hmarl.network.ksp import get_k_shortest_paths
from sa_hmarl.training.utils import generate_requests, make_env


METHODS = ("wo", "df", "rf", "iwd")


def _finite_or_large(value: Any, large: float = 1e9) -> float:
    if value is None:
        return large
    try:
        val = float(value)
    except (TypeError, ValueError):
        return large
    if np.isnan(val) or np.isinf(val):
        return large
    return val


def _distance_km(env, src_node: int, server_id: int) -> float:
    server_node = env.mec.servers[server_id].node_id
    paths = get_k_shortest_paths(env.net.G, src_node, server_node, k=1, weight="length_km")
    if not paths:
        return 1e9
    return env.net.path_length_km(paths[0])


def _best_split_for_server(obs_c: Dict[str, Any], server_id: int, num_servers: int) -> int:
    """Choose split with minimum estimated compute delay for a fixed server."""
    best_split = 0
    best_cost = float("inf")
    num_splits = obs_c["request_features"]["num_splits"]
    for split_id in range(num_splits):
        idx = split_id * num_servers + server_id
        feat = obs_c["candidate_features"][idx]
        local_ms = _finite_or_large(feat.get("local_compute_ms"))
        edge_ms = _finite_or_large(feat.get("edge_compute_ms"))
        cost = local_ms + edge_ms
        if cost < best_cost:
            best_cost = cost
            best_split = split_id
    return best_split


def _select_wo(obs_c: Dict[str, Any], rng: np.random.RandomState, num_servers: int) -> int:
    """No-optimization offloading: random split/server."""
    num_splits = obs_c["request_features"]["num_splits"]
    split_id = int(rng.randint(0, num_splits))
    server_id = int(rng.randint(0, num_servers))
    return split_id * num_servers + server_id


def _select_df(env, obs_c: Dict[str, Any], num_servers: int) -> int:
    """Distance-first: nearest server, then best compute split."""
    src_node = obs_c["request_features"]["src_node"]
    distances = [_distance_km(env, src_node, sid) for sid in range(num_servers)]
    server_id = int(np.argmin(distances))
    split_id = _best_split_for_server(obs_c, server_id, num_servers)
    return split_id * num_servers + server_id


def _select_rf(obs_c: Dict[str, Any], num_servers: int) -> int:
    """Resource-first: lightest server, then best compute split."""
    utils = obs_c["server_utilizations"]
    server_id = int(np.argmin(utils))
    split_id = _best_split_for_server(obs_c, server_id, num_servers)
    return split_id * num_servers + server_id


def _select_iwd(
    env,
    obs_c: Dict[str, Any],
    rng: np.random.RandomState,
    num_servers: int,
    n_droplets: int = 30,
    n_iters: int = 12,
    rho: float = 0.15,
) -> int:
    """Lightweight IWD-like search over all split/server actions.

    This is not a full offline IWD solver; it is a one-request population-style
    approximation that keeps the baseline deterministic enough for online
    evaluation while still optimizing a composite cost.
    """
    num_splits = obs_c["request_features"]["num_splits"]
    src_node = obs_c["request_features"]["src_node"]
    action_ids = np.arange(num_splits * num_servers, dtype=int)

    distances = [_distance_km(env, src_node, sid) for sid in range(num_servers)]
    features = obs_c["candidate_features"]
    utils = obs_c["server_utilizations"]

    raw = []
    for action_id in action_ids:
        split_id, server_id = decode_agent_c_action(int(action_id), num_servers)
        feat = features[int(action_id)]
        delay = _finite_or_large(feat.get("local_compute_ms")) + _finite_or_large(feat.get("edge_compute_ms"))
        load = _finite_or_large(utils[server_id])
        dist = _finite_or_large(distances[server_id])
        fs = _finite_or_large(feat.get("best_fs_estimate"))
        raw.append((delay, load, dist, fs))

    arr = np.asarray(raw, dtype=float)
    denom = np.maximum(arr.max(axis=0), 1e-6)
    norm = arr / denom
    # Weighting follows the paper intuition: delay/resource/distance all matter;
    # FS is a small optical-domain tie-breaker in this SA-HMARL environment.
    costs = 0.35 * norm[:, 0] + 0.30 * norm[:, 1] + 0.25 * norm[:, 2] + 0.10 * norm[:, 3]
    costs = np.nan_to_num(costs, nan=1e9, posinf=1e9, neginf=1e9)

    soil = np.ones_like(costs)
    visits = np.zeros_like(costs, dtype=int)
    for _ in range(n_iters):
        for _ in range(n_droplets):
            inv = 1.0 / np.maximum(soil + costs + 1e-6, 1e-9)
            probs = inv / inv.sum()
            choice = int(rng.choice(len(action_ids), p=probs))
            visits[choice] += 1
            soil[choice] = (1.0 - rho) * soil[choice] + rho * costs[choice]

    return int(action_ids[int(np.argmax(visits))])


def select_yin_action(
    method: str,
    env,
    obs_c: Dict[str, Any],
    rng: np.random.RandomState,
    num_servers: int,
) -> int:
    method = method.lower()
    if method == "wo":
        return _select_wo(obs_c, rng, num_servers)
    if method == "df":
        return _select_df(env, obs_c, num_servers)
    if method == "rf":
        return _select_rf(obs_c, num_servers)
    if method == "iwd":
        return _select_iwd(env, obs_c, rng, num_servers)
    raise ValueError(f"Unknown method: {method}")


def evaluate_one(
    method: str,
    request_count: int,
    episodes_data: List[List],
    env_proto,
    args,
    seed: int,
) -> Dict[str, Any]:
    num_servers = len(env_proto.mec.servers)
    topology = env_proto.net.topology
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    total = blocked = success = deadline_met = 0
    total_delay = total_fs = total_path = 0.0
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()

    for ep_idx, requests in enumerate(episodes_data):
        env = make_env(
            topology=topology,
            num_slots=args.num_slots,
            num_servers=args.num_servers,
            seed=seed + ep_idx,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            server_nodes=server_nodes,
            capacities=capacities,
        )
        env.max_blocks = args.max_blocks
        env.reset(requests)
        rng = np.random.RandomState(seed * 100000 + ep_idx * 997 + request_count)

        for req in requests:
            total += 1
            obs_c = build_agent_c_observation(env, req)
            action_idx_c = select_yin_action(method, env, obs_c, rng, num_servers)
            split_id, server_id = decode_agent_c_action(action_idx_c, num_servers)
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_r = sp_hm_ff_action(obs_r)
            if action_r is None:
                blocked += 1
                reason_counter["no_valid_r_action"] += 1
                continue

            _, _, _, info = env.step((split_id, server_id), action_r)
            if not info.get("success", False):
                blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1
                continue

            success += 1
            total_delay += info.get("delay_ms", 0.0)
            total_fs += info.get("num_slots", 0.0)
            total_path += info.get("path_dist_km", 0.0)
            if info.get("deadline_met", True):
                deadline_met += 1

    return {
        "method": method,
        "request_count": request_count,
        "seed": seed,
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocked / total if total else 0.0,
        "success_rate": success / total if total else 0.0,
        "avg_delay_ms": total_delay / success if success else 0.0,
        "deadline_met_rate": deadline_met / success if success else 0.0,
        "avg_fs": total_fs / success if success else 0.0,
        "avg_path_km": total_path / success if success else 0.0,
        "reason_dist": dict(reason_counter),
        "split_dist": _norm_counter(split_counter, total),
        "server_dist": _norm_counter(server_counter, total),
    }


def _norm_counter(counter: Counter, denom: int) -> Dict[str, float]:
    if denom <= 0:
        return {}
    return {k: v / denom for k, v in sorted(counter.items())}


def _aggregate(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric in ["blocking_rate", "success_rate", "avg_delay_ms", "deadline_met_rate", "avg_fs", "avg_path_km"]:
        vals = [r[metric] for r in seed_results]
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))
    reasons = Counter()
    for r in seed_results:
        reasons.update(r["reason_dist"])
    out["reason_dist_total"] = dict(reasons)
    out["total"] = int(sum(r["total"] for r in seed_results))
    out["blocked"] = int(sum(r["blocked"] for r in seed_results))
    out["success"] = int(sum(r["success"] for r in seed_results))
    return out


def _write_outputs(results: Dict[str, Any], args) -> None:
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)

    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for req_n in sorted(results.keys(), key=lambda x: int(x)):
        for method, r in sorted(results[req_n].items()):
            rows.append({
                "requests": int(req_n),
                "method": method.upper(),
                "blocking": r["blocking_rate_mean"],
                "blocking_std": r["blocking_rate_std"],
                "success": r["success_rate_mean"],
                "delay_ms": r["avg_delay_ms_mean"],
                "avg_fs": r["avg_fs_mean"],
                "avg_path_km": r["avg_path_km_mean"],
                "blocked": r["blocked"],
                "total": r["total"],
            })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Yin-Style Offloading Load Sweep",
        "",
        f"Topology: `{args.topology}`, servers: `{args.num_servers}`, slots: `{args.num_slots}`",
        f"Request counts: `{args.request_counts}`",
        f"Seeds: `{args.seeds}`, episodes/request-count/seed: `{args.episodes}`",
        "",
        "| Requests | WO Blk | DF Blk | RF Blk | IWD Blk | WO Delay | DF Delay | RF Delay | IWD Delay |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for req_n in sorted(results.keys(), key=lambda x: int(x)):
        r = results[req_n]
        lines.append(
            f"| {req_n} | {r['wo']['blocking_rate_mean']:.3f} | "
            f"{r['df']['blocking_rate_mean']:.3f} | {r['rf']['blocking_rate_mean']:.3f} | "
            f"{r['iwd']['blocking_rate_mean']:.3f} | {r['wo']['avg_delay_ms_mean']:.1f} | "
            f"{r['df']['avg_delay_ms_mean']:.1f} | {r['rf']['avg_delay_ms_mean']:.1f} | "
            f"{r['iwd']['avg_delay_ms_mean']:.1f} |"
        )
    lines.extend(["", "## Failure Reasons", ""])
    for req_n in sorted(results.keys(), key=lambda x: int(x)):
        lines.append(f"### Requests = {req_n}")
        for method, r in sorted(results[req_n].items()):
            lines.append(f"- {method.upper()}: {r['reason_dist_total']}")
        lines.append("")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="net1")
    parser.add_argument("--num_servers", type=int, default=3)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--request_counts", type=str, default="15,20,25,30,35,40,45,50,55,60")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--methods", type=str, default="wo,df,rf,iwd")
    parser.add_argument("--output_json", type=str, default="sa_hmarl/experiments/yin_style_offloading_sweep.json")
    parser.add_argument("--output_csv", type=str, default="sa_hmarl/experiments/yin_style_offloading_sweep.csv")
    parser.add_argument("--output_md", type=str, default="sa_hmarl/experiments/yin_style_offloading_sweep.md")
    args = parser.parse_args()

    if args.max_blocks is None:
        args.max_blocks = args.num_slots
    request_counts = [int(x.strip()) for x in args.request_counts.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]

    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=seeds[0],
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )
    env_proto.max_blocks = args.max_blocks

    all_results: Dict[str, Any] = {}
    for req_n in request_counts:
        print(f"\n=== Requests per episode: {req_n} ===", flush=True)
        all_results[str(req_n)] = {}

        seed_episode_sets: Dict[int, List[List]] = {}
        for seed in seeds:
            rng = np.random.RandomState(seed + req_n * 17)
            episodes = []
            for _ in range(args.episodes):
                src = int(rng.randint(0, env_proto.net.NUM_NODES))
                episodes.append(generate_requests(
                    env_proto,
                    rng,
                    src,
                    req_n,
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
                ))
            seed_episode_sets[seed] = episodes

        for method in methods:
            seed_results = [
                evaluate_one(method, req_n, seed_episode_sets[seed], env_proto, args, seed)
                for seed in seeds
            ]
            agg = _aggregate(seed_results)
            all_results[str(req_n)][method] = agg
            print(
                f"{method.upper()}: blk={agg['blocking_rate_mean']:.3f} "
                f"delay={agg['avg_delay_ms_mean']:.1f} fs={agg['avg_fs_mean']:.2f}",
                flush=True,
            )

    _write_outputs(all_results, args)
    print(f"\nSaved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
