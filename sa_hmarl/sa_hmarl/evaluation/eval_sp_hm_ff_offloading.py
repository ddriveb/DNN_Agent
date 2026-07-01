"""Evaluate C-side offloading baselines with fixed SP-HM-FF RMSA.

RMSA backend:
  1. Path: shortest path only (KSP path index 0).
  2. Modulation: highest spectral-efficiency modulation feasible on that path.
  3. Spectrum: First-Fit block (lowest start slot).

This script isolates DNN split/server offloading strategies (WO/DF/RF/IWD)
from learned or search-based RMSA backends.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.training.utils import generate_requests, make_env


def sp_hm_ff_action(obs_r: Dict[str, Any]) -> Optional[tuple[int, int, int]]:
    """Return (path_idx, mod_idx, block_idx) for shortest/highest/first-fit.

    If the shortest path has no feasible modulation/block, return the intended
    shortest-path/highest-mod action with block 0 so the environment records the
    failure reason (for example no_suitable_block) instead of silently trying a
    fallback path.
    """
    paths = obs_r["candidate_paths"]
    mod_names = obs_r["mod_names"]
    if not paths or not mod_names:
        return None

    path_idx = 0
    feasible = obs_r["feasible_mask_per_path_mod"][path_idx]
    required = obs_r["required_fs_per_path_mod"][path_idx]
    blocks_by_mod = obs_r["candidate_blocks_per_path_mod"][path_idx]

    feasible_mods = []
    for mod_idx, is_feasible in enumerate(feasible):
        if is_feasible and required[mod_idx] is not None:
            feasible_mods.append(mod_idx)
    if not feasible_mods:
        return path_idx, 0, 0

    # Modulation registry is ordered by increasing spectral efficiency in this
    # project (BPSK, QPSK, 8QAM, 16QAM, ...). Use the largest index among
    # feasible formats as the highest-SE modulation.
    mod_idx = max(feasible_mods)
    blocks = blocks_by_mod[mod_idx]
    if not blocks:
        return path_idx, mod_idx, 0

    # Candidate blocks are (start_slot, size). First-Fit is the lowest start.
    block_idx = min(range(len(blocks)), key=lambda i: blocks[i][0])
    return path_idx, mod_idx, block_idx


def evaluate_method(
    method_name: str,
    episodes_data: List[List],
    env_proto,
    args,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate one C-side offloading method with SP-HM-FF RMSA."""
    num_servers = len(env_proto.mec.servers)
    topology = env_proto.net.topology
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    total = 0
    blocked = 0
    success = 0
    total_delay = 0.0
    total_fs = 0.0
    total_waste = 0.0
    total_path = 0.0
    deadline_met = 0

    split_counter = Counter()
    server_counter = Counter()
    reason_counter = Counter()
    mod_counter = Counter()
    valid_action_counts = []

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
        # Include all candidate free blocks so lowest-start First-Fit is
        # selectable by env.step(), not truncated by max_blocks=5.
        env.max_blocks = args.max_blocks
        env.reset(requests)
        server_selected_count = np.zeros(num_servers, dtype=int)
        rng = np.random.RandomState(seed + ep_idx + 10000)

        for req in requests:
            total += 1
            obs_c = build_agent_c_observation(env, req)
            raw_mask_c = obs_c["agent_c_mask"]
            if args.c_mask_mode == "paper":
                # Paper-style WO/DF/RF/IWD: choose by the baseline rule first,
                # then let server/RMSA infeasibility become blocking.
                mask_c = np.ones_like(raw_mask_c, dtype=bool)
            else:
                mask_c = raw_mask_c
            valid_action_counts.append(int(np.sum(mask_c)))
            action_idx_c = select_offloading_action(
                method_name,
                env,
                req,
                obs_c,
                mask_c,
                rng=rng,
                server_selected_count=server_selected_count,
            )
            if action_idx_c is None:
                blocked += 1
                reason_counter["no_valid_c_action"] += 1
                continue

            split_id, server_id = decode_agent_c_action(action_idx_c, num_servers)
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count[server_id] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_r = sp_hm_ff_action(obs_r)
            if action_r is None:
                blocked += 1
                reason_counter["no_valid_r_action"] += 1
                continue

            _, _, _, info = env.step((split_id, server_id), action_r)
            if info.get("success") is False:
                blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1
                continue

            success += 1
            total_delay += info.get("delay_ms", 0.0)
            total_fs += info.get("num_slots", 0)
            total_waste += info.get("block_waste", 0.0)
            total_path += info.get("path_dist_km", 0.0)
            if info.get("deadline_met", True):
                deadline_met += 1
            mod_counter[info.get("modulation", "unknown")] += 1

    blocking_rate = blocked / total if total else 0.0
    success_rate = success / total if total else 0.0
    avg_delay = total_delay / success if success else 0.0
    avg_fs = total_fs / success if success else 0.0
    avg_waste = total_waste / success if success else 0.0
    avg_path = total_path / success if success else 0.0
    deadline_rate = deadline_met / success if success else 0.0

    return {
        "method": method_name,
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocking_rate,
        "success_rate": success_rate,
        "avg_delay_ms": avg_delay,
        "deadline_met_rate": deadline_rate,
        "avg_fs": avg_fs,
        "avg_waste": avg_waste,
        "avg_path_km": avg_path,
        "objective_score": blocking_rate + 0.5 * (avg_delay / 65.0),
        "split_dist": _normalize_counter(split_counter, total),
        "server_dist": _normalize_counter(server_counter, total),
        "mod_dist": _normalize_counter(mod_counter, success),
        "reason_dist": dict(reason_counter),
        "valid_action_mean": float(np.mean(valid_action_counts)) if valid_action_counts else 0.0,
        "valid_action_min": int(np.min(valid_action_counts)) if valid_action_counts else 0,
        "valid_action_max": int(np.max(valid_action_counts)) if valid_action_counts else 0,
    }


def _normalize_counter(counter: Counter, denom: int) -> Dict[str, float]:
    if denom <= 0:
        return {}
    return {k: v / denom for k, v in sorted(counter.items())}


def _aggregate(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = [
        "blocking_rate",
        "success_rate",
        "avg_delay_ms",
        "deadline_met_rate",
        "avg_fs",
        "avg_waste",
        "avg_path_km",
        "objective_score",
        "valid_action_mean",
    ]
    out: Dict[str, Any] = {}
    for metric in metrics:
        vals = [r[metric] for r in seed_results]
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals))

    for key in ["split_dist", "server_dist", "mod_dist"]:
        names = set()
        for r in seed_results:
            names.update(r[key].keys())
        out[f"{key}_mean"] = {
            name: float(np.mean([r[key].get(name, 0.0) for r in seed_results]))
            for name in sorted(names)
        }

    reasons = Counter()
    for r in seed_results:
        reasons.update(r["reason_dist"])
    out["reason_dist_total"] = dict(reasons)
    out["total"] = int(sum(r["total"] for r in seed_results))
    out["blocked"] = int(sum(r["blocked"] for r in seed_results))
    out["success"] = int(sum(r["success"] for r in seed_results))
    return out


def _fmt_dist(dist: Dict[str, float]) -> str:
    if not dist:
        return "-"
    return ", ".join(f"{k}={v:.1%}" for k, v in sorted(dist.items()))


def _write_markdown(path: str, results: Dict[str, Any], args) -> None:
    lines = [
        "# SP-HM-FF RMSA Offloading Comparison",
        "",
        "RMSA backend: shortest path + highest feasible modulation + first-fit spectrum.",
        "",
        f"**Topology:** {args.topology}",
        f"**Split profile:** {args.split_profile} | **Splits:** {args.num_splits}",
        f"**Slots:** {args.num_slots} | **Servers:** {args.num_servers}",
        f"**C mask mode:** {args.c_mask_mode}",
        f"**Seeds:** {args.seeds} | **Episodes/seed:** {args.episodes} | **Requests/episode:** {args.requests_per_episode}",
        "",
        "## Main Results",
        "",
        "| Method | Blocking | Success | Delay(ms) | DeadlineSat | AvgFS | ObjScore |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, r in sorted(results.items(), key=lambda kv: kv[1]["blocking_rate_mean"]):
        lines.append(
            f"| {method.upper()} | {r['blocking_rate_mean']:.3f} | "
            f"{r['success_rate_mean']:.3f} | {r['avg_delay_ms_mean']:.1f} | "
            f"{r['deadline_met_rate_mean']:.3f} | {r['avg_fs_mean']:.2f} | "
            f"{r['objective_score_mean']:.4f} |"
        )

    lines.extend(["", "## Behavior", ""])
    for method, r in sorted(results.items(), key=lambda kv: kv[1]["blocking_rate_mean"]):
        lines.extend([
            f"### {method.upper()}",
            f"- Split: {_fmt_dist(r['split_dist_mean'])}",
            f"- Server: {_fmt_dist(r['server_dist_mean'])}",
            f"- Modulation: {_fmt_dist(r['mod_dist_mean'])}",
            f"- Failure reasons: {r['reason_dist_total']}",
            f"- Valid C actions mean: {r['valid_action_mean_mean']:.2f}",
            "",
        ])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
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
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--methods", type=str, default="wo,df,rf,iwd")
    parser.add_argument(
        "--c_mask_mode",
        type=str,
        choices=["masked", "paper"],
        default="masked",
        help=(
            "masked: choose among Agent-C feasible actions; "
            "paper: ignore Agent-C mask for WO/DF/RF/IWD and count infeasible execution as blocking"
        ),
    )
    parser.add_argument("--output_json", type=str, default="sa_hmarl/experiments/sp_hm_ff_offloading.json")
    parser.add_argument("--output_md", type=str, default="sa_hmarl/experiments/sp_hm_ff_offloading.md")
    args = parser.parse_args()

    if args.max_blocks is None:
        args.max_blocks = args.num_slots

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

    all_results = {}
    for method in methods:
        print(f"\n--- {method.upper()} + SP-HM-FF ---")
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            episodes_data = []
            for _ in range(args.episodes):
                src = rng.randint(0, env_proto.net.NUM_NODES)
                requests = generate_requests(
                    env_proto,
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
                episodes_data.append(requests)

            res = evaluate_method(method, episodes_data, env_proto, args, seed)
            seed_results.append(res)
            print(
                f"seed {seed}: blk={res['blocking_rate']:.3f} "
                f"delay={res['avg_delay_ms']:.1f} fs={res['avg_fs']:.2f} "
                f"reasons={res['reason_dist']}"
            )
        agg = _aggregate(seed_results)
        all_results[method] = agg
        print(
            f">> {method.upper()} AVG: blk={agg['blocking_rate_mean']:.3f} "
            f"delay={agg['avg_delay_ms_mean']:.1f} fs={agg['avg_fs_mean']:.2f}"
        )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    _write_markdown(args.output_md, all_results, args)
    print(f"\nSaved JSON: {args.output_json}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
