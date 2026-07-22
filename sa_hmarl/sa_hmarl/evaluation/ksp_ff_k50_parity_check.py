#!/usr/bin/env python3
"""KSP-FF K=50 hops parity check under COST239 fixed-C / all-OD.

Runs KSP-FF K=50 hops on 5 pilot seeds and reports blocking rates.
If the average does not match the previous ~5.6342% result, the script exits
with an error so the main experiment cannot proceed without config drift
investigation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
from sa_hmarl.evaluation.diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od import (
    _generate_all_od_requests,
)
from sa_hmarl.training.utils import make_env


@dataclass
class PerSeedResult:
    seed: int
    total: int = 0
    blocked: int = 0
    r_no_valid: int = 0
    server_overload: int = 0
    deadline_infeasible: int = 0
    no_suitable_block: int = 0
    avg_fs: float = 0.0
    avg_hops: float = 0.0
    avg_path_km: float = 0.0
    avg_block_start: float = 0.0
    avg_block_waste: float = 0.0


def _run_ksp_ff_on_seed(seed: int, args: argparse.Namespace) -> PerSeedResult:
    """Run KSP-FF K=50 hops on one seed with fixed-C / all-OD."""
    rng = np.random.RandomState(seed)
    total_requests = args.warmup_requests + args.requests_per_episode

    env = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy_r,
        path_sort_strategy=args.path_sort_strategy_r,
        k=args.k_paths_r,
    )
    num_servers = len(env.mec.servers)
    server_node_ids = [int(s.node_id) for s in env.mec.servers]
    node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}

    requests = _generate_all_od_requests(
        num_nodes=env.net.NUM_NODES,
        server_node_ids=server_node_ids,
        rng=rng,
        num_requests=total_requests,
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
        poisson_arrivals=args.poisson_arrivals,
        exponential_holding=args.exponential_holding,
    )
    env.reset(requests)

    res = PerSeedResult(seed=seed)
    fs_vals, hops_vals, km_vals, block_start_vals, block_waste_vals = [], [], [], [], []

    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        is_warmup = step_idx < args.warmup_requests

        dst_node = int(getattr(req, "_dst_node", req.src_node))
        split_id = int(args.fixed_split_id)
        server_id = node_to_server.get(dst_node)
        if server_id is None:
            server_id = min(node_to_server.values(), key=lambda i: abs(i - dst_node))

        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        if not raw_r_mask.any():
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                res.total += 1
                res.blocked += 1
                res.r_no_valid += 1
            continue

        r_idx = ksp_ff_highest_mod_action(obs_r)
        if r_idx is None:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                res.total += 1
                res.blocked += 1
                res.r_no_valid += 1
            continue

        r_action = decode_agent_r_action(
            int(r_idx), len(obs_r["mod_names"]), env.max_blocks
        )
        _, _, _, info = env.step((split_id, server_id), r_action)
        if not is_warmup:
            res.total += 1
            if info.get("success", False):
                fs_vals.append(info.get("num_slots", 0))
                hops_vals.append(info.get("num_hops", 0))
                km_vals.append(info.get("path_dist_km", 0.0))
                block_start_vals.append(info.get("start_slot", 0))
                block_waste_vals.append(info.get("block_waste", 0.0))
            else:
                res.blocked += 1
                reason = info.get("reason", "unknown")
                if reason == "r_no_valid_action":
                    res.r_no_valid += 1
                elif reason == "server_overload":
                    res.server_overload += 1
                elif reason == "deadline_infeasible":
                    res.deadline_infeasible += 1
                elif reason == "no_suitable_block":
                    res.no_suitable_block += 1

    res.avg_fs = float(np.mean(fs_vals)) if fs_vals else 0.0
    res.avg_hops = float(np.mean(hops_vals)) if hops_vals else 0.0
    res.avg_path_km = float(np.mean(km_vals)) if km_vals else 0.0
    res.avg_block_start = float(np.mean(block_start_vals)) if block_start_vals else 0.0
    res.avg_block_waste = float(np.mean(block_waste_vals)) if block_waste_vals else 0.0
    return res


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="5001,5002,5003,5004,5005")
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths_r", type=int, default=50)
    parser.add_argument("--path_sort_strategy_r", default="hops")
    parser.add_argument("--block_sort_strategy_r", default="start_asc")
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.3)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--warmup_requests", type=int, default=500)
    parser.add_argument("--requests_per_episode", type=int, default=6000)
    parser.add_argument("--fixed_split_id", type=int, default=0)
    parser.add_argument("--poisson_arrivals", action="store_true")
    parser.add_argument("--exponential_holding", action="store_true")
    parser.add_argument("--output_dir",
                        default="sa_hmarl/experiments/"
                                "strict_v13_vs_ksp_ff_vs_deeprmsa_cost239_fixed_c_all_od/ksp_ff_only_parity")
    parser.add_argument("--expected_blocking_rate", type=float, default=0.056341666666666665)
    parser.add_argument("--tolerance_pp", type=float, default=1.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"]:
        os.environ.setdefault(key, "1")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        res = _run_ksp_ff_on_seed(seed, args)
        results.append({
            "seed": res.seed,
            "total": res.total,
            "blocked": res.blocked,
            "blocking_rate": res.blocked / max(res.total, 1),
            "r_no_valid": res.r_no_valid,
            "server_overload": res.server_overload,
            "deadline_infeasible": res.deadline_infeasible,
            "no_suitable_block": res.no_suitable_block,
            "avg_fs": res.avg_fs,
            "avg_hops": res.avg_hops,
            "avg_path_km": res.avg_path_km,
            "avg_block_start": res.avg_block_start,
            "avg_block_waste": res.avg_block_waste,
        })
        print(
            f"[seed={seed}] blocked={res.blocked}/{res.total} "
            f"({res.blocked / max(res.total, 1):.4%}) "
            f"r_no_valid={res.r_no_valid}",
            flush=True,
        )

    avg_blocking = float(np.mean([r["blocking_rate"] for r in results]))
    std_blocking = float(np.std([r["blocking_rate"] for r in results]))
    print(f"Average blocking rate: {avg_blocking:.4%} ± {std_blocking:.4%}")

    summary = {
        "method": "KSP-FF K=50 hops",
        "function": "ksp_ff_highest_mod_action",
        "expected_blocking_rate": args.expected_blocking_rate,
        "tolerance_pp": args.tolerance_pp,
        "average_blocking_rate": avg_blocking,
        "std_blocking_rate": std_blocking,
        "per_seed": results,
    }

    summary_path = output_dir / "ksp_ff_parity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    diff_pp = abs(avg_blocking - args.expected_blocking_rate) * 100.0
    if diff_pp > args.tolerance_pp:
        print(
            f"ERROR: KSP-FF blocking rate {avg_blocking:.4%} differs from expected "
            f"{args.expected_blocking_rate:.4%} by {diff_pp:.2f} pp (> tolerance "
            f"{args.tolerance_pp:.2f} pp). Investigate config drift before proceeding.",
            file=sys.stderr,
        )
        return 1

    print("KSP-FF parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
