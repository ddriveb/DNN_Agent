#!/usr/bin/env python3
"""KSP-FF K=50 hops parity check (fixed-C / all-OD, 5 seeds)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import gzip
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import (
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od import (
    FixedRequestRecord,
    MethodSummary,
    _generate_all_od_requests,
    _spectrum_state,
)
from sa_hmarl.training.utils import make_env


def _run_ksp_ff_only(env, requests, fixed_split_id, seed):
    summary = MethodSummary(method="KSP-FF K=50 hops")
    records: List[FixedRequestRecord] = []
    node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < 500
        env.advance_time(req.arrival_time)
        free_ratio, lfb, frag, phi_spec = _spectrum_state(env)
        dst_node = int(getattr(req, "_dst_node", req.src_node))
        server_id = node_to_server.get(dst_node)
        if server_id is None:
            server_id = min(node_to_server.values(), key=lambda i: abs(i - dst_node))

        env.k = 50
        env.path_sort_strategy = "hops"
        env.block_sort_strategy = "start_asc"
        obs_r = build_agent_r_observation(env, req, fixed_split_id, server_id)
        agent_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        legal_count = int(agent_r_mask.sum())

        if not agent_r_mask.any():
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = FixedRequestRecord(
                    seed=seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    warmup=False,
                    fixed_src_node=int(req.src_node),
                    fixed_split_id=fixed_split_id,
                    fixed_server_id=server_id,
                    fixed_dst_node=dst_node,
                    r_idx=None,
                    r_valid=False,
                    path_idx=-1,
                    modulation_idx=-1,
                    block_idx=-1,
                    modulation="",
                    required_fs=0,
                    block_start=-1,
                    block_size=0,
                    block_waste=0.0,
                    path_hops=0,
                    path_length_km=0.0,
                    path_nodes=[],
                    free_ratio_before=free_ratio,
                    lfb_before=lfb,
                    fragmentation_before=frag,
                    phi_spec_before=phi_spec,
                    legal_r_action_count=legal_count,
                    strict_top30_contains_ksp_action=False,
                    strict_rank_of_ksp_action_if_contained=-1,
                    strict_selected_rank_in_ppo_r_top30=-1,
                    success=False,
                    failure_reason="r_no_valid_action",
                    delay_ms=0.0,
                    decision_ms=0.0,
                )
                records.append(rec)
                summary.add_record(rec)
            continue

        a = ksp_ff_highest_mod_action(obs_r)
        if a is None:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = FixedRequestRecord(
                    seed=seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    warmup=False,
                    fixed_src_node=int(req.src_node),
                    fixed_split_id=fixed_split_id,
                    fixed_server_id=server_id,
                    fixed_dst_node=dst_node,
                    r_idx=None,
                    r_valid=False,
                    path_idx=-1,
                    modulation_idx=-1,
                    block_idx=-1,
                    modulation="",
                    required_fs=0,
                    block_start=-1,
                    block_size=0,
                    block_waste=0.0,
                    path_hops=0,
                    path_length_km=0.0,
                    path_nodes=[],
                    free_ratio_before=free_ratio,
                    lfb_before=lfb,
                    fragmentation_before=frag,
                    phi_spec_before=phi_spec,
                    legal_r_action_count=legal_count,
                    strict_top30_contains_ksp_action=False,
                    strict_rank_of_ksp_action_if_contained=-1,
                    strict_selected_rank_in_ppo_r_top30=-1,
                    success=False,
                    failure_reason="r_no_valid_action",
                    delay_ms=0.0,
                    decision_ms=0.0,
                )
                records.append(rec)
                summary.add_record(rec)
            continue

        r_action = decode_agent_r_action(a, num_mods, max_blocks)
        _, _, _, info = env.step((fixed_split_id, server_id), r_action)
        if not is_warmup:
            path_idx, mod_idx, block_idx = r_action
            path_feats = obs_r.get("path_features", [])
            path_length_km = 0.0
            hop_count = 0
            if path_feats and 0 <= path_idx < len(path_feats):
                path_length_km = float(path_feats[path_idx].get("path_length_km", 0.0))
                hop_count = int(path_feats[path_idx].get("hop_count", 0))
            mod_name = ""
            mod_names = obs_r.get("mod_names", [])
            if mod_names and 0 <= mod_idx < len(mod_names):
                mod_name = str(mod_names[mod_idx])
            blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
            block_start = -1
            block_size = 0
            if block_idx < len(blocks):
                block_start = int(blocks[block_idx][0])
                block_size = int(blocks[block_idx][1])
            req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
            required_fs = int(req_fs) if req_fs is not None else 0
            block_waste = (block_size - required_fs) / max(block_size, 1)
            candidate_paths = obs_r.get("candidate_paths", [])
            path_nodes = []
            if candidate_paths and 0 <= path_idx < len(candidate_paths):
                path_nodes = [int(n) for n in candidate_paths[path_idx]]
            success = bool(info.get("success", False))
            rec = FixedRequestRecord(
                seed=seed,
                step_idx=step_idx,
                req_id=int(req.req_id),
                arrival_time=float(req.arrival_time),
                warmup=False,
                fixed_src_node=int(req.src_node),
                fixed_split_id=fixed_split_id,
                fixed_server_id=server_id,
                fixed_dst_node=dst_node,
                r_idx=int(a),
                r_valid=True,
                path_idx=path_idx,
                modulation_idx=mod_idx,
                block_idx=block_idx,
                modulation=mod_name,
                required_fs=required_fs,
                block_start=block_start,
                block_size=block_size,
                block_waste=float(block_waste),
                path_hops=hop_count,
                path_length_km=path_length_km,
                path_nodes=path_nodes,
                free_ratio_before=free_ratio,
                lfb_before=lfb,
                fragmentation_before=frag,
                phi_spec_before=phi_spec,
                legal_r_action_count=legal_count,
                strict_top30_contains_ksp_action=False,
                strict_rank_of_ksp_action_if_contained=-1,
                strict_selected_rank_in_ppo_r_top30=-1,
                success=success,
                failure_reason=str(info.get("reason", "")) if not success else "",
                delay_ms=float(info.get("delay_ms", 0.0)) if success else 0.0,
                decision_ms=0.0,
            )
            records.append(rec)
            summary.add_record(rec)
    return summary, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--arrival_interval", type=float, default=0.3)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--fixed_split_id", type=int, default=0)
    parser.add_argument("--warmup_requests", type=int, default=500)
    parser.add_argument("--requests_per_episode", type=int, default=6000)
    parser.add_argument("--seeds", default="5001,5002,5003,5004,5005")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/strict_v13_vs_ksp_ff_vs_deeprmsa_cost239_fixed_c_all_od/ksp_ff_only_parity")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    env_for_nodes = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        42,
        modulation_profile="default",
        max_blocks=10,
        block_sort_strategy="start_asc",
        path_sort_strategy="hops",
        k=50,
    )
    server_node_ids = [int(s.node_id) for s in env_for_nodes.mec.servers]
    total_requests = args.warmup_requests + args.requests_per_episode
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    per_seed: List[Dict[str, Any]] = []
    all_records: List[FixedRequestRecord] = []

    t_start = time.time()
    for seed in seeds:
        rng = np.random.RandomState(seed)
        requests = _generate_all_od_requests(
            num_nodes=env_for_nodes.net.NUM_NODES,
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
            poisson_arrivals=False,
            exponential_holding=False,
        )
        env = make_env(
            args.topology,
            args.num_slots,
            args.num_servers,
            seed,
            modulation_profile="default",
            max_blocks=10,
            block_sort_strategy="start_asc",
            path_sort_strategy="hops",
            k=50,
        )
        env.reset(requests)
        summary, records = _run_ksp_ff_only(env, requests, args.fixed_split_id, seed)
        per_seed.append({"seed": seed, "summary": summary.to_dict()})
        all_records.extend(records)
        print(f"[ksp_ff_parity][seed={seed}] blocking={summary.blocking_rate():.4%}")

    total_summary = MethodSummary(method="KSP-FF K=50 hops")
    for rec in all_records:
        total_summary.add_record(rec)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parity.json").write_text(json.dumps({
        "config": vars(args),
        "overall": total_summary.to_dict(),
        "per_seed": per_seed,
        "elapsed_seconds": time.time() - t_start,
    }, indent=2, default=str), encoding="utf-8")

    csv_path = out_dir / "per_seed_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "total", "blocked", "blocking_rate", "r_no_valid_rate",
            "avg_fs", "avg_path_km", "avg_hops", "avg_block_start", "avg_block_waste"
        ])
        for row in per_seed:
            s = row["summary"]
            writer.writerow([
                row["seed"], s["total"], s["blocked"], s["blocking_rate"],
                s["r_no_valid_rate"], s["avg_fs"], s["avg_path_km"], s["avg_hops"],
                s["avg_block_start"], s["avg_block_waste"],
            ])

    trace_path = out_dir / "trace.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec.to_dict(), default=str) + "\n")

    print(f"[ksp_ff_parity] Overall blocking={total_summary.blocking_rate():.4%}")
    print(f"[ksp_ff_parity] Wrote {out_dir / 'parity.json'}")
    print(f"[ksp_ff_parity] Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())
