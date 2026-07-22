#!/usr/bin/env python3
"""Five-seed paired closed-loop regression for Strict v1.3 optimization."""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.env.observation_builder import (
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.evaluation import (
    generate_multitopology_strict_v13_vs_heuristics as multitopology,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_r
from sa_hmarl.evaluation.strict_v13_online import select_strict_v13
from sa_hmarl.network.modulation import ModulationRegistry


TOPOLOGIES = (
    "xlron_nsfnet_deeprmsa",
    "xlron_usnet_gcnrmsa",
    "xlron_jpn48",
)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _make_requests(args, topology: str, seed: int):
    env = multitopology._build_env(args, topology, seed)
    server_nodes = [int(server.node_id) for server in env.mec.servers]
    requests = multitopology._generate_all_od_requests(
        env.net.NUM_NODES,
        server_nodes,
        np.random.RandomState(seed),
        args.warmup_requests + args.requests_per_episode,
        args.arrival_interval,
        args.holding_min,
        args.holding_max,
        args.deadline_min,
        args.deadline_max,
        args.size_min_mb,
        args.size_max_mb,
        args.edge_cost_min,
        args.edge_cost_max,
        args.num_splits,
        args.split_profile,
        args.poisson_arrivals,
        args.exponential_holding,
    )
    return env, requests


def _run_version(args, topology: str, seed: int, agent_r, ranker, optimized: bool):
    env, requests = _make_requests(args, topology, seed)
    env.reset(requests)
    node_to_server = {
        int(server.node_id): index for index, server in enumerate(env.mec.servers)
    }
    records: List[Tuple[int, int | None, bool, str]] = []

    for step_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        if not env.event_queue or int(env.event_queue[0][2].req_id) != int(req.req_id):
            raise RuntimeError(
                f"Queue mismatch topology={topology} seed={seed} step={step_index}"
            )
        destination = int(getattr(req, "_dst_node", req.src_node))
        server_id = int(node_to_server[destination])
        obs_r = build_agent_r_observation(
            env, req, args.fixed_split_id, server_id
        )
        mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if not mask.any():
            action = None
            info = env.reject_next_request(req.req_id, "r_no_valid_action")[3]
        else:
            selected = select_strict_v13(
                ranker,
                agent_r,
                env,
                req,
                obs_r,
                args.fixed_split_id,
                server_id,
                optimized=optimized,
                collect_action_coverage_diagnostics=False,
            )
            action = selected.action
            if action is None:
                info = env.reject_next_request(req.req_id, "r_no_valid_action")[3]
            else:
                decoded = decode_agent_r_action(
                    int(action), env.mod_reg.num_formats, env.max_blocks
                )
                info = env.step((args.fixed_split_id, server_id), decoded)[3]

        if step_index >= args.warmup_requests:
            records.append((
                int(req.req_id),
                int(action) if action is not None else None,
                bool(info.get("success", False)),
                str(info.get("reason", "")) if not info.get("success", False) else "",
            ))
    return records


def _worker(task):
    topology, seed, args_dict = task
    args = argparse.Namespace(**args_dict)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    registry = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, registry, args.device)
    ranker = multitopology._load_ranker_model(
        args.strict_ranker_checkpoint, args.device
    )
    reference = _run_version(args, topology, seed, agent_r, ranker, optimized=False)
    optimized = _run_version(args, topology, seed, agent_r, ranker, optimized=True)

    first_mismatch = None
    action_mismatches = 0
    outcome_mismatches = 0
    blocked_identity_mismatches = 0
    for index, (left, right) in enumerate(zip(reference, optimized)):
        if left[1] != right[1]:
            action_mismatches += 1
        if left[2:] != right[2:]:
            outcome_mismatches += 1
        if (not left[2]) != (not right[2]):
            blocked_identity_mismatches += 1
        if left != right and first_mismatch is None:
            first_mismatch = {
                "index": index,
                "reference": left,
                "optimized": right,
            }

    reference_blocked = sum(not row[2] for row in reference)
    optimized_blocked = sum(not row[2] for row in optimized)
    passed = (
        len(reference) == len(optimized) == args.requests_per_episode
        and action_mismatches == 0
        and outcome_mismatches == 0
        and blocked_identity_mismatches == 0
    )
    return {
        "topology": topology,
        "seed": seed,
        "evaluated_requests": len(reference),
        "request_trace_hash": multitopology._request_trace_hash(
            _make_requests(args, topology, seed)[1]
        ),
        "reference_blocked": reference_blocked,
        "optimized_blocked": optimized_blocked,
        "reference_blocking_rate": reference_blocked / max(len(reference), 1),
        "optimized_blocking_rate": optimized_blocked / max(len(optimized), 1),
        "action_mismatches": action_mismatches,
        "outcome_mismatches": outcome_mismatches,
        "blocked_identity_mismatches": blocked_identity_mismatches,
        "first_mismatch": first_mismatch,
        "passed": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = multitopology.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir="sa_hmarl/experiments/strict_v13_latency_optimization/closed_loop_regression",
        warmup_requests=500,
        requests_per_episode=6000,
        max_workers=5,
    )
    parser.add_argument("--topologies", default=",".join(TOPOLOGIES))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topologies = tuple(item.strip() for item in args.topologies.split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    tasks = [(topology, seed, vars(args)) for topology in topologies for seed in seeds]
    workers = min(args.max_workers or len(tasks), len(tasks))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{result['topology']} seed={result['seed']}] "
                f"actions={result['action_mismatches']} outcomes={result['outcome_mismatches']} "
                f"passed={result['passed']}",
                flush=True,
            )
    results.sort(key=lambda item: (topologies.index(item["topology"]), item["seed"]))
    passed = all(item["passed"] for item in results)
    payload = {
        "schema_version": 1,
        "passed": passed,
        "config": vars(args),
        "checkpoint_hashes": {
            "ppo_r": multitopology._sha256(args.agent_r_checkpoint),
            "ranker": multitopology._sha256(args.strict_ranker_checkpoint),
        },
        "results": results,
        "totals": {
            "evaluated_requests": sum(item["evaluated_requests"] for item in results),
            "action_mismatches": sum(item["action_mismatches"] for item in results),
            "outcome_mismatches": sum(item["outcome_mismatches"] for item in results),
            "blocked_identity_mismatches": sum(
                item["blocked_identity_mismatches"] for item in results
            ),
        },
    }
    _atomic_json(output_dir / "CLOSED_LOOP_REGRESSION.json", payload)

    csv_path = output_dir / "per_seed_regression.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "topology", "seed", "evaluated_requests", "reference_blocked",
            "optimized_blocked", "reference_blocking_rate",
            "optimized_blocking_rate", "action_mismatches", "outcome_mismatches",
            "blocked_identity_mismatches", "passed", "request_trace_hash",
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    os.replace(temporary, csv_path)

    report = [
        "# Strict v1.3 Closed-loop Exact Regression",
        "",
        f"Overall PASS: **{passed}**",
        "",
        "| Topology | Seed | Reference blocked | Optimized blocked | Action mismatch | Outcome mismatch | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        report.append(
            f"| {result['topology']} | {result['seed']} | {result['reference_blocked']} | "
            f"{result['optimized_blocked']} | {result['action_mismatches']} | "
            f"{result['outcome_mismatches']} | {result['passed']} |"
        )
    report.extend([
        "",
        f"- Evaluated requests: {payload['totals']['evaluated_requests']}",
        f"- Action mismatches: {payload['totals']['action_mismatches']}",
        f"- Outcome mismatches: {payload['totals']['outcome_mismatches']}",
        f"- Blocked identity mismatches: {payload['totals']['blocked_identity_mismatches']}",
    ])
    report_path = output_dir / "CLOSED_LOOP_REGRESSION.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(f"[done] PASS={passed} output={output_dir}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
