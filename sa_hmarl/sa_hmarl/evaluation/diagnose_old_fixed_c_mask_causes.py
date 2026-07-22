"""Read-only cause decomposition for the legacy fixed-C/all-OD R mask."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import build_agent_r_observation
from sa_hmarl.evaluation.generate_multitopology_strict_v13_vs_heuristics import (
    _build_env,
    _generate_all_od_requests,
)
from sa_hmarl.network.ksp import get_k_shortest_paths


PRIMARY_ORDER = (
    "server_capacity_infeasible",
    "deadline_budget_infeasible",
    "no_reach_feasible_path_mod",
    "fs_exceeds_total_slots",
    "no_common_free_capacity",
    "continuity_mismatch",
    "contiguity_fragmentation",
    "k50_path_pool_empty",
)


def _max_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _spectrum_digest(env: Any) -> bytes:
    return b"".join(env.net.link_states[key].tobytes() for key in sorted(env.net.link_states))


def diagnose_mask_empty(env: Any, req: Any, split_id: int, server_id: int) -> Dict[str, Any]:
    before = _spectrum_digest(env)
    split = req.splits[split_id]
    server = env.mec.servers[server_id]
    paths = get_k_shortest_paths(
        env.net.G, req.src_node, server.node_id, 50,
        weight="length_km", sort_by="hops",
    )
    flags = {name: False for name in PRIMARY_ORDER}
    edge_ms = env._estimate_compute_ms(server, split.edge_compute_cost)
    local_ms = env._estimate_local_compute_ms(server, split.local_compute_cost)
    if edge_ms == float("inf") or split.edge_compute_cost > server.available_compute:
        flags["server_capacity_infeasible"] = True
    else:
        reach_count = deadline_count = fitting_count = 0
        for path in paths:
            distance = env.net.path_length_km(path)
            edge_lengths = {
                (min(u, v), max(u, v)): env.net.G.edges[(min(u, v), max(u, v))]["length_km"]
                for u, v in zip(path[:-1], path[1:])
            }
            per_edge_free = [
                ~env.net.link_states[(min(u, v), max(u, v))]
                for u, v in zip(path[:-1], path[1:])
            ]
            for mod_idx in range(env.mod_reg.num_formats):
                mod = env.mod_reg[mod_idx]
                if distance > mod.reach_km:
                    continue
                reach_count += 1
                feasible, required_fs, _ = env.fs_calc.compute_full(
                    data_bits=split.intermediate_data_bits,
                    deadline_s=req.deadline_ms / 1000.0,
                    ctrl_delay_s=0.0,
                    local_compute_s=local_ms / 1000.0,
                    edge_compute_s=edge_ms / 1000.0,
                    queue_delay_s=server._queue_delay_ema / 1000.0,
                    path=path,
                    edge_lengths=edge_lengths,
                    modulation=mod,
                )
                if not feasible:
                    flags["deadline_budget_infeasible"] = True
                    continue
                deadline_count += 1
                if required_fs > env.net.num_slots:
                    flags["fs_exceeds_total_slots"] = True
                    continue
                fitting_count += 1
                if any(int(free.sum()) < required_fs for free in per_edge_free):
                    flags["no_common_free_capacity"] = True
                    continue
                common = np.logical_and.reduce(per_edge_free)
                if int(common.sum()) < required_fs:
                    flags["continuity_mismatch"] = True
                elif _max_run(common) < required_fs:
                    flags["contiguity_fragmentation"] = True
        if paths and reach_count == 0:
            flags["no_reach_feasible_path_mod"] = True
        if reach_count and deadline_count == 0:
            flags["deadline_budget_infeasible"] = True
        if deadline_count and fitting_count == 0:
            flags["fs_exceeds_total_slots"] = True

    old_k = env.k
    try:
        env.k = 500
        obs500 = build_agent_r_observation(env, req, split_id, server_id)
        flags["k50_path_pool_empty"] = bool(np.asarray(obs500["agent_r_mask"], dtype=bool).any())
    finally:
        env.k = old_k
    if _spectrum_digest(env) != before:
        raise RuntimeError("Read-only mask diagnosis mutated spectrum state")
    primary = next((name for name in PRIMARY_ORDER if flags[name]), "unclassified")
    return {"primary_reason": primary, "flags": flags}


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        num_slots=320, num_servers=4, k_paths_r=50,
        path_sort_strategy_r="hops", block_sort_strategy_r="start_asc",
        modulation_profile="default", max_blocks=10,
    )


def run_seed(seed: int) -> Dict[str, Any]:
    args = _args()
    topology = "xlron_cost239_ptrnet_real"
    env = _build_env(args, topology, seed)
    server_nodes = [int(server.node_id) for server in env.mec.servers]
    requests = _generate_all_od_requests(
        env.net.NUM_NODES, server_nodes, np.random.RandomState(seed), 6500,
        0.3, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2,
        3, "default3", False, False,
    )
    env.reset(requests)
    node_to_server = {int(server.node_id): index for index, server in enumerate(env.mec.servers)}
    counts = Counter()
    examples: List[Dict[str, Any]] = []
    for index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        dst = int(req._dst_node)
        server_id = node_to_server[dst]
        obs = build_agent_r_observation(env, req, 0, server_id)
        mask = np.asarray(obs["agent_r_mask"], dtype=bool)
        diagnosis = None
        if not mask.any():
            diagnosis = diagnose_mask_empty(env, req, 0, server_id)
            env.reject_next_request(req.req_id, "r_no_valid_action")
        else:
            flat_action = int(ksp_ff_highest_mod_action(obs))
            path_idx = flat_action // (env.mod_reg.num_formats * env.max_blocks)
            remainder = flat_action % (env.mod_reg.num_formats * env.max_blocks)
            action = (path_idx, remainder // env.max_blocks, remainder % env.max_blocks)
            _, _, _, info = env.step((0, server_id), action)
            if not bool(info.get("success", False)):
                raise RuntimeError(f"Legal KSP action failed at seed={seed}, request={index}: {info}")
        if index < 500:
            continue
        counts["evaluated"] += 1
        if diagnosis is None:
            counts["accepted"] += 1
        else:
            counts["blocked"] += 1
            counts[f"primary:{diagnosis['primary_reason']}"] += 1
            for name, value in diagnosis["flags"].items():
                if value:
                    counts[f"flag:{name}"] += 1
            if len(examples) < 20:
                examples.append({
                    "request_index": index, "req_id": req.req_id,
                    "src": req.src_node, "dst": dst, **diagnosis,
                })
    return {
        "seed": seed, "evaluated": counts["evaluated"], "accepted": counts["accepted"],
        "blocked": counts["blocked"], "blocking_rate": counts["blocked"] / counts["evaluated"],
        "primary_causes": {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("primary:")},
        "nonexclusive_flags": {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("flag:")},
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="5001,5002,5003,5004,5005")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in seeds}
        runs = []
        for future in as_completed(futures):
            result = future.result()
            runs.append(result)
            print(f"seed={result['seed']} blocking={result['blocking_rate']:.4%}", flush=True)
    runs.sort(key=lambda row: row["seed"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    primary = Counter()
    flags = Counter()
    for run in runs:
        primary.update(run["primary_causes"])
        flags.update(run["nonexclusive_flags"])
    payload = {
        "protocol": {
            "topology": "xlron_cost239_ptrnet_real", "num_slots": 320,
            "fixed_split_id": 0, "traffic": "fixed-C/all-OD", "selector": "KSP-FF K=50 hops",
            "warmup": 500, "evaluated_per_seed": 6000, "seeds": seeds,
            "primary_order": list(PRIMARY_ORDER),
        },
        "runs": runs, "aggregate_primary_causes": dict(primary),
        "aggregate_nonexclusive_flags": dict(flags),
    }
    (output / "OLD_MASK_CAUSE_DECOMPOSITION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output / "old_mask_per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "evaluated", "accepted", "blocked", "blocking_rate"] + list(PRIMARY_ORDER)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run[key] for key in fields[:5]}
            row.update({name: run["primary_causes"].get(name, 0) for name in PRIMARY_ORDER})
            writer.writerow(row)
    total = sum(run["evaluated"] for run in runs)
    blocked = sum(run["blocked"] for run in runs)
    lines = [
        "# Old Fixed-C/All-OD Mask Cause Decomposition", "",
        f"- Blocking: {blocked}/{total} = {blocked / total:.4%}",
        f"- Primary cause order: `{' > '.join(PRIMARY_ORDER)}`", "",
        "| Primary cause | Count | Share of blocked |", "|---|---:|---:|",
    ]
    for name in PRIMARY_ORDER:
        count = primary[name]
        lines.append(f"| {name} | {count} | {count / max(blocked, 1):.2%} |")
    lines.extend(["", "Nonexclusive flags are retained in the JSON output. Diagnosis is read-only."])
    (output / "OLD_MASK_CAUSE_DECOMPOSITION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
