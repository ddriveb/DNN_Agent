#!/usr/bin/env python3
"""Profile and verify exact online latency optimization for Strict v1.3."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from sa_hmarl.baselines.rmsa_baselines import (
    ff_ksp_highest_mod_action,
    ksp_ff_highest_mod_action,
)
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
METHODS = ("strict_reference", "strict_optimized", "ksp_ff", "ff_ksp")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _requests(args, topology: str, seed: int, count: int):
    env = multitopology._build_env(args, topology, seed)
    server_nodes = [int(server.node_id) for server in env.mec.servers]
    requests = multitopology._generate_all_od_requests(
        env.net.NUM_NODES,
        server_nodes,
        np.random.RandomState(seed),
        count,
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


def _server_map(env) -> Dict[int, int]:
    return {int(server.node_id): index for index, server in enumerate(env.mec.servers)}


def _server_id(node_to_server: Dict[int, int], req) -> int:
    destination = int(getattr(req, "_dst_node", req.src_node))
    if destination not in node_to_server:
        raise RuntimeError(f"No deterministic server for destination {destination}")
    return int(node_to_server[destination])


def _consume(env, req, split_id: int, server_id: int, action: int | None) -> Dict[str, Any]:
    if action is None:
        return env.reject_next_request(req.req_id, "r_no_valid_action")[3]
    decoded = decode_agent_r_action(
        int(action), env.mod_reg.num_formats, env.max_blocks
    )
    return env.step((split_id, server_id), decoded)[3]


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def run_equivalence(args, topology: str, agent_r, ranker) -> Dict[str, Any]:
    total_requests = args.warmup_requests + args.equivalence_states * 3
    env, requests = _requests(args, topology, args.profile_seed, total_requests)
    env.reset(requests)
    node_to_server = _server_map(env)

    checked = 0
    action_mismatches = 0
    topk_mismatches = 0
    legal_mismatches = 0
    max_logit_error = 0.0
    max_feature_error = 0.0
    max_normalized_error = 0.0
    max_score_error = 0.0
    request_mismatches = 0
    first_failure = None
    samples: List[Dict[str, Any]] = []

    for step_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        if not env.event_queue or int(env.event_queue[0][2].req_id) != int(req.req_id):
            request_mismatches += 1
            first_failure = first_failure or {
                "step": step_index,
                "reason": "request_queue_mismatch",
                "loop_req": int(req.req_id),
                "queue_req": int(env.event_queue[0][2].req_id) if env.event_queue else None,
            }
            break

        server_id = _server_id(node_to_server, req)
        obs_r = build_agent_r_observation(
            env, req, args.fixed_split_id, server_id
        )
        mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        if step_index < args.warmup_requests:
            if not mask.any():
                _consume(env, req, args.fixed_split_id, server_id, None)
                continue
            selected = select_strict_v13(
                ranker,
                agent_r,
                env,
                req,
                obs_r,
                args.fixed_split_id,
                server_id,
                optimized=True,
                collect_action_coverage_diagnostics=False,
            )
            _consume(env, req, args.fixed_split_id, server_id, selected.action)
            continue

        if not mask.any():
            _consume(env, req, args.fixed_split_id, server_id, None)
            continue

        reference = select_strict_v13(
            ranker,
            agent_r,
            env,
            req,
            obs_r,
            args.fixed_split_id,
            server_id,
            optimized=False,
            collect_action_coverage_diagnostics=False,
        )
        optimized = select_strict_v13(
            ranker,
            agent_r,
            env,
            req,
            obs_r,
            args.fixed_split_id,
            server_id,
            optimized=True,
            collect_action_coverage_diagnostics=False,
        )

        legal_equal = np.array_equal(
            reference.topk.legal_actions, optimized.topk.legal_actions
        )
        topk_equal = np.array_equal(
            reference.topk.candidate_actions, optimized.topk.candidate_actions
        )
        action_equal = reference.action == optimized.action
        if not legal_equal:
            legal_mismatches += 1
        if not topk_equal:
            topk_mismatches += 1
        if not action_equal:
            action_mismatches += 1

        legal = reference.topk.legal_actions
        logit_error = _max_abs(
            reference.topk.logits[legal], optimized.topk.logits[legal]
        )
        feature_error = _max_abs(
            reference.ranker_features, optimized.ranker_features
        )
        normalized_error = _max_abs(
            reference.normalized_features, optimized.normalized_features
        )
        score_error = _max_abs(
            reference.ranker_scores, optimized.ranker_scores
        )
        max_logit_error = max(max_logit_error, logit_error)
        max_feature_error = max(max_feature_error, feature_error)
        max_normalized_error = max(max_normalized_error, normalized_error)
        max_score_error = max(max_score_error, score_error)

        passed = (
            legal_equal
            and topk_equal
            and action_equal
            and logit_error <= 1e-6
            and feature_error <= 1e-7
            and normalized_error <= 1e-6
            and score_error <= 1e-6
        )
        if not passed and first_failure is None:
            first_failure = {
                "step": step_index,
                "req_id": int(req.req_id),
                "legal_equal": legal_equal,
                "topk_equal": topk_equal,
                "action_equal": action_equal,
                "reference_action": reference.action,
                "optimized_action": optimized.action,
                "logit_error": logit_error,
                "feature_error": feature_error,
                "normalized_error": normalized_error,
                "score_error": score_error,
            }
            break

        if len(samples) < args.equivalence_sample_limit:
            samples.append({
                "step": step_index,
                "req_id": int(req.req_id),
                "candidate_count": int(reference.topk.candidate_actions.size),
                "selected_action": int(reference.action),
                "max_feature_error": feature_error,
                "max_score_error": score_error,
            })
        checked += 1
        _consume(env, req, args.fixed_split_id, server_id, optimized.action)
        if checked >= args.equivalence_states:
            break

    passed = (
        checked >= args.equivalence_states
        and action_mismatches == 0
        and topk_mismatches == 0
        and legal_mismatches == 0
        and request_mismatches == 0
        and max_logit_error <= 1e-6
        and max_feature_error <= 1e-7
        and max_normalized_error <= 1e-6
        and max_score_error <= 1e-6
    )
    return {
        "topology": topology,
        "seed": args.profile_seed,
        "target_states": args.equivalence_states,
        "checked_states": checked,
        "passed": passed,
        "action_mismatches": action_mismatches,
        "topk_mismatches": topk_mismatches,
        "legal_mismatches": legal_mismatches,
        "request_mismatches": request_mismatches,
        "max_legal_logit_error": max_logit_error,
        "max_raw_feature_error": max_feature_error,
        "max_normalized_feature_error": max_normalized_error,
        "max_ranker_score_error": max_score_error,
        "first_failure": first_failure,
        "samples": samples,
    }


def _append(sample_map: Dict[str, List[float]], values: Dict[str, float]) -> None:
    for key, value in values.items():
        sample_map.setdefault(key, []).append(float(value))


def _summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "median", "p90", "p95", "p99", "std")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "std": float(array.std()),
    }


def run_profile(args, topology: str, method: str, agent_r, ranker, repeat: int):
    count = args.warmup_requests + args.profile_eval_requests
    env, requests = _requests(args, topology, args.profile_seed, count)
    env.reset(requests)
    node_to_server = _server_map(env)
    samples: Dict[str, List[float]] = {}
    blocked = 0
    selected_actions: List[int | None] = []

    for step_index, req in enumerate(requests):
        end_to_end_start = time.perf_counter()

        start = time.perf_counter()
        env.advance_time(req.arrival_time)
        advance_ms = _ms(start)
        if not env.event_queue or int(env.event_queue[0][2].req_id) != int(req.req_id):
            raise RuntimeError(
                f"Queue mismatch at topology={topology} method={method} step={step_index}"
            )

        start = time.perf_counter()
        multitopology._spectrum_state(env)
        spectrum_ms = _ms(start)

        server_id = _server_id(node_to_server, req)
        start = time.perf_counter()
        obs_r = build_agent_r_observation(
            env, req, args.fixed_split_id, server_id
        )
        obs_ms = _ms(start)
        mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        selector_timings: Dict[str, float] = {}
        if not mask.any():
            action = None
            selector_ms = 0.0
        elif method.startswith("strict_"):
            optimized = method == "strict_optimized"
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
            selector_timings = selected.timings_ms
            selector_ms = float(selected.timings_ms["selector_total_ms"])
        else:
            selector = (
                ksp_ff_highest_mod_action if method == "ksp_ff"
                else ff_ksp_highest_mod_action
            )
            start = time.perf_counter()
            candidate = selector(obs_r)
            selector_ms = _ms(start)
            action = int(candidate) if candidate is not None else None

        start = time.perf_counter()
        info = _consume(env, req, args.fixed_split_id, server_id, action)
        env_step_ms = _ms(start)
        if not info.get("success", False):
            blocked += 1

        start = time.perf_counter()
        if action is not None:
            decoded = decode_agent_r_action(
                int(action), env.mod_reg.num_formats, env.max_blocks
            )
            multitopology._extract_r_action_meta(obs_r, decoded)
        metadata_ms = _ms(start)
        end_to_end_ms = _ms(end_to_end_start)

        if step_index >= args.warmup_requests:
            selected_actions.append(action)
            outer = {
                "advance_time_ms": advance_ms,
                "spectrum_summary_ms": spectrum_ms,
                "obs_r_build_ms": obs_ms,
                "selector_total_ms": selector_ms,
                "env_step_ms": env_step_ms,
                "metadata_ms": metadata_ms,
                "end_to_end_ms": end_to_end_ms,
            }
            _append(samples, outer)
            for key, value in selector_timings.items():
                if key != "selector_total_ms":
                    samples.setdefault(key, []).append(float(value))

    return {
        "topology": topology,
        "method": method,
        "repeat": repeat,
        "seed": args.profile_seed,
        "warmup_requests": args.warmup_requests,
        "evaluated_requests": args.profile_eval_requests,
        "blocked": blocked,
        "blocking_rate": blocked / max(len(requests), 1),
        "selected_action_hash": multitopology._sha256_text(
            json.dumps(selected_actions, separators=(",", ":"))
        )[:24],
        "timings_ms": {key: _summary(values) for key, values in samples.items()},
    }


def _mean(result: Dict[str, Any], key: str) -> float:
    return float(result["timings_ms"].get(key, {}).get("mean", 0.0))


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Strict v1.3 Exact Latency Optimization",
        "",
        "## Protocol",
        "",
        "- fixed-C / all-OD; no PPO-C or DF_C",
        "- PPO-R legal Top-30; K=50 hops; 25-d frozen Ranker",
        "- CPU, serial latency measurement",
        "- Diagnostic KSP-FF/FF-KSP coverage calls excluded from production selector timing",
        "",
        "## Equivalence",
        "",
        "| Topology | States | Passed | Top-K mismatch | Action mismatch | Feature max error | Score max error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["equivalence"]:
        lines.append(
            f"| {result['topology']} | {result['checked_states']} | {result['passed']} | "
            f"{result['topk_mismatches']} | {result['action_mismatches']} | "
            f"{result['max_raw_feature_error']:.3e} | {result['max_ranker_score_error']:.3e} |"
        )

    lines.extend([
        "",
        "## Latency",
        "",
        "| Topology | Method | Selector mean ms | Selector p95 ms | E2E mean ms | E2E p95 ms |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for result in payload["profiles"]:
        selector = result["timings_ms"]["selector_total_ms"]
        end_to_end = result["timings_ms"]["end_to_end_ms"]
        lines.append(
            f"| {result['topology']} | {result['method']} | {selector['mean']:.3f} | "
            f"{selector['p95']:.3f} | {end_to_end['mean']:.3f} | {end_to_end['p95']:.3f} |"
        )

    lines.extend(["", "## Speedup", ""])
    for topology in TOPOLOGIES:
        profiles = {
            item["method"]: item for item in payload["profiles"]
            if item["topology"] == topology and item["repeat"] == 0
        }
        if "strict_reference" not in profiles or "strict_optimized" not in profiles:
            continue
        reference = profiles["strict_reference"]
        optimized = profiles["strict_optimized"]
        ref_selector = _mean(reference, "selector_total_ms")
        opt_selector = _mean(optimized, "selector_total_ms")
        ref_e2e = _mean(reference, "end_to_end_ms")
        opt_e2e = _mean(optimized, "end_to_end_ms")
        lines.append(
            f"- `{topology}`: selector {ref_selector / max(opt_selector, 1e-12):.2f}x; "
            f"end-to-end {ref_e2e / max(opt_e2e, 1e-12):.2f}x."
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = multitopology.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir="sa_hmarl/experiments/strict_v13_latency_optimization",
        warmup_requests=500,
    )
    parser.add_argument("--profile_seed", type=int, default=6101)
    parser.add_argument("--profile_eval_requests", type=int, default=1000)
    parser.add_argument("--equivalence_states", type=int, default=500)
    parser.add_argument("--equivalence_sample_limit", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--topologies", default=",".join(TOPOLOGIES))
    parser.add_argument("--profile_methods", default=",".join(METHODS))
    parser.add_argument("--skip_equivalence", action="store_true")
    parser.add_argument("--skip_profile", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topologies = tuple(item.strip() for item in args.topologies.split(",") if item.strip())
    methods = tuple(item.strip() for item in args.profile_methods.split(",") if item.strip())
    invalid_methods = [method for method in methods if method not in METHODS]
    if invalid_methods:
        raise ValueError(f"Invalid profile methods: {invalid_methods}")

    modulation_registry = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, modulation_registry, args.device)
    ranker = multitopology._load_ranker_model(
        args.strict_ranker_checkpoint, args.device
    )

    equivalence = []
    if not args.skip_equivalence:
        for topology in topologies:
            print(f"[equivalence] {topology}", flush=True)
            result = run_equivalence(args, topology, agent_r, ranker)
            equivalence.append(result)
            print(
                f"  states={result['checked_states']} passed={result['passed']} "
                f"feature_err={result['max_raw_feature_error']:.3e}",
                flush=True,
            )
            if not result["passed"]:
                payload = {"config": vars(args), "equivalence": equivalence, "profiles": []}
                _atomic_json(output_dir / "REFERENCE_VS_OPTIMIZED_CORRECTNESS.json", payload)
                raise RuntimeError(f"Equivalence failed for {topology}: {result['first_failure']}")

    profiles = []
    if not args.skip_profile:
        for repeat in range(args.repeats):
            for topology in topologies:
                for method in methods:
                    print(f"[profile] repeat={repeat} {topology} {method}", flush=True)
                    result = run_profile(args, topology, method, agent_r, ranker, repeat)
                    profiles.append(result)
                    print(
                        f"  selector={_mean(result, 'selector_total_ms'):.3f} ms "
                        f"e2e={_mean(result, 'end_to_end_ms'):.3f} ms",
                        flush=True,
                    )

    payload = {
        "schema_version": 1,
        "config": vars(args),
        "checkpoint_hashes": {
            "ppo_r": multitopology._sha256(args.agent_r_checkpoint),
            "ranker": multitopology._sha256(args.strict_ranker_checkpoint),
        },
        "equivalence": equivalence,
        "profiles": profiles,
    }
    _atomic_json(output_dir / "LATENCY_RESULTS.json", payload)
    _atomic_json(
        output_dir / "REFERENCE_VS_OPTIMIZED_CORRECTNESS.json",
        {"config": vars(args), "equivalence": equivalence},
    )
    _write_markdown(output_dir / "LATENCY_COMPARISON.md", payload)
    print(f"[done] outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
