#!/usr/bin/env python3
"""N1-FUSE0: three-topology action parity + fair latency benchmark.

Parity: per-request action signature of the fused selector must equal the
original N1 selector exactly (0 mismatches required).  Blocking outcomes
then coincide by construction.

Latency: single worker, CPU, pure Python/NumPy, native kernel unset, same
route-cache warmup, same traces for all arms, including the fair
cached bit-parallel early-exit KSP-FF K=50 reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from sa_hmarl.pure_rmsa_v13.core import execute
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.run_phaseA import (
    _prewarm_routes,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import (
    PythonKSPFFSelector,
)
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env

from ..model import TinyOpportunityWeights
from ..protocol import K_PATHS, NUM_SLOTS, TOPOLOGY_CONFIG
from ..selector import NeuralOpportunitySelector
from .selector import FusedNeuralOpportunitySelector

_N1_ROOT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "training_conflict_v1"
)
_NAME_BY_TOPOLOGY = {
    "xlron_nsfnet_deeprmsa": "nsfnet",
    "xlron_usnet_gcnrmsa": "usnet",
    "xlron_jpn48": "jpn48",
}
PARITY_SEEDS = (61201, 61202, 61203)


def _signature(action):
    if hasattr(action, "path"):
        return (action.path_rank, action.start_slot, action.required_fs)
    return ("blocked", getattr(action, "reason", None))


def run_parity(topology: str, seed: int, warmup: int, eval_requests: int) -> dict:
    """Lockstep both selectors on one env; count action mismatches."""
    config = TOPOLOGY_CONFIG[topology]
    env = make_env(
        seed, warmup + eval_requests, topology=topology,
        num_slots=NUM_SLOTS, load_erlang=config["load_erlang"], k_paths=K_PATHS,
    )
    _prewarm_routes(env)
    weights = TinyOpportunityWeights.load(
        _N1_ROOT / _NAME_BY_TOPOLOGY[topology] / "deployment_weights.npz"
    )
    original = NeuralOpportunitySelector(env, weights)
    fused = FusedNeuralOpportunitySelector(env, weights)
    mismatches = 0
    first_mismatch = None
    blocked_orig = 0
    for index, request in enumerate(env.trace.requests):
        env.advance_external(request)
        a_orig = original.select(env, request)
        a_fused = fused.select(env, request)
        if index >= warmup:
            if _signature(a_orig) != _signature(a_fused):
                mismatches += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "request_index": index,
                        "original": _signature(a_orig),
                        "fused": _signature(a_fused),
                    }
            blocked_orig += int(not execute(env, request, a_orig)["success"])
        else:
            execute(env, request, a_orig)
    return {
        "topology": topology,
        "seed": seed,
        "warmup": warmup,
        "eval_requests": eval_requests,
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "blocked_original": blocked_orig,
        "blocking": blocked_orig / eval_requests,
    }


def run_latency(topology: str, seed: int, warmup: int, eval_requests: int) -> dict:
    """Fair latency for fused / original / KSP-FF K=50 on identical traces."""
    config = TOPOLOGY_CONFIG[topology]
    rows = {}
    for arm in ("fused", "n1_original", "ksp_ff_k50"):
        env = make_env(
            seed, warmup + eval_requests, topology=topology,
            num_slots=NUM_SLOTS, load_erlang=config["load_erlang"], k_paths=K_PATHS,
        )
        _prewarm_routes(env)
        if arm == "ksp_ff_k50":
            selector = PythonKSPFFSelector(env)
        else:
            weights = TinyOpportunityWeights.load(
                _N1_ROOT / _NAME_BY_TOPOLOGY[topology] / "deployment_weights.npz"
            )
            selector = (
                FusedNeuralOpportunitySelector(env, weights)
                if arm == "fused"
                else NeuralOpportunitySelector(env, weights)
            )
        blocked = 0
        lat = []
        for index, request in enumerate(env.trace.requests):
            env.advance_external(request)
            t0 = time.perf_counter_ns()
            action = selector.select(env, request)
            dt = time.perf_counter_ns() - t0
            result = execute(env, request, action)
            if index >= warmup:
                lat.append(dt)
                blocked += int(not result["success"])
        lat_ms = np.asarray(lat) / 1e6
        rows[arm] = {
            "mean_ms": float(lat_ms.mean()),
            "p50_ms": float(np.percentile(lat_ms, 50)),
            "p95_ms": float(np.percentile(lat_ms, 95)),
            "blocked": blocked,
            "blocking": blocked / eval_requests,
        }
    return {"topology": topology, "seed": seed, "arms": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="n1_fuse0_bench")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-requests", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts")
    args = parser.parse_args(argv)
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        print("ERROR: native kernel must be unset", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parity_rows = []
    for topology in TOPOLOGY_CONFIG:
        for seed in PARITY_SEEDS:
            row = run_parity(topology, seed, args.warmup, args.eval_requests)
            parity_rows.append(row)
            print(f"parity {topology} seed {seed}: mismatches={row['mismatches']}",
                  flush=True)
    total_mismatches = sum(r["mismatches"] for r in parity_rows)

    latency_rows = []
    for topology in TOPOLOGY_CONFIG:
        row = run_latency(topology, PARITY_SEEDS[0], args.warmup, args.eval_requests)
        latency_rows.append(row)
        arms = row["arms"]
        print(
            f"latency {topology}: fused {arms['fused']['mean_ms']:.4f}ms "
            f"orig {arms['n1_original']['mean_ms']:.4f}ms "
            f"ksp {arms['ksp_ff_k50']['mean_ms']:.4f}ms",
            flush=True,
        )

    result = {
        "experiment": "n1_fuse0",
        "parity_seeds": list(PARITY_SEEDS),
        "warmup": args.warmup,
        "eval_requests": args.eval_requests,
        "parity": parity_rows,
        "total_mismatches": total_mismatches,
        "parity_passed": total_mismatches == 0,
        "latency": latency_rows,
    }
    (args.output_dir / "FUSE0_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "total_mismatches": total_mismatches,
        "parity_passed": total_mismatches == 0,
    }, indent=2))
    return 0 if total_mismatches == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
