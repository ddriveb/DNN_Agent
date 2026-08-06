"""Pre-training latency ceiling for the complete Neural Opportunity path."""
from __future__ import annotations

import argparse
import json
import os
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

from .model import TinyOpportunityWeights
from .protocol import (
    K_PATHS,
    LATENCY_MEAN_MS_GATE,
    NUM_SLOTS,
    PROTOCOL_ID,
    SMOKE_SEED,
    TOPOLOGY_CONFIG,
)
from .selector import NeuralOpportunitySelector

NATIVE_ENV_VAR = "SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"


def _run_arm(topology, config, arm, warmup, eval_requests):
    env = make_env(
        SMOKE_SEED,
        warmup + eval_requests,
        topology=topology,
        num_slots=NUM_SLOTS,
        load_erlang=config["load_erlang"],
        k_paths=K_PATHS,
    )
    _prewarm_routes(env)
    if arm == "ksp_ff_k50":
        selector = PythonKSPFFSelector(env)
    elif arm == "neural_opportunity_zero":
        selector = NeuralOpportunitySelector(
            env, TinyOpportunityWeights.zeros()
        )
    else:
        raise ValueError(arm)
    latencies = []
    blocked = 0
    for index, request in enumerate(env.trace.requests):
        env.advance_external(request)
        started = time.perf_counter_ns()
        action = selector.select(env, request)
        latencies.append(time.perf_counter_ns() - started)
        result = execute(env, request, action)
        if index >= warmup:
            blocked += int(not result["success"])
    measured = np.asarray(latencies[warmup:], dtype=np.float64) / 1e6
    return {
        "arm": arm,
        "blocked": blocked,
        "blocking": blocked / eval_requests,
        "mean_ms": float(np.mean(measured)),
        "p50_ms": float(np.percentile(measured, 50)),
        "p95_ms": float(np.percentile(measured, 95)),
        "p99_ms": float(np.percentile(measured, 99)),
    }


def run(output_dir: Path, warmup: int, eval_requests: int):
    if os.environ.get(NATIVE_ENV_VAR):
        raise RuntimeError(f"{NATIVE_ENV_VAR} must be unset")
    rows = []
    for topology, config in TOPOLOGY_CONFIG.items():
        topology_rows = [
            _run_arm(topology, config, arm, warmup, eval_requests)
            for arm in ("ksp_ff_k50", "neural_opportunity_zero")
        ]
        by_arm = {row["arm"]: row for row in topology_rows}
        neural = by_arm["neural_opportunity_zero"]
        ksp = by_arm["ksp_ff_k50"]
        rows.append(
            {
                "topology": topology,
                "arms": by_arm,
                "latency_ratio": neural["mean_ms"] / ksp["mean_ms"],
                "passes_absolute_gate": neural["mean_ms"]
                <= LATENCY_MEAN_MS_GATE,
            }
        )
    verdict = (
        "GO_TO_TEACHER_COLLECTION"
        if all(row["passes_absolute_gate"] for row in rows)
        else "HOLD_OPTIMIZE_FEATURE_PATH"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "seed": SMOKE_SEED,
        "warmup": warmup,
        "eval_requests": eval_requests,
        "native_kernel": False,
        "gate_mean_ms": LATENCY_MEAN_MS_GATE,
        "rows": rows,
        "verdict": verdict,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "LATENCY_CEILING.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--eval-requests", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/latency_ceiling"
        ),
    )
    args = parser.parse_args()
    result = run(args.output_dir, args.warmup, args.eval_requests)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
