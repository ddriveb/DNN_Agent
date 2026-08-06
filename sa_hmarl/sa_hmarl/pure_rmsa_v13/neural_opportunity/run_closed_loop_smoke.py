"""Single-seed closed-loop diagnostic after offline training."""
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
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.selectors_phaseA import (
    OpportunityOnlyDirectSelector,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import (
    PythonKSPFFSelector,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.feasible_window_sketch import (
    ExactOptimizedDirectCompressedSketchSelector,
)
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env

from .model import TinyOpportunityWeights
from .protocol import K_PATHS, NUM_SLOTS, PROTOCOL_ID, SMOKE_SEED, TOPOLOGY_CONFIG
from .selector import NeuralOpportunitySelector


def _run(topology, config, arm, weights_path, warmup, eval_requests, seed=SMOKE_SEED):
    env = make_env(
        seed,
        warmup + eval_requests,
        topology=topology,
        num_slots=NUM_SLOTS,
        load_erlang=config["load_erlang"],
        k_paths=K_PATHS,
    )
    _prewarm_routes(env)
    if arm == "ksp_ff_k50":
        selector = PythonKSPFFSelector(env)
    elif arm == "opportunity_teacher":
        selector = OpportunityOnlyDirectSelector(
            env,
            budget=config["budget"],
            opportunity_weight=config["opportunity_weight"],
        )
    elif arm == "full_direct":
        selector = ExactOptimizedDirectCompressedSketchSelector(
            env,
            budget=config["budget"],
            opportunity_weight=config["opportunity_weight"],
        )
    elif arm == "neural_opportunity":
        selector = NeuralOpportunitySelector(
            env, TinyOpportunityWeights.load(weights_path)
        )
    else:
        raise ValueError(arm)
    blocked = 0
    latencies = []
    for index, request in enumerate(env.trace.requests):
        env.advance_external(request)
        started = time.perf_counter_ns()
        action = selector.select(env, request)
        elapsed = time.perf_counter_ns() - started
        result = execute(env, request, action)
        if index >= warmup:
            latencies.append(elapsed)
            blocked += int(not result["success"])
    latency_ms = np.asarray(latencies, dtype=np.float64) / 1e6
    return {
        "arm": arm,
        "blocked": blocked,
        "blocking": blocked / eval_requests,
        "mean_ms": float(np.mean(latency_ms)),
        "p95_ms": float(np.percentile(latency_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-requests", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise RuntimeError("native Direct Sketch kernel must be unset")
    name_by_topology = {
        "xlron_nsfnet_deeprmsa": "nsfnet",
        "xlron_usnet_gcnrmsa": "usnet",
        "xlron_jpn48": "jpn48",
    }
    rows = []
    for topology, config in TOPOLOGY_CONFIG.items():
        weights = (
            args.weights_root
            / name_by_topology[topology]
            / "deployment_weights.npz"
        )
        arms = [
            _run(
                topology,
                config,
                arm,
                weights,
                args.warmup,
                args.eval_requests,
            )
            for arm in (
                "ksp_ff_k50",
                "opportunity_teacher",
                "neural_opportunity",
            )
        ]
        by_arm = {row["arm"]: row for row in arms}
        rows.append(
            {
                "topology": topology,
                "arms": by_arm,
                "neural_vs_teacher_pp": 100.0
                * (
                    by_arm["neural_opportunity"]["blocking"]
                    - by_arm["opportunity_teacher"]["blocking"]
                ),
                "neural_vs_ksp_gain_pp": 100.0
                * (
                    by_arm["ksp_ff_k50"]["blocking"]
                    - by_arm["neural_opportunity"]["blocking"]
                ),
            }
        )
    result = {
        "protocol_id": PROTOCOL_ID,
        "diagnostic_only": True,
        "seed": SMOKE_SEED,
        "warmup": args.warmup,
        "eval_requests": args.eval_requests,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CLOSED_LOOP_SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
