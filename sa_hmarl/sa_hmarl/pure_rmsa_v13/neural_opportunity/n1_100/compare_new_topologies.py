#!/usr/bin/env python3
"""N1-100 vs KSP-FF K=5/50 comparison for the new topologies.

Uses the project environment (``make_env``), same trace per arm, and the
XLRON-consistent settings: 100 slots, holding_truncation=2.0, warmup=3000,
eval=10000, seeds 69511-69520.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100 import protocol  # noqa: F401, registers abilene
from sa_hmarl.pure_rmsa_v13.core import execute
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.run_phaseA import (
    _prewarm_routes,
)
from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.model import (
    TinyOpportunityWeights,
)
from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.selector import (
    NeuralOpportunitySelector,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import PythonKSPFFSelector
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env

# Fair-format rerun: only the three topologies whose results were
# produced on the legacy pool.  cost239/abilene COMPARE_RESULTS are
# already final (seeds 61821-61830) and must NOT be overwritten.
TOPOLOGIES = ("xlron_nsfnet_deeprmsa", "xlron_jpn48", "xlron_usnet_gcnrmsa")
SEEDS = tuple(range(62021, 62031))
WARMUP = 3000
EVAL = 10000
NUM_SLOTS = 100
K_PATHS = 50
HOLDING_TRUNCATION = 2.0
PATH_SORT_STRATEGY = "xlron"
BASE = Path("sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron")

SHORT_NAMES = {
    "cost239_deeprmsa": "cost239",
    "abilene": "abilene",
    "xlron_nsfnet_deeprmsa": "nsfnet",
    "xlron_jpn48": "jpn48",
    "xlron_usnet_gcnrmsa": "usnet",
}


def run_n1(seed: int, topology: str, load_erlang: float, weights_path: Path) -> dict:
    env = make_env(
        seed,
        WARMUP + EVAL,
        topology=topology,
        num_slots=NUM_SLOTS,
        load_erlang=load_erlang,
        k_paths=K_PATHS,
        holding_truncation=HOLDING_TRUNCATION,
        path_sort_strategy=PATH_SORT_STRATEGY,
    )
    _prewarm_routes(env)
    selector = NeuralOpportunitySelector(env, TinyOpportunityWeights.load(weights_path))
    blocked = served = 0
    t0 = time.perf_counter()
    for request_index, request in enumerate(env.trace.requests):
        env.advance_external(request)
        result = execute(env, request, selector.select(env, request))
        if request_index >= WARMUP:
            if result["success"]:
                served += 1
            else:
                blocked += 1
    elapsed = time.perf_counter() - t0
    total = served + blocked
    return {
        "seed": seed,
        "topology": topology,
        "arm": "n1",
        "load": load_erlang,
        "blocked": blocked,
        "served": served,
        "total": total,
        "blocked_rate": blocked / total * 100.0 if total else 0.0,
        "elapsed_s": elapsed,
    }


def run_ksp(seed: int, topology: str, load_erlang: float, k: int) -> dict:
    env = make_env(
        seed,
        WARMUP + EVAL,
        topology=topology,
        num_slots=NUM_SLOTS,
        load_erlang=load_erlang,
        k_paths=k,
        holding_truncation=HOLDING_TRUNCATION,
        path_sort_strategy=PATH_SORT_STRATEGY,
    )
    _prewarm_routes(env)
    selector = PythonKSPFFSelector(env)
    blocked = served = 0
    t0 = time.perf_counter()
    for request_index, request in enumerate(env.trace.requests):
        env.advance_external(request)
        result = execute(env, request, selector.select(env, request))
        if request_index >= WARMUP:
            if result["success"]:
                served += 1
            else:
                blocked += 1
    elapsed = time.perf_counter() - t0
    total = served + blocked
    return {
        "seed": seed,
        "topology": topology,
        "arm": f"ksp_ff_k{k}",
        "load": load_erlang,
        "blocked": blocked,
        "served": served,
        "total": total,
        "blocked_rate": blocked / total * 100.0 if total else 0.0,
        "elapsed_s": elapsed,
    }


def main() -> None:
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise SystemExit("native kernel must be unset")

    for topology in TOPOLOGIES:
        config = protocol.TOPOLOGY_CONFIG[topology]
        load_erlang = config["load_erlang"]
        short = SHORT_NAMES[topology]
        weights_path = BASE / short / "training" / "deployment_weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {weights_path}")

        rows = []
        for seed in SEEDS:
            for arm, fn in (
                ("n1", lambda: run_n1(seed, topology, load_erlang, weights_path)),
                ("ksp_ff_k5", lambda: run_ksp(seed, topology, load_erlang, 5)),
                ("ksp_ff_k50", lambda: run_ksp(seed, topology, load_erlang, 50)),
            ):
                result = fn()
                rows.append(result)
                print(result, flush=True)

        out_dir = BASE / short
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "COMPARE_RESULTS.json"
        csv_path = out_dir / "per_seed.csv"

        json_path.write_text(
            json.dumps(
                {
                    "protocol_id": protocol.PROTOCOL_ID,
                    "topology": topology,
                    "load": load_erlang,
                    "num_slots": NUM_SLOTS,
                    "warmup": WARMUP,
                    "eval_requests": EVAL,
                    "holding_truncation": HOLDING_TRUNCATION,
                    "seeds": list(SEEDS),
                    "results": rows,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "seed",
                    "topology",
                    "arm",
                    "load",
                    "blocked",
                    "served",
                    "total",
                    "blocked_rate",
                    "elapsed_s",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"[compare] {topology} results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
