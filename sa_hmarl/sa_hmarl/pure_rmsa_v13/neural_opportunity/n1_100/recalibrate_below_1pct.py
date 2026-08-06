#!/usr/bin/env python3
"""Recalibrate loads so that KSP-FF K=50 mean blocking is below 1%."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100 import protocol  # noqa: F401, registers abilene
from sa_hmarl.pure_rmsa_v13.core import execute
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.run_phaseA import (
    _prewarm_routes,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import PythonKSPFFSelector
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env

TOPOLOGY_LOADS = {
    "cost239_deeprmsa": [560, 580, 600, 620],
    "abilene": [80, 85, 90, 95],
    # 2026-08-06 probes (xlron pool, seed 62001): 1% crossing near
    # nsfnet ~235E / jpn48 ~320E / usnet ~510E; sweeps bracket it.
    "xlron_nsfnet_deeprmsa": [210, 225, 240, 255, 270],
    "xlron_jpn48": [280, 300, 320, 340, 360],
    "xlron_usnet_gcnrmsa": [480, 500, 520, 540, 560],
}
SEEDS = (62001, 62002, 62003, 62004, 62005)
SHORT_NAMES = {
    "cost239_deeprmsa": "cost239",
    "abilene": "abilene",
    "xlron_nsfnet_deeprmsa": "nsfnet",
    "xlron_jpn48": "jpn48",
    "xlron_usnet_gcnrmsa": "usnet",
}
PATH_SORT_STRATEGY = "xlron"
WARMUP = 3000
EVAL = 10000
K_PATHS = 50
HOLDING_TRUNCATION = 2.0
NUM_SLOTS = 100
BASE = Path("sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron")
TARGET_MAX = 1.0  # percent


def run_ksp50(topology: str, load_erlang: float, seed: int) -> float:
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
    selector = PythonKSPFFSelector(env)
    blocked = 0
    for request_index, request in enumerate(env.trace.requests):
        env.advance_external(request)
        result = execute(env, request, selector.select(env, request))
        if request_index >= WARMUP:
            blocked += int(not result["success"])
    return blocked / EVAL * 100.0


def choose_load(rows: list[dict]) -> tuple[int, float]:
    """Choose the highest load whose mean blocking is still below TARGET_MAX.

    This pushes the operating point as close as possible to the 1% ceiling while
    staying strictly below it, which reduces floor-effect noise compared with
    selecting the first (lowest) qualifying load.
    """
    loads = sorted({r["load"] for r in rows})
    by_load = {
        load: [r["blocked_rate"] for r in rows if r["load"] == load]
        for load in loads
    }
    means = {load: sum(v) / len(v) for load, v in by_load.items()}
    candidates = [load for load in loads if means[load] < TARGET_MAX]
    if candidates:
        chosen = max(candidates)
        return chosen, means[chosen]
    # Fallback to the lowest mean if none below 1%.
    chosen = min(loads, key=lambda load: means[load])
    return chosen, means[chosen]


def main() -> None:
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise SystemExit("native kernel must be unset")

    for topology, loads in TOPOLOGY_LOADS.items():
        short = SHORT_NAMES[topology]
        out_dir = BASE / short
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "CALIBRATION_RESULTS.json"
        backup = out_dir / "CALIBRATION_RESULTS_5_8pct.json"
        if out_path.exists() and not backup.exists():
            shutil.copy(out_path, backup)

        rows = []
        for load_erlang in loads:
            for seed in SEEDS:
                blocked_rate = run_ksp50(topology, load_erlang, seed)
                rows.append(
                    {
                        "topology": topology,
                        "load": load_erlang,
                        "seed": seed,
                        "blocked_rate": float(blocked_rate),
                    }
                )
                print(
                    f"[recalibrate] {topology} load={load_erlang} seed={seed} "
                    f"KSP-FF K=50 blocked={blocked_rate:.4f}%",
                    flush=True,
                )

        chosen_load, mean = choose_load(rows)
        output = {
            "topology": topology,
            "loads": loads,
            "seeds": list(SEEDS),
            "warmup": WARMUP,
            "eval_requests": EVAL,
            "holding_truncation": HOLDING_TRUNCATION,
            "k_paths": K_PATHS,
            "num_slots": NUM_SLOTS,
            "target_max_blocking_pct": TARGET_MAX,
            "results": rows,
            "chosen_load": chosen_load,
            "chosen_mean_blocking": float(mean),
            "selection_rule": f"highest mean < {TARGET_MAX}%",
        }
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"[recalibrate] {topology} -> load={chosen_load} "
            f"mean_blocking={mean:.4f}% (saved {out_path})",
            flush=True,
        )


if __name__ == "__main__":
    main()
