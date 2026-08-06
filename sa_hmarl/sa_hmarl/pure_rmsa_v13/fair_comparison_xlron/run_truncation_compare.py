#!/usr/bin/env python3
"""Truncation-mode comparison on the SAME XLRON-compatible env:
- no_trunc : untruncated exponential holding
- hard     : resample until <= 2*mean (project convention)
- rejection: XLRON 5-candidate rejection sampling
Same load (250), same seeds, K in {5, 50}; isolates the truncation
mechanism's effect on the K=5 vs K=50 gap."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .env import RMSAEnv
from .ksp_ff import run_episode
from .traffic import generate_trace

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TOPOLOGY = "xlron_nsfnet_deeprmsa"
NUM_SLOTS = 100
LOAD = 250.0
WARMUP = 3000
EVAL = 10000
SEEDS = (69301, 69302, 69303)
K_VALUES = (5, 50)
MODES = ("no_trunc", "hard", "rejection")


def main() -> None:
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise SystemExit("native kernel must be unset")
    results = []
    for mode in MODES:
        trunc = mode != "no_trunc"
        tmode = mode if mode != "no_trunc" else "rejection"
        for seed in SEEDS:
            for k in K_VALUES:
                trace = generate_trace(TOPOLOGY, 14, WARMUP + EVAL, LOAD,
                                       seed, num_slots=NUM_SLOTS,
                                       truncate_holding_time=trunc,
                                       truncation_mode=tmode)
                env = RMSAEnv(trace, k_paths=k, topology_name=TOPOLOGY)
                r = run_episode(env, warmup=WARMUP, eval_requests=EVAL)
                r.update({"seed": int(seed), "k": int(k), "mode": mode})
                results.append(r)
                print(r, flush=True)
    out = {"protocol_id": "pure_rmsa_n1_xlron_truncation_compare_v1",
           "date": "2026-08-05", "load": LOAD, "num_slots": NUM_SLOTS,
           "warmup": WARMUP, "eval_requests": EVAL, "results": results}
    (ARTIFACTS / "truncation_compare").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "truncation_compare"
              / "TRUNCATION_COMPARE_RESULTS.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(ARTIFACTS / "truncation_compare"
              / "TRUNCATION_COMPARE_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "k", "mode", "blocked",
                                          "served", "total",
                                          "blocked_rate"])
        w.writeheader()
        w.writerows(results)
    print("\n== summary (mean over seeds) ==")
    for mode in MODES:
        row = {}
        for k in K_VALUES:
            rs = [r["blocked_rate"] for r in results
                  if r["mode"] == mode and r["k"] == k]
            row[k] = sum(rs) / len(rs)
        print(f"{mode}: K5={row[5]:.3f}% K50={row[50]:.3f}% "
              f"dK={row[5] - row[50]:.3f}pp")
    print("truncation compare done")


if __name__ == "__main__":
    main()
