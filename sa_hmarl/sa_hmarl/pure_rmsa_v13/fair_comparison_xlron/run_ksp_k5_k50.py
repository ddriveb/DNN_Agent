#!/usr/bin/env python3
"""XLRON-compatible KSP-FF comparison: K=5 vs K=50, 10 seeds.

Setting (Doherty 2025 / XLRON): NSFNET directed (14 nodes, 44 links),
100 FSU, load=250 Erlang, warmup 3000, eval 10000, hops-first KSP,
25-100 Gbps uniform (1 Gbps step), rejection-sampled holding-time
truncation (2x mean), BPSK/QPSK/8QAM/16QAM, first-fit, SBP.
"""
from __future__ import annotations

import csv
import json
import os
import time
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
SEEDS = tuple(range(69301, 69311))
K_VALUES = (5, 50)


def run_one(seed: int, k: int) -> dict:
    t0 = time.perf_counter()
    trace = generate_trace(TOPOLOGY, 14, WARMUP + EVAL, LOAD, seed,
                           num_slots=NUM_SLOTS)
    env = RMSAEnv(trace, k_paths=k, topology_name=TOPOLOGY)
    res = run_episode(env, warmup=WARMUP, eval_requests=EVAL)
    res["seed"] = int(seed)
    res["k"] = int(k)
    res["elapsed_s"] = float(time.perf_counter() - t0)
    return res


def main() -> None:
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise SystemExit("native kernel must be unset")
    results = []
    for seed in SEEDS:
        for k in K_VALUES:
            r = run_one(seed, k)
            results.append(r)
            print(r, flush=True)
    out = {
        "protocol_id": "pure_rmsa_n1_xlron_ksp_k5_k50_v1",
        "date": "2026-08-05",
        "topology": TOPOLOGY, "num_slots": NUM_SLOTS, "load": LOAD,
        "warmup": WARMUP, "eval_requests": EVAL,
        "truncation": "rejection_sampling_2x_mean",
        "seeds": list(SEEDS), "k_values": list(K_VALUES),
        "results": results,
    }
    (ARTIFACTS / "ksp_k5_k50").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "ksp_k5_k50" / "KSP_K5_K50_RESULTS.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(ARTIFACTS / "ksp_k5_k50" / "KSP_K5_K50_RESULTS.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "k", "blocked", "served",
                                          "total", "blocked_rate",
                                          "elapsed_s"])
        w.writeheader()
        w.writerows(results)
    # summary
    for k in K_VALUES:
        rs = [r["blocked_rate"] for r in results if r["k"] == k]
        print(f"K={k}: mean={sum(rs)/len(rs):.4f}% "
              f"std={__import__('numpy').std(rs, ddof=1):.4f} "
              f"per-seed={[round(v, 3) for v in rs]}")
    k5 = [r["blocked_rate"] for r in results if r["k"] == 5]
    k50 = [r["blocked_rate"] for r in results if r["k"] == 50]
    dk = sum(k5) / len(k5) - sum(k50) / len(k50)
    print(f"dK (K5-K50) = {dk:.4f} pp")
    print("ksp k5 k50 done")


if __name__ == "__main__":
    main()
