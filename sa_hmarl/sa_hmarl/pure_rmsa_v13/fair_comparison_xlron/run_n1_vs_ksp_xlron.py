#!/usr/bin/env python3
"""N1-100 (frozen, XLRON-trained) vs KSP-FF K=5/50 on the XLRON
environment: same seeds 69301-69310, load=250, 100 slots, rejection
truncation.  Reports SBP and selector latency."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from .env import RMSAEnv
from .ksp_ff import run_episode
from .n1_adapter import run_n1_episode
from .traffic import generate_trace

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TOPOLOGY = "xlron_nsfnet_deeprmsa"
NUM_SLOTS = 100
LOAD = 250.0
WARMUP = 3000
EVAL = 10000
SEEDS = tuple(range(69301, 69311))
N1_WEIGHTS = ("sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/"
              "n1_100_xlron/nsfnet/training/deployment_weights.npz")


def main() -> None:
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise SystemExit("native kernel must be unset")
    results = []
    for seed in SEEDS:
        for arm in ("n1", "ksp_ff_k5", "ksp_ff_k50"):
            t0 = time.perf_counter()
            trace = generate_trace(TOPOLOGY, 14, WARMUP + EVAL, LOAD, seed,
                                   num_slots=NUM_SLOTS)
            if arm == "n1":
                env = RMSAEnv(trace, k_paths=50, topology_name=TOPOLOGY)
                r = run_n1_episode(env, N1_WEIGHTS, warmup=WARMUP,
                                   eval_requests=EVAL)
            else:
                k = 5 if arm == "ksp_ff_k5" else 50
                env = RMSAEnv(trace, k_paths=k, topology_name=TOPOLOGY)
                r = run_episode(env, warmup=WARMUP, eval_requests=EVAL)
            r.update({"seed": int(seed), "arm": arm,
                      "elapsed_s": float(time.perf_counter() - t0)})
            results.append(r)
            print(r, flush=True)
    out = {"protocol_id": "pure_rmsa_n1_xlron_vs_ksp_v1",
           "date": "2026-08-05", "topology": TOPOLOGY,
           "num_slots": NUM_SLOTS, "load": LOAD, "warmup": WARMUP,
           "eval_requests": EVAL, "seeds": list(SEEDS),
           "n1_weights": N1_WEIGHTS, "results": results}
    (ARTIFACTS / "n1_vs_ksp").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "n1_vs_ksp" / "N1_VS_KSP_RESULTS.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(ARTIFACTS / "n1_vs_ksp" / "N1_VS_KSP_RESULTS.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "blocked", "served",
                                          "total", "blocked_rate",
                                          "elapsed_s"])
        w.writeheader()
        w.writerows(results)
    import numpy as np
    for arm in ("n1", "ksp_ff_k5", "ksp_ff_k50"):
        rs = [r["blocked_rate"] for r in results if r["arm"] == arm]
        print(f"{arm}: mean={sum(rs)/len(rs):.4f}% "
              f"std={np.std(rs, ddof=1):.4f}")
    n1 = [r["blocked_rate"] for r in results if r["arm"] == "n1"]
    k5 = [r["blocked_rate"] for r in results if r["arm"] == "ksp_ff_k5"]
    k50 = [r["blocked_rate"] for r in results if r["arm"] == "ksp_ff_k50"]
    print(f"dK (K5-K50): {sum(k5)/10 - sum(k50)/10:.4f} pp")
    print(f"N1 vs KSP-FF K50: {sum(k50)/10 - sum(n1)/10:.4f} pp")
    print(f"N1 vs KSP-FF K5: {sum(k5)/10 - sum(n1)/10:.4f} pp")
    print("n1 vs ksp done")


if __name__ == "__main__":
    main()
