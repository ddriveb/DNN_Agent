#!/usr/bin/env python3
"""Independent recomputation of N1-vs-KSP blocking statistics.

Sources (raw rows only, no Markdown):
- sa_hmarl/pure_rmsa_v13/artifacts/direct_sketch_exact_optimization/
  phaseA_confirmatory_10seed/per_seed.csv   (KSP k5/k50, bf_k50, full_direct, opp_only)
- sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/
  confirmatory_10seed/neural_per_seed.csv   (neural_opportunity)

Formulas (locked, explicit):
- delta_pp = (N1_blocking - KSP_blocking) * 100   -> NEGATIVE means N1 better
- paired t 95% CI over per-seed deltas (ddof=1)
- wins/ties/losses from sign of per-seed delta

Written for the independent audit; reads only, writes to the audit dir.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # pure_rmsa_v13/
FACT = ROOT / "artifacts" / "direct_sketch_exact_optimization" / "phaseA_confirmatory_10seed" / "per_seed.csv"
NEURAL = ROOT / "neural_opportunity" / "artifacts" / "confirmatory_10seed" / "neural_per_seed.csv"
OUT = Path(__file__).resolve().parent


def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def recompute() -> dict:
    fact = load(FACT)
    neural = load(NEURAL)
    rows = fact + neural

    by_topo: dict[str, dict[int, dict[str, dict]]] = defaultdict(
        lambda: defaultdict(dict))
    for r in rows:
        topo = r["topology"]
        seed = int(r["seed"])
        arm = r["arm"]
        by_topo[topo][seed][arm] = {
            "blocked": int(r["blocked"]),
            "blocking": float(r["blocking"]),
            "selector_mean_ms": float(r.get("selector_mean_ms") or r.get("mean_ms") or 0.0),
            "selector_p95_ms": float(r.get("selector_p95_ms") or r.get("p95_ms") or 0.0),
            "ksp_backend": r.get("ksp_ff_backend", ""),
        }

    result: dict = {"formula": "delta_pp = (N1 - KSP) * 100; negative = N1 better",
                    "per_topology": {}}
    for topo in sorted(by_topo):
        seeds = sorted(by_topo[topo])
        seed_arms = {a for s in seeds for a in by_topo[topo][s]}
        entry = {"seeds": seeds, "n_seeds": len(seeds), "arms_present": sorted(seed_arms)}
        deltas = {}
        for ksp_arm in ("ksp_ff_k50", "ksp_ff_k5"):
            if ksp_arm not in seed_arms:
                continue
            d = []
            for s in seeds:
                n1 = by_topo[topo][s]["neural_opportunity"]["blocking"]
                ksp = by_topo[topo][s][ksp_arm]["blocking"]
                d.append((n1 - ksp) * 100.0)
            arr = np.asarray(d, dtype=np.float64)
            n = len(arr)
            mean = float(arr.mean())
            half = float(1.0) if n < 2 else float(
                __import__("scipy").stats.t.ppf(0.975, n - 1)
                * arr.std(ddof=1) / np.sqrt(n))
            wins = int((arr < 0).sum())      # N1 better
            ties = int((arr == 0).sum())
            losses = int((arr > 0).sum())
            deltas[ksp_arm] = {
                "mean_delta_pp": mean,
                "ci95_halfwidth_pp": half,
                "ci95_low_pp": mean - half,
                "ci95_high_pp": mean + half,
                "n1_wins": wins, "ties": ties, "n1_losses": losses,
                "per_seed_delta_pp": {str(s): float(x) for s, x in zip(seeds, arr)},
                "all_10_seeds_present": n == 10,
                "ci_entirely_negative": mean + half < 0.0,
            }
        entry["vs_ksp_ff_k50"] = deltas.get("ksp_ff_k50")
        entry["vs_ksp_ff_k5"] = deltas.get("ksp_ff_k5")
        # mean blocking per arm for provenance cross-check
        entry["mean_blocking"] = {
            a: float(np.mean([by_topo[topo][s][a]["blocking"] for s in seeds]))
            for a in sorted(seed_arms)
        }
        result["per_topology"][topo] = entry

    out = OUT / "PAIRWISE_BLOCKING_RECOMPUTE.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "seed", "arm", "blocking", "blocked",
                    "selector_mean_ms", "ksp_backend"])
        for r in rows:
            w.writerow([r["topology"], r["seed"], r["arm"], r["blocking"],
                        r["blocked"], r.get("selector_mean_ms") or r.get("mean_ms", ""),
                        r.get("ksp_ff_backend", "")])
    return result


def main() -> int:
    result = recompute()
    (OUT / "AUDIT_BLOCKING_RECOMPUTE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    for topo, e in result["per_topology"].items():
        for arm, d in (("ksp_ff_k50", e["vs_ksp_ff_k50"]),
                       ("ksp_ff_k5", e["vs_ksp_ff_k5"])):
            if d is None:
                continue
            print(f"{topo} N1-vs-{arm}: mean={d['mean_delta_pp']:+.4f} pp "
                  f"CI=[{d['ci95_low_pp']:+.4f},{d['ci95_high_pp']:+.4f}] "
                  f"W/T/L={d['n1_wins']}/{d['ties']}/{d['n1_losses']} "
                  f"seeds={d['n1_wins']+d['ties']+d['n1_losses']}/10 "
                  f"CI<0={d['ci_entirely_negative']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
