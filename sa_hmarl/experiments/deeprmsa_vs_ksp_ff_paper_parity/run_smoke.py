#!/usr/bin/env python3
"""Stage B — 1-seed smoke test per topology (warmup=300, evaluated=1000).

Verifies program wiring, traffic, and metrics end-to-end for all three KSP-FF
variants plus (optionally) a random-untrained DeepRMSA policy sanity run.
Writes results/smoke.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parents[1]))

from paper_rmsa_core import (
    generate_paper_requests,
    ksp_ff_decide,
    load_run_config,
    run_simulation,
)

CFG = load_run_config()
SMOKE = CFG["measurement"]["smoke"]


def _ksp_closure(view, env, req):
    return ksp_ff_decide(view)


def run_topology_smoke(topo_key: str, topo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    arrival_interval = CFG["traffic"]["mean_holding_time"] / topo_cfg["load_erlang"]
    requests = generate_paper_requests(
        topo_cfg["num_nodes"], SMOKE["seed"],
        SMOKE["warmup_requests"] + SMOKE["evaluated_requests"],
        arrival_interval,
        mean_holding_time=CFG["traffic"]["mean_holding_time"],
        bitrate_min_gbps=CFG["traffic"]["bitrate_min_gbps"],
        bitrate_max_gbps=CFG["traffic"]["bitrate_max_gbps"],
    )
    out: Dict[str, Any] = {}
    for method_name, method in CFG["methods"].items():
        if method["type"] != "ksp_ff":
            continue
        res = run_simulation(
            topo_key, requests, _ksp_closure,
            k_paths=method["k_paths"], path_sort=method["path_sort"],
            warmup=SMOKE["warmup_requests"],
            num_slots=CFG["spectrum"]["num_slots"],
        )
        out[method_name] = res
    return out


def main() -> int:
    t0 = time.perf_counter()
    payload: Dict[str, Any] = {"stage": "B_smoke", "config": SMOKE, "topologies": {}}
    all_ok = True
    for topo_name, topo_cfg in CFG["topologies"].items():
        res = run_topology_smoke(topo_cfg["topology_key"], topo_cfg)
        payload["topologies"][topo_name] = res
        for method_name, r in res.items():
            cons = r["conservation"]
            ok = (
                r["evaluated"] == SMOKE["evaluated_requests"]
                and r["admitted"] + r["blocked"] == r["evaluated"]
                and cons["allocations"] == cons["releases"]
                and cons["final_active"] == 0
                and cons["spectrum_empty_after_drain"]
                and 0.0 <= r["blocking_rate"] < 1.0
                and r["avg_fs"] > 0
            )
            all_ok &= ok
            print(
                f"[smoke] {topo_name:8s} {method_name:16s} "
                f"blk={r['blocking_rate']:.4%} admitted={r['admitted']:4d} "
                f"util={r['mean_utilization']:.3f} conservation_ok={ok}",
                flush=True,
            )
    payload["elapsed_seconds"] = time.perf_counter() - t0
    payload["all_ok"] = all_ok
    (EXP_DIR / "results" / "smoke.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"[stage-b] all_ok={all_ok} ({payload['elapsed_seconds']:.1f}s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
