#!/usr/bin/env python3
"""Stage D (evaluation) — evaluate trained DeepRMSA checkpoints on test seeds.

For each topology x training seed, loads checkpoints/<topology>/seed_<s>/best.pt
and evaluates the argmax policy on the 10 shared test seeds
(warmup=3000, evaluated=10000) — the same traces used by the KSP-FF methods.

Writes results/deeprmsa_eval.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parents[1]))

from paper_rmsa_core import (
    generate_paper_requests,
    load_run_config,
    run_simulation,
)

CFG = load_run_config()
DCFG = CFG["methods"]["deeprmsa_local"]
MEAS = CFG["measurement"]


def _eval_one(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    from paper_deeprmsa_agent import PaperDeepRMSAAgent, make_deeprmsa_decide
    from paper_rmsa_core import paper_modulation_registry

    torch.set_num_threads(1)
    topology = job["topology"]
    topo_cfg = CFG["topologies"][topology]
    mod_reg = paper_modulation_registry()

    ckpt = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    agent = PaperDeepRMSAAgent(
        num_nodes=topo_cfg["num_nodes"], num_slots=CFG["spectrum"]["num_slots"],
        k_path=DCFG["k_paths"], m_blocks=DCFG["m_blocks"], mod_registry=mod_reg,
        gamma=DCFG["gamma"], lr=DCFG["lr"], entropy_coef=DCFG["entropy_coef"],
        value_loss_coef=DCFG["value_loss_coef"], max_grad_norm=DCFG["max_grad_norm"],
        num_layers=DCFG["num_layers"], layer_size=DCFG["layer_size"], device="cpu",
    )
    agent.load_state_dict(ckpt)
    agent.eval()
    for p in agent.policy_net.parameters():
        p.requires_grad = False

    arrival_interval = CFG["traffic"]["mean_holding_time"] / topo_cfg["load_erlang"]
    per_seed: Dict[str, Any] = {}
    for seed in MEAS["test_seeds"]:
        requests = generate_paper_requests(
            topo_cfg["num_nodes"], seed,
            MEAS["warmup_requests"] + MEAS["evaluated_requests"], arrival_interval,
            mean_holding_time=CFG["traffic"]["mean_holding_time"],
            bitrate_min_gbps=CFG["traffic"]["bitrate_min_gbps"],
            bitrate_max_gbps=CFG["traffic"]["bitrate_max_gbps"],
        )
        res = run_simulation(
            topo_cfg["topology_key"], requests,
            make_deeprmsa_decide(agent, mod_reg),
            k_paths=DCFG["k_paths"], path_sort=DCFG["path_sort"],
            warmup=MEAS["warmup_requests"], num_slots=CFG["spectrum"]["num_slots"],
        )
        per_seed[str(seed)] = res
    rates = [r["blocking_rate"] for r in per_seed.values()]
    return {
        "topology": topology,
        "training_seed": job["training_seed"],
        "checkpoint": job["checkpoint"],
        "checkpoint_meta": ckpt.get("meta", {}),
        "per_seed": per_seed,
        "mean_blocking_rate": float(np.mean(rates)),
        "std_blocking_rate": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
    }


def main() -> int:
    t0 = time.perf_counter()
    jobs: List[Dict[str, Any]] = []
    for topology in CFG["topologies"]:
        for train_seed in DCFG["train_seeds"]:
            ckpt_path = EXP_DIR / "checkpoints" / topology / f"seed_{train_seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"[stage-d-eval] MISSING checkpoint: {ckpt_path}", flush=True)
                return 1
            jobs.append({
                "topology": topology,
                "training_seed": train_seed,
                "checkpoint": str(ckpt_path),
            })

    max_workers = min(len(jobs), os.cpu_count() or 1)
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one, job): job for job in jobs}
        for fut in as_completed(futures):
            out = fut.result()
            results.append(out)
            print(
                f"[stage-d-eval] {out['topology']:8s} train_seed={out['training_seed']:3d} "
                f"mean_blk={out['mean_blocking_rate']:.4%} ± {out['std_blocking_rate']:.4%}",
                flush=True,
            )

    payload = {
        "stage": "D_deeprmsa_eval",
        "protocol_id": CFG["protocol_id"],
        "warmup": MEAS["warmup_requests"],
        "evaluated": MEAS["evaluated_requests"],
        "test_seeds": MEAS["test_seeds"],
        "results": sorted(results, key=lambda r: (r["topology"], r["training_seed"])),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (EXP_DIR / "results" / "deeprmsa_eval.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"[stage-d-eval] done ({payload['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
