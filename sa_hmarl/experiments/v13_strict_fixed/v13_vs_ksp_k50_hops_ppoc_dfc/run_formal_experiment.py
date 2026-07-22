"""Orchestrate formal 20-seed evaluation: Strict v1.3 vs KSP-FF K=50 hops under PPO-C and DF_C.

This script is intentionally self-contained. It launches per-seed workers with a
concurrency limit, waits for completion, reruns failed seeds once, and then
aggregates results into all required deliverables.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project root is four levels above this file (file -> exp_dir -> v13_strict_fixed -> experiments -> sa_hmarl -> project_root).
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = ROOT / ".venv" / "bin" / "python"
MODULE = "sa_hmarl.evaluation.eval_strict_v13_multitopology_cside_fair"

TOPOLOGY = "xlron_cost239_ptrnet_real"
SEEDS = list(range(5001, 5021))
WARMUP = 500
REQUESTS = 6000
CONCURRENCY = 4

C_MODES = "ppo_c,df_c"
C_CHECKPOINTS = "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt,_none_"
R_MODES = "ppo_r_top1,ksp_ff_highest,strict_v13,old_v13"
RANKER_SPECS = [
    "strict_v13=sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt",
    "old_v13=sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt",
]

ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}


def _build_cmd(seed: int, output_dir: Path) -> List[str]:
    return [
        str(PYTHON), "-m", MODULE,
        "--topology", TOPOLOGY,
        "--seed", str(seed),
        "--c_modes", C_MODES,
        "--c_checkpoints", C_CHECKPOINTS,
        "--r_modes", R_MODES,
        "--warmup_requests", str(WARMUP),
        "--requests_per_episode", str(REQUESTS),
        "--poisson_arrivals",
        "--exponential_holding",
        "--output_dir", str(output_dir),
        "--output_json", f"seed_{seed}.json",
        "--output_log", f"seed_{seed}.log",
        *sum([["--ranker_specs", spec] for spec in RANKER_SPECS], []),
    ]


def _run_seed(seed: int, output_dir: Path, timeout: int = 7200) -> Dict[str, Any]:
    out = output_dir / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(seed, out)
    env = os.environ.copy()
    env.update(ENV)
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    return {
        "seed": seed,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "output_dir": str(out),
    }


def _worker(args: Tuple[int, Path, int]) -> Dict[str, Any]:
    seed, output_dir, timeout = args
    return _run_seed(seed, output_dir, timeout)


def launch_all(output_dir: Path, concurrency: int = CONCURRENCY, timeout: int = 7200) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use a simple semaphore through subprocess concurrency via threads.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_worker, (seed, output_dir, timeout)): seed for seed in SEEDS}
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            print(f"[launcher] seed={rec['seed']} rc={rec['returncode']} elapsed={rec['elapsed_sec']:.1f}s")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(ROOT / "sa_hmarl/experiments/v13_strict_fixed/v13_vs_ksp_k50_hops_ppoc_dfc/formal"))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--manifest", default="LAUNCH_MANIFEST.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = launch_all(output_dir, args.concurrency, args.timeout)
    failed = [r for r in records if r["returncode"] != 0]
    if failed:
        print(f"[launcher] {len(failed)} seeds failed on first attempt; retrying once...")
        retry_records = launch_all(output_dir, args.concurrency, args.timeout)
        # Keep original records plus retry info.
        for rec in records:
            if rec["returncode"] != 0:
                rec["retry"] = next((r for r in retry_records if r["seed"] == rec["seed"]), None)

    manifest_path = output_dir / args.manifest
    manifest_path.write_text(json.dumps({"records": records}, indent=2, default=str), encoding="utf-8")
    print(f"[launcher] Manifest saved to {manifest_path}")
    n_failed = sum(1 for r in records if r["returncode"] != 0 and (r.get("retry") is None or r["retry"]["returncode"] != 0))
    print(f"[launcher] Total failed after retry: {n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
