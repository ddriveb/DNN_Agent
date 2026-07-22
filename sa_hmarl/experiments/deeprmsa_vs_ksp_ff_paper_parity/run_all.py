#!/usr/bin/env python3
"""End-to-end orchestrator for the DeepRMSA vs KSP-FF paper-parity pipeline.

Stages:
  A  unit tests           (run_unit_tests.py; aborts on failure)
  B  1-seed smoke         (run_smoke.py)
  C  KSP parity           (run_ksp_parity.py; 10 seeds x 2 topologies, parallel)
  D  DeepRMSA training    (train_deeprmsa_paper.py x 6 runs, parallel)
  D2 DeepRMSA evaluation  (eval_deeprmsa_paper.py)
  R  report generation    (make_reports.py)

Usage:
    python run_all.py                  # full pipeline
    python run_all.py --stages A,B,C   # subset
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
PY = sys.executable

from paper_rmsa_core import load_run_config  # noqa: E402

CFG = load_run_config()
DCFG = CFG["methods"]["deeprmsa_local"]


def run(cmd: list[str], log: str | None = None) -> int:
    print(f"[run_all] $ {' '.join(cmd)}", flush=True)
    if log:
        with (EXP_DIR / "results" / "logs" / log).open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=EXP_DIR, stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=EXP_DIR)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="A,B,C,D,D2,R")
    args = parser.parse_args()
    stages = {s.strip() for s in args.stages.split(",")}
    (EXP_DIR / "results" / "logs").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    if "A" in stages:
        if run([PY, "run_unit_tests.py"]) != 0:
            print("[run_all] Stage A FAILED — aborting", flush=True)
            return 1
    if "B" in stages:
        if run([PY, "run_smoke.py"]) != 0:
            print("[run_all] Stage B FAILED — aborting", flush=True)
            return 1
    if "C" in stages:
        if run([PY, "run_ksp_parity.py"]) != 0:
            print("[run_all] Stage C parity FAILED — aborting before Stage D", flush=True)
            return 1
    if "D" in stages:
        procs = []
        for topology in CFG["topologies"]:
            for seed in DCFG["train_seeds"]:
                log = (EXP_DIR / "results" / "logs" / f"train_{topology}_{seed}.log").open("w", encoding="utf-8")
                procs.append((subprocess.Popen(
                    [PY, "train_deeprmsa_paper.py", "--topology", topology, "--seed", str(seed)],
                    cwd=EXP_DIR, stdout=log, stderr=subprocess.STDOUT,
                ), log))
        rc = 0
        for proc, log in procs:
            rc |= proc.wait()
            log.close()
        if rc != 0:
            print("[run_all] Stage D training FAILED — aborting", flush=True)
            return 1
    if "D2" in stages:
        if run([PY, "eval_deeprmsa_paper.py"]) != 0:
            print("[run_all] Stage D evaluation FAILED", flush=True)
            return 1
    if "R" in stages:
        if run([PY, "make_reports.py"]) != 0:
            return 1

    print(f"[run_all] pipeline finished ({time.perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
