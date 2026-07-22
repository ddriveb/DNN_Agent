"""Serial vs parallel consistency test for v1.35 paired feature dataset.

Runs the paired generator with max_workers=1 and max_workers=3 on the same
 tiny config and compares group ids, action ids, labels, masks, and features.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _run_generator(output_dir: Path, max_workers: int) -> Path:
    cmd = [
        sys.executable,
        "sa_hmarl/sa_hmarl/evaluation/v135_paired_feature_dataset.py",
        "--output_dir", str(output_dir),
        "--train_seeds", "1001", "1002",
        "--val_seeds", "2001",
        "--test_seeds", "3001",
        "--train_episodes", "1",
        "--val_episodes", "1",
        "--test_episodes", "1",
        "--requests_per_episode", "20",
        "--warmup_requests", "0",
        "--horizon", "3",
        "--max_candidates", "8",
        "--ppo_top_k", "8",
        "--max_workers", str(max_workers),
    ]
    env = {**dict(os.environ), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run(cmd, check=True, env=env)
    return output_dir


def _load_split(path: Path, split: str):
    npz = np.load(str(path / f"{split}.npz"), allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def _compare(name: str, a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> list:
    errors = []
    if a.shape != b.shape:
        errors.append(f"{name}: shape mismatch {a.shape} vs {b.shape}")
        return errors
    if a.dtype == object or b.dtype == object:
        if not np.array_equal(a, b):
            errors.append(f"{name}: object array mismatch")
    elif np.issubdtype(a.dtype, np.floating):
        if not np.allclose(a, b, atol=atol):
            diff = np.abs(a - b).max()
            errors.append(f"{name}: float mismatch max_diff={diff:.6e}")
    else:
        if not np.array_equal(a, b):
            errors.append(f"{name}: discrete mismatch")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep temporary directories")
    args = parser.parse_args()

    base = Path(tempfile.mkdtemp(prefix="v135_consistency_"))
    serial_dir = base / "serial"
    parallel_dir = base / "parallel"

    try:
        print("[consistency] Running serial generation...", flush=True)
        t0 = time.perf_counter()
        _run_generator(serial_dir, max_workers=1)
        serial_time = time.perf_counter() - t0

        print("[consistency] Running parallel generation (workers=3)...", flush=True)
        t0 = time.perf_counter()
        _run_generator(parallel_dir, max_workers=3)
        parallel_time = time.perf_counter() - t0

        print("[consistency] Comparing outputs...", flush=True)
        all_errors = []
        for split in ("train", "val", "test"):
            s = _load_split(serial_dir, split)
            p = _load_split(parallel_dir, split)
            if sorted(s.keys()) != sorted(p.keys()):
                all_errors.append(f"{split}: key set mismatch")
                continue
            if len(s["group_id"]) != len(p["group_id"]):
                all_errors.append(f"{split}: group count mismatch {len(s['group_id'])} vs {len(p['group_id'])}")
                continue
            for key in sorted(s.keys()):
                all_errors.extend(_compare(f"{split}.{key}", s[key], p[key]))

        report = {
            "serial_time_sec": serial_time,
            "parallel_time_sec": parallel_time,
            "speedup": serial_time / max(parallel_time, 1e-6),
            "errors": all_errors,
            "passed": len(all_errors) == 0,
        }
        out = Path("sa_hmarl/experiments/v135_parallel_paired")
        out.mkdir(parents=True, exist_ok=True)
        out_json = out / "PARALLEL_CONSISTENCY.json"
        out_md = out / "PARALLEL_CONSISTENCY.md"
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        lines = [
            "# Parallel Consistency Test",
            "",
            f"- Serial wall-clock: {serial_time:.2f} s",
            f"- Parallel wall-clock: {parallel_time:.2f} s",
            f"- Speedup: {report['speedup']:.2f}x",
            "",
            f"**Result: {'PASS' if report['passed'] else 'FAIL'}**",
            "",
        ]
        if all_errors:
            lines.append("## Errors")
            for e in all_errors:
                lines.append(f"- {e}")
        else:
            lines.append("All group IDs, action IDs, labels, label components, masks, v1 features, and poststate_v1 features match between serial and parallel execution.")
        out_md.write_text("\n".join(lines), encoding="utf-8")

        print(f"[consistency] Result: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"[consistency] Serial {serial_time:.2f}s, Parallel {parallel_time:.2f}s, Speedup {report['speedup']:.2f}x")
        if all_errors:
            for e in all_errors:
                print(f"  - {e}")
        return 0 if report["passed"] else 1
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
