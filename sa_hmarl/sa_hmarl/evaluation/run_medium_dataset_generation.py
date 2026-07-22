"""Generate the SA-HMARL v1.3 medium dataset and merge it.

Configuration:
  - train: 10 shards, base seed 5001
  - val:   3 shards, base seed 6001
  - test:  3 shards, base seed 7001
  - 1200 requests / shard, 250 warmup
  - max_workers=5, log_interval=300s

After generation, runs the shard merger and copies the dataset report to the
experiments directory as MEDIUM_DATASET_REPORT.md.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


DATASET_NAME = "r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium"
SEEDS = "5001,6001,7001"
N_TRAIN = 10
N_VAL = 3
N_TEST = 3
REQUESTS_PER_EPISODE = 1200
WARMUP = 250
MAX_WORKERS = 5
LOG_INTERVAL = 300.0


def _run(cmd: list, root: Path, env: dict, timeout: int = 18000) -> None:
    print(f"[medium] {' '.join(cmd[-4:])} ...")
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stderr[-2000:]}")
    print(proc.stdout[-500:])


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    dataset_dir = root / "sa_hmarl" / "datasets" / DATASET_NAME
    exp_dir = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking"
    exp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    gen_cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5",
        "--output_dir", str(dataset_dir),
        "--seeds", SEEDS,
        "--n_train_shards", str(N_TRAIN),
        "--n_val_shards", str(N_VAL),
        "--n_test_shards", str(N_TEST),
        "--requests_per_episode", str(REQUESTS_PER_EPISODE),
        "--warmup", str(WARMUP),
        "--max_workers", str(MAX_WORKERS),
        "--log_interval", str(LOG_INTERVAL),
        "--device", "cpu",
    ]
    _run(gen_cmd, root, env, timeout=18000)

    merge_cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.merge_r_counterfactual_ranking_shards",
        "--dataset_dir", str(dataset_dir),
    ]
    _run(merge_cmd, root, env, timeout=600)

    src_report = dataset_dir / "DATASET_REPORT.md"
    dst_report = exp_dir / "MEDIUM_DATASET_REPORT.md"
    if src_report.exists():
        shutil.copy(src_report, dst_report)
        print(f"[medium] Copied dataset report to {dst_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
