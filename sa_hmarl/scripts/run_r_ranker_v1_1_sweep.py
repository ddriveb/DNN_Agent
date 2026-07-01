"""Run the R-ranker v1.1 return-weight sweep.

This script keeps the v1 candidate construction untouched and only changes the
counterfactual return coefficients.  For each configuration it:
  1. Generates a counterfactual ranking dataset.
  2. Trains a listwise ranker.
  3. Evaluates the ranker in closed loop (rank-only) against PPO-R, DeepRMSA,
     and KSP-BF.

All outputs are written under:
    sa_hmarl/datasets/r_counterfactual_ranking_v1_1/<name>/
    sa_hmarl/checkpoints/r_counterfactual_ranking_v1_1/<name>/
    sa_hmarl/experiments/r_counterfactual_ranking_v1_1_eval_<name>.json/.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SA_HMARL = PROJECT_ROOT / "sa_hmarl"

# Keep the same protocol used for v1.
EPISODES_PER_SPLIT = 10
REQUESTS_PER_EPISODE = 80

CONFIGS: List[Dict[str, object]] = [
    {
        "name": "v1_baseline",
        "return_future_block_coef": 4.0,
        "return_future_nsb_coef": 3.0,
        "return_delay_coef": 0.03,
        "return_fs_coef": 0.05,
    },
    {
        "name": "no_fs",
        "return_future_block_coef": 4.0,
        "return_future_nsb_coef": 3.0,
        "return_delay_coef": 0.03,
        "return_fs_coef": 0.0,
    },
    {
        "name": "weak_fs_001",
        "return_future_block_coef": 4.0,
        "return_future_nsb_coef": 3.0,
        "return_delay_coef": 0.03,
        "return_fs_coef": 0.01,
    },
    {
        "name": "weak_fs_002",
        "return_future_block_coef": 4.0,
        "return_future_nsb_coef": 3.0,
        "return_delay_coef": 0.03,
        "return_fs_coef": 0.02,
    },
    {
        "name": "stronger_block",
        "return_future_block_coef": 6.0,
        "return_future_nsb_coef": 3.0,
        "return_delay_coef": 0.03,
        "return_fs_coef": 0.01,
    },
    {
        "name": "stronger_nsb",
        "return_future_block_coef": 5.0,
        "return_future_nsb_coef": 5.0,
        "return_delay_coef": 0.03,
        "return_fs_coef": 0.01,
    },
]


def run_cmd(cmd: List[str], log_path: Path, env: Dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{'='*60}\n{' '.join(cmd)}\n{'='*60}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit {proc.returncode}: {' '.join(cmd)}")


def main() -> None:
    sweep_log = SA_HMARL / "experiments" / "r_ranker_v1_1_sweep.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SA_HMARL)

    python = str(PROJECT_ROOT / ".venv" / "bin" / "python")

    # Use the existing v1 checkpoint as the baseline instead of regenerating it.
    baseline_checkpoint = SA_HMARL / "checkpoints" / "r_counterfactual_ranking_k5m10_h5" / "ranking_model.pt"

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n### Starting {name} ###", flush=True)

        dataset_dir = SA_HMARL / "datasets" / "r_counterfactual_ranking_v1_1" / name
        checkpoint_dir = SA_HMARL / "checkpoints" / "r_counterfactual_ranking_v1_1" / name
        eval_json = SA_HMARL / "experiments" / f"r_counterfactual_ranking_v1_1_eval_{name}.json"
        eval_md = SA_HMARL / "experiments" / f"r_counterfactual_ranking_v1_1_eval_{name}.md"

            # 1. Dataset generation (skip baseline; reuse existing v1 dataset).
        if name != "v1_baseline":
            metadata_path = dataset_dir / "metadata.json"
            if metadata_path.exists():
                with sweep_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{name}] dataset exists, skipping generation\n")
            else:
                gen_cmd = [
                    python, "-m", "sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset",
                    "--output_dir", str(dataset_dir),
                    "--episodes", str(EPISODES_PER_SPLIT),
                    "--requests_per_episode", str(REQUESTS_PER_EPISODE),
                    "--candidate_mode", "v1",
                    "--return_current_block_coef", "3.0",
                    "--return_future_block_coef", str(cfg["return_future_block_coef"]),
                    "--return_future_nsb_coef", str(cfg["return_future_nsb_coef"]),
                    "--return_delay_coef", str(cfg["return_delay_coef"]),
                    "--return_fs_coef", str(cfg["return_fs_coef"]),
                ]
                with sweep_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[{name}] dataset generation start\n")
                run_cmd(gen_cmd, sweep_log, env)
                with sweep_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{name}] dataset generation done\n")

        # 2. Training.
        if name == "v1_baseline":
            ckpt_path = baseline_checkpoint
        else:
            ckpt_path = checkpoint_dir / "ranking_model.pt"
            if ckpt_path.exists():
                with sweep_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{name}] checkpoint exists, skipping training\n")
            else:
                train_cmd = [
                    python, "-m", "sa_hmarl.training.train_r_counterfactual_ranking",
                    "--dataset_dir", str(dataset_dir),
                    "--output_dir", str(checkpoint_dir),
                    "--device", "cpu",
                ]
                with sweep_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{name}] training start\n")
                run_cmd(train_cmd, sweep_log, env)
                with sweep_log.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{name}] training done\n")

        # 3. Closed-loop evaluation.
        # Only the baseline needs the full comparison; v1.1 variants only need
        # their own rank-only result because PPO-R/DeepRMSA/KSP-BF are shared
        # baselines across all configs.
        methods = (
            "ppo_r,counterfactual_rank_only,deep_rmsa,ksp_bf"
            if name == "v1_baseline" else "counterfactual_rank_only"
        )
        eval_cmd = [
            python, "-m", "sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop_v2",
            "--ranking_checkpoint", str(ckpt_path),
            "--methods", methods,
            "--output_json", str(eval_json),
            "--output_md", str(eval_md),
            "--device", "cpu",
        ]
        with sweep_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{name}] eval start\n")
        run_cmd(eval_cmd, sweep_log, env)
        with sweep_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{name}] eval done\n")

        # Print a quick summary line.
        try:
            result = json.loads(eval_json.read_text(encoding="utf-8"))
            rank_only = result["methods"]["counterfactual_rank_only"]["aggregate"]
            deep_rmsa = result["methods"]["deep_rmsa"]["aggregate"]
            ppo_r = result["methods"]["ppo_r"]["aggregate"]
            print(
                f"[{name}] rank_only={rank_only['blocking_rate']:.2%} "
                f"deep_rmsa={deep_rmsa['blocking_rate']:.2%} "
                f"ppo_r={ppo_r['blocking_rate']:.2%}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{name}] summary failed: {exc}", flush=True)

    print("\nSweep complete.", flush=True)


if __name__ == "__main__":
    main()
