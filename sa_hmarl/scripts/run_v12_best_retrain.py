"""Stage 3 optional scenario-specific retrain for the v1.2 R-ranker.

Given a Stage 1 scenario (or explicit parameters), this script:
  1. Generates a counterfactual ranking dataset in that scenario.
  2. Trains a new ranking model on that dataset.
  3. Evaluates the retrained model in closed-loop on the same scenario.

The goal is to see whether a scenario-specific ranker can widen the
v1.2-vs-DeepRMSA gap and/or bring overload overhead under the strict
Stage 1 threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "sa_hmarl")}


def _parse_csv_strings(text: str) -> List[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def _run(cmd: List[str], desc: str) -> None:
    print(f"\n[run] {desc}\n{' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=ENV, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")


def _scenario_from_summary(args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    rows = payload.get("rows", [])
    row = next((r for r in rows if r["scenario_id"] == args.scenario_id), None)
    if row is None:
        raise ValueError(f"Scenario {args.scenario_id} not found in {summary_path}")
    profile = row["split_profile"]
    if profile.startswith("default"):
        num_splits = 3
    elif profile.startswith("complex5"):
        num_splits = 5
    else:
        num_splits = row.get("num_splits", 3)
    scenario = {
        "agent_c_checkpoint": config.get("agent_c_checkpoint", args.agent_c_checkpoint),
        "agent_r_checkpoint": config.get("agent_r_checkpoint", args.agent_r_checkpoint),
        "topology": row.get("topology", config.get("topology", args.topology)),
        "num_slots": int(row.get("num_slots", config.get("num_slots", args.num_slots))),
        "num_servers": int(config.get("num_servers", args.num_servers)),
        "k_paths": int(config.get("k_paths", args.k_paths)),
        "max_blocks": int(config.get("max_blocks", args.max_blocks)),
        "block_sort_strategy": config.get("block_sort_strategy", args.block_sort_strategy),
        "split_profile": profile,
        "num_splits": num_splits,
        "arrival_interval": float(row["arrival_interval"]),
        "holding_min": float(config.get("holding_min", args.holding_min)),
        "holding_max": float(row["holding_max"]),
        "deadline_min": float(config.get("deadline_min", args.deadline_min)),
        "deadline_max": float(config.get("deadline_max", args.deadline_max)),
        "size_min_mb": float(config.get("size_min_mb", args.size_min_mb)),
        "size_max_mb": float(row["size_max_mb"]),
        "edge_cost_min": float(config.get("edge_cost_min", args.edge_cost_min)),
        "edge_cost_max": float(config.get("edge_cost_max", args.edge_cost_max)),
        "modulation_profile": config.get("modulation_profile", args.modulation_profile),
        "device": args.device,
    }
    return scenario


def _scenario_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "agent_c_checkpoint": args.agent_c_checkpoint,
        "agent_r_checkpoint": args.agent_r_checkpoint,
        "topology": args.topology,
        "num_slots": args.num_slots,
        "num_servers": args.num_servers,
        "k_paths": args.k_paths,
        "max_blocks": args.max_blocks,
        "block_sort_strategy": args.block_sort_strategy,
        "split_profile": args.split_profile,
        "num_splits": args.num_splits,
        "arrival_interval": args.arrival_interval,
        "holding_min": args.holding_min,
        "holding_max": args.holding_max,
        "deadline_min": args.deadline_min,
        "deadline_max": args.deadline_max,
        "size_min_mb": args.size_min_mb,
        "size_max_mb": args.size_max_mb,
        "edge_cost_min": args.edge_cost_min,
        "edge_cost_max": args.edge_cost_max,
        "modulation_profile": args.modulation_profile,
        "device": args.device,
    }


def _build_dataset_cmd(scenario: Dict[str, Any], args: argparse.Namespace, dataset_dir: Path) -> List[str]:
    return [
        str(PYTHON),
        "sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py",
        "--agent_c_checkpoint", str(scenario["agent_c_checkpoint"]),
        "--agent_r_checkpoint", str(scenario["agent_r_checkpoint"]),
        "--output_dir", str(dataset_dir),
        "--splits", args.splits,
        "--train_episodes", str(args.train_episodes),
        "--val_episodes", str(args.val_episodes),
        "--test_episodes", str(args.test_episodes),
        "--requests_per_episode", str(args.requests_per_episode),
        "--topology", str(scenario["topology"]),
        "--num_slots", str(scenario["num_slots"]),
        "--num_servers", str(scenario["num_servers"]),
        "--k_paths", str(scenario["k_paths"]),
        "--max_blocks", str(scenario["max_blocks"]),
        "--block_sort_strategy", str(scenario["block_sort_strategy"]),
        "--split_profile", str(scenario["split_profile"]),
        "--num_splits", str(scenario["num_splits"]),
        "--arrival_interval", str(scenario["arrival_interval"]),
        "--holding_min", str(scenario["holding_min"]),
        "--holding_max", str(scenario["holding_max"]),
        "--deadline_min", str(scenario["deadline_min"]),
        "--deadline_max", str(scenario["deadline_max"]),
        "--size_min_mb", str(scenario["size_min_mb"]),
        "--size_max_mb", str(scenario["size_max_mb"]),
        "--edge_cost_min", str(scenario["edge_cost_min"]),
        "--edge_cost_max", str(scenario["edge_cost_max"]),
        "--modulation_profile", str(scenario["modulation_profile"]),
        "--horizon", str(args.horizon),
        "--gamma", str(args.gamma),
        "--util_threshold", str(args.util_threshold),
        "--alpha", str(args.alpha),
        "--ppo_top_k", str(args.ppo_top_k),
        "--num_random_candidates", str(args.num_random_candidates),
        "--min_candidates", str(args.min_candidates),
        "--max_candidates", str(args.max_candidates),
        "--candidate_seed", str(args.candidate_seed),
        "--return_current_block_coef", str(args.return_current_block_coef),
        "--return_future_block_coef", str(args.return_future_block_coef),
        "--return_future_nsb_coef", str(args.return_future_nsb_coef),
        "--return_delay_coef", str(args.return_delay_coef),
        "--return_fs_coef", str(args.return_fs_coef),
        "--return_future_server_overload_coef", str(args.return_future_server_overload_coef),
        "--path_penalty_coef", str(args.path_penalty_coef),
        "--fs_penalty_coef", str(args.fs_penalty_coef),
        "--candidate_mode", str(args.candidate_mode),
        "--device", str(scenario["device"]),
    ]


def _build_train_cmd(args: argparse.Namespace, dataset_dir: Path, ckpt_dir: Path) -> List[str]:
    return [
        str(PYTHON),
        "sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py",
        "--dataset_dir", str(dataset_dir),
        "--output_dir", str(ckpt_dir),
        "--hidden_dims", args.hidden_dims,
        "--dropout", str(args.dropout),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--max_grad_norm", str(args.max_grad_norm),
        "--tau_label", str(args.tau_label),
        "--tau_model", str(args.tau_model),
        "--reg_weight", str(args.reg_weight),
        "--patience", str(args.patience),
        "--min_delta", str(args.min_delta),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--log_every", str(args.log_every),
    ]


def _build_eval_cmd(scenario: Dict[str, Any], args: argparse.Namespace, ckpt_path: Path, out_json: Path, out_md: Path) -> List[str]:
    return [
        str(PYTHON),
        "sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_closed_loop.py",
        "--agent_c_checkpoint", str(scenario["agent_c_checkpoint"]),
        "--agent_r_checkpoint", str(scenario["agent_r_checkpoint"]),
        "--ranking_checkpoint", str(ckpt_path),
        "--supervised_checkpoint", str(args.supervised_checkpoint),
        "--deep_rmsa_checkpoint", str(args.deep_rmsa_checkpoint),
        "--methods", args.methods,
        "--seeds", args.eval_seeds,
        "--episodes", str(args.eval_episodes),
        "--requests_per_episode", str(args.requests_per_episode),
        "--topology", str(scenario["topology"]),
        "--num_slots", str(scenario["num_slots"]),
        "--num_servers", str(scenario["num_servers"]),
        "--k_paths", str(scenario["k_paths"]),
        "--max_blocks", str(scenario["max_blocks"]),
        "--block_sort_strategy", str(scenario["block_sort_strategy"]),
        "--split_profile", str(scenario["split_profile"]),
        "--num_splits", str(scenario["num_splits"]),
        "--arrival_interval", str(scenario["arrival_interval"]),
        "--holding_min", str(scenario["holding_min"]),
        "--holding_max", str(scenario["holding_max"]),
        "--deadline_min", str(scenario["deadline_min"]),
        "--deadline_max", str(scenario["deadline_max"]),
        "--size_min_mb", str(scenario["size_min_mb"]),
        "--size_max_mb", str(scenario["size_max_mb"]),
        "--edge_cost_min", str(scenario["edge_cost_min"]),
        "--edge_cost_max", str(scenario["edge_cost_max"]),
        "--modulation_profile", str(scenario["modulation_profile"]),
        "--device", str(scenario["device"]),
        "--output_json", str(out_json),
        "--output_md", str(out_md),
    ]


def _extract_metrics(eval_json: Path) -> Dict[str, Any]:
    report = json.loads(eval_json.read_text(encoding="utf-8"))
    v12 = report["methods"]["counterfactual_rank_only"]["aggregate"]
    deep = report["methods"]["deep_rmsa"]["aggregate"]
    return {
        "v12_blocking": v12["blocking_rate"],
        "deep_blocking": deep["blocking_rate"],
        "deep_minus_v12_pp": (deep["blocking_rate"] - v12["blocking_rate"]) * 100.0,
        "v12_delay": v12["mean_delay_ms"],
        "deep_delay": deep["mean_delay_ms"],
        "delay_delta_ms": v12["mean_delay_ms"] - deep["mean_delay_ms"],
        "v12_overload": v12["server_overload_rate"],
        "deep_overload": deep["server_overload_rate"],
        "overload_delta_pp": (v12["server_overload_rate"] - deep["server_overload_rate"]) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", default=None, help="Stage 1 summary JSON.")
    parser.add_argument("--scenario_id", default=None, help="Scenario ID from summary.")
    parser.add_argument("--run_dir", required=True, help="Output directory for dataset, checkpoint, and eval.")

    # Scenario overrides (used if summary_json/scenario_id not given).
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.15)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")

    # Dataset generation parameters.
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--train_episodes", type=int, default=2)
    parser.add_argument("--val_episodes", type=int, default=1)
    parser.add_argument("--test_episodes", type=int, default=1)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--ppo_top_k", type=int, default=8)
    parser.add_argument("--num_random_candidates", type=int, default=5)
    parser.add_argument("--min_candidates", type=int, default=15)
    parser.add_argument("--max_candidates", type=int, default=30)
    parser.add_argument("--candidate_seed", type=int, default=12345)
    parser.add_argument("--return_current_block_coef", type=float, default=3.0)
    parser.add_argument("--return_future_block_coef", type=float, default=4.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=3.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.03)
    parser.add_argument("--return_fs_coef", type=float, default=0.05)
    parser.add_argument("--return_future_server_overload_coef", type=float, default=0.0)
    parser.add_argument("--path_penalty_coef", type=float, default=0.0)
    parser.add_argument("--fs_penalty_coef", type=float, default=0.0)
    parser.add_argument("--candidate_mode", default="v1")

    # Training parameters.
    parser.add_argument("--hidden_dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--tau_label", type=float, default=0.5)
    parser.add_argument("--tau_model", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=5)

    # Evaluation parameters.
    parser.add_argument("--supervised_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_formal/c_post_decision_joint.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="ppo_r,counterfactual_rank_only,deep_rmsa,ksp_bf")
    parser.add_argument("--eval_seeds", default="3030,4040,5050")
    parser.add_argument("--eval_episodes", type=int, default=5)

    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if bool(args.summary_json) != bool(args.scenario_id):
        print("[abort] --summary_json and --scenario_id must be provided together.", flush=True)
        sys.exit(1)

    if args.summary_json:
        scenario = _scenario_from_summary(args)
    else:
        scenario = _scenario_from_args(args)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = run_dir / "dataset"
    ckpt_dir = run_dir / "checkpoint"
    eval_json = run_dir / "eval.json"
    eval_md = run_dir / "eval.md"
    ckpt_path = ckpt_dir / "ranking_model.pt"

    # 1. Dataset generation.
    _run(_build_dataset_cmd(scenario, args, dataset_dir), "dataset generation")

    # 2. Training.
    _run(_build_train_cmd(args, dataset_dir, ckpt_dir), "ranker training")

    # 3. Evaluation.
    _run(_build_eval_cmd(scenario, args, ckpt_path, eval_json, eval_md), "closed-loop evaluation")

    # 4. Summary.
    metrics = _extract_metrics(eval_json)
    summary = {
        "scenario": scenario,
        "args": vars(args),
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_path": str(ckpt_path),
        "eval_json": str(eval_json),
        "eval_md": str(eval_md),
        "metrics": metrics,
    }
    summary_path = run_dir / "retrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {summary_path}")
    print(
        f"[result] v1.2={metrics['v12_blocking']:.2%}, deep={metrics['deep_blocking']:.2%}, "
        f"Deep-v1.2={metrics['deep_minus_v12_pp']:+.2f}pp, "
        f"delay_delta={metrics['delay_delta_ms']:+.2f}ms, overload_delta={metrics['overload_delta_pp']:+.2f}pp"
    )


if __name__ == "__main__":
    main()
