"""Stage 3 targeted retrain sweep for the two v1.2 candidate scenarios.

Scenarios:
  A: default3, ai=0.09, size_max=30, holding_max=14
  B: complex5_v2_lite, ai=0.15, size_max=30, holding_max=14

Overload penalty coefficient ∈ {0.5, 1.0, 2.0}.
Fixed return weights mirror v1.2_mixed_low with an added server-overload term.

Stop-loss:
  - PASS:        gap ≥ 1.5 pp, overload Δ ≤ 0.5 pp, delay Δ ≤ 1 ms
  - STRONG PASS: gap ≥ 2.0 pp, overload Δ ≤ 0.5 pp, delay Δ ≤ 1 ms
  - STOP:        after all three coefficients, no PASS; recommend accepting ~1 pp.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv" / "bin" / "python")
RETRAIN_SCRIPT = str(ROOT / "sa_hmarl" / "scripts" / "run_v12_best_retrain.py")
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "sa_hmarl")}

SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_id": "snap24_gnutella_reach_s24_default3_ai0p09_sz30p0_h14p0",
        "split_profile": "default3",
        "num_splits": 3,
        "arrival_interval": 0.09,
        "size_max_mb": 30.0,
        "holding_max": 14.0,
    },
    {
        "scenario_id": "snap24_gnutella_reach_s24_complex5_v2_lite_ai0p15_sz30p0_h14p0",
        "split_profile": "complex5_v2_lite",
        "num_splits": 5,
        "arrival_interval": 0.15,
        "size_max_mb": 30.0,
        "holding_max": 14.0,
    },
]

OVERLOAD_COEFS = [0.5, 1.0, 2.0]


def _classify(metrics: Dict[str, Any]) -> str:
    gap = metrics["deep_minus_v12_pp"]
    delay = metrics["delay_delta_ms"]
    overload = metrics["overload_delta_pp"]
    if gap >= 2.0 and delay <= 1.0 and overload <= 0.5:
        return "STRONG_PASS"
    if gap >= 1.5 and delay <= 1.0 and overload <= 0.5:
        return "PASS"
    return "FAIL"


def _build_cmd(scenario: Dict[str, Any], overload_coef: float, run_dir: Path, args: argparse.Namespace) -> List[str]:
    return [
        PYTHON,
        RETRAIN_SCRIPT,
        "--run_dir", str(run_dir),
        # scenario overrides
        "--split_profile", scenario["split_profile"],
        "--num_splits", str(scenario["num_splits"]),
        "--arrival_interval", str(scenario["arrival_interval"]),
        "--size_max_mb", str(scenario["size_max_mb"]),
        "--holding_max", str(scenario["holding_max"]),
        # dataset generation
        "--train_episodes", str(args.train_episodes),
        "--val_episodes", str(args.val_episodes),
        "--test_episodes", str(args.test_episodes),
        "--requests_per_episode", str(args.requests_per_episode),
        "--horizon", str(args.horizon),
        "--gamma", str(args.gamma),
        "--return_current_block_coef", str(args.return_current_block_coef),
        "--return_future_block_coef", str(args.return_future_block_coef),
        "--return_future_nsb_coef", str(args.return_future_nsb_coef),
        "--return_delay_coef", str(args.return_delay_coef),
        "--return_fs_coef", str(args.return_fs_coef),
        "--return_future_server_overload_coef", str(overload_coef),
        "--path_penalty_coef", str(args.path_penalty_coef),
        "--fs_penalty_coef", str(args.fs_penalty_coef),
        # training
        "--hidden_dims", args.hidden_dims,
        "--dropout", str(args.dropout),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        # evaluation
        "--eval_seeds", args.eval_seeds,
        "--eval_episodes", str(args.eval_episodes),
        "--device", args.device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="sa_hmarl/experiments/v12_stage3_retrain")
    # dataset sizing
    parser.add_argument("--train_episodes", type=int, default=5)
    parser.add_argument("--val_episodes", type=int, default=2)
    parser.add_argument("--test_episodes", type=int, default=2)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.95)
    # return weights
    parser.add_argument("--return_current_block_coef", type=float, default=3.0)
    parser.add_argument("--return_future_block_coef", type=float, default=4.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=3.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.03)
    parser.add_argument("--return_fs_coef", type=float, default=0.05)
    parser.add_argument("--path_penalty_coef", type=float, default=0.05)
    parser.add_argument("--fs_penalty_coef", type=float, default=0.05)
    # training
    parser.add_argument("--hidden_dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    # eval
    parser.add_argument("--eval_seeds", default="3030,4040,5050")
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    best = None

    for scenario in SCENARIOS:
        sid = scenario["scenario_id"]
        print(f"\n{'='*60}\nScenario: {sid}\n{'='*60}", flush=True)
        for overload_coef in OVERLOAD_COEFS:
            run_name = f"{sid}_ol{str(overload_coef).replace('.', 'p')}"
            run_dir = output_root / run_name
            print(f"\n[coef={overload_coef}] -> {run_dir}", flush=True)
            cmd = _build_cmd(scenario, overload_coef, run_dir, args)
            result = subprocess.run(cmd, env=ENV, cwd=str(ROOT))
            if result.returncode != 0:
                print(f"[error] {run_name} failed; continuing.", flush=True)
                continue

            summary_path = run_dir / "retrain_summary.json"
            if not summary_path.exists():
                print(f"[error] {summary_path} missing; continuing.", flush=True)
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary["metrics"]
            verdict = _classify(metrics)
            row = {
                "run": run_name,
                "scenario_id": sid,
                "overload_coef": overload_coef,
                "v12_blocking": metrics["v12_blocking"],
                "deep_blocking": metrics["deep_blocking"],
                "deep_minus_v12_pp": metrics["deep_minus_v12_pp"],
                "delay_delta_ms": metrics["delay_delta_ms"],
                "overload_delta_pp": metrics["overload_delta_pp"],
                "verdict": verdict,
                "run_dir": str(run_dir),
            }
            all_results.append(row)
            print(
                f"[result] {run_name}: v1.2={metrics['v12_blocking']:.2%}, "
                f"deep={metrics['deep_blocking']:.2%}, "
                f"gap={metrics['deep_minus_v12_pp']:+.2f}pp, "
                f"delayΔ={metrics['delay_delta_ms']:+.2f}ms, "
                f"overloadΔ={metrics['overload_delta_pp']:+.2f}pp -> {verdict}",
                flush=True,
            )
            if best is None or metrics["deep_minus_v12_pp"] > best["deep_minus_v12_pp"]:
                best = row

    # Final sweep summary.
    pass_results = [r for r in all_results if r["verdict"] in ("PASS", "STRONG_PASS")]
    sweep_summary = {
        "args": vars(args),
        "runs": all_results,
        "best_run": best,
        "pass_runs": pass_results,
        "recommendation": (
            "STOP: no run achieved gap>=1.5pp with overload<=0.5pp. Accept ~1pp advantage."
            if not pass_results else
            "Proceed with best PASS run for formal validation."
        ),
    }
    summary_path = output_root / "stage3_sweep_summary.json"
    summary_path.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {summary_path}")

    if pass_results:
        print("\nPASS / STRONG PASS runs:")
        for r in pass_results:
            print(f"  {r['run']}: gap={r['deep_minus_v12_pp']:+.2f}pp, overloadΔ={r['overload_delta_pp']:+.2f}pp, {r['verdict']}")
    else:
        print("\n[STOP] No run met the PASS threshold. Best run:")
        if best:
            print(f"  {best['run']}: gap={best['deep_minus_v12_pp']:+.2f}pp, overloadΔ={best['overload_delta_pp']:+.2f}pp")


if __name__ == "__main__":
    main()
