"""Run the Doherty/XLRON-style external-OD transfer diagnostic pilot.

Uses the existing strict v1.3 evaluator with Doherty-style environment parameters.
The locked v1.3/v1.35 rankers are evaluated zero-shot; results are tagged as
transfer diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path("sa_hmarl/experiments/v135_doherty_external_od_pilot")
RANKER_CKPTS = {
    "strict_v13": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/strict_v13/seed_42/ranking_model.pt",
    "v135_afterstate": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate/seed_42/ranking_model.pt",
    "v135_afterstate_explicit": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate_explicit/seed_42/ranking_model.pt",
}
AGENT_R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"

# Doherty-style external-OD parameters.
NUM_SLOTS = 100
HOLDING_MIN = 5.0
HOLDING_MAX = 15.0
DEADLINE_MIN = 100.0
DEADLINE_MAX = 300.0
SIZE_MIN_MB = 15.0
SIZE_MAX_MB = 60.0
EDGE_COST_MIN = 0.1
EDGE_COST_MAX = 0.2
K_PATHS_R = 50
PATH_SORT_R = "hops"
BLOCK_SORT_R = "start_asc"


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    return env


def _ranker_specs() -> List[str]:
    specs = []
    for name, path in RANKER_CKPTS.items():
        specs.extend(["--ranker_specs", f"{name}={path}"])
    return specs


def _run_eval(
    seed: int,
    arrival_interval: float,
    warmup: int,
    requests: int,
    r_modes: str,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.eval_strict_v13_multitopology_cside_verified",
        "--topology", "xlron_cost239_ptrnet_real",
        "--seed", str(seed),
        "--c_modes", "df_c",
        "--c_checkpoints", "_none_",
        "--r_modes", r_modes,
        "--agent_r_checkpoint", AGENT_R_CKPT,
        *_ranker_specs(),
        "--num_slots", str(NUM_SLOTS),
        "--num_servers", "4",
        "--k_paths_c", "5",
        "--k_paths_r", str(K_PATHS_R),
        "--path_sort_strategy_c", "hops",
        "--path_sort_strategy_r", PATH_SORT_R,
        "--block_sort_strategy_c", "start_asc",
        "--block_sort_strategy_r", BLOCK_SORT_R,
        "--modulation_profile", "default",
        "--max_blocks", "10",
        "--split_profile", "default3",
        "--num_splits", "3",
        "--arrival_interval", str(arrival_interval),
        "--holding_min", str(HOLDING_MIN),
        "--holding_max", str(HOLDING_MAX),
        "--deadline_min", str(DEADLINE_MIN),
        "--deadline_max", str(DEADLINE_MAX),
        "--size_min_mb", str(SIZE_MIN_MB),
        "--size_max_mb", str(SIZE_MAX_MB),
        "--edge_cost_min", str(EDGE_COST_MIN),
        "--edge_cost_max", str(EDGE_COST_MAX),
        "--warmup_requests", str(warmup),
        "--requests_per_episode", str(requests),
        "--poisson_arrivals",
        "--exponential_holding",
        "--device", "cpu",
        "--output_dir", str(output_dir),
        "--output_json", "results.json",
        "--output_log", "eval.log",
    ]
    print(f"[external-od] eval seed={seed} arrival={arrival_interval} r_modes={r_modes}")
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[3], env=_env(), check=True)
    result_path = output_dir / "results.json"
    return json.loads(result_path.read_text(encoding="utf-8"))


def _ksp_blocking(payload: Dict[str, Any]) -> Optional[float]:
    for row in payload.get("results", []):
        if row.get("r_mode") == "ksp_ff_highest":
            return float(row["blocking_rate"])
    return None


def _calibrate_arrival_interval(seed: int = 3030) -> float:
    """Grid-search arrival_interval to put KSP-FF blocking in 5-15% range."""
    candidates = [0.08, 0.06, 0.05, 0.04, 0.03, 0.025]
    best_interval = candidates[0]
    best_distance = float("inf")
    calibration_results = []
    target = 0.10
    for interval in candidates:
        output_dir = BASE_DIR / "calibration" / f"seed_{seed}_arr_{interval}"
        if (output_dir / "results.json").exists():
            payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
        else:
            payload = _run_eval(
                seed=seed,
                arrival_interval=interval,
                warmup=500,
                requests=1000,
                r_modes="ksp_ff_highest",
                output_dir=output_dir,
            )
        blocking = _ksp_blocking(payload)
        calibration_results.append({"arrival_interval": interval, "ksp_blocking": blocking})
        print(f"[calibrate] interval={interval} ksp_blocking={blocking:.4%}" if blocking is not None else f"[calibrate] interval={interval} ksp_blocking=N/A")
        if blocking is None:
            continue
        distance = abs(blocking - target)
        if 0.05 <= blocking <= 0.15 and distance < best_distance:
            best_distance = distance
            best_interval = interval
    # If none in target range, pick closest.
    if best_distance == float("inf"):
        for rec in calibration_results:
            blocking = rec["ksp_blocking"]
            if blocking is None:
                continue
            distance = abs(blocking - target)
            if distance < best_distance:
                best_distance = distance
                best_interval = rec["arrival_interval"]
    calibration_path = BASE_DIR / "calibration" / "calibration_results.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration_results, indent=2, default=str), encoding="utf-8")
    print(f"[calibrate] selected arrival_interval={best_interval}")
    return best_interval


def run_phase(phase: str, arrival_interval: Optional[float] = None) -> List[Path]:
    if phase == "micro":
        seeds = [3030]
        warmup = 100
        requests = 200
    elif phase == "smoke":
        seeds = [3030]
        warmup = 500
        requests = 1000
        if arrival_interval is None:
            arrival_interval = _calibrate_arrival_interval(seed=3030)
    elif phase == "pilot":
        seeds = [3030, 4040, 5050, 6060, 7070]
        warmup = 1000
        requests = 5000
        if arrival_interval is None:
            # Try to load calibrated interval from smoke.
            calib_path = BASE_DIR / "calibration" / "calibration_results.json"
            if calib_path.exists():
                recs = json.loads(calib_path.read_text(encoding="utf-8"))
                # Choose interval closest to 10% blocking.
                best = min((r for r in recs if r["ksp_blocking"] is not None),
                           key=lambda r: abs(r["ksp_blocking"] - 0.10))
                arrival_interval = best["arrival_interval"]
            else:
                arrival_interval = 0.04
    else:
        raise ValueError(f"Unknown phase: {phase}")

    result_paths: List[Path] = []
    r_modes = "ksp_ff_highest,ppo_r_top1,strict_v13,v135_afterstate,v135_afterstate_explicit"
    for seed in seeds:
        output_dir = BASE_DIR / phase / f"seed_{seed}"
        _run_eval(seed, arrival_interval, warmup, requests, r_modes, output_dir)
        result_paths.append(output_dir / "results.json")

    # Record chosen interval for the phase.
    meta = {
        "phase": phase,
        "arrival_interval": arrival_interval,
        "num_slots": NUM_SLOTS,
        "holding_min": HOLDING_MIN,
        "holding_max": HOLDING_MAX,
        "deadline_min": DEADLINE_MIN,
        "deadline_max": DEADLINE_MAX,
        "size_min_mb": SIZE_MIN_MB,
        "size_max_mb": SIZE_MAX_MB,
        "edge_cost_min": EDGE_COST_MIN,
        "edge_cost_max": EDGE_COST_MAX,
        "transfer_diagnostic_only": True,
    }
    (BASE_DIR / phase / "external_od_params.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return result_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["micro", "smoke", "pilot"])
    parser.add_argument("--arrival_interval", type=float, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    t0 = time.perf_counter()
    paths = run_phase(args.phase, arrival_interval=args.arrival_interval)
    elapsed = time.perf_counter() - t0
    print(f"[external-od] {args.phase} completed in {elapsed:.1f}s ({len(paths)} result files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
