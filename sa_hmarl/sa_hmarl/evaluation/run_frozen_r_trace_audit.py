"""Runner for the frozen-trace R-only discriminative validation audit.

Generates frozen C-side traces from three origins and replays them under five
R-side methods, gating micro -> smoke -> pilot.

Parallelism is two-level:
  1. Seeds/origins are processed in parallel to reduce wall time.
  2. For each trace, the five R-method replays run in parallel, because replay
     is the dominant cost (each method starts from a fresh environment).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, List, Tuple


ORIGINS = ("ksp_ff_highest", "ppo_r_top1", "strict_v13")
R_MODES = ("ksp_ff_highest", "ppo_r_top1", "strict_v13", "v135_afterstate", "v135_afterstate_explicit")
RANKER_CKPTS = {
    "strict_v13": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/strict_v13/seed_42/ranking_model.pt",
    "v135_afterstate": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate/seed_42/ranking_model.pt",
    "v135_afterstate_explicit": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate_explicit/seed_42/ranking_model.pt",
}
AGENT_R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"
AGENT_C_CKPT = "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt"
BASE_DIR = Path("sa_hmarl/experiments/v135_r_only_frozen_trace_audit")


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    return env


def _run_generate(
    origin_r_mode: str,
    seed: int,
    warmup: int,
    requests: int,
    trace_dir: Path,
) -> Path:
    trace_path = trace_dir / f"origin_{origin_r_mode}_seed_{seed}.json"
    if trace_path.exists():
        return trace_path

    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.generate_frozen_r_trace",
        "--seed", str(seed),
        "--c_mode", "df_c",
        "--origin_r_mode", origin_r_mode,
        "--agent_r_checkpoint", AGENT_R_CKPT,
        "--c_checkpoint", AGENT_C_CKPT,
        "--warmup_requests", str(warmup),
        "--requests_per_episode", str(requests),
        "--poisson_arrivals",
        "--exponential_holding",
        "--output_dir", str(trace_dir),
        "--output_json", trace_path.name,
    ]
    if origin_r_mode in ("strict_v13", "v135_afterstate", "v135_afterstate_explicit"):
        cmd.extend(["--ranker_spec", f"{origin_r_mode}={RANKER_CKPTS[origin_r_mode]}"])

    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[3], env=_env(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return trace_path


def _run_single_replay(
    trace_path: Path,
    r_mode: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    if result_path.exists():
        return result_path

    ranker_specs = []
    for name, path in RANKER_CKPTS.items():
        ranker_specs.extend(["--ranker_specs", f"{name}={path}"])

    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.replay_frozen_r_trace",
        "--trace", str(trace_path),
        "--agent_r_checkpoint", AGENT_R_CKPT,
        *ranker_specs,
        "--r_mode", r_mode,
        "--output_dir", str(output_dir),
        "--output_json", "results.json",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[3], env=_env(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result_path


def _merge_replay_outputs(
    trace_path: Path,
    r_mode_result_paths: Dict[str, Path],
    final_output_dir: Path,
) -> Path:
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_output_dir / "results.json"
    if final_path.exists():
        return final_path

    rows = []
    request_trace_hash = None
    c_context_hash = None
    config_hash = None
    trace_origin = None
    schema_version = None
    for r_mode in R_MODES:
        path = r_mode_result_paths.get(r_mode)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing replay output for {r_mode}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema_version = payload.get("schema_version", schema_version)
        request_trace_hash = payload.get("request_trace_hash", request_trace_hash)
        c_context_hash = payload.get("c_context_hash", c_context_hash)
        config_hash = payload.get("config_hash", config_hash)
        trace_origin = payload.get("trace_origin", trace_origin)
        for row in payload.get("results", []):
            rows.append(row)

    merged = {
        "schema_version": schema_version,
        "trace_path": str(trace_path),
        "trace_origin": trace_origin,
        "request_trace_hash": request_trace_hash,
        "c_context_hash": c_context_hash,
        "config_hash": config_hash,
        "results": rows,
    }
    final_path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    return final_path


def _worker_task(args: Tuple[str, int, int, int]) -> Dict[str, Any]:
    phase, origin_r_mode, seed, warmup, requests = args
    trace_dir = BASE_DIR / "traces" / phase
    trace_path = _run_generate(origin_r_mode, seed, warmup, requests, trace_dir)

    # Replay each R method in parallel using subprocesses (avoids nested
    # ProcessPoolExecutor issues inside the already-spawned worker process).
    final_output_dir = BASE_DIR / phase / f"origin_{origin_r_mode}" / f"seed_{seed}"
    r_mode_paths: Dict[str, Path] = {}
    r_mode_dirs = {r: final_output_dir / f"rmode_{r}" for r in R_MODES}
    ranker_specs = []
    for name, path in RANKER_CKPTS.items():
        ranker_specs.extend(["--ranker_specs", f"{name}={path}"])

    procs = []
    for r_mode in R_MODES:
        out_dir = r_mode_dirs[r_mode]
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "sa_hmarl.evaluation.replay_frozen_r_trace",
            "--trace", str(trace_path),
            "--agent_r_checkpoint", AGENT_R_CKPT,
            *ranker_specs,
            "--r_mode", r_mode,
            "--output_dir", str(out_dir),
            "--output_json", "results.json",
        ]
        procs.append((r_mode, subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[3], env=_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))

    for r_mode, proc in procs:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"replay failed for {r_mode} with exit code {rc}")
        r_mode_paths[r_mode] = r_mode_dirs[r_mode] / "results.json"

    result_path = _merge_replay_outputs(trace_path, r_mode_paths, final_output_dir)
    return {
        "phase": phase,
        "origin": origin_r_mode,
        "seed": seed,
        "trace_path": str(trace_path),
        "result_path": str(result_path),
    }


def run_phase(phase: str, max_workers: int = 3) -> List[Path]:
    if phase == "micro":
        seeds = [3030]
        warmup = 5
        requests = 20
    elif phase == "smoke":
        seeds = [3030]
        warmup = 20
        requests = 100
    elif phase == "pilot":
        seeds = [3030, 4040, 5050, 6060, 7070]
        warmup = 1000
        requests = 5000
    else:
        raise ValueError(f"Unknown phase: {phase}")

    tasks = [(phase, origin, seed, warmup, requests) for seed in seeds for origin in ORIGINS]
    result_paths: List[Path] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker_task, t): t for t in tasks}
        while futures:
            done, _ = wait(list(futures.keys()), timeout=60, return_when=FIRST_COMPLETED)
            for fut in done:
                task = futures.pop(fut)
                try:
                    res = fut.result()
                    result_paths.append(Path(res["result_path"]))
                    print(f"[runner] done {task[1]} seed={task[2]} -> {res['result_path']} ({time.perf_counter()-start:.0f}s)", flush=True)
                except Exception as exc:
                    print(f"[runner] FAILED {task[1]} seed={task[2]}: {exc}", flush=True)
            completed = sum(1 for p in result_paths if p.exists())
            print(f"[runner] progress {completed}/{len(tasks)} ({time.perf_counter()-start:.0f}s)", flush=True)
    return result_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["micro", "smoke", "pilot"])
    parser.add_argument("--max_workers", type=int, default=3)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    t0 = time.perf_counter()
    paths = run_phase(args.phase, max_workers=args.max_workers)
    elapsed = time.perf_counter() - t0
    print(f"[runner] {args.phase} completed in {elapsed:.1f}s ({len(paths)} result files)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
