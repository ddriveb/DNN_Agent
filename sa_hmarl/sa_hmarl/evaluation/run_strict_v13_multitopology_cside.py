"""Run strict v1.3 multi-topology C-side fair evaluation (verified).

Runs smoke and full 4-topology x 5-seed experiments in parallel, supports resume
with strict schema/hash validation, monitors progress, and produces raw results.

Key properties (v1.3 verified):
- Output base dir is `sa_hmarl/experiments/v13_strict_multitopology_cside_verified/`.
- Per-task validation manifest records runner/analyzer/source hashes.
- Resume validates schema_version, config_hash, runner/evaluator/analyzer code
  hashes, source manifest, checkpoint SHA-256s, and request-trace consistency.
- A task is only accepted after its subprocess output passes the same validation.
- A successful smoke phase writes `smoke_validated.json`; the formal phase refuses
  to start unless the marker exists and all hashes match the current configuration.
- Default main matrix: C = ppo_c,df_c; R = ppo_r_top1,ksp_ff_highest,strict_v13.
  old_v13 is optional and not run by default; ppo_c_snap24 is available only via
  the `--c_modes` override (intended for COST239 diagnostic runs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


try:
    import psutil
except ImportError:
    psutil = None


# Topology protocol table.  Default main matrix uses ppo_c/df_c for every
# topology.  The source-matched COST239 PPO-C checkpoint is selected below;
# all other topologies use the snap24 delay-aware checkpoint as a zero-shot
# transfer baseline.  ppo_c_snap24 is intentionally absent here and can be
# requested explicitly with `--c_modes ppo_c,df_c,ppo_c_snap24`.
TOPOLOGIES = {
    "xlron_cost239_ptrnet_real": {
        "arrival_interval": 0.0625,
        "edge_cost_max": 2.2,
        "c_modes": ["ppo_c", "df_c"],
        "c_checkpoints": [
            "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt",
            "_df_",
        ],
    },
    "xlron_german17": {
        "arrival_interval": 0.07142857142857142,
        "edge_cost_max": 3.0,
        "c_modes": ["ppo_c", "df_c"],
        "c_checkpoints": [
            "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt",
            "_df_",
        ],
    },
    "xlron_nsfnet_deeprmsa": {
        "arrival_interval": 0.07692307692307693,
        "edge_cost_max": 4.0,
        "c_modes": ["ppo_c", "df_c"],
        "c_checkpoints": [
            "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt",
            "_df_",
        ],
    },
    "xlron_jpn48": {
        "arrival_interval": 0.1,
        "edge_cost_max": 5.2,
        "c_modes": ["ppo_c", "df_c"],
        "c_checkpoints": [
            "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt",
            "_df_",
        ],
    },
}

SEEDS = [3030, 4040, 5050, 6060, 7070]
SCHEMA_VERSION = "v1.3-verified-2026-07-14"
RANKER_CKPT = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt"
OLD_RANKER_CKPT = "sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt"
AGENT_R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"
BASE_DIR = Path("sa_hmarl/experiments/v13_strict_multitopology_cside_verified")

# Source files whose hashes are recorded for reproducibility audits.
_SOURCE_MANIFEST_PATHS = [
    "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_fair.py",
    "sa_hmarl/sa_hmarl/evaluation/run_strict_v13_multitopology_cside.py",
    "sa_hmarl/sa_hmarl/evaluation/analyze_strict_v13_multitopology_cside.py",
    "sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py",
    "sa_hmarl/sa_hmarl/baselines/ksp.py",
    "sa_hmarl/sa_hmarl/env/action_mask.py",
    "sa_hmarl/sa_hmarl/env/observation_builder.py",
    "sa_hmarl/sa_hmarl/agents/r_agent.py",
    "sa_hmarl/sa_hmarl/agents/ppo_agents.py",
    "sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py",
    "sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py",
]

_EVALUATOR_PATH = Path(__file__).resolve().parent / "eval_strict_v13_multitopology_cside_fair.py"
_ANALYZER_PATH = Path(__file__).resolve().parent / "analyze_strict_v13_multitopology_cside.py"

R_MODES_DEFAULT = "ppo_r_top1,ksp_ff_highest,strict_v13"


def _physical_cores() -> int:
    if psutil is not None:
        return psutil.cpu_count(logical=False) or 4
    return 4


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "N/A"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _code_hash(path: Path) -> str:
    return _sha256(str(path))


def _source_manifest(root: Path) -> Dict[str, str]:
    """Stable manifest of source file hashes."""
    manifest: Dict[str, str] = {}
    for rel in _SOURCE_MANIFEST_PATHS:
        manifest[rel] = _sha256(str(root / rel))
    return manifest


def _task_dir(topology: str, seed: int, smoke: bool) -> Path:
    sub = "smoke" if smoke else "per_task"
    return BASE_DIR / sub / topology / f"seed_{seed}"


def _build_ranker_specs(r_modes: str, root: Path) -> List[str]:
    """Build ranker spec list matching the active R modes."""
    specs: List[str] = []
    for mode in [m.strip() for m in r_modes.split(",") if m.strip()]:
        if mode == "strict_v13":
            specs.append(f"strict_v13={root / RANKER_CKPT}")
        elif mode == "old_v13":
            specs.append(f"old_v13={root / OLD_RANKER_CKPT}")
    return specs


def _expected_hashes(task: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Compute expected config/code/checkpoint/source hashes for validation.

    The config hash must match exactly what the evaluator computes from its
    argparse Namespace. We therefore mirror the command-line arguments the
    runner will pass to the evaluator.
    """
    c_modes = ",".join(task["c_modes"])
    c_checkpoints = ",".join(task["c_checkpoints"])
    args_dict = {
        "topology": task["topology"],
        "seed": task["seed"],
        "c_modes": c_modes,
        "c_checkpoints": c_checkpoints,
        "r_modes": task["r_modes"],
        "ranker_specs": task["ranker_specs"],
        "agent_r_checkpoint": task["agent_r_checkpoint"],
        "num_slots": 320,
        "num_servers": 4,
        "k_paths_c": 5,
        "k_paths_r": 50,
        "path_sort_strategy_c": "hops",
        "path_sort_strategy_r": "hops",
        "block_sort_strategy_c": "start_asc",
        "block_sort_strategy_r": "start_asc",
        "modulation_profile": "default",
        "max_blocks": 10,
        "split_profile": "default3",
        "num_splits": 3,
        "arrival_interval": task["arrival_interval"],
        "holding_min": 20.0,
        "holding_max": 30.0,
        "deadline_min": 30.0,
        "deadline_max": 100.0,
        "size_min_mb": 5.0,
        "size_max_mb": 30.0,
        "edge_cost_min": 0.1,
        "edge_cost_max": task["edge_cost_max"],
        "warmup_requests": task["warmup_requests"],
        "requests_per_episode": task["requests_per_episode"],
        "poisson_arrivals": True,
        "exponential_holding": True,
        "device": "cpu",
        "output_dir": str(task["task_dir"]),
        "output_json": "results.json",
        "output_log": "eval.log",
    }

    checkpoint_sha256s: Dict[str, str] = {"agent_r": _sha256(task["agent_r_checkpoint"])}
    for mode, ckpt in zip(task["c_modes"], task["c_checkpoints"]):
        if ckpt == "_df_" or ckpt.startswith("_"):
            checkpoint_sha256s[mode] = "heuristic"
        else:
            checkpoint_sha256s[mode] = _sha256(str(root / ckpt))
    for spec in task["ranker_specs"]:
        name, path = spec.split("=", 1)
        checkpoint_sha256s[name] = _sha256(path)

    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": _hash_text(json.dumps(args_dict, sort_keys=True, default=str)),
        "runner_code_hash": _code_hash(Path(__file__).resolve()),
        "evaluator_code_hash": _code_hash(_EVALUATOR_PATH),
        "analyzer_code_hash": _code_hash(_ANALYZER_PATH),
        "source_manifest": _source_manifest(root),
        "checkpoint_sha256s": checkpoint_sha256s,
    }


def _check_task_payload(task_dir: Path, expected: Dict[str, Any]) -> Optional[str]:
    """Return None if results.json + validation_manifest match expected, else error."""
    json_path = task_dir / "results.json"
    manifest_path = task_dir / "validation_manifest.json"
    done_path = task_dir / "done.marker"
    if not json_path.exists():
        return "results.json missing"
    if not manifest_path.exists():
        return "validation_manifest.json missing"
    if not done_path.exists():
        return "done.marker missing"
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"JSON parse error: {exc}"

    if payload.get("schema_version") != expected.get("schema_version"):
        return "schema_version mismatch"
    if payload.get("config_hash") != expected.get("config_hash"):
        return "config_hash mismatch"
    # The evaluator writes its own hash as both `code_hash` and `evaluator_code_hash`.
    if payload.get("code_hash") != expected.get("evaluator_code_hash"):
        return "evaluator code_hash mismatch"
    if payload.get("evaluator_code_hash") != expected.get("evaluator_code_hash"):
        return "evaluator_code_hash mismatch"
    if manifest.get("runner_code_hash") != expected.get("runner_code_hash"):
        return "runner_code_hash mismatch"
    if manifest.get("analyzer_code_hash") != expected.get("analyzer_code_hash"):
        return "analyzer_code_hash mismatch"
    if manifest.get("source_manifest") != expected.get("source_manifest"):
        return "source_manifest mismatch"
    if manifest.get("checkpoint_sha256s") != expected.get("checkpoint_sha256s"):
        return "manifest checkpoint_sha256s mismatch"
    if payload.get("checkpoint_sha256s") != expected.get("checkpoint_sha256s"):
        return "payload checkpoint_sha256s mismatch"

    trace_hash = payload.get("request_trace_hash")
    if not trace_hash:
        return "missing request_trace_hash"
    for row in payload.get("results", []):
        if row.get("request_trace_hash") != trace_hash:
            return "result row request_trace_hash mismatch"
    return None


def _is_done(task_dir: Path, expected: Dict[str, Any]) -> bool:
    """A task is only reusable if all hashes, schema, and traces match."""
    return _check_task_payload(task_dir, expected) is None


def _run_task(task: Dict[str, Any], root: Path, smoke: bool) -> Dict[str, Any]:
    task_dir = task["task_dir"]
    task_dir.mkdir(parents=True, exist_ok=True)
    expected = task["expected"]
    manifest_path = task_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps(expected, indent=2, default=str), encoding="utf-8")

    json_path = task_dir / "results.json"
    log_path = task_dir / "eval.log"

    c_modes = ",".join(task["c_modes"])
    c_checkpoints = ",".join(task["c_checkpoints"])
    ranker_specs = task["ranker_specs"]

    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.eval_strict_v13_multitopology_cside_fair",
        "--topology", task["topology"],
        "--seed", str(task["seed"]),
        "--c_modes", c_modes,
        "--c_checkpoints", c_checkpoints,
        "--r_modes", task["r_modes"],
        "--agent_r_checkpoint", task["agent_r_checkpoint"],
    ]
    for spec in ranker_specs:
        cmd.extend(["--ranker_specs", spec])
    cmd.extend([
        "--num_slots", "320",
        "--num_servers", "4",
        "--k_paths_c", "5",
        "--k_paths_r", "50",
        "--path_sort_strategy_c", "hops",
        "--path_sort_strategy_r", "hops",
        "--block_sort_strategy_c", "start_asc",
        "--block_sort_strategy_r", "start_asc",
        "--modulation_profile", "default",
        "--max_blocks", "10",
        "--split_profile", "default3",
        "--num_splits", "3",
        "--arrival_interval", str(task["arrival_interval"]),
        "--holding_min", "20.0",
        "--holding_max", "30.0",
        "--deadline_min", "30.0",
        "--deadline_max", "100.0",
        "--size_min_mb", "5.0",
        "--size_max_mb", "30.0",
        "--edge_cost_min", "0.1",
        "--edge_cost_max", str(task["edge_cost_max"]),
        "--warmup_requests", str(task["warmup_requests"]),
        "--requests_per_episode", str(task["requests_per_episode"]),
        "--poisson_arrivals",
        "--exponential_holding",
        "--device", "cpu",
        "--output_dir", str(task_dir),
        "--output_json", "results.json",
        "--output_log", "eval.log",
    ])

    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    t0 = time.perf_counter()
    max_mem_mb = 0.0
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=root,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=18000,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"Non-zero exit {proc.returncode}")

        err = _check_task_payload(task_dir, expected)
        if err:
            raise RuntimeError(f"Validation failed: {err}")

        elapsed = time.perf_counter() - t0
        if psutil is not None:
            max_mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        return {
            "topology": task["topology"],
            "seed": task["seed"],
            "status": "done",
            "elapsed_sec": elapsed,
            "max_mem_mb": max_mem_mb,
            "task_dir": str(task_dir),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log_path.write_text(
            log_path.read_text(encoding="utf-8") + f"\n[runner error] {exc}\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        # Do not leave a done.marker if validation or execution failed.
        done_path = task_dir / "done.marker"
        if done_path.exists():
            done_path.unlink()
        return {
            "topology": task["topology"],
            "seed": task["seed"],
            "status": "failed",
            "error": str(exc),
            "elapsed_sec": elapsed,
            "task_dir": str(task_dir),
        }


def _build_tasks(
    smoke: bool,
    c_modes_override: Optional[str] = None,
    topologies: Optional[List[str]] = None,
    warmup_requests: Optional[int] = None,
    requests_per_episode: Optional[int] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    root = Path(__file__).resolve().parents[3]
    r_modes = R_MODES_DEFAULT
    ranker_specs = _build_ranker_specs(r_modes, root)
    agent_r_checkpoint = str(root / AGENT_R_CKPT)

    tasks: List[Dict[str, Any]] = []
    selected_topologies = topologies if topologies is not None else list(TOPOLOGIES.keys())
    for topology in selected_topologies:
        cfg = TOPOLOGIES[topology]
        task_seeds = seeds if seeds is not None else ([3030] if smoke else SEEDS)
        task_warmup = warmup_requests if warmup_requests is not None else (20 if smoke else 1000)
        task_requests = requests_per_episode if requests_per_episode is not None else (50 if smoke else 5000)
        c_modes = cfg["c_modes"]
        c_checkpoints = cfg["c_checkpoints"]
        if c_modes_override is not None:
            requested = [m.strip() for m in c_modes_override.split(",") if m.strip()]
            c_modes = requested
            c_checkpoints = []
            for m in requested:
                if m.startswith("df") or m.startswith("rf") or m in ("wo", "greedy", "iwd"):
                    c_checkpoints.append("_df_")
                else:
                    c_checkpoints.append("sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
            # COST239 source-matched PPO-C special case.
            if topology == "xlron_cost239_ptrnet_real" and "ppo_c" in requested:
                idx = requested.index("ppo_c")
                c_checkpoints[idx] = "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt"
        for seed in task_seeds:
            tasks.append({
                "topology": topology,
                "seed": seed,
                "c_modes": c_modes,
                "c_checkpoints": c_checkpoints,
                "r_modes": r_modes,
                "ranker_specs": ranker_specs,
                "agent_r_checkpoint": agent_r_checkpoint,
                "arrival_interval": cfg["arrival_interval"],
                "edge_cost_max": cfg["edge_cost_max"],
                "warmup_requests": task_warmup,
                "requests_per_episode": task_requests,
                "task_dir": _task_dir(topology, seed, smoke),
            })
    return tasks


def _filter_pending(tasks: List[Dict[str, Any]], root: Path) -> List[Dict[str, Any]]:
    pending = []
    for t in tasks:
        expected = _expected_hashes(t, root)
        t["expected"] = expected
        if not _is_done(t["task_dir"], expected):
            pending.append(t)
    return pending


def _wait_any(futures, timeout: float):
    """Return a partition of done futures and still-pending futures."""
    done_set, not_done_set = wait(list(futures.keys()), timeout=timeout, return_when=FIRST_COMPLETED)
    return done_set, not_done_set


def _run_phase(tasks: List[Dict[str, Any]], root: Path, smoke: bool, max_workers: int) -> List[Dict[str, Any]]:
    pending = _filter_pending(tasks, root)
    total = len(tasks)
    print(f"[runner] {'Smoke' if smoke else 'Full'} phase: {total} tasks, {len(pending)} pending, max_workers={max_workers}")
    if not pending:
        return []

    results: List[Dict[str, Any]] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_task, t, root, smoke): t for t in pending}
        while futures:
            done, _ = _wait_any(futures, timeout=60)
            for fut in done:
                task = futures.pop(fut)
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {
                        "topology": task["topology"],
                        "seed": task["seed"],
                        "status": "failed",
                        "error": str(exc),
                    }
                results.append(res)
                print(f"[runner] {res['topology']} seed={res['seed']} -> {res['status']}")
            completed = sum(1 for r in results if r.get("status") == "done")
            failed = sum(1 for r in results if r.get("status") == "failed")
            running = len(futures)
            cpu_pct = psutil.cpu_percent(interval=0.5) if psutil else "N/A"
            print(
                f"[runner] progress: {completed}/{len(pending)} done, {failed} failed, {running} running, "
                f"wall={time.perf_counter()-start:.0f}s, cpu={cpu_pct}%"
            )
    return results


def _smoke_config_snapshot(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Protocol-level config captured in the smoke marker."""
    if not tasks:
        return {}
    return {
        "topologies": sorted({t["topology"] for t in tasks}),
        "c_modes": ",".join(tasks[0]["c_modes"]),
        "r_modes": tasks[0]["r_modes"],
        "ranker_ckpt": RANKER_CKPT,
        "agent_r_checkpoint": AGENT_R_CKPT,
    }


def _check_smoke_marker(tasks: List[Dict[str, Any]], root: Path) -> Optional[str]:
    """Verify the smoke marker exists and matches current source/checkpoint config."""
    marker_path = BASE_DIR / "smoke_validated.json"
    if not marker_path.exists():
        return "smoke_validated.json missing"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"cannot parse smoke marker: {exc}"

    marker_config = marker.get("config", {})
    marker_source = marker.get("source_manifest", {})
    marker_ckpts = marker.get("checkpoint_sha256s", {})

    current_config = _smoke_config_snapshot(tasks)
    if marker_config != current_config:
        return "smoke marker config mismatch"

    current_source = _source_manifest(root)
    if marker_source != current_source:
        return "smoke marker source_manifest mismatch"

    for t in tasks:
        topology = t["topology"]
        expected = _expected_hashes(t, root)
        if marker_ckpts.get(topology) != expected["checkpoint_sha256s"]:
            return f"smoke marker checkpoint_sha256s mismatch for {topology}"
    return None


def _write_smoke_marker(tasks: List[Dict[str, Any]], root: Path) -> None:
    """Write smoke_validated.json after a successful smoke phase."""
    marker_path = BASE_DIR / "smoke_validated.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    request_trace_hashes: Dict[str, Optional[str]] = {}
    checkpoint_sha256s: Dict[str, Dict[str, str]] = {}
    for t in tasks:
        topology = t["topology"]
        expected = _expected_hashes(t, root)
        json_path = t["task_dir"] / "results.json"
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            request_trace_hashes[topology] = payload.get("request_trace_hash")
        checkpoint_sha256s[topology] = expected["checkpoint_sha256s"]

    marker = {
        "schema_version": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": _smoke_config_snapshot(tasks),
        "source_manifest": _source_manifest(root),
        "checkpoint_sha256s": checkpoint_sha256s,
        "request_trace_hashes": request_trace_hashes,
    }
    marker_path.write_text(json.dumps(marker, indent=2, default=str), encoding="utf-8")
    print(f"[runner] Wrote smoke marker {marker_path}")


def run_experiment(
    smoke: bool = False,
    max_workers: Optional[int] = None,
    ranker_ckpt: Optional[str] = None,
    old_ranker_ckpt: Optional[str] = None,
    output_dir: Optional[str] = None,
    r_modes: Optional[str] = None,
    c_modes_override: Optional[str] = None,
    topologies: Optional[List[str]] = None,
    warmup_requests: Optional[int] = None,
    requests_per_episode: Optional[int] = None,
    seeds: Optional[List[int]] = None,
    require_smoke_marker: bool = True,
) -> int:
    global RANKER_CKPT, OLD_RANKER_CKPT, BASE_DIR, R_MODES_DEFAULT
    if ranker_ckpt is not None:
        RANKER_CKPT = ranker_ckpt
    if old_ranker_ckpt is not None:
        OLD_RANKER_CKPT = old_ranker_ckpt
    if output_dir is not None:
        BASE_DIR = Path(output_dir)
    if r_modes is not None:
        R_MODES_DEFAULT = r_modes

    root = Path(__file__).resolve().parents[3]
    tasks = _build_tasks(
        smoke=smoke,
        c_modes_override=c_modes_override,
        topologies=topologies,
        warmup_requests=warmup_requests,
        requests_per_episode=requests_per_episode,
        seeds=seeds,
    )

    if not smoke and require_smoke_marker:
        err = _check_smoke_marker(tasks, root)
        if err:
            print(f"[runner] Formal phase refused: {err}")
            return 1

    if max_workers is None:
        phys = _physical_cores()
        max_workers = min(phys - 1, len(tasks))
        if max_workers < 1:
            max_workers = 1
        if psutil is not None:
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            if avail_gb < 12 and max_workers > 4:
                max_workers = 4
            if avail_gb < 8 and max_workers > 3:
                max_workers = 3

    phase_results = _run_phase(tasks, root, smoke, max_workers)
    phase_file = BASE_DIR / ("smoke_phase_results.json" if smoke else "full_phase_results.json")
    phase_file.parent.mkdir(parents=True, exist_ok=True)
    phase_file.write_text(json.dumps(phase_results, indent=2, default=str), encoding="utf-8")

    failed = [r for r in phase_results if r.get("status") != "done"]
    if failed:
        print(f"[runner] {len(failed)} tasks failed. See per-task logs.")
        return 1

    if smoke:
        _write_smoke_marker(tasks, root)

    print("[runner] All tasks completed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--ranker_ckpt", type=str, default=None, help="Override strict v1.3 ranker checkpoint")
    parser.add_argument("--old_ranker_ckpt", type=str, default=None, help="Override old/legacy ranker checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output base directory")
    parser.add_argument("--r_modes", type=str, default=None, help="Override R modes, e.g. strict_v13")
    parser.add_argument("--c_modes", type=str, default=None, help="Override C modes, e.g. ppo_c,df_c")
    parser.add_argument(
        "--require_smoke_marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Formal phase requires a matching smoke_validated.json (default: True)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_experiment(
        smoke=args.smoke,
        max_workers=args.max_workers,
        ranker_ckpt=args.ranker_ckpt,
        old_ranker_ckpt=args.old_ranker_ckpt,
        output_dir=args.output_dir,
        r_modes=args.r_modes,
        c_modes_override=args.c_modes,
        require_smoke_marker=args.require_smoke_marker,
    )


if __name__ == "__main__":
    raise SystemExit(main())
