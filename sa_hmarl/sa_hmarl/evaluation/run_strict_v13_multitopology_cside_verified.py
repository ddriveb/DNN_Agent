"""Run strict v1.3 multi-topology C-side fair evaluation (verified).

Supports micro-E2E, smoke and full phases with source-manifest resume,
smoke-validation marker, and acceptance validation.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None


# Topology protocol table.
TOPOLOGIES = {
    "xlron_cost239_ptrnet_real": {
        "arrival_interval": 0.0625,
        "edge_cost_max": 2.2,
    },
    "xlron_german17": {
        "arrival_interval": 0.07142857142857142,
        "edge_cost_max": 3.0,
    },
    "xlron_nsfnet_deeprmsa": {
        "arrival_interval": 0.07692307692307693,
        "edge_cost_max": 4.0,
    },
    "xlron_jpn48": {
        "arrival_interval": 0.1,
        "edge_cost_max": 5.2,
    },
}

SEEDS = [3030, 4040, 5050, 6060, 7070]
SCHEMA_VERSION = "v1.3-verified-2026-07-14"
RANKER_CKPT = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt"
OLD_RANKER_CKPT = "sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt"
V135_RANKER_CKPT = "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate/seed_42/ranking_model.pt"
V135_EXPLICIT_RANKER_CKPT = "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate_explicit/seed_42/ranking_model.pt"
AGENT_R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"
AGENT_C_COST239_CKPT = "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt"
AGENT_C_TRANSFER_CKPT = "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt"
BASE_DIR = Path("sa_hmarl/experiments/v13_strict_multitopology_cside_verified_v2")

SOURCE_MANIFEST_FILES = [
    "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_verified.py",
    "sa_hmarl/sa_hmarl/evaluation/run_strict_v13_multitopology_cside_verified.py",
    "sa_hmarl/sa_hmarl/evaluation/analyze_strict_v13_multitopology_cside_verified.py",
    "sa_hmarl/sa_hmarl/evaluation/diagnose_r_action_horizon_oracle.py",
    "sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py",
    "sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py",
    "sa_hmarl/sa_hmarl/network/ksp.py",
    "sa_hmarl/sa_hmarl/env/event_env.py",
    "sa_hmarl/sa_hmarl/env/action_mask.py",
    "sa_hmarl/sa_hmarl/env/observation_builder.py",
    "sa_hmarl/sa_hmarl/agents/r_agent.py",
    "sa_hmarl/sa_hmarl/agents/ppo_agents.py",
    "sa_hmarl/sa_hmarl/agents/c_agent.py",
    "sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py",
    "sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py",
    "sa_hmarl/sa_hmarl/evaluation/r_poststate_features.py",
]


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
    manifest: Dict[str, str] = {}
    for rel in SOURCE_MANIFEST_FILES:
        p = root / rel
        manifest[rel] = _sha256(str(p)) if p.exists() else "missing"
    return manifest


def _c_checkpoint_for_mode(c_mode: str, topology: str) -> str:
    if c_mode.startswith("df") or c_mode.startswith("rf") or c_mode in ("wo", "greedy", "iwd"):
        return "_heuristic_"
    if topology == "xlron_cost239_ptrnet_real" and c_mode == "ppo_c":
        return AGENT_C_COST239_CKPT
    if c_mode == "ppo_c_snap24" and topology == "xlron_cost239_ptrnet_real":
        return AGENT_C_TRANSFER_CKPT
    if c_mode == "ppo_c":
        return AGENT_C_TRANSFER_CKPT
    if c_mode == "ppo_c_snap24":
        return AGENT_C_TRANSFER_CKPT
    return "_none_"


def _task_dir(topology: str, seed: int, phase: str, base_dir: Path) -> Path:
    return base_dir / phase / topology / f"seed_{seed}"


def _expected_hashes(
    task: Dict[str, Any],
    root: Path,
    ranker_specs: List[str],
    agent_r_ckpt: str,
    r_modes: str,
    source_manifest: Dict[str, str],
) -> Dict[str, str]:
    """Compute expected hashes for resume validation."""
    c_modes = ",".join(task["c_modes"])
    c_checkpoints = ",".join(task["c_checkpoints"])
    runner_hash = _code_hash(Path(__file__).resolve())
    analyzer_hash = _code_hash(root / "sa_hmarl/sa_hmarl/evaluation/analyze_strict_v13_multitopology_cside_verified.py")

    args_dict = {
        "topology": task["topology"],
        "seed": task["seed"],
        "c_modes": c_modes,
        "c_checkpoints": c_checkpoints,
        "r_modes": r_modes,
        "ranker_specs": ranker_specs,
        "agent_r_checkpoint": str(root / agent_r_ckpt),
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
        "runner_code_hash": runner_hash,
        "analyzer_code_hash": analyzer_hash,
    }

    c_checkpoint_sha256s: Dict[str, str] = {}
    for mode in task["c_modes"]:
        ckpt = _c_checkpoint_for_mode(mode, task["topology"])
        c_checkpoint_sha256s[mode] = _sha256(str(root / ckpt)) if not ckpt.startswith("_") else "heuristic"

    ranker_sha256s: Dict[str, str] = {}
    for spec in ranker_specs:
        name, path = spec.split("=", 1)
        ranker_sha256s[name] = _sha256(path)

    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": _hash_text(json.dumps(args_dict, sort_keys=True, default=str)),
        "runner_code_hash": _code_hash(Path(__file__).resolve()),
        "evaluator_code_hash": _code_hash(root / "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_verified.py"),
        "analyzer_code_hash": analyzer_hash,
        "source_manifest": source_manifest,
        "agent_r_sha256": _sha256(str(root / agent_r_ckpt)),
        "c_checkpoint_sha256s": c_checkpoint_sha256s,
        "ranker_sha256s": ranker_sha256s,
    }


def _validate_payload(payload: Dict[str, Any], expected: Dict[str, str]) -> Tuple[bool, str]:
    """Validate a results.json payload against expected hashes."""
    if payload.get("schema_version") != expected["schema_version"]:
        return False, f"schema_version mismatch: {payload.get('schema_version')}"
    if payload.get("config_hash") != expected["config_hash"]:
        return False, "config_hash mismatch"
    if payload.get("evaluator_code_hash") != expected["evaluator_code_hash"]:
        return False, "evaluator_code_hash mismatch"
    if payload.get("runner_code_hash") != expected["runner_code_hash"]:
        return False, "runner_code_hash mismatch"
    if payload.get("analyzer_code_hash") != expected["analyzer_code_hash"]:
        return False, "analyzer_code_hash mismatch"
    if payload.get("source_manifest") != expected["source_manifest"]:
        return False, "source_manifest mismatch"

    trace_hash = payload.get("request_trace_hash")
    if not trace_hash:
        return False, "missing request_trace_hash"
    for row in payload.get("results", []):
        if row.get("request_trace_hash") != trace_hash:
            return False, "request_trace_hash inconsistent across rows"

    ckpts = payload.get("checkpoint_sha256s", {})
    if ckpts.get("agent_r") != expected["agent_r_sha256"]:
        return False, "agent_r checkpoint SHA mismatch"
    for mode, sha in expected["c_checkpoint_sha256s"].items():
        if ckpts.get(mode) != sha:
            return False, f"C checkpoint SHA mismatch for {mode}"
    for name, sha in expected["ranker_sha256s"].items():
        if ckpts.get(name) != sha:
            return False, f"ranker checkpoint SHA mismatch for {name}"
    return True, "ok"


def _is_done(task_dir: Path, expected: Dict[str, str]) -> bool:
    json_path = task_dir / "results.json"
    done_path = task_dir / "done.marker"
    if not json_path.exists() or not done_path.exists():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    ok, _ = _validate_payload(payload, expected)
    return ok


def _build_cmd(task: Dict[str, Any], root: Path, r_modes: str, ranker_specs: List[str], agent_r: str, runner_hash: str, analyzer_hash: str) -> List[str]:
    return [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.eval_strict_v13_multitopology_cside_verified",
        "--topology", task["topology"],
        "--seed", str(task["seed"]),
        "--c_modes", ",".join(task["c_modes"]),
        "--c_checkpoints", ",".join(task["c_checkpoints"]),
        "--r_modes", r_modes,
        "--agent_r_checkpoint", str(root / agent_r),
        *([arg for spec in ranker_specs for arg in ("--ranker_specs", spec)]),
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
        "--output_dir", str(task["task_dir"]),
        "--output_json", "results.json",
        "--output_log", "eval.log",
        "--runner_code_hash", runner_hash,
        "--analyzer_code_hash", analyzer_hash,
    ]


def _run_task(
    task: Dict[str, Any],
    root: Path,
    r_modes: str,
    ranker_specs: List[str],
    agent_r: str,
    expected: Dict[str, str],
    runner_hash: str,
    analyzer_hash: str,
) -> Dict[str, Any]:
    task_dir = task["task_dir"]
    task_dir.mkdir(parents=True, exist_ok=True)
    json_path = task_dir / "results.json"
    log_path = task_dir / "eval.log"

    # Accept existing validated result without rerun.
    if _is_done(task_dir, expected):
        return {
            "topology": task["topology"],
            "seed": task["seed"],
            "status": "done",
            "task_dir": str(task_dir),
            "from_resume": True,
        }

    cmd = _build_cmd(task, root, r_modes, ranker_specs, agent_r, runner_hash, analyzer_hash)
    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"

    t0 = time.perf_counter()
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

        # Acceptance validation.
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        ok, reason = _validate_payload(payload, expected)
        if not ok:
            raise RuntimeError(f"Acceptance validation failed: {reason}")

        elapsed = time.perf_counter() - t0
        return {
            "topology": task["topology"],
            "seed": task["seed"],
            "status": "done",
            "elapsed_sec": elapsed,
            "task_dir": str(task_dir),
            "from_resume": False,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log_path.write_text(
            log_path.read_text(encoding="utf-8") + f"\n[runner error] {exc}\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        return {
            "topology": task["topology"],
            "seed": task["seed"],
            "status": "failed",
            "error": str(exc),
            "elapsed_sec": elapsed,
            "task_dir": str(task_dir),
        }


def _build_tasks(
    topologies: List[str],
    seeds: List[int],
    c_modes: List[str],
    warmup: int,
    requests: int,
    base_dir: Path,
    phase: str,
) -> List[Dict[str, Any]]:
    tasks = []
    for topology in topologies:
        cfg = TOPOLOGIES[topology]
        c_checkpoints = [_c_checkpoint_for_mode(m, topology) for m in c_modes]
        for seed in seeds:
            tasks.append({
                "topology": topology,
                "seed": seed,
                "c_modes": c_modes,
                "c_checkpoints": c_checkpoints,
                "arrival_interval": cfg["arrival_interval"],
                "edge_cost_max": cfg["edge_cost_max"],
                "warmup_requests": warmup,
                "requests_per_episode": requests,
                "task_dir": _task_dir(topology, seed, phase, base_dir),
                "phase": phase,
            })
    return tasks


def _run_phase(
    tasks: List[Dict[str, Any]],
    root: Path,
    r_modes: str,
    ranker_specs: List[str],
    agent_r: str,
    max_workers: int,
    source_manifest: Dict[str, str],
) -> List[Dict[str, Any]]:
    expected_list = [
        _expected_hashes(
            t, root, ranker_specs, agent_r, r_modes, source_manifest
        )
        for t in tasks
    ]
    pending = [(t, e) for t, e in zip(tasks, expected_list) if not _is_done(t["task_dir"], e)]
    total = len(tasks)
    print(f"[runner] phase: {total} tasks, {len(pending)} pending, max_workers={max_workers}")
    if not pending:
        return []

    results: List[Dict[str, Any]] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_task, t, root, r_modes, ranker_specs, agent_r, e, e["runner_code_hash"], e["analyzer_code_hash"]): t
            for t, e in pending
        }
        while futures:
            from concurrent.futures import wait, FIRST_COMPLETED
            done, _ = wait(list(futures.keys()), timeout=60, return_when=FIRST_COMPLETED)
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


def _ranker_specs(root: Path, r_modes: str) -> List[str]:
    modes = {m.strip() for m in r_modes.split(",") if m.strip()}
    specs: List[str] = []
    if "strict_v13" in modes:
        specs.append(f"strict_v13={root / RANKER_CKPT}")
    if "old_v13" in modes:
        specs.append(f"old_v13={root / OLD_RANKER_CKPT}")
    if "v135_afterstate" in modes:
        specs.append(f"v135_afterstate={root / V135_RANKER_CKPT}")
    if "v135_afterstate_explicit" in modes:
        specs.append(f"v135_afterstate_explicit={root / V135_EXPLICIT_RANKER_CKPT}")
    return specs


def run_experiment(
    phase: str = "full",
    max_workers: Optional[int] = None,
    topologies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    c_modes: Optional[List[str]] = None,
    r_modes: Optional[str] = None,
    warmup: Optional[int] = None,
    requests: Optional[int] = None,
    base_dir: Optional[Path] = None,
    ranker_ckpt: Optional[str] = None,
    old_ranker_ckpt: Optional[str] = None,
    v135_ranker_ckpt: Optional[str] = None,
    v135_explicit_ranker_ckpt: Optional[str] = None,
    agent_r_ckpt: Optional[str] = None,
) -> int:
    root = Path(__file__).resolve().parents[3]
    global RANKER_CKPT, OLD_RANKER_CKPT, V135_RANKER_CKPT, V135_EXPLICIT_RANKER_CKPT

    base = base_dir or BASE_DIR
    base.mkdir(parents=True, exist_ok=True)

    if topologies is None:
        if phase == "micro":
            topologies = ["xlron_cost239_ptrnet_real"]
        else:
            topologies = list(TOPOLOGIES.keys())
    if seeds is None:
        if phase in ("micro", "smoke"):
            seeds = [3030]
        else:
            seeds = SEEDS
    if c_modes is None:
        c_modes = ["ppo_c", "df_c"]
    if r_modes is None:
        r_modes = "ppo_r_top1,ksp_ff_highest,strict_v13"
    if warmup is None:
        warmup = {"micro": 5, "smoke": 20, "full": 1000}.get(phase, 1000)
    if requests is None:
        requests = {"micro": 20, "smoke": 50, "full": 5000}.get(phase, 5000)

    ranker_ckpt = ranker_ckpt or RANKER_CKPT
    old_ranker_ckpt = old_ranker_ckpt or OLD_RANKER_CKPT
    v135_ranker_ckpt = v135_ranker_ckpt or V135_RANKER_CKPT
    v135_explicit_ranker_ckpt = v135_explicit_ranker_ckpt or V135_EXPLICIT_RANKER_CKPT
    agent_r_ckpt = agent_r_ckpt or AGENT_R_CKPT

    RANKER_CKPT = ranker_ckpt
    OLD_RANKER_CKPT = old_ranker_ckpt
    V135_RANKER_CKPT = v135_ranker_ckpt
    V135_EXPLICIT_RANKER_CKPT = v135_explicit_ranker_ckpt

    source_manifest = _source_manifest(root)
    specs = _ranker_specs(root, r_modes)

    if max_workers is None:
        phys = _physical_cores()
        max_workers = min(phys - 1, len(topologies) * len(seeds))
        max_workers = max(1, max_workers)
        if psutil is not None:
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            if avail_gb < 12 and max_workers > 4:
                max_workers = 4
            if avail_gb < 8 and max_workers > 3:
                max_workers = 3

    tasks = _build_tasks(topologies, seeds, c_modes, warmup, requests, base, phase)
    phase_results = _run_phase(
        tasks, root, r_modes, specs, agent_r_ckpt, max_workers,
        source_manifest,
    )

    phase_dir = base / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    phase_file = phase_dir / f"{phase}_phase_results.json"
    phase_file.write_text(json.dumps(phase_results, indent=2, default=str), encoding="utf-8")

    failed = [r for r in phase_results if r.get("status") != "done"]
    if failed:
        print(f"[runner] {len(failed)} tasks failed. See per-task logs.")
        return 1

    # Smoke validation marker.
    if phase == "smoke":
        smoke_matrix = {}
        for t in tasks:
            task_results = [r for r in phase_results
                            if r.get("topology") == t["topology"]
                            and r.get("seed") == t["seed"]
                            and r.get("status") == "done"]
            smoke_matrix[f"{t['topology']}|seed_{t['seed']}"] = {
                "status": "done" if task_results else "failed",
                "task_dir": str(t["task_dir"]),
            }
        marker = {
            "schema_version": SCHEMA_VERSION,
            "phase": "smoke",
            "topologies": topologies,
            "seeds": seeds,
            "c_modes": c_modes,
            "r_modes": r_modes,
            "warmup": warmup,
            "requests": requests,
            "env_params": {
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
                "holding_min": 20.0,
                "holding_max": 30.0,
                "deadline_min": 30.0,
                "deadline_max": 100.0,
                "size_min_mb": 5.0,
                "size_max_mb": 30.0,
                "edge_cost_min": 0.1,
            },
            "topology_params": {t: TOPOLOGIES[t] for t in topologies},
            "source_manifest": source_manifest,
            "code_hashes": {
                "evaluator": _code_hash(root / "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_verified.py"),
                "runner": _code_hash(Path(__file__).resolve()),
                "analyzer": _code_hash(root / "sa_hmarl/sa_hmarl/evaluation/analyze_strict_v13_multitopology_cside_verified.py"),
            },
            "checkpoint_sha256s": {
                "agent_r": _sha256(str(root / agent_r_ckpt)),
                **{
                    spec.split("=", 1)[0]: _sha256(spec.split("=", 1)[1])
                    for spec in specs
                },
            },
            "c_checkpoint_sha256s": {
                m: (_sha256(str(root / _c_checkpoint_for_mode(m, "xlron_cost239_ptrnet_real")))
                    if not _c_checkpoint_for_mode(m, "xlron_cost239_ptrnet_real").startswith("_") else "heuristic")
                for m in c_modes
            },
            "smoke_matrix": smoke_matrix,
            "smoke_acceptance": len(failed) == 0,
            "request_trace_hashes": {},
        }
        marker_path = base / "smoke" / "smoke_validated.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker, indent=2, default=str), encoding="utf-8")
        print(f"[runner] smoke marker written to {marker_path}")

    print(f"[runner] {phase} phase completed.")
    return 0


def _check_smoke_marker(
    base: Path,
    current_manifest: Dict[str, str],
    ranker_specs: List[str],
    agent_r_ckpt: str,
    topologies: List[str],
    seeds: List[int],
    c_modes: List[str],
    r_modes: str,
    warmup: int,
    requests: int,
) -> Tuple[bool, str]:
    marker_path = base / "smoke" / "smoke_validated.json"
    if not marker_path.exists():
        return False, "smoke marker missing"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != SCHEMA_VERSION:
        return False, "schema_version mismatch"
    if marker.get("topologies") != topologies:
        return False, "topology list mismatch"
    # Smoke seeds may be a subset of full seeds; do not require equality.
    if marker.get("c_modes") != c_modes:
        return False, "c_mode list mismatch"
    if marker.get("r_modes") != r_modes:
        return False, "r_mode list mismatch"
    # Warmup/request counts differ between smoke and full by design; record only.
    if marker.get("source_manifest") != current_manifest:
        return False, "source_manifest mismatch"

    root = base.parents[2]
    ckpts = marker.get("checkpoint_sha256s", {})
    if ckpts.get("agent_r") != _sha256(str(root / agent_r_ckpt)):
        return False, "agent_r checkpoint mismatch"
    for spec in ranker_specs:
        name, path = spec.split("=", 1)
        if ckpts.get(name) != _sha256(path):
            return False, f"{name} checkpoint mismatch"

    code_hashes = marker.get("code_hashes", {})
    expected = {
        "evaluator": _code_hash(root / "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_verified.py"),
        "runner": _code_hash(Path(__file__).resolve()),
        "analyzer": _code_hash(root / "sa_hmarl/sa_hmarl/evaluation/analyze_strict_v13_multitopology_cside_verified.py"),
    }
    for k, v in expected.items():
        if code_hashes.get(k) != v:
            return False, f"{k} code hash mismatch"

    if not marker.get("smoke_acceptance", False):
        return False, "smoke phase not accepted"
    return True, "ok"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="full", choices=["micro", "smoke", "full"])
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--topologies", type=str, default=None, help="Comma-separated list")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list")
    parser.add_argument("--c_modes", type=str, default=None, help="Comma-separated list")
    parser.add_argument("--r_modes", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--ranker_ckpt", type=str, default=None)
    parser.add_argument("--old_ranker_ckpt", type=str, default=None)
    parser.add_argument("--v135_ranker_ckpt", type=str, default=None)
    parser.add_argument("--v135_explicit_ranker_ckpt", type=str, default=None)
    parser.add_argument("--agent_r_ckpt", type=str, default=None)
    return parser


def main() -> int:
    global RANKER_CKPT, OLD_RANKER_CKPT, V135_RANKER_CKPT, V135_EXPLICIT_RANKER_CKPT

    parser = build_parser()
    args = parser.parse_args()

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()] if args.topologies else None
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
    c_modes = [m.strip() for m in args.c_modes.split(",") if m.strip()] if args.c_modes else None
    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    ranker_ckpt = args.ranker_ckpt or RANKER_CKPT
    old_ranker_ckpt = args.old_ranker_ckpt or OLD_RANKER_CKPT
    v135_ranker_ckpt = args.v135_ranker_ckpt or V135_RANKER_CKPT
    v135_explicit_ranker_ckpt = args.v135_explicit_ranker_ckpt or V135_EXPLICIT_RANKER_CKPT
    agent_r_ckpt = args.agent_r_ckpt or AGENT_R_CKPT

    # Resolve r_modes default and ranker specs before the smoke check.
    r_modes = args.r_modes if args.r_modes is not None else "ppo_r_top1,ksp_ff_highest,strict_v13"
    root = Path(__file__).resolve().parents[3]

    RANKER_CKPT = ranker_ckpt
    OLD_RANKER_CKPT = old_ranker_ckpt
    V135_RANKER_CKPT = v135_ranker_ckpt
    V135_EXPLICIT_RANKER_CKPT = v135_explicit_ranker_ckpt
    ranker_specs = _ranker_specs(root, r_modes)

    if args.phase == "full":
        manifest = _source_manifest(root)
        ok, reason = _check_smoke_marker(
            base_dir, manifest, ranker_specs, agent_r_ckpt,
            topologies if topologies is not None else list(TOPOLOGIES.keys()),
            seeds if seeds is not None else SEEDS,
            c_modes if c_modes is not None else ["ppo_c", "df_c"],
            r_modes,
            args.warmup if args.warmup is not None else 1000,
            args.requests if args.requests is not None else 5000,
        )
        if not ok:
            print(f"[runner] Refusing to start full phase: {reason}")
            return 1

    return run_experiment(
        phase=args.phase,
        max_workers=args.max_workers,
        topologies=topologies,
        seeds=seeds,
        c_modes=c_modes,
        r_modes=r_modes,
        warmup=args.warmup,
        requests=args.requests,
        base_dir=base_dir,
        ranker_ckpt=ranker_ckpt,
        old_ranker_ckpt=old_ranker_ckpt,
        v135_ranker_ckpt=v135_ranker_ckpt,
        v135_explicit_ranker_ckpt=v135_explicit_ranker_ckpt,
        agent_r_ckpt=agent_r_ckpt,
    )


if __name__ == "__main__":
    raise SystemExit(main())
