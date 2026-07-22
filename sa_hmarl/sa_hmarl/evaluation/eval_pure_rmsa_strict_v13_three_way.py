"""Paper-parity pure RMSA comparison for frozen Strict v1.3 and heuristics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sa_hmarl.baselines.rmsa_baselines import (
    ff_ksp_highest_mod_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.evaluation.optical_only_rmsa_evaluator import load_ppo_r, load_ranker
from sa_hmarl.evaluation.pure_rmsa_paper_env import (
    MEAN_HOLDING_TIME,
    PureRMSAPaperEnv,
    PureRMSARequest,
    generate_paper_requests,
    paper_modulation_registry,
)
from sa_hmarl.network.topology_data import get_topology_edges


PPO_R_CHECKPOINT = "sa_hmarl/checkpoints/agent_r_mixed.pt"
STRICT_RANKER_CHECKPOINT = (
    "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/"
    "full_state_uniform_strict/seed_42/ranking_model.pt"
)
METHODS = ("strict_v13", "ksp_ff_k50_hops", "ff_ksp_k50_hops")
TOPOLOGIES = {
    "cost239": {"key": "cost239_deeprmsa", "paper_load": 600.0},
    "nsfnet": {"key": "xlron_nsfnet_deeprmsa", "paper_load": 250.0},
    "usnet": {"key": "xlron_usnet_gcnrmsa", "paper_load": None},
    "jpn48": {"key": "xlron_jpn48", "paper_load": None},
}
CALIBRATION_GRID = (50.0, 100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0)
CALIBRATION_FINE_GRIDS = {
    "usnet": (425.0, 450.0, 475.0, 500.0, 525.0, 550.0, 575.0, 650.0, 700.0, 750.0, 900.0, 1000.0),
    "jpn48": (325.0, 350.0, 375.0, 425.0, 450.0, 475.0, 500.0, 525.0, 650.0, 700.0),
}
CALIBRATION_TARGETS = (0.01, 0.05, 0.10, 0.20)
C_ONLY_FEATURES = {
    "split_norm", "server_norm", "deadline_norm", "intermediate_size_norm",
    "server_utilization", "k_c_valid_ratio", "k_r_total_ratio", "phi_spec_norm",
}


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _trace_hash(requests: Sequence[PureRMSARequest]) -> str:
    payload = [asdict(req) for req in requests]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _topology_hash(topology: str) -> str:
    return hashlib.sha256(json.dumps(get_topology_edges(topology)).encode()).hexdigest()


def _decode(action: int, num_mods: int, max_blocks: int) -> Tuple[int, int, int]:
    path_idx = int(action) // (num_mods * max_blocks)
    rem = int(action) % (num_mods * max_blocks)
    mod_idx, block_idx = divmod(rem, max_blocks)
    return path_idx, mod_idx, block_idx


def _difference_bucket(left: Optional[int], right: Optional[int], num_mods: int, max_blocks: int) -> str:
    if left is None or right is None:
        return "availability" if left != right else "same"
    if int(left) == int(right):
        return "same"
    lp, lm, lb = _decode(int(left), num_mods, max_blocks)
    rp, rm, rb = _decode(int(right), num_mods, max_blocks)
    changed = []
    if lp != rp:
        changed.append("path")
    if lm != rm:
        changed.append("mod")
    if lb != rb:
        changed.append("block")
    return "+".join(changed)


def _ppo_top30(agent_r: Any, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    features, mask = agent_r.build_action_features(obs)
    features = np.asarray(features, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return np.empty(0, dtype=np.int64), features
    with torch.no_grad():
        logits = agent_r.policy_net(
            torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        ).squeeze(0).cpu().numpy()
    logits = np.asarray(logits, dtype=np.float32)
    top_local = np.argsort(-logits[legal], kind="stable")[:min(30, legal.size)]
    return legal[top_local].astype(np.int64), features


def _strict_feature_matrix(
    obs: Dict[str, Any],
    req: PureRMSARequest,
    candidates: np.ndarray,
    base_features: np.ndarray,
    ranker: Dict[str, Any],
    max_blocks: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_paths = max(len(obs["candidate_paths"]), 1)
    num_mods = max(len(obs["mod_names"]), 1)
    mask = np.asarray(obs["agent_r_mask"], dtype=bool)
    valid_ratio = float(mask.sum()) / max(len(mask), 1)
    rows = []
    for action in candidates:
        path_idx, mod_idx, block_idx = _decode(int(action), num_mods, max_blocks)
        base = base_features[int(action)]
        values = {
            "path_length_km": float(base[0]),
            "hop_count": float(base[1]),
            "lfb": float(base[2]),
            "free_ratio": float(base[3]),
            "frag_index": float(base[4]),
            "spectral_efficiency": float(base[5]),
            "reach_km": float(base[6]),
            "required_fs": float(base[7]),
            "block_size": float(base[8]),
            "block_waste": float(base[9]),
            "path_mod_feasible": float(base[10]),
            "holding_norm": float(req.holding_time) / 10.0,
            "selected_valid_r_ratio": valid_ratio,
            "raw_r_valid_ratio": valid_ratio,
            "path_idx_norm": path_idx / max(num_paths - 1, 1),
            "mod_idx_norm": mod_idx / max(num_mods - 1, 1),
            "block_idx_norm": block_idx / max(max_blocks - 1, 1),
        }
        row = []
        for index, name in enumerate(ranker["feature_names"]):
            if name in C_ONLY_FEATURES:
                row.append(float(ranker["feature_mean"][index]))
            elif name in values:
                row.append(float(values[name]))
            else:
                raise RuntimeError(f"No pure-RMSA mapping for Strict feature {name!r}")
        rows.append(row)
    raw = np.asarray(rows, dtype=np.float32)
    normalized = (raw - ranker["feature_mean"]) / ranker["feature_std"]
    if raw.shape != (len(candidates), 25) or not np.all(np.isfinite(normalized)):
        raise RuntimeError(f"Invalid Strict feature matrix shape={raw.shape}")
    return raw, normalized.astype(np.float32)


def _select_action(
    method: str,
    obs: Dict[str, Any],
    req: PureRMSARequest,
    env: PureRMSAPaperEnv,
    agent_r: Any,
    ranker: Dict[str, Any],
) -> Tuple[Optional[int], Dict[str, Any]]:
    if method == "ksp_ff_k50_hops":
        return ksp_ff_highest_mod_action(obs), {}
    if method == "ff_ksp_k50_hops":
        return ff_ksp_highest_mod_action(obs), {}
    if method != "strict_v13":
        raise ValueError(method)
    candidates, base_features = _ppo_top30(agent_r, obs)
    if candidates.size == 0:
        return None, {"candidate_count": 0}
    raw, normalized = _strict_feature_matrix(
        obs, req, candidates, base_features, ranker, env.max_blocks
    )
    with torch.no_grad():
        scores = ranker["model"](
            torch.as_tensor(normalized, dtype=torch.float32, device=ranker["device"])
        ).cpu().numpy()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Strict ranker emitted NaN/Inf")
    best = int(np.argmax(scores))
    return int(candidates[best]), {
        "candidate_count": int(candidates.size),
        "raw_features": raw,
        "normalized_features": normalized,
        "score_margin": float(np.sort(scores)[-1] - np.sort(scores)[-2]) if len(scores) > 1 else 0.0,
    }


def _update_feature_audit(audit: Dict[str, Any], info: Dict[str, Any]) -> None:
    raw = info.get("raw_features")
    norm = info.get("normalized_features")
    if raw is None or norm is None:
        return
    raw = np.asarray(raw, dtype=float)
    norm = np.asarray(norm, dtype=float)
    if not audit:
        audit.update({
            "count": 0,
            "raw_sum": np.zeros(raw.shape[1]),
            "raw_min": np.full(raw.shape[1], np.inf),
            "raw_max": np.full(raw.shape[1], -np.inf),
            "norm_sum": np.zeros(raw.shape[1]),
            "norm_min": np.full(raw.shape[1], np.inf),
            "norm_max": np.full(raw.shape[1], -np.inf),
            "outside_3sigma": np.zeros(raw.shape[1], dtype=np.int64),
            "nonfinite": np.zeros(raw.shape[1], dtype=np.int64),
        })
    audit["count"] += raw.shape[0]
    audit["raw_sum"] += raw.sum(axis=0)
    audit["raw_min"] = np.minimum(audit["raw_min"], raw.min(axis=0))
    audit["raw_max"] = np.maximum(audit["raw_max"], raw.max(axis=0))
    audit["norm_sum"] += norm.sum(axis=0)
    audit["norm_min"] = np.minimum(audit["norm_min"], norm.min(axis=0))
    audit["norm_max"] = np.maximum(audit["norm_max"], norm.max(axis=0))
    audit["outside_3sigma"] += (np.abs(norm) > 3.0).sum(axis=0)
    audit["nonfinite"] += (~np.isfinite(norm)).sum(axis=0)


def _serialize_feature_audit(audit: Dict[str, Any], ranker: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not audit:
        return []
    count = max(int(audit["count"]), 1)
    rows = []
    for index, name in enumerate(ranker["feature_names"]):
        rows.append({
            "name": name,
            "source": "checkpoint_mean" if name in C_ONLY_FEATURES else "pure_rmsa_dynamic",
            "training_mean": float(ranker["feature_mean"][index]),
            "training_std": float(ranker["feature_std"][index]),
            "raw_min": float(audit["raw_min"][index]),
            "raw_max": float(audit["raw_max"][index]),
            "raw_mean": float(audit["raw_sum"][index] / count),
            "normalized_min": float(audit["norm_min"][index]),
            "normalized_max": float(audit["norm_max"][index]),
            "normalized_mean": float(audit["norm_sum"][index] / count),
            "outside_3sigma_ratio": float(audit["outside_3sigma"][index] / count),
            "nonfinite_count": int(audit["nonfinite"][index]),
        })
    return rows


def run_episode(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    env = PureRMSAPaperEnv(job["topology_key"], num_slots=100, k_paths=50, max_blocks=10)
    requests = generate_paper_requests(
        env.num_nodes, job["seed"], job["warmup"] + job["evaluated"], job["load_erlang"]
    )
    agent_r = None
    ranker = None
    if job["method"] == "strict_v13":
        agent_r = load_ppo_r(PPO_R_CHECKPOINT, paper_modulation_registry(), "cpu")
        ranker = load_ranker(STRICT_RANKER_CHECKPOINT, "cpu")
    env.reset()
    counters = Counter()
    sums = Counter()
    feature_audit: Dict[str, Any] = {}
    trace_rows = []
    start_time = time.perf_counter()
    for index, req in enumerate(requests):
        warmup = index < job["warmup"]
        env.advance_time(req.arrival_time)
        end_to_end_start = time.perf_counter()
        obs = env.build_observation(req)
        decision_start = time.perf_counter()
        action, selector_info = _select_action(job["method"], obs, req, env, agent_r, ranker)
        decision_ms = (time.perf_counter() - decision_start) * 1000.0
        if not warmup and job["method"] == "strict_v13":
            _update_feature_audit(feature_audit, selector_info)
            ksp_shadow = ksp_ff_highest_mod_action(obs)
            ffksp_shadow = ff_ksp_highest_mod_action(obs)
            counters[f"strict_vs_ksp:{_difference_bucket(action, ksp_shadow, env.mod_reg.num_formats, env.max_blocks)}"] += 1
            counters[f"strict_vs_ffksp:{_difference_bucket(action, ffksp_shadow, env.mod_reg.num_formats, env.max_blocks)}"] += 1

        if action is None:
            success = False
            step_info = {"reason": "r_no_valid_action"}
            diagnosis = dict(obs["pure_mask_diagnosis"])
            if job.get("check_k500", False):
                obs500 = env.build_observation(req, k=500)
                rescue = bool(np.asarray(obs500["agent_r_mask"], dtype=bool).any())
                diagnosis["flags"]["k50_path_truncation"] = rescue
                if rescue:
                    diagnosis["primary_reason"] = "k50_path_truncation"
        else:
            if not np.asarray(obs["agent_r_mask"], dtype=bool)[int(action)]:
                raise RuntimeError(f"{job['method']} selected illegal action {action}")
            step_info = env.step(int(action), obs, req)
            success = bool(step_info["success"])
            diagnosis = dict(obs["pure_mask_diagnosis"])
            if not success:
                raise RuntimeError(f"Legal action failed: {step_info}")

        if warmup:
            continue
        spectrum = env.net.spectrum_summary()
        end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0
        counters["evaluated"] += 1
        counters["accepted" if success else "blocked"] += 1
        sums["decision_ms"] += decision_ms
        sums["end_to_end_ms"] += end_to_end_ms
        sums["legal_actions"] += int(np.asarray(obs["agent_r_mask"], dtype=bool).sum())
        sums["spectrum_occupancy"] += spectrum["occupancy"]
        sums["fragmentation"] += spectrum["fragmentation"]
        sums["largest_free_block"] += spectrum["largest_free_block"]
        sums["active_connections"] += len(env.active_connections)
        if not success:
            counters[f"cause:{diagnosis['primary_reason']}"] += 1
            for cause, enabled in diagnosis["flags"].items():
                if enabled:
                    counters[f"flag:{cause}"] += 1
        else:
            counters[f"modulation:{step_info['mod_idx']}"] += 1
            for key in ("path_idx", "mod_idx", "block_idx", "start_slot", "required_fs", "path_hops", "path_km"):
                sums[key] += float(step_info[key])
            sums["block_waste"] += (
                float(step_info["block_size"] - step_info["required_fs"])
                / max(float(step_info["block_size"]), 1.0)
            )
        if job.get("store_trace", False):
            row = {
                "topology": job["topology"], "load_erlang": job["load_erlang"],
                "seed": job["seed"], "method": job["method"], "request_index": index,
                "req_id": req.req_id, "success": success, "action": action,
                "legal_actions": int(np.asarray(obs["agent_r_mask"], dtype=bool).sum()),
                "decision_ms": decision_ms, "failure_reason": None if success else diagnosis["primary_reason"],
                "end_to_end_ms": end_to_end_ms,
            }
            row.update({k: step_info.get(k) for k in ("path_idx", "mod_idx", "block_idx", "start_slot", "required_fs", "path_hops", "path_km")})
            trace_rows.append(row)

    evaluated = max(counters["evaluated"], 1)
    accepted = max(counters["accepted"], 1)
    return {
        "topology": job["topology"],
        "topology_key": job["topology_key"],
        "load_erlang": float(job["load_erlang"]),
        "seed": int(job["seed"]),
        "method": job["method"],
        "warmup": int(job["warmup"]),
        "evaluated": int(counters["evaluated"]),
        "accepted": int(counters["accepted"]),
        "blocked": int(counters["blocked"]),
        "blocking_rate": counters["blocked"] / evaluated,
        "primary_causes": {k.split(":", 1)[1]: v for k, v in counters.items() if k.startswith("cause:")},
        "nonexclusive_flags": {k.split(":", 1)[1]: v for k, v in counters.items() if k.startswith("flag:")},
        "avg_decision_ms": sums["decision_ms"] / evaluated,
        "avg_end_to_end_ms": sums["end_to_end_ms"] / evaluated,
        "avg_legal_actions": sums["legal_actions"] / evaluated,
        "avg_spectrum_occupancy": sums["spectrum_occupancy"] / evaluated,
        "avg_fragmentation": sums["fragmentation"] / evaluated,
        "avg_largest_free_block": sums["largest_free_block"] / evaluated,
        "avg_active_connections": sums["active_connections"] / evaluated,
        "avg_path_idx": sums["path_idx"] / accepted,
        "avg_mod_idx": sums["mod_idx"] / accepted,
        "avg_block_idx": sums["block_idx"] / accepted,
        "avg_start_slot": sums["start_slot"] / accepted,
        "avg_required_fs": sums["required_fs"] / accepted,
        "avg_path_hops": sums["path_hops"] / accepted,
        "avg_path_km": sums["path_km"] / accepted,
        "avg_block_waste": sums["block_waste"] / accepted,
        "modulation_distribution": {
            key.split(":", 1)[1]: value for key, value in counters.items()
            if key.startswith("modulation:")
        },
        "same_state_action_differences": {
            key: value for key, value in counters.items()
            if key.startswith("strict_vs_")
        },
        "request_trace_hash": _trace_hash(requests),
        "feature_audit": (
            _serialize_feature_audit(feature_audit, ranker)
            if job["method"] == "strict_v13" else []
        ),
        "trace_rows": trace_rows,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def _run_job_persisted(job: Dict[str, Any]) -> Dict[str, Any]:
    result_path_value = job.get("result_path")
    if result_path_value:
        result_path = Path(result_path_value)
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
    result = run_episode(job)
    if result_path_value:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        os.replace(temporary, result_path)
    return result


def _run_jobs(jobs: List[Dict[str, Any]], max_workers: int) -> List[Dict[str, Any]]:
    if max_workers <= 1:
        return [_run_job_persisted(job) for job in jobs]
    outputs = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_job_persisted, job): job for job in jobs}
        for future in as_completed(futures):
            output = future.result()
            outputs.append(output)
            print(
                f"[{output['topology']} rho={output['load_erlang']:g} seed={output['seed']} "
                f"{output['method']}] blocking={output['blocking_rate']:.4%}", flush=True
            )
    return outputs


def calibrate_transfer_loads(args: argparse.Namespace) -> Dict[str, Any]:
    calibration_path = Path(args.output_dir) / "LOAD_CALIBRATION.json"
    existing = None
    if calibration_path.exists():
        existing = json.loads(calibration_path.read_text(encoding="utf-8"))
        if existing.get("stage") == "refined":
            return existing
    coarse_runs = existing.get("runs", []) if existing else []
    grids = CALIBRATION_FINE_GRIDS if coarse_runs else {
        topology: CALIBRATION_GRID for topology in ("usnet", "jpn48")
    }
    jobs = []
    for topology, grid in grids.items():
        for load in grid:
            jobs.append({
                "topology": topology, "topology_key": TOPOLOGIES[topology]["key"],
                "load_erlang": load, "seed": args.calibration_seed,
                "warmup": args.calibration_warmup, "evaluated": args.calibration_requests,
                "method": "ksp_ff_k50_hops", "check_k500": False, "store_trace": False,
            })
    new_runs = _run_jobs(jobs, args.max_workers)
    if not coarse_runs:
        interim = {
            "stage": "coarse", "protocol": "Doherty paper traffic; KSP-only transfer-topology calibration",
            "coarse_grid": list(CALIBRATION_GRID), "targets": list(CALIBRATION_TARGETS),
            "seed": args.calibration_seed, "runs": new_runs,
        }
        calibration_path.write_text(json.dumps(interim, indent=2), encoding="utf-8")
        return interim
    runs = coarse_runs + new_runs
    locked: Dict[str, List[float]] = {}
    for topology in ("usnet", "jpn48"):
        rows = sorted((x for x in runs if x["topology"] == topology), key=lambda x: x["load_erlang"])
        chosen = []
        for target in CALIBRATION_TARGETS:
            best = min(rows, key=lambda x: abs(x["blocking_rate"] - target))
            chosen.append(float(best["load_erlang"]))
        locked[topology] = sorted(set(chosen))
    return {
        "stage": "refined",
        "protocol": "Doherty paper traffic; KSP-only transfer-topology calibration",
        "coarse_grid": list(CALIBRATION_GRID),
        "fine_grids": {key: list(value) for key, value in CALIBRATION_FINE_GRIDS.items()},
        "targets": list(CALIBRATION_TARGETS),
        "seed": args.calibration_seed, "runs": runs, "locked_loads": locked,
    }


def _locked_loads(calibration: Dict[str, Any]) -> Dict[str, List[float]]:
    return {
        "cost239": [600.0],
        "nsfnet": [250.0],
        "usnet": list(calibration["locked_loads"]["usnet"]),
        "jpn48": list(calibration["locked_loads"]["jpn48"]),
    }


def _write_csv(path: Path, runs: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "topology", "topology_key", "load_erlang", "seed", "method", "warmup",
        "evaluated", "accepted", "blocked", "blocking_rate", "avg_decision_ms",
        "avg_legal_actions", "avg_path_idx", "avg_mod_idx", "avg_block_idx",
        "avg_start_slot", "avg_required_fs", "avg_path_hops", "avg_path_km",
        "avg_block_waste", "request_trace_hash", "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda x: (x["topology"], x["load_erlang"], x["seed"], x["method"])):
            writer.writerow({key: run.get(key) for key in fields})


def _write_traces(path: Path, runs: Sequence[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for run in runs:
            for row in run.get("trace_rows", []):
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def _aggregate(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["topology"], run["load_erlang"], run["method"]), []).append(run)
    rows = []
    for (topology, load, method), group in sorted(groups.items()):
        rates = np.asarray([x["blocking_rate"] for x in group], dtype=float)
        rows.append({
            "topology": topology, "load_erlang": load, "method": method,
            "n_seeds": len(group), "mean_blocking_rate": float(rates.mean()),
            "std_blocking_rate": float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
            "mean_decision_ms": float(np.mean([x["avg_decision_ms"] for x in group])),
        })
    return rows


def _write_markdown(path: Path, phase: str, runs: Sequence[Dict[str, Any]]) -> None:
    rows = _aggregate(runs)
    lines = [
        f"# Pure RMSA Three-Way {phase.title()} Results", "",
        "Frozen Strict v1.3 is a cross-domain transfer; it was not trained in this pure RMSA environment.", "",
        "| Topology | Load | Method | Seeds | Blocking | Std | Decision ms |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['topology']} | {row['load_erlang']:.0f} | {row['method']} | {row['n_seeds']} | "
            f"{row['mean_blocking_rate']:.4%} | {row['std_blocking_rate']:.4%} | {row['mean_decision_ms']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _checkpoint_audit() -> Dict[str, Any]:
    ranker = load_ranker(STRICT_RANKER_CHECKPOINT, "cpu")
    mod_reg = paper_modulation_registry()
    hard_failures = []
    if ranker["input_dim"] != 25:
        hard_failures.append(f"ranker input_dim={ranker['input_dim']}")
    if list(mod_reg.names) != ["BPSK", "QPSK", "8QAM", "16QAM"]:
        hard_failures.append(f"modulation order={mod_reg.names}")
    return {
        "status": "FAIL" if hard_failures else "PASS_WITH_DISTRIBUTION_SHIFT",
        "hard_failures": hard_failures,
        "ppo_r_checkpoint": PPO_R_CHECKPOINT,
        "ppo_r_sha256": _sha256(PPO_R_CHECKPOINT),
        "ranker_checkpoint": STRICT_RANKER_CHECKPOINT,
        "ranker_sha256": _sha256(STRICT_RANKER_CHECKPOINT),
        "ranker_input_dim": ranker["input_dim"],
        "ranker_feature_names": ranker["feature_names"],
        "paper_modulation_names": list(mod_reg.names),
        "paper_modulation_reaches": [mod_reg[i].reach_km for i in range(mod_reg.num_formats)],
        "known_distribution_shifts": [
            "checkpoint training used SA-HMARL C/MEC context",
            "checkpoint evaluation used 320 slots; paper protocol uses 100 slots",
            "paper reach table is 10000/2500/1250/625 km",
            "C-only ranker fields are neutralized to checkpoint means",
        ],
    }


def _protocol_payload(locked_loads: Dict[str, List[float]]) -> Dict[str, Any]:
    return {
        "protocol": "Doherty paper-parity pure RMSA",
        "methods": list(METHODS), "k_paths": 50, "path_sort": "hops",
        "same_hop_tiebreak": "km", "max_blocks": 10, "block_sort": "start_asc",
        "num_slots": 100, "slot_bw_hz": 12.5e9, "guard_band_fs": 1,
        "bitrate_gbps": [25, 100], "mean_holding_parameter": 10.0,
        "holding_rule": "exponential; redraw while ttl == 0 or ttl >= 20",
        "arrival_interval": "10/load_erlang",
        "directed_arc_independent_spectrum": True,
        "locked_loads": locked_loads,
        "topologies": TOPOLOGIES,
        "equivalence_margin_pp": 0.10,
    }


def _write_protocol_files(output_dir: Path, locked_loads: Dict[str, List[float]]) -> None:
    payload = _protocol_payload(locked_loads)
    (output_dir / "PURE_RMSA_PROTOCOL_LOCK.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Pure RMSA Protocol Lock", "",
        "- Primary protocol: Doherty et al. paper-parity implementation.",
        "- C-side, MEC, split, compute delay, queue delay and deadline are absent.",
        "- Spectrum is independent per directed arc (dual fiber).",
        "- Strict label: Frozen Strict v1.3 pure-RMSA transfer.",
        "- KSP-FF and FF-KSP: K=50 hops, same-hop km tie-break, highest feasible modulation, start-ascending First-Fit.",
        "- Strict candidate pool: frozen PPO-R legal Top-30 only.", "",
        "## Locked Loads", "",
    ]
    for topology, loads in locked_loads.items():
        lines.append(f"- `{topology}`: {loads}")
    (output_dir / "PURE_RMSA_PROTOCOL_LOCK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "smoke", "pilot", "formal"), required=True)
    parser.add_argument("--output-dir", default="sa_hmarl/experiments/pure_rmsa_strict_v13_vs_ksp_ff_ffksp_k50_hops")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--calibration-seed", type=int, default=5001)
    parser.add_argument("--calibration-warmup", type=int, default=3000)
    parser.add_argument("--calibration-requests", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--evaluated", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--check-k500", action="store_true")
    parser.add_argument("--store-trace", action="store_true")
    parser.add_argument("--topologies", default=None, help="Comma-separated task filter; does not alter episodes")
    parser.add_argument("--methods", default=None, help="Comma-separated task filter; does not alter selectors")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_audit = _checkpoint_audit()
    (output_dir / "CHECKPOINT_COMPATIBILITY_AUDIT.json").write_text(
        json.dumps(checkpoint_audit, indent=2), encoding="utf-8"
    )
    if checkpoint_audit["hard_failures"]:
        raise SystemExit(f"Hard checkpoint incompatibility: {checkpoint_audit['hard_failures']}")

    calibration_path = output_dir / "LOAD_CALIBRATION.json"
    if args.phase == "calibration":
        calibration = calibrate_transfer_loads(args)
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
        print(f"Wrote {calibration_path} (stage={calibration.get('stage', 'legacy')})")
        return 0
    if not calibration_path.exists():
        raise SystemExit("LOAD_CALIBRATION.json is required before smoke/pilot/formal")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    locked_loads = _locked_loads(calibration)
    _write_protocol_files(output_dir, locked_loads)

    defaults = {
        "smoke": (500, 1000, [5001]),
        "pilot": (500, 6000, list(range(5001, 5006))),
        "formal": (500, 6000, list(range(5001, 5021))),
    }
    warmup, evaluated, seeds = defaults[args.phase]
    if args.warmup is not None:
        warmup = args.warmup
    if args.evaluated is not None:
        evaluated = args.evaluated
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
    jobs = []
    selected_topologies = set(args.topologies.split(",")) if args.topologies else set(locked_loads)
    selected_methods = tuple(args.methods.split(",")) if args.methods else METHODS
    for topology, loads in locked_loads.items():
        if topology not in selected_topologies:
            continue
        for load in loads:
            for seed in seeds:
                for method in selected_methods:
                    jobs.append({
                        "topology": topology, "topology_key": TOPOLOGIES[topology]["key"],
                        "load_erlang": load, "seed": seed, "warmup": warmup,
                        "evaluated": evaluated, "method": method,
                        "check_k500": args.check_k500,
                        "store_trace": args.store_trace,
                        "result_path": str(
                            output_dir / "shards" / f"{args.phase}_paper_exact"
                            / f"{topology}_rho{load:g}_seed{seed}_{method}.json"
                        ),
                    })
    started = time.perf_counter()
    runs = _run_jobs(jobs, args.max_workers)
    payload = {
        "phase": args.phase, "protocol": _protocol_payload(locked_loads),
        "runs": [{k: v for k, v in run.items() if k != "trace_rows"} for run in runs],
        "aggregate": _aggregate(runs), "elapsed_seconds": time.perf_counter() - started,
    }
    json_name = f"{args.phase.upper()}_RESULTS.json"
    md_name = f"{args.phase.upper()}_RESULTS.md"
    (output_dir / json_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(output_dir / md_name, args.phase, runs)
    _write_csv(output_dir / f"{args.phase}_per_seed_per_load_summary.csv", runs)
    if args.store_trace:
        _write_traces(output_dir / f"{args.phase}_action_trace.jsonl.gz", runs)
    manifest = {
        "phase": args.phase, "elapsed_seconds": payload["elapsed_seconds"],
        "checkpoint_audit": checkpoint_audit,
        "topology_hashes": {name: _topology_hash(cfg["key"]) for name, cfg in TOPOLOGIES.items()},
        "request_trace_parity": len({
            (r["topology"], r["load_erlang"], r["seed"], r["request_trace_hash"])
            for r in runs
        }) == len({(r["topology"], r["load_erlang"], r["seed"]) for r in runs}),
    }
    (output_dir / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed {args.phase}: {len(runs)} tasks in {payload['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
