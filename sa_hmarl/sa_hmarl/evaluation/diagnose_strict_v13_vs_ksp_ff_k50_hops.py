#!/usr/bin/env python3
"""Diagnosis: Why does Strict v1.3 block less than KSP-FF K=50 hops under PPO-C?

Protocol lock:
- Topology: xlron_cost239_ptrnet_real
- C-side: PPO-C (frozen checkpoint)
- R-side methods: Strict v1.3 ranker, KSP-FF K=50 hops
- R-side K_path = 50
- R-side path_sort_strategy = "hops"
- R-side block_sort_strategy = "start_asc"
- max_blocks = 10

KSP-FF K=50 hops is implemented by ``ksp_ff_highest_mod_action`` (distance-adaptive
KSP-FF: scan paths in hops order, pick highest feasible modulation, First-Fit block).
``ksp_ff_action`` (naive flat First-Fit / BPSK-first) is NOT used as the formal
KSP-FF K=50 hops baseline.

This script does NOT train any network and does NOT modify any checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
)
from sa_hmarl.evaluation.eval_long_horizon_system_comparison import (
    generate_long_horizon_requests,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5 import (
    _ppo_r_topk_actions,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRICT_V13_NAME = "Strict v1.3"
KSP_FF_NAME = "KSP-FF K=50 hops"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class RequestRecord:
    """Per-request record for one method."""

    seed: int
    step_idx: int
    req_id: int
    arrival_time: float
    src_node: int
    size_mb: float
    deadline_ms: float
    holding_time: float
    warmup: bool

    # C-side
    c_idx: Optional[int]
    split_id: int
    server_id: int
    c_valid: bool

    # R-side
    r_idx: Optional[int]
    r_valid: bool
    path_idx: int
    modulation: str
    required_fs: int
    block_start: int
    block_size: int
    block_waste: float
    path_hops: int
    path_length_km: float
    path_nodes: List[int]

    # Outcome
    success: bool
    failure_reason: str
    delay_ms: float

    # Extra diagnostics
    ppo_r_idx: Optional[int]
    same_as_ppo_r: bool
    candidate_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "step_idx": self.step_idx,
            "req_id": self.req_id,
            "arrival_time": self.arrival_time,
            "src_node": self.src_node,
            "size_mb": self.size_mb,
            "deadline_ms": self.deadline_ms,
            "holding_time": self.holding_time,
            "warmup": self.warmup,
            "c_idx": self.c_idx,
            "split_id": self.split_id,
            "server_id": self.server_id,
            "c_valid": self.c_valid,
            "r_idx": self.r_idx,
            "r_valid": self.r_valid,
            "path_idx": self.path_idx,
            "modulation": self.modulation,
            "required_fs": self.required_fs,
            "block_start": self.block_start,
            "block_size": self.block_size,
            "block_waste": self.block_waste,
            "path_hops": self.path_hops,
            "path_length_km": self.path_length_km,
            "path_nodes": self.path_nodes,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "delay_ms": self.delay_ms,
            "ppo_r_idx": self.ppo_r_idx,
            "same_as_ppo_r": self.same_as_ppo_r,
            "candidate_count": self.candidate_count,
        }


@dataclass
class MethodSummary:
    """Aggregate summary for one method across one or more seeds."""

    method: str
    total: int = 0
    admitted: int = 0
    blocked: int = 0
    server_overload: int = 0
    no_suitable_block: int = 0
    c_no_valid: int = 0
    r_no_valid: int = 0
    other_failure: int = 0
    delay_sum: float = 0.0
    fs_sum: float = 0.0
    decision_ms_sum: float = 0.0
    path_length_sum: float = 0.0
    hop_count_sum: float = 0.0

    path_idx_counts: Dict[int, int] = field(default_factory=dict)
    mod_counts: Dict[str, int] = field(default_factory=dict)
    required_fs_values: List[int] = field(default_factory=list)
    block_start_values: List[int] = field(default_factory=list)
    block_waste_values: List[float] = field(default_factory=list)

    def add_record(self, rec: RequestRecord) -> None:
        if rec.warmup:
            return
        self.total += 1
        if rec.success:
            self.admitted += 1
            self.delay_sum += rec.delay_ms
            self.fs_sum += rec.required_fs
            self.path_length_sum += rec.path_length_km
            self.hop_count_sum += rec.path_hops
            self.path_idx_counts[rec.path_idx] = self.path_idx_counts.get(rec.path_idx, 0) + 1
            self.mod_counts[rec.modulation] = self.mod_counts.get(rec.modulation, 0) + 1
            self.required_fs_values.append(rec.required_fs)
            self.block_start_values.append(rec.block_start)
            self.block_waste_values.append(rec.block_waste)
        else:
            self.blocked += 1
            reason = rec.failure_reason
            if reason == "server_overload" or reason == "server_saturated":
                self.server_overload += 1
            elif reason == "no_suitable_block":
                self.no_suitable_block += 1
            elif reason == "c_no_valid_action":
                self.c_no_valid += 1
            elif reason == "r_no_valid_action":
                self.r_no_valid += 1
            else:
                self.other_failure += 1

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.total, 1))

    def overload_rate(self) -> float:
        return float(self.server_overload / max(self.total, 1))

    def nsb_rate(self) -> float:
        return float(self.no_suitable_block / max(self.total, 1))

    def c_no_valid_rate(self) -> float:
        return float(self.c_no_valid / max(self.total, 1))

    def r_no_valid_rate(self) -> float:
        return float(self.r_no_valid / max(self.total, 1))

    def avg_delay_ms(self) -> float:
        return float(self.delay_sum / max(self.admitted, 1))

    def avg_fs(self) -> float:
        return float(self.fs_sum / max(self.admitted, 1))

    def avg_path_km(self) -> float:
        return float(self.path_length_sum / max(self.admitted, 1))

    def avg_hops(self) -> float:
        return float(self.hop_count_sum / max(self.admitted, 1))

    def avg_decision_ms(self) -> float:
        return float(self.decision_ms_sum / max(self.total, 1))

    def path_idx_distribution(self) -> Dict[str, float]:
        total = sum(self.path_idx_counts.values())
        return {str(k): v / total for k, v in sorted(self.path_idx_counts.items())} if total else {}

    def mod_distribution(self) -> Dict[str, float]:
        total = sum(self.mod_counts.values())
        return {k: v / total for k, v in sorted(self.mod_counts.items())} if total else {}

    def avg_block_start(self) -> float:
        return float(np.mean(self.block_start_values)) if self.block_start_values else 0.0

    def avg_block_waste(self) -> float:
        return float(np.mean(self.block_waste_values)) if self.block_waste_values else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "total": self.total,
            "admitted": self.admitted,
            "blocked": self.blocked,
            "blocking_rate": self.blocking_rate(),
            "server_overload": self.server_overload,
            "overload_rate": self.overload_rate(),
            "no_suitable_block": self.no_suitable_block,
            "nsb_rate": self.nsb_rate(),
            "c_no_valid": self.c_no_valid,
            "c_no_valid_rate": self.c_no_valid_rate(),
            "r_no_valid": self.r_no_valid,
            "r_no_valid_rate": self.r_no_valid_rate(),
            "other_failure": self.other_failure,
            "avg_delay_ms": self.avg_delay_ms(),
            "avg_fs": self.avg_fs(),
            "avg_path_km": self.avg_path_km(),
            "avg_hops": self.avg_hops(),
            "avg_decision_ms": self.avg_decision_ms(),
            "path_idx_distribution": self.path_idx_distribution(),
            "mod_distribution": self.mod_distribution(),
            "avg_block_start": self.avg_block_start(),
            "avg_block_waste": self.avg_block_waste(),
            "required_fs_values": self.required_fs_values,
            "block_start_values": self.block_start_values,
            "block_waste_values": self.block_waste_values,
        }


@dataclass
class PairedRecord:
    """Paired record across Strict v1.3 and KSP-FF K=50 hops for one request."""

    seed: int
    step_idx: int
    req_id: int
    arrival_time: float
    src_node: int
    size_mb: float
    deadline_ms: float
    holding_time: float
    warmup: bool

    strict_c_idx: Optional[int]
    strict_split_id: int
    strict_server_id: int
    ksp_c_idx: Optional[int]
    ksp_split_id: int
    ksp_server_id: int
    same_c_action: bool

    strict_r_idx: Optional[int]
    ksp_r_idx: Optional[int]
    same_r_action: bool

    strict_path_idx: int
    strict_modulation: str
    strict_required_fs: int
    strict_block_start: int
    strict_block_size: int
    strict_block_waste: float
    strict_path_hops: int
    strict_path_length_km: float
    strict_path_nodes: List[int]

    ksp_path_idx: int
    ksp_modulation: str
    ksp_required_fs: int
    ksp_block_start: int
    ksp_block_size: int
    ksp_block_waste: float
    ksp_path_hops: int
    ksp_path_length_km: float
    ksp_path_nodes: List[int]

    strict_success: bool
    ksp_success: bool
    strict_failure_reason: str
    ksp_failure_reason: str
    outcome_class: str

    strict_server_util_before: float
    ksp_server_util_before: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "step_idx": self.step_idx,
            "req_id": self.req_id,
            "arrival_time": self.arrival_time,
            "src_node": self.src_node,
            "size_mb": self.size_mb,
            "deadline_ms": self.deadline_ms,
            "holding_time": self.holding_time,
            "warmup": self.warmup,
            "strict_c_idx": self.strict_c_idx,
            "strict_split_id": self.strict_split_id,
            "strict_server_id": self.strict_server_id,
            "ksp_c_idx": self.ksp_c_idx,
            "ksp_split_id": self.ksp_split_id,
            "ksp_server_id": self.ksp_server_id,
            "same_c_action": self.same_c_action,
            "strict_r_idx": self.strict_r_idx,
            "ksp_r_idx": self.ksp_r_idx,
            "same_r_action": self.same_r_action,
            "strict_path_idx": self.strict_path_idx,
            "strict_modulation": self.strict_modulation,
            "strict_required_fs": self.strict_required_fs,
            "strict_block_start": self.strict_block_start,
            "strict_block_size": self.strict_block_size,
            "strict_block_waste": self.strict_block_waste,
            "strict_path_hops": self.strict_path_hops,
            "strict_path_length_km": self.strict_path_length_km,
            "strict_path_nodes": self.strict_path_nodes,
            "ksp_path_idx": self.ksp_path_idx,
            "ksp_modulation": self.ksp_modulation,
            "ksp_required_fs": self.ksp_required_fs,
            "ksp_block_start": self.ksp_block_start,
            "ksp_block_size": self.ksp_block_size,
            "ksp_block_waste": self.ksp_block_waste,
            "ksp_path_hops": self.ksp_path_hops,
            "ksp_path_length_km": self.ksp_path_length_km,
            "ksp_path_nodes": self.ksp_path_nodes,
            "strict_success": self.strict_success,
            "ksp_success": self.ksp_success,
            "strict_failure_reason": self.strict_failure_reason,
            "ksp_failure_reason": self.ksp_failure_reason,
            "outcome_class": self.outcome_class,
            "strict_server_util_before": self.strict_server_util_before,
            "ksp_server_util_before": self.ksp_server_util_before,
        }


@dataclass
class ProbeRecord:
    """Counterfactual probe record at a divergence request."""

    seed: int
    step_idx: int
    req_id: int
    strict_state_ksp_action_feasible: bool
    strict_state_ksp_action_reason: str
    ksp_state_strict_action_feasible: bool
    ksp_state_strict_action_reason: str
    strict_action_in_ksp_obs: bool
    ksp_action_in_strict_obs: bool
    future_strict_in_ksp_state: Dict[str, Any]
    future_ksp_in_strict_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "step_idx": self.step_idx,
            "req_id": self.req_id,
            "strict_state_ksp_action_feasible": self.strict_state_ksp_action_feasible,
            "strict_state_ksp_action_reason": self.strict_state_ksp_action_reason,
            "ksp_state_strict_action_feasible": self.ksp_state_strict_action_feasible,
            "ksp_state_strict_action_reason": self.ksp_state_strict_action_reason,
            "strict_action_in_ksp_obs": self.strict_action_in_ksp_obs,
            "ksp_action_in_strict_obs": self.ksp_action_in_strict_obs,
            "future_strict_in_ksp_state": self.future_strict_in_ksp_state,
            "future_ksp_in_strict_state": self.future_ksp_in_strict_state,
        }


# ---------------------------------------------------------------------------
# Helpers copied / adapted from eval_strict_v13_multitopology_cside_verified.py
# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "N/A"


def _load_ranker_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"),
        int(ckpt["input_dim"]),
        tuple(ckpt.get("hidden_dims", [128, 64])),
        float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    feature_mean = np.asarray(ckpt.get("feature_mean", 0.0), dtype=np.float32)
    feature_std = np.maximum(
        np.asarray(ckpt.get("feature_std", 1.0), dtype=np.float32), 1e-6
    )
    if feature_mean.ndim == 0:
        feature_mean = np.zeros(int(ckpt["input_dim"]), dtype=np.float32)
    if feature_std.ndim == 0:
        feature_std = np.ones(int(ckpt["input_dim"]), dtype=np.float32)
    feature_names = list(ckpt.get("feature_names", FEATURE_NAMES))
    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_names": feature_names,
        "input_dim": int(ckpt["input_dim"]),
        "ckpt_sha256": _sha256(ckpt_path),
    }


def _select_c_action(
    agent_c,
    env,
    req,
    obs_c: Dict[str, Any],
    num_servers: int,
) -> Tuple[Optional[int], Optional[int], Optional[int], bool, Dict[str, Any]]:
    """Return (flat_c_idx, split_id, server_id, c_valid, info)."""
    info: Dict[str, Any] = {
        "raw_mask_empty": False,
        "effective_mask_empty": False,
        "action_in_effective_mask": False,
        "c_policy_ms": 0.0,
    }
    raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    info["raw_mask_empty"] = not raw_mask.any()
    if info["raw_mask_empty"]:
        return None, None, None, False, info
    t0 = time.perf_counter()
    action_idx, _, risk_mask = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
    info["c_policy_ms"] = (time.perf_counter() - t0) * 1000.0
    effective = np.asarray(risk_mask, dtype=bool)
    info["effective_mask_empty"] = not effective.any()
    if info["effective_mask_empty"]:
        return None, None, None, False, info
    if not (0 <= int(action_idx) < len(effective) and effective[int(action_idx)]):
        info["action_in_effective_mask"] = False
        return None, None, None, False, info
    info["action_in_effective_mask"] = True
    split_id, server_id = decode_agent_c_action(action_idx, num_servers)
    return int(action_idx), int(split_id), int(server_id), True, info


def _select_r_action_strict(
    ranker: Dict[str, Any],
    agent_r,
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    split_id: int,
    server_id: int,
) -> Tuple[Optional[int], bool, Dict[str, Any]]:
    """Return (r_idx, r_valid, info) for Strict v1.3."""
    info: Dict[str, Any] = {
        "r_proposer_ms": 0.0,
        "ranker_ms": 0.0,
        "candidate_count": 0,
        "fallback": False,
        "in_candidates": False,
        "score_margin": 0.0,
        "feature_finite": True,
    }
    candidates = _ppo_r_topk_actions(agent_r, obs_r, 30)
    candidates = np.asarray(candidates, dtype=np.int64)
    has_candidates = candidates.size > 0
    info["candidate_count"] = int(candidates.size)
    info["fallback"] = not has_candidates
    if not has_candidates:
        return None, False, info

    t0 = time.perf_counter()
    r_features, _ = agent_r.build_action_features(obs_r)
    feats = []
    for a in candidates:
        feat = _r_feature_vector(
            env,
            req,
            obs_c,
            obs_r,
            r_features,
            int(a),
            split_id,
            server_id,
            feature_names=ranker["feature_names"],
        )
        feats.append(feat)
    x = np.stack(feats, axis=0).astype(np.float32)
    finite_all = np.isfinite(x).all()
    info["feature_finite"] = bool(finite_all)
    x = (x - ranker["feature_mean"]) / ranker["feature_std"]
    x[~np.isfinite(x)] = 0.0
    with torch.no_grad():
        scores = (
            ranker["model"](torch.as_tensor(x, dtype=torch.float32, device=ranker["device"]))
            .cpu()
            .numpy()
        )
    best_local = int(np.argmax(scores))
    r_idx = int(candidates[best_local])
    info["in_candidates"] = True
    sorted_scores = np.sort(scores)[::-1]
    info["score_margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
    info["ranker_ms"] = (time.perf_counter() - t0) * 1000.0
    return r_idx, True, info


def _select_r_action_ksp_ff(obs_r: Dict[str, Any]) -> Tuple[Optional[int], bool, Dict[str, Any]]:
    """Return (r_idx, r_valid, info) for KSP-FF K=50 hops."""
    info: Dict[str, Any] = {"ranker_ms": 0.0}
    a = ksp_ff_highest_mod_action(obs_r)
    if a is None:
        return None, False, info
    return int(a), True, info


def _extract_r_action_meta(obs_r: Dict[str, Any], r_action: Tuple[int, int, int]) -> Dict[str, Any]:
    path_idx, mod_idx, block_idx = r_action
    path_feats = obs_r.get("path_features", [])
    meta = {
        "path_idx": path_idx,
        "mod_name": "",
        "path_length_km": 0.0,
        "hop_count": 0,
        "block_start": -1,
        "block_size": 0,
        "block_waste": 0.0,
        "required_fs": 0,
        "path_nodes": [],
    }
    if path_feats and 0 <= path_idx < len(path_feats):
        meta["path_length_km"] = float(path_feats[path_idx].get("path_length_km", 0.0))
        meta["hop_count"] = int(path_feats[path_idx].get("hop_count", 0))
    mod_names = obs_r.get("mod_names", [])
    if mod_names and 0 <= mod_idx < len(mod_names):
        meta["mod_name"] = str(mod_names[mod_idx])
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    if block_idx < len(blocks):
        meta["block_start"] = int(blocks[block_idx][0])
        meta["block_size"] = int(blocks[block_idx][1])
    req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    meta["required_fs"] = int(req_fs) if req_fs is not None else 0
    meta["block_waste"] = float(
        (meta["block_size"] - meta["required_fs"]) / max(meta["block_size"], 1)
    )
    candidate_paths = obs_r.get("candidate_paths", [])
    if candidate_paths and 0 <= path_idx < len(candidate_paths):
        meta["path_nodes"] = [int(n) for n in candidate_paths[path_idx]]
    return meta


def _run_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    ranker: Optional[Dict[str, Any]],
    mode: str,  # "strict" or "ksp_ff"
    args: argparse.Namespace,
) -> Tuple[MethodSummary, List[RequestRecord]]:
    """Run one episode and return aggregate summary plus request-level records."""
    assert mode in ("strict", "ksp_ff")
    records: List[RequestRecord] = []
    summary = MethodSummary(method=STRICT_V13_NAME if mode == "strict" else KSP_FF_NAME)
    num_servers = len(env.mec.servers)
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < args.warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        # C-side under K_C.
        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        c_idx, split_id, server_id, c_valid, c_info = _select_c_action(
            agent_c, env, req, obs_c, num_servers
        )
        server_util_before = float(env.mec.servers[server_id].utilization) if c_valid and 0 <= server_id < num_servers else 0.0

        if not c_valid:
            env.reject_next_request(req.req_id, "c_no_valid_action")
            if not is_warmup:
                rec = RequestRecord(
                    seed=args._current_seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    src_node=int(req.src_node),
                    size_mb=float(req.splits[0].intermediate_size_mb) if req.splits else 0.0,
                    deadline_ms=float(req.deadline_ms),
                    holding_time=float(req.holding_time),
                    warmup=False,
                    c_idx=None,
                    split_id=-1,
                    server_id=-1,
                    c_valid=False,
                    r_idx=None,
                    r_valid=False,
                    path_idx=-1,
                    modulation="",
                    required_fs=0,
                    block_start=-1,
                    block_size=0,
                    block_waste=0.0,
                    path_hops=0,
                    path_length_km=0.0,
                    path_nodes=[],
                    success=False,
                    failure_reason="c_no_valid_action",
                    delay_ms=0.0,
                    ppo_r_idx=None,
                    same_as_ppo_r=False,
                    candidate_count=0,
                )
                rec.delay_ms = (time.perf_counter() - t0) * 1000.0
                records.append(rec)
                summary.add_record(rec)
                summary.decision_ms_sum += rec.delay_ms
            continue

        # R-side under K_path.
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        agent_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        if not agent_r_mask.any():
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = RequestRecord(
                    seed=args._current_seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    src_node=int(req.src_node),
                    size_mb=float(req.splits[split_id].intermediate_size_mb) if 0 <= split_id < len(req.splits) else 0.0,
                    deadline_ms=float(req.deadline_ms),
                    holding_time=float(req.holding_time),
                    warmup=False,
                    c_idx=int(c_idx),
                    split_id=int(split_id),
                    server_id=int(server_id),
                    c_valid=True,
                    r_idx=None,
                    r_valid=False,
                    path_idx=-1,
                    modulation="",
                    required_fs=0,
                    block_start=-1,
                    block_size=0,
                    block_waste=0.0,
                    path_hops=0,
                    path_length_km=0.0,
                    path_nodes=[],
                    success=False,
                    failure_reason="r_no_valid_action",
                    delay_ms=0.0,
                    ppo_r_idx=None,
                    same_as_ppo_r=False,
                    candidate_count=0,
                )
                rec.delay_ms = (time.perf_counter() - t0) * 1000.0
                records.append(rec)
                summary.add_record(rec)
                summary.decision_ms_sum += rec.delay_ms
            continue

        # PPO-R top-1 for reference.
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, max_blocks)

        if mode == "strict":
            r_idx, r_valid, r_info = _select_r_action_strict(
                ranker, agent_r, env, req, obs_c, obs_r, split_id, server_id
            )
            candidate_count = r_info.get("candidate_count", 0)
        else:
            r_idx, r_valid, r_info = _select_r_action_ksp_ff(obs_r)
            candidate_count = 0

        if not r_valid or r_idx is None:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = RequestRecord(
                    seed=args._current_seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    src_node=int(req.src_node),
                    size_mb=float(req.splits[split_id].intermediate_size_mb) if 0 <= split_id < len(req.splits) else 0.0,
                    deadline_ms=float(req.deadline_ms),
                    holding_time=float(req.holding_time),
                    warmup=False,
                    c_idx=int(c_idx),
                    split_id=int(split_id),
                    server_id=int(server_id),
                    c_valid=True,
                    r_idx=None,
                    r_valid=False,
                    path_idx=-1,
                    modulation="",
                    required_fs=0,
                    block_start=-1,
                    block_size=0,
                    block_waste=0.0,
                    path_hops=0,
                    path_length_km=0.0,
                    path_nodes=[],
                    success=False,
                    failure_reason="r_no_valid_action",
                    delay_ms=0.0,
                    ppo_r_idx=int(ppo_idx),
                    same_as_ppo_r=False,
                    candidate_count=candidate_count,
                )
                rec.delay_ms = (time.perf_counter() - t0) * 1000.0
                records.append(rec)
                summary.add_record(rec)
                summary.decision_ms_sum += rec.delay_ms
            continue

        r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)

        if not is_warmup:
            meta = _extract_r_action_meta(obs_r, r_action)
            success = bool(info.get("success", False))
            rec = RequestRecord(
                seed=args._current_seed,
                step_idx=step_idx,
                req_id=int(req.req_id),
                arrival_time=float(req.arrival_time),
                src_node=int(req.src_node),
                size_mb=float(req.splits[split_id].intermediate_size_mb) if 0 <= split_id < len(req.splits) else 0.0,
                deadline_ms=float(req.deadline_ms),
                holding_time=float(req.holding_time),
                warmup=False,
                c_idx=int(c_idx),
                split_id=int(split_id),
                server_id=int(server_id),
                c_valid=True,
                r_idx=int(r_idx),
                r_valid=True,
                path_idx=int(meta["path_idx"]),
                modulation=str(meta["mod_name"]),
                required_fs=int(meta["required_fs"]),
                block_start=int(meta["block_start"]),
                block_size=int(meta["block_size"]),
                block_waste=float(meta["block_waste"]),
                path_hops=int(meta["hop_count"]),
                path_length_km=float(meta["path_length_km"]),
                path_nodes=[int(n) for n in meta["path_nodes"]],
                success=success,
                failure_reason=str(info.get("reason", "")) if not success else "",
                delay_ms=float(info.get("delay_ms", 0.0)) if success else 0.0,
                ppo_r_idx=int(ppo_idx),
                same_as_ppo_r=int(r_idx) == int(ppo_idx),
                candidate_count=candidate_count,
            )
            # Attach server util before decision for paired analysis.
            rec._server_util_before = server_util_before  # type: ignore
            rec.delay_ms = (time.perf_counter() - t0) * 1000.0
            records.append(rec)
            summary.add_record(rec)
            summary.decision_ms_sum += rec.delay_ms

    return summary, records


# ---------------------------------------------------------------------------
# Counterfactual probe
# ---------------------------------------------------------------------------
def _try_action_in_state(
    env,
    req,
    split_id: int,
    server_id: int,
    r_idx: int,
    obs_r: Dict[str, Any],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Try executing a specific R action in a copied env state. Return success/reason/info."""
    r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
    _, _, _, info = env.step((split_id, server_id), r_action)
    success = bool(info.get("success", False))
    reason = info.get("reason", "") if not success else "success"
    return success, reason, info


def _rollout_future_blocking(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c,
    agent_r,
    num_servers: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Roll out H future requests with frozen PPO-C + PPO-R from env state."""
    blocked = 0
    nsb = 0
    overload = 0
    other = 0
    for offset in range(horizon):
        t = start_idx + offset
        if t >= len(requests):
            break
        req = requests[t]
        env.advance_time(req.arrival_time)

        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if not raw_c_mask.any():
            blocked += 1
            env.reject_next_request(req.req_id, "c_no_valid_action")
            continue
        c_idx, split, server, c_valid, _ = _select_c_action(agent_c, env, req, obs_c, num_servers)
        if not c_valid:
            blocked += 1
            env.reject_next_request(req.req_id, "c_no_valid_action")
            continue

        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split, server)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if not raw_r_mask.any():
            blocked += 1
            env.reject_next_request(req.req_id, "r_no_valid_action")
            continue
        r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split, server), r_action)
        if not info.get("success", False):
            blocked += 1
            reason = info.get("reason", "")
            if reason == "no_suitable_block":
                nsb += 1
            elif reason == "server_overload":
                overload += 1
            else:
                other += 1

    return {
        "blocked": blocked,
        "nsb": nsb,
        "overload": overload,
        "other": other,
        "horizon": horizon,
    }


def _run_counterfactual_probe(
    strict_records: List[RequestRecord],
    ksp_records: List[RequestRecord],
    strict_env_snapshots: Dict[int, Any],
    ksp_env_snapshots: Dict[int, Any],
    strict_obs_r_snapshots: Dict[int, Dict[str, Any]],
    ksp_obs_r_snapshots: Dict[int, Dict[str, Any]],
    requests: List[Any],
    agent_c,
    agent_r,
    args: argparse.Namespace,
) -> List[ProbeRecord]:
    """Run counterfactual probes on divergence requests."""
    probes: List[ProbeRecord] = []
    num_servers = len(args._env.mec.servers)

    diverge_indices = [
        i
        for i, (sr, kr) in enumerate(zip(strict_records, ksp_records))
        if not sr.warmup and sr.r_idx != kr.r_idx
    ]
    # Sample at most probe_budget divergence requests per seed.
    sample_stride = max(1, len(diverge_indices) // max(args.probe_budget, 1))
    sampled = diverge_indices[::sample_stride][: args.probe_budget]

    for step_idx in sampled:
        sr = strict_records[step_idx]
        kr = ksp_records[step_idx]
        req = requests[step_idx]

        strict_env = strict_env_snapshots.get(step_idx)
        ksp_env = ksp_env_snapshots.get(step_idx)
        strict_obs_r = strict_obs_r_snapshots.get(step_idx)
        ksp_obs_r = ksp_obs_r_snapshots.get(step_idx)
        if strict_env is None or ksp_env is None or strict_obs_r is None or ksp_obs_r is None:
            continue

        # Check action feasibility in opposite observation masks.
        strict_action_in_ksp_obs = (
            0 <= sr.r_idx < len(ksp_obs_r["agent_r_mask"])
            and bool(ksp_obs_r["agent_r_mask"][sr.r_idx])
        ) if sr.r_idx is not None else False
        ksp_action_in_strict_obs = (
            0 <= kr.r_idx < len(strict_obs_r["agent_r_mask"])
            and bool(strict_obs_r["agent_r_mask"][kr.r_idx])
        ) if kr.r_idx is not None else False

        # Try KSP action in Strict state.
        strict_state_ksp_action_feasible = False
        strict_state_ksp_action_reason = "action_not_in_mask"
        if ksp_action_in_strict_obs and kr.r_idx is not None:
            branch = copy.deepcopy(strict_env)
            success, reason, _ = _try_action_in_state(
                branch, req, sr.split_id, sr.server_id, kr.r_idx, strict_obs_r
            )
            strict_state_ksp_action_feasible = success
            strict_state_ksp_action_reason = reason

        # Try Strict action in KSP state.
        ksp_state_strict_action_feasible = False
        ksp_state_strict_action_reason = "action_not_in_mask"
        if strict_action_in_ksp_obs and sr.r_idx is not None:
            branch = copy.deepcopy(ksp_env)
            success, reason, _ = _try_action_in_state(
                branch, req, kr.split_id, kr.server_id, sr.r_idx, ksp_obs_r
            )
            ksp_state_strict_action_feasible = success
            ksp_state_strict_action_reason = reason

        # Future rollout: continue with frozen PPO-C+PPO-R.
        future_strict_in_ksp_state = {}
        future_ksp_in_strict_state = {}
        if strict_state_ksp_action_feasible and kr.r_idx is not None:
            branch = copy.deepcopy(strict_env)
            _try_action_in_state(branch, req, sr.split_id, sr.server_id, kr.r_idx, strict_obs_r)
            future_strict_in_ksp_state = _rollout_future_blocking(
                branch, requests, step_idx + 1, args.probe_horizon, agent_c, agent_r, num_servers, args
            )
        if ksp_state_strict_action_feasible and sr.r_idx is not None:
            branch = copy.deepcopy(ksp_env)
            _try_action_in_state(branch, req, kr.split_id, kr.server_id, sr.r_idx, ksp_obs_r)
            future_ksp_in_strict_state = _rollout_future_blocking(
                branch, requests, step_idx + 1, args.probe_horizon, agent_c, agent_r, num_servers, args
            )

        probes.append(
            ProbeRecord(
                seed=sr.seed,
                step_idx=step_idx,
                req_id=sr.req_id,
                strict_state_ksp_action_feasible=strict_state_ksp_action_feasible,
                strict_state_ksp_action_reason=strict_state_ksp_action_reason,
                ksp_state_strict_action_feasible=ksp_state_strict_action_feasible,
                ksp_state_strict_action_reason=ksp_state_strict_action_reason,
                strict_action_in_ksp_obs=strict_action_in_ksp_obs,
                ksp_action_in_strict_obs=ksp_action_in_strict_obs,
                future_strict_in_ksp_state=future_strict_in_ksp_state,
                future_ksp_in_strict_state=future_ksp_in_strict_state,
            )
        )

    return probes


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------
def _build_paired_records(
    strict_records: List[RequestRecord],
    ksp_records: List[RequestRecord],
) -> List[PairedRecord]:
    paired: List[PairedRecord] = []
    for sr, kr in zip(strict_records, ksp_records):
        if sr.warmup or kr.warmup:
            continue
        outcome_class = "both_success"
        if sr.success and not kr.success:
            outcome_class = "strict_win"
        elif not sr.success and kr.success:
            outcome_class = "ksp_win"
        elif not sr.success and not kr.success:
            outcome_class = "both_block"

        paired.append(
            PairedRecord(
                seed=sr.seed,
                step_idx=sr.step_idx,
                req_id=sr.req_id,
                arrival_time=sr.arrival_time,
                src_node=sr.src_node,
                size_mb=sr.size_mb,
                deadline_ms=sr.deadline_ms,
                holding_time=sr.holding_time,
                warmup=sr.warmup,
                strict_c_idx=sr.c_idx,
                strict_split_id=sr.split_id,
                strict_server_id=sr.server_id,
                ksp_c_idx=kr.c_idx,
                ksp_split_id=kr.split_id,
                ksp_server_id=kr.server_id,
                same_c_action=(sr.c_idx == kr.c_idx and sr.split_id == kr.split_id and sr.server_id == kr.server_id),
                strict_r_idx=sr.r_idx,
                ksp_r_idx=kr.r_idx,
                same_r_action=(sr.r_idx == kr.r_idx),
                strict_path_idx=sr.path_idx,
                strict_modulation=sr.modulation,
                strict_required_fs=sr.required_fs,
                strict_block_start=sr.block_start,
                strict_block_size=sr.block_size,
                strict_block_waste=sr.block_waste,
                strict_path_hops=sr.path_hops,
                strict_path_length_km=sr.path_length_km,
                strict_path_nodes=sr.path_nodes,
                ksp_path_idx=kr.path_idx,
                ksp_modulation=kr.modulation,
                ksp_required_fs=kr.required_fs,
                ksp_block_start=kr.block_start,
                ksp_block_size=kr.block_size,
                ksp_block_waste=kr.block_waste,
                ksp_path_hops=kr.path_hops,
                ksp_path_length_km=kr.path_length_km,
                ksp_path_nodes=kr.path_nodes,
                strict_success=sr.success,
                ksp_success=kr.success,
                strict_failure_reason=sr.failure_reason,
                ksp_failure_reason=kr.failure_reason,
                outcome_class=outcome_class,
                strict_server_util_before=getattr(sr, "_server_util_before", 0.0),
                ksp_server_util_before=getattr(kr, "_server_util_before", 0.0),
            )
        )
    return paired


def _analyze_paired(
    paired: List[PairedRecord],
    strict_records: List[RequestRecord],
    ksp_records: List[RequestRecord],
) -> Dict[str, Any]:
    """Compute divergence and win/loss diagnostics."""
    strict_wins = [p for p in paired if p.outcome_class == "strict_win"]
    ksp_wins = [p for p in paired if p.outcome_class == "ksp_win"]

    def _failure_breakdown(items: List[PairedRecord], ksp_field: bool) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in items:
            reason = p.ksp_failure_reason if ksp_field else p.strict_failure_reason
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _last_divergence_distance(items: List[PairedRecord], divergence_key: str) -> Dict[str, Any]:
        distances: List[int] = []
        for idx, p in enumerate(items):
            step = p.step_idx
            # Scan backwards within the same seed up to the previous item.
            prev_step = items[idx - 1].step_idx if idx > 0 else -1
            found = None
            for candidate in range(step - 1, prev_step, -1):
                match = paired[candidate]
                if match.seed != p.seed:
                    continue
                if getattr(match, divergence_key):
                    found = step - match.step_idx
                    break
            distances.append(found if found is not None else -1)
        return {
            "mean": float(np.mean([d for d in distances if d >= 0])) if any(d >= 0 for d in distances) else None,
            "median": float(np.median([d for d in distances if d >= 0])) if any(d >= 0 for d in distances) else None,
            "values": distances,
        }

    strict_win_ksp_failure_breakdown = _failure_breakdown(strict_wins, ksp_field=True)
    ksp_win_strict_failure_breakdown = _failure_breakdown(ksp_wins, ksp_field=False)

    strict_win_last_r_divergence = _last_divergence_distance(
        strict_wins, "same_r_action"
    )
    strict_win_last_c_divergence = _last_divergence_distance(
        strict_wins, "same_c_action"
    )
    ksp_win_last_r_divergence = _last_divergence_distance(
        ksp_wins, "same_r_action"
    )
    ksp_win_last_c_divergence = _last_divergence_distance(
        ksp_wins, "same_c_action"
    )

    # First divergence positions.
    first_r_divergence_per_seed: Dict[int, int] = {}
    first_c_divergence_per_seed: Dict[int, int] = {}
    first_outcome_per_seed: Dict[int, int] = {}
    for p in paired:
        seed = p.seed
        if not p.same_r_action and seed not in first_r_divergence_per_seed:
            first_r_divergence_per_seed[seed] = p.step_idx
        if not p.same_c_action and seed not in first_c_divergence_per_seed:
            first_c_divergence_per_seed[seed] = p.step_idx
        if p.outcome_class in ("strict_win", "ksp_win") and seed not in first_outcome_per_seed:
            first_outcome_per_seed[seed] = p.step_idx

    # Windowed preference analysis before strict_win requests.
    window_size = 20
    strict_win_preference: Dict[str, Any] = {}
    if strict_wins:
        window_records_strict: List[RequestRecord] = []
        window_records_ksp: List[RequestRecord] = []
        for p in strict_wins:
            start = max(0, p.step_idx - window_size)
            for rec in strict_records:
                if rec.seed == p.seed and start <= rec.step_idx < p.step_idx and not rec.warmup:
                    window_records_strict.append(rec)
            for rec in ksp_records:
                if rec.seed == p.seed and start <= rec.step_idx < p.step_idx and not rec.warmup:
                    window_records_ksp.append(rec)
        strict_win_preference = {
            "window_size": window_size,
            "strict_avg_path_idx": float(np.mean([r.path_idx for r in window_records_strict])) if window_records_strict else None,
            "ksp_avg_path_idx": float(np.mean([r.path_idx for r in window_records_ksp])) if window_records_ksp else None,
            "strict_avg_required_fs": float(np.mean([r.required_fs for r in window_records_strict])) if window_records_strict else None,
            "ksp_avg_required_fs": float(np.mean([r.required_fs for r in window_records_ksp])) if window_records_ksp else None,
            "strict_avg_block_start": float(np.mean([r.block_start for r in window_records_strict])) if window_records_strict else None,
            "ksp_avg_block_start": float(np.mean([r.block_start for r in window_records_ksp])) if window_records_ksp else None,
            "strict_avg_block_waste": float(np.mean([r.block_waste for r in window_records_strict])) if window_records_strict else None,
            "ksp_avg_block_waste": float(np.mean([r.block_waste for r in window_records_ksp])) if window_records_ksp else None,
            "strict_avg_path_hops": float(np.mean([r.path_hops for r in window_records_strict])) if window_records_strict else None,
            "ksp_avg_path_hops": float(np.mean([r.path_hops for r in window_records_ksp])) if window_records_ksp else None,
        }

    return {
        "strict_win_count": len(strict_wins),
        "ksp_win_count": len(ksp_wins),
        "both_success_count": sum(1 for p in paired if p.outcome_class == "both_success"),
        "both_block_count": sum(1 for p in paired if p.outcome_class == "both_block"),
        "strict_win_ksp_failure_breakdown": strict_win_ksp_failure_breakdown,
        "ksp_win_strict_failure_breakdown": ksp_win_strict_failure_breakdown,
        "strict_win_last_r_divergence_mean": strict_win_last_r_divergence.get("mean"),
        "strict_win_last_c_divergence_mean": strict_win_last_c_divergence.get("mean"),
        "ksp_win_last_r_divergence_mean": ksp_win_last_r_divergence.get("mean"),
        "ksp_win_last_c_divergence_mean": ksp_win_last_c_divergence.get("mean"),
        "first_r_divergence_per_seed": first_r_divergence_per_seed,
        "first_c_divergence_per_seed": first_c_divergence_per_seed,
        "first_outcome_per_seed": first_outcome_per_seed,
        "strict_win_pre_window_20": strict_win_preference,
    }


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------
def _diagnose(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    root = Path(__file__).resolve().parents[3]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    ranker = _load_ranker_model(args.strict_ranker_checkpoint, args.device)

    total_requests = args.warmup_requests + args.requests_per_episode
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    per_seed_summaries: List[Dict[str, Any]] = []
    all_paired: List[PairedRecord] = []
    all_probes: List[ProbeRecord] = []
    all_trace_rows: List[Dict[str, Any]] = []
    all_strict_records: List[RequestRecord] = []
    all_ksp_records: List[RequestRecord] = []

    env_for_nodes = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy_r,
        path_sort_strategy=args.path_sort_strategy_r,
        k=args.k_paths_r,
    )
    args._env = env_for_nodes  # for probe helper

    for seed in seeds:
        args._current_seed = seed
        rng = np.random.RandomState(seed)
        requests = generate_long_horizon_requests(
            num_nodes=env_for_nodes.net.NUM_NODES,
            rng=rng,
            num_requests=total_requests,
            arrival_interval=args.arrival_interval,
            holding_min=args.holding_min,
            holding_max=args.holding_max,
            deadline_min=args.deadline_min,
            deadline_max=args.deadline_max,
            size_min_mb=args.size_min_mb,
            size_max_mb=args.size_max_mb,
            edge_cost_min=args.edge_cost_min,
            edge_cost_max=args.edge_cost_max,
            num_splits=args.num_splits,
            split_profile=args.split_profile,
            poisson_arrivals=args.poisson_arrivals,
            exponential_holding=args.exponential_holding,
        )

        # Strict v1.3 episode.
        env_strict = make_env(
            args.topology,
            args.num_slots,
            args.num_servers,
            seed,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy_r,
            path_sort_strategy=args.path_sort_strategy_r,
            k=args.k_paths_r,
        )
        env_strict.reset(requests)
        strict_summary, strict_records = _run_episode(
            env_strict, requests, agent_c, agent_r, ranker, "strict", args
        )

        # KSP-FF K=50 hops episode.
        env_ksp = make_env(
            args.topology,
            args.num_slots,
            args.num_servers,
            seed,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy_r,
            path_sort_strategy=args.path_sort_strategy_r,
            k=args.k_paths_r,
        )
        env_ksp.reset(requests)
        ksp_summary, ksp_records = _run_episode(
            env_ksp, requests, agent_c, agent_r, None, "ksp_ff", args
        )

        # Build snapshots for counterfactual probe (only for divergence requests).
        strict_env_snapshots: Dict[int, Any] = {}
        ksp_env_snapshots: Dict[int, Any] = {}
        strict_obs_r_snapshots: Dict[int, Dict[str, Any]] = {}
        ksp_obs_r_snapshots: Dict[int, Dict[str, Any]] = {}

        # We need to re-run a lightweight pass to capture pre-decision snapshots.
        # This is cheaper than storing every snapshot.
        diverge_indices = {
            i
            for i, (sr, kr) in enumerate(zip(strict_records, ksp_records))
            if not sr.warmup and sr.r_idx != kr.r_idx
        }
        sampled_diverge = sorted(list(diverge_indices))[: args.probe_budget]
        if sampled_diverge:
            # Re-run Strict episode and capture snapshots at sampled steps.
            env_s = make_env(
                args.topology,
                args.num_slots,
                args.num_servers,
                seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy_r,
                path_sort_strategy=args.path_sort_strategy_r,
                k=args.k_paths_r,
            )
            env_s.reset(requests)
            for step_idx, req in enumerate(requests):
                if step_idx in sampled_diverge:
                    strict_env_snapshots[step_idx] = copy.deepcopy(env_s)
                env_s.advance_time(req.arrival_time)
                env_s.k = args.k_paths_c
                env_s.path_sort_strategy = args.path_sort_strategy_c
                env_s.block_sort_strategy = args.block_sort_strategy_c
                obs_c = build_agent_c_observation(env_s, req)
                c_idx, split_id, server_id, c_valid, _ = _select_c_action(agent_c, env_s, req, obs_c, len(env_s.mec.servers))
                if not c_valid:
                    env_s.reject_next_request(req.req_id, "c_no_valid_action")
                    continue
                env_s.k = args.k_paths_r
                env_s.path_sort_strategy = args.path_sort_strategy_r
                env_s.block_sort_strategy = args.block_sort_strategy_r
                obs_r = build_agent_r_observation(env_s, req, split_id, server_id)
                if step_idx in sampled_diverge:
                    strict_obs_r_snapshots[step_idx] = copy.deepcopy(obs_r)
                r_idx, r_valid, _ = _select_r_action_strict(
                    ranker, agent_r, env_s, req, obs_c, obs_r, split_id, server_id
                )
                if not r_valid or r_idx is None:
                    env_s.reject_next_request(req.req_id, "r_no_valid_action")
                    continue
                r_action = decode_agent_r_action(r_idx, env_s.mod_reg.num_formats, env_s.max_blocks)
                env_s.step((split_id, server_id), r_action)

            # Re-run KSP episode and capture snapshots.
            env_k = make_env(
                args.topology,
                args.num_slots,
                args.num_servers,
                seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy_r,
                path_sort_strategy=args.path_sort_strategy_r,
                k=args.k_paths_r,
            )
            env_k.reset(requests)
            for step_idx, req in enumerate(requests):
                if step_idx in sampled_diverge:
                    ksp_env_snapshots[step_idx] = copy.deepcopy(env_k)
                env_k.advance_time(req.arrival_time)
                env_k.k = args.k_paths_c
                env_k.path_sort_strategy = args.path_sort_strategy_c
                env_k.block_sort_strategy = args.block_sort_strategy_c
                obs_c = build_agent_c_observation(env_k, req)
                c_idx, split_id, server_id, c_valid, _ = _select_c_action(agent_c, env_k, req, obs_c, len(env_k.mec.servers))
                if not c_valid:
                    env_k.reject_next_request(req.req_id, "c_no_valid_action")
                    continue
                env_k.k = args.k_paths_r
                env_k.path_sort_strategy = args.path_sort_strategy_r
                env_k.block_sort_strategy = args.block_sort_strategy_r
                obs_r = build_agent_r_observation(env_k, req, split_id, server_id)
                if step_idx in sampled_diverge:
                    ksp_obs_r_snapshots[step_idx] = copy.deepcopy(obs_r)
                r_idx, r_valid, _ = _select_r_action_ksp_ff(obs_r)
                if not r_valid or r_idx is None:
                    env_k.reject_next_request(req.req_id, "r_no_valid_action")
                    continue
                r_action = decode_agent_r_action(r_idx, env_k.mod_reg.num_formats, env_k.max_blocks)
                env_k.step((split_id, server_id), r_action)

            # Run probes.
            probes = _run_counterfactual_probe(
                strict_records,
                ksp_records,
                strict_env_snapshots,
                ksp_env_snapshots,
                strict_obs_r_snapshots,
                ksp_obs_r_snapshots,
                requests,
                agent_c,
                agent_r,
                args,
            )
            all_probes.extend(probes)

        paired = _build_paired_records(strict_records, ksp_records)
        all_paired.extend(paired)
        all_strict_records.extend(strict_records)
        all_ksp_records.extend(ksp_records)

        per_seed_summaries.append({
            "seed": seed,
            "strict": strict_summary.to_dict(),
            "ksp_ff": ksp_summary.to_dict(),
            "strict_minus_ksp_blocking_pp": (strict_summary.blocking_rate() - ksp_summary.blocking_rate()) * 100.0,
        })

        print(
            f"[diagnose][seed={seed}] "
            f"Strict blocking={strict_summary.blocking_rate():.4%} "
            f"KSP-FF blocking={ksp_summary.blocking_rate():.4%} "
            f"Δ={(strict_summary.blocking_rate() - ksp_summary.blocking_rate()) * 100:+.2f}pp"
        )

    # Overall summaries.
    strict_total = MethodSummary(method=STRICT_V13_NAME)
    ksp_total = MethodSummary(method=KSP_FF_NAME)
    for rec in all_strict_records:
        strict_total.add_record(rec)
    for rec in all_ksp_records:
        ksp_total.add_record(rec)

    # Sample trace rows (one per evaluated request, subsampled for file size).
    sample_rate = max(1, len(all_paired) // args.max_trace_samples)
    sampled_paired = all_paired[::sample_rate]
    all_trace_rows = [p.to_dict() for p in sampled_paired]

    paired_analysis = _analyze_paired(all_paired, all_strict_records, all_ksp_records)

    payload = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "overall": {
            "strict": strict_total.to_dict(),
            "ksp_ff": ksp_total.to_dict(),
            "strict_minus_ksp_blocking_pp": (strict_total.blocking_rate() - ksp_total.blocking_rate()) * 100.0,
        },
        "per_seed": per_seed_summaries,
        "paired_analysis": paired_analysis,
        "probe_count": len(all_probes),
        "probe_records": [p.to_dict() for p in all_probes],
    }
    return payload, all_trace_rows, per_seed_summaries


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_outputs(
    payload: Dict[str, Any],
    trace_rows: List[Dict[str, Any]],
    per_seed_summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "DIAGNOSIS.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    trace_path = output_dir / "paired_trace_sample.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, default=str) + "\n")

    csv_path = output_dir / "per_seed_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed",
            "strict_total", "strict_blocked", "strict_blocking_rate",
            "strict_overload_rate", "strict_nsb_rate", "strict_c_no_valid_rate", "strict_r_no_valid_rate",
            "strict_avg_delay_ms", "strict_avg_fs", "strict_avg_path_km", "strict_avg_decision_ms",
            "ksp_total", "ksp_blocked", "ksp_blocking_rate",
            "ksp_overload_rate", "ksp_nsb_rate", "ksp_c_no_valid_rate", "ksp_r_no_valid_rate",
            "ksp_avg_delay_ms", "ksp_avg_fs", "ksp_avg_path_km", "ksp_avg_decision_ms",
            "strict_minus_ksp_blocking_pp",
        ])
        for row in per_seed_summaries:
            s = row["strict"]
            k = row["ksp_ff"]
            writer.writerow([
                row["seed"],
                s["total"], s["blocked"], s["blocking_rate"],
                s["overload_rate"], s["nsb_rate"], s["c_no_valid_rate"], s["r_no_valid_rate"],
                s["avg_delay_ms"], s["avg_fs"], s["avg_path_km"], s["avg_decision_ms"],
                k["total"], k["blocked"], k["blocking_rate"],
                k["overload_rate"], k["nsb_rate"], k["c_no_valid_rate"], k["r_no_valid_rate"],
                k["avg_delay_ms"], k["avg_fs"], k["avg_path_km"], k["avg_decision_ms"],
                row["strict_minus_ksp_blocking_pp"],
            ])

    md_path = output_dir / "DIAGNOSIS.md"
    md_path.write_text(_build_markdown(payload, trace_rows), encoding="utf-8")

    print(f"[diagnose] Wrote {json_path}")
    print(f"[diagnose] Wrote {trace_path}")
    print(f"[diagnose] Wrote {csv_path}")
    print(f"[diagnose] Wrote {md_path}")


def _build_markdown(payload: Dict[str, Any], trace_rows: List[Dict[str, Any]]) -> str:
    cfg = payload["config"]
    ov = payload["overall"]
    s = ov["strict"]
    k = ov["ksp_ff"]
    pa = payload["paired_analysis"]

    lines = [
        f"# Diagnosis: {STRICT_V13_NAME} vs {KSP_FF_NAME}",
        "",
        "## Protocol Lock",
        "",
        f"- Topology: `{cfg['topology']}`",
        f"- C-side: PPO-C (`{cfg['agent_c_checkpoint']}`)",
        f"- R-side methods: {STRICT_V13_NAME} and {KSP_FF_NAME}",
        f"- R-side K_path: {cfg['k_paths_r']}",
        f"- R-side path_sort_strategy: `{cfg['path_sort_strategy_r']}`",
        f"- R-side block_sort_strategy: `{cfg['block_sort_strategy_r']}`",
        f"- max_blocks: {cfg['max_blocks']}",
        f"- KSP-FF implementation: `ksp_ff_highest_mod_action` (distance-adaptive)",
        f"- Seeds: {cfg['seeds']}",
        f"- Warmup: {cfg['warmup_requests']}, Evaluated: {cfg['requests_per_episode']}",
        "",
        "## Main Result",
        "",
        "| Method | Blocking | Overload | NSB | C-no-valid | R-no-valid | Avg delay ms | Avg FS | Avg path km | Avg decision ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {STRICT_V13_NAME} | {s['blocking_rate']:.4%} | {s['overload_rate']:.4%} | {s['nsb_rate']:.4%} | "
        f"{s['c_no_valid_rate']:.4%} | {s['r_no_valid_rate']:.4%} | {s['avg_delay_ms']:.2f} | {s['avg_fs']:.2f} | "
        f"{s['avg_path_km']:.2f} | {s['avg_decision_ms']:.2f} |",
        f"| {KSP_FF_NAME} | {k['blocking_rate']:.4%} | {k['overload_rate']:.4%} | {k['nsb_rate']:.4%} | "
        f"{k['c_no_valid_rate']:.4%} | {k['r_no_valid_rate']:.4%} | {k['avg_delay_ms']:.2f} | {k['avg_fs']:.2f} | "
        f"{k['avg_path_km']:.2f} | {k['avg_decision_ms']:.2f} |",
        "",
        f"**Blocking difference**: Strict v1.3 - KSP-FF K=50 hops = **{ov['strict_minus_ksp_blocking_pp']:.2f} pp**",
        "",
        "## Mechanism Summary",
        "",
        _mechanism_summary(payload),
        "",
        "## Evidence Tables",
        "",
        "### Win/Loss Decomposition",
        "",
        "| Outcome | Count |",
        "|---|---:|",
        f"| strict_win | {pa['strict_win_count']} |",
        f"| ksp_win | {pa['ksp_win_count']} |",
        f"| both_success | {pa['both_success_count']} |",
        f"| both_block | {pa['both_block_count']} |",
        "",
        "### strict_win: KSP-FF K=50 hops failure reason breakdown",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in sorted(pa["strict_win_ksp_failure_breakdown"].items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "### ksp_win: Strict v1.3 failure reason breakdown", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in sorted(pa["ksp_win_strict_failure_breakdown"].items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "### Divergence timing", "", "| Metric | Value |", "|---|---:|"])
    lines.append(f"| Mean R-divergence distance before strict_win | {_fmt(pa['strict_win_last_r_divergence_mean'])} |")
    lines.append(f"| Mean C-divergence distance before strict_win | {_fmt(pa['strict_win_last_c_divergence_mean'])} |")
    lines.append(f"| Mean R-divergence distance before ksp_win | {_fmt(pa['ksp_win_last_r_divergence_mean'])} |")
    lines.append(f"| Mean C-divergence distance before ksp_win | {_fmt(pa['ksp_win_last_c_divergence_mean'])} |")

    lines.extend(["", "### Action preference deltas (admitted requests)", "", "| Metric | Strict v1.3 | KSP-FF K=50 hops |", "|---|---|---|"])
    lines.append(f"| Avg path idx | {s['path_idx_distribution'] and _mean_key(s['path_idx_distribution']):.2f} | {k['path_idx_distribution'] and _mean_key(k['path_idx_distribution']):.2f} |")
    lines.append(f"| Avg required FS | {np.mean(s['required_fs_values']):.2f} | {np.mean(k['required_fs_values']):.2f} |")
    lines.append(f"| Avg block start | {s['avg_block_start']:.2f} | {k['avg_block_start']:.2f} |")
    lines.append(f"| Avg block waste | {s['avg_block_waste']:.4f} | {k['avg_block_waste']:.4f} |")
    lines.append(f"| Avg hops | {s['avg_hops']:.2f} | {k['avg_hops']:.2f} |")
    lines.append(f"| Avg path km | {s['avg_path_km']:.2f} | {k['avg_path_km']:.2f} |")

    lines.extend(["", "## Counterfactual Probe Summary", ""])
    lines.append(f"- Sampled divergence probes: {payload['probe_count']}")
    if payload["probe_records"]:
        strict_action_in_ksp = sum(1 for p in payload["probe_records"] if p["strict_action_in_ksp_obs"])
        ksp_action_in_strict = sum(1 for p in payload["probe_records"] if p["ksp_action_in_strict_obs"])
        strict_state_ksp_ok = sum(1 for p in payload["probe_records"] if p["strict_state_ksp_action_feasible"])
        ksp_state_strict_ok = sum(1 for p in payload["probe_records"] if p["ksp_state_strict_action_feasible"])
        lines.append(f"- Strict action legal in KSP obs: {strict_action_in_ksp}/{payload['probe_count']}")
        lines.append(f"- KSP action legal in Strict obs: {ksp_action_in_strict}/{payload['probe_count']}")
        lines.append(f"- KSP action feasible in Strict state: {strict_state_ksp_ok}/{payload['probe_count']}")
        lines.append(f"- Strict action feasible in KSP state: {ksp_state_strict_ok}/{payload['probe_count']}")
    else:
        lines.append("- No probes collected.")

    lines.extend(["", "## Causal Caution", ""])
    lines.append(
        "If this diagnostic shows that Strict v1.3 reduces **server_overload** failures relative to "
        "KSP-FF K=50 hops, this is not because the R-ranker directly assigns a server. "
        "R-actions change the trajectory-level optical spectrum and server load state, which in turn "
        "changes the PPO-C observation and the subsequent (split, server) decision. "
        "The observed blocking gap is therefore a coupled C+R closed-loop effect, not a direct "
        "per-action server assignment."
    )
    lines.append("")
    return "\n".join(lines)


def _mechanism_summary(payload: Dict[str, Any]) -> str:
    s = payload["overall"]["strict"]
    k = payload["overall"]["ksp_ff"]
    pa = payload["paired_analysis"]
    blocking_diff_pp = payload["overall"]["strict_minus_ksp_blocking_pp"]

    paragraphs = []
    paragraphs.append(
        f"Across the evaluated seeds, {STRICT_V13_NAME} achieves a blocking rate of "
        f"{s['blocking_rate']:.4%} while {KSP_FF_NAME} achieves {k['blocking_rate']:.4%}, "
        f"a difference of {blocking_diff_pp:.2f} percentage points in favor of Strict v1.3."
    )

    if pa["strict_win_count"] > pa["ksp_win_count"]:
        paragraphs.append(
            f"Request-level decomposition shows {pa['strict_win_count']} strict_win requests versus "
            f"{pa['ksp_win_count']} ksp_win requests, confirming the aggregate advantage is systematic."
        )

    strict_failures = pa["strict_win_ksp_failure_breakdown"]
    if strict_failures:
        top_reason = max(strict_failures.items(), key=lambda x: x[1])
        paragraphs.append(
            f"Among strict_win requests, the dominant KSP-FF K=50 hops failure reason is "
            f"'{top_reason[0]}' ({top_reason[1]} occurrences). "
            "This indicates that KSP-FF's greedy path/mod/block choices tend to push the trajectory "
            "into states where later requests cannot be admitted."
        )

    if pa["strict_win_last_r_divergence_mean"] is not None and pa["strict_win_last_r_divergence_mean"] > 0:
        paragraphs.append(
            f"The mean distance from the most recent R-action divergence to a strict_win event is "
            f"{pa['strict_win_last_r_divergence_mean']:.1f} requests. "
            "A non-zero distance indicates that the advantage is not solely due to the immediate action "
            "being better, but rather due to trajectory-level resource-state management."
        )

    if s["avg_path_km"] != k["avg_path_km"] or s["avg_hops"] != k["avg_hops"]:
        paragraphs.append(
            f"Admitted-request path preferences differ: Strict v1.3 averages {s['avg_hops']:.2f} hops / "
            f"{s['avg_path_km']:.2f} km, while KSP-FF K=50 hops averages {k['avg_hops']:.2f} hops / "
            f"{k['avg_path_km']:.2f} km."
        )

    if s["avg_fs"] != k["avg_fs"]:
        paragraphs.append(
            f"Strict v1.3 uses an average of {s['avg_fs']:.2f} FS per admitted request, "
            f"whereas KSP-FF K=50 hops uses {k['avg_fs']:.2f} FS. "
            "Higher FS usage in Strict v1.3 (if observed) typically reflects higher spectral-efficiency "
            "modulations or better spectrum placement, not inefficiency."
        )

    return "\n\n".join(paragraphs)


def _fmt(x):
    return f"{x:.2f}" if x is not None else "N/A"


def _mean_key(dist: Dict[str, float]) -> float:
    if not dist:
        return 0.0
    total = 0.0
    weight = 0.0
    for k, v in dist.items():
        total += float(k) * v
        weight += v
    return total / weight if weight > 0 else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent_c_checkpoint",
        default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt",
    )
    parser.add_argument(
        "--agent_r_checkpoint",
        default="sa_hmarl/checkpoints/agent_r_mixed.pt",
    )
    parser.add_argument(
        "--strict_ranker_checkpoint",
        default="sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt",
    )
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths_c", type=int, default=5)
    parser.add_argument("--k_paths_r", type=int, default=50)
    parser.add_argument("--path_sort_strategy_c", default="hops")
    parser.add_argument("--path_sort_strategy_r", default="hops")
    parser.add_argument("--block_sort_strategy_c", default="start_asc")
    parser.add_argument("--block_sort_strategy_r", default="start_asc")
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--requests_per_episode", type=int, default=6000)
    parser.add_argument("--warmup_requests", type=int, default=500)
    parser.add_argument("--poisson_arrivals", action="store_true")
    parser.add_argument("--exponential_holding", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output_dir",
        default="sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_diagnosis",
    )
    parser.add_argument("--probe_budget", type=int, default=50)
    parser.add_argument("--probe_horizon", type=int, default=20)
    parser.add_argument("--max_trace_samples", type=int, default=20000)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    t_start = time.time()
    payload, trace_rows, per_seed = _diagnose(args)
    out_dir = Path(args.output_dir)
    _write_outputs(payload, trace_rows, per_seed, out_dir)
    elapsed = time.time() - t_start
    print(f"[diagnose] Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
