#!/usr/bin/env python3
"""Diagnosis: Strict v1.3 vs KSP-FF K=50 hops under fixed-C / fixed-OD.

Protocol lock:
- C-side is FIXED: no PPO-C is loaded or called.
- Every request uses the same fixed_src_node, fixed_split_id, fixed_server_id.
- fixed_dst_node = env.mec.servers[fixed_server_id].node_id.
- R-side K_path = 50, path_sort_strategy="hops", block_sort_strategy="start_asc", max_blocks=10.
- Strict v1.3 = PPO-R proposal-supported full-state common-future counterfactual RMSA reranker.
- Strict v1.3 candidate set = PPO-R legal Top-30 only.
- KSP-FF K=50 hops = ksp_ff_highest_mod_action.
- ksp_ff_action (naive flat First-Fit / BPSK-first) is NOT used.

This script does NOT train any network and does NOT modify any checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import (
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_r,
    _select_r_action_from_obs,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5 import (
    _ppo_r_topk_actions,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import SPLIT_PROFILES, make_env


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRICT_V13_NAME = "Strict v1.3"
KSP_FF_NAME = "KSP-FF K=50 hops"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class FixedRequestRecord:
    """Per-request record for one method under fixed-C / fixed-OD."""

    seed: int
    step_idx: int
    req_id: int
    arrival_time: float
    warmup: bool

    # Fixed C-side
    fixed_src_node: int
    fixed_split_id: int
    fixed_server_id: int
    fixed_dst_node: int

    # R-side action
    r_idx: Optional[int]
    r_valid: bool
    path_idx: int
    modulation_idx: int
    block_idx: int
    modulation: str
    required_fs: int
    block_start: int
    block_size: int
    block_waste: float
    path_hops: int
    path_length_km: float
    path_nodes: List[int]

    # State before decision
    free_ratio_before: float
    lfb_before: float
    fragmentation_before: float
    phi_spec_before: float
    legal_r_action_count: int

    # Strict diagnostics
    strict_top30_contains_ksp_action: bool
    strict_rank_of_ksp_action_if_contained: int
    strict_selected_rank_in_ppo_r_top30: int

    # Outcome
    success: bool
    failure_reason: str
    delay_ms: float
    decision_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "step_idx": self.step_idx,
            "req_id": self.req_id,
            "arrival_time": self.arrival_time,
            "warmup": self.warmup,
            "fixed_src_node": self.fixed_src_node,
            "fixed_split_id": self.fixed_split_id,
            "fixed_server_id": self.fixed_server_id,
            "fixed_dst_node": self.fixed_dst_node,
            "r_idx": self.r_idx,
            "r_valid": self.r_valid,
            "path_idx": self.path_idx,
            "modulation_idx": self.modulation_idx,
            "block_idx": self.block_idx,
            "modulation": self.modulation,
            "required_fs": self.required_fs,
            "block_start": self.block_start,
            "block_size": self.block_size,
            "block_waste": self.block_waste,
            "path_hops": self.path_hops,
            "path_length_km": self.path_length_km,
            "path_nodes": self.path_nodes,
            "free_ratio_before": self.free_ratio_before,
            "lfb_before": self.lfb_before,
            "fragmentation_before": self.fragmentation_before,
            "phi_spec_before": self.phi_spec_before,
            "legal_r_action_count": self.legal_r_action_count,
            "strict_top30_contains_ksp_action": self.strict_top30_contains_ksp_action,
            "strict_rank_of_ksp_action_if_contained": self.strict_rank_of_ksp_action_if_contained,
            "strict_selected_rank_in_ppo_r_top30": self.strict_selected_rank_in_ppo_r_top30,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "delay_ms": self.delay_ms,
            "decision_ms": self.decision_ms,
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
    r_no_valid: int = 0
    deadline_infeasible: int = 0
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

    def add_record(self, rec: FixedRequestRecord) -> None:
        if rec.warmup:
            return
        self.total += 1
        self.decision_ms_sum += rec.decision_ms
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
            elif reason == "r_no_valid_action":
                self.r_no_valid += 1
            elif reason in ("deadline_infeasible", "deadline_failure"):
                self.deadline_infeasible += 1
            else:
                self.other_failure += 1

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.total, 1))

    def overload_rate(self) -> float:
        return float(self.server_overload / max(self.total, 1))

    def nsb_rate(self) -> float:
        return float(self.no_suitable_block / max(self.total, 1))

    def r_no_valid_rate(self) -> float:
        return float(self.r_no_valid / max(self.total, 1))

    def deadline_rate(self) -> float:
        return float(self.deadline_infeasible / max(self.total, 1))

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
            "r_no_valid": self.r_no_valid,
            "r_no_valid_rate": self.r_no_valid_rate(),
            "deadline_infeasible": self.deadline_infeasible,
            "deadline_rate": self.deadline_rate(),
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
    warmup: bool
    fixed_src_node: int
    fixed_split_id: int
    fixed_server_id: int
    fixed_dst_node: int

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

    ksp_path_idx: int
    ksp_modulation: str
    ksp_required_fs: int
    ksp_block_start: int
    ksp_block_size: int
    ksp_block_waste: float
    ksp_path_hops: int
    ksp_path_length_km: float

    strict_success: bool
    ksp_success: bool
    strict_failure_reason: str
    ksp_failure_reason: str
    outcome_class: str

    strict_top30_contains_ksp_action: bool
    strict_rank_of_ksp_action_if_contained: int
    strict_selected_rank_in_ppo_r_top30: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "step_idx": self.step_idx,
            "req_id": self.req_id,
            "arrival_time": self.arrival_time,
            "warmup": self.warmup,
            "fixed_src_node": self.fixed_src_node,
            "fixed_split_id": self.fixed_split_id,
            "fixed_server_id": self.fixed_server_id,
            "fixed_dst_node": self.fixed_dst_node,
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
            "ksp_path_idx": self.ksp_path_idx,
            "ksp_modulation": self.ksp_modulation,
            "ksp_required_fs": self.ksp_required_fs,
            "ksp_block_start": self.ksp_block_start,
            "ksp_block_size": self.ksp_block_size,
            "ksp_block_waste": self.ksp_block_waste,
            "ksp_path_hops": self.ksp_path_hops,
            "ksp_path_length_km": self.ksp_path_length_km,
            "strict_success": self.strict_success,
            "ksp_success": self.ksp_success,
            "strict_failure_reason": self.strict_failure_reason,
            "ksp_failure_reason": self.ksp_failure_reason,
            "outcome_class": self.outcome_class,
            "strict_top30_contains_ksp_action": self.strict_top30_contains_ksp_action,
            "strict_rank_of_ksp_action_if_contained": self.strict_rank_of_ksp_action_if_contained,
            "strict_selected_rank_in_ppo_r_top30": self.strict_selected_rank_in_ppo_r_top30,
        }


# ---------------------------------------------------------------------------
# Helpers
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


def _select_r_action_strict(
    ranker: Dict[str, Any],
    agent_r,
    env,
    req,
    obs_r: Dict[str, Any],
    split_id: int,
    server_id: int,
) -> Tuple[Optional[int], bool, Dict[str, Any]]:
    """Return (r_idx, r_valid, info) for Strict v1.3."""
    info: Dict[str, Any] = {
        "candidate_count": 0,
        "fallback": False,
        "in_candidates": False,
        "score_margin": 0.0,
        "strict_selected_rank_in_ppo_r_top30": -1,
        "strict_top30_contains_ksp_action": False,
        "strict_rank_of_ksp_action_if_contained": -1,
    }
    candidates = _ppo_r_topk_actions(agent_r, obs_r, 30)
    candidates = np.asarray(candidates, dtype=np.int64)
    has_candidates = candidates.size > 0
    info["candidate_count"] = int(candidates.size)
    info["fallback"] = not has_candidates
    if not has_candidates:
        return None, False, info

    # Check KSP-FF action coverage in Top-30.
    ksp_idx = ksp_ff_highest_mod_action(obs_r)
    if ksp_idx is not None:
        ksp_idx = int(ksp_idx)
        info["strict_top30_contains_ksp_action"] = int(ksp_idx) in candidates
        if info["strict_top30_contains_ksp_action"]:
            info["strict_rank_of_ksp_action_if_contained"] = int(
                np.where(candidates == ksp_idx)[0][0]
            )

    # Build a minimal obs_c for the 25-d feature vector.
    num_servers = len(env.mec.servers)
    num_splits = len(req.splits)
    num_paths = len(obs_r["candidate_paths"])
    num_mods = len(obs_r["mod_names"])
    server_utilizations = [s.utilization for s in env.mec.servers]
    feasible_count = int(obs_r.get("feasible_count", 0))
    feasible_counts = [[0] * num_servers for _ in range(num_splits)]
    if 0 <= split_id < num_splits and 0 <= server_id < num_servers:
        feasible_counts[split_id][server_id] = feasible_count

    candidate_features = []
    for sp_id in range(num_splits):
        for sv_id in range(num_servers):
            split = req.splits[sp_id]
            server = env.mec.servers[sv_id]
            edge_ms = env._estimate_compute_ms(server, split.edge_compute_cost)
            is_target = (sp_id == split_id and sv_id == server_id)
            feat = {
                "server_available_compute": server.available_compute,
                "edge_compute_cost": split.edge_compute_cost,
                "edge_compute_ms": edge_ms,
                "_feasible_mask_per_path_mod": obs_r["feasible_mask_per_path_mod"] if is_target else [[False] * num_mods for _ in range(num_paths)],
                "_required_fs_per_path_mod": obs_r["required_fs_per_path_mod"] if is_target else [[None] * num_mods for _ in range(num_paths)],
                "_candidate_blocks_per_path_mod": obs_r["candidate_blocks_per_path_mod"] if is_target else [[[] for _ in range(num_mods)] for _ in range(num_paths)],
                "_mod_formats": num_mods,
            }
            candidate_features.append(feat)

    obs_c_minimal = {
        "server_utilizations": server_utilizations,
        "feasible_counts": feasible_counts,
        "candidate_features": candidate_features,
    }

    # Rank candidates with Strict v1.3 ranker.
    r_features, _ = agent_r.build_action_features(obs_r)
    feats = []
    for a in candidates:
        feat = _r_feature_vector(
            env,
            req,
            obs_c_minimal,
            obs_r,
            r_features,
            int(a),
            split_id,
            server_id,
            feature_names=ranker["feature_names"],
        )
        feats.append(feat)
    x = np.stack(feats, axis=0).astype(np.float32)
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
    info["strict_selected_rank_in_ppo_r_top30"] = best_local
    sorted_scores = np.sort(scores)[::-1]
    info["score_margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
    return r_idx, True, info


def _select_r_action_ksp_ff(obs_r: Dict[str, Any]) -> Tuple[Optional[int], bool, Dict[str, Any]]:
    """Return (r_idx, r_valid, info) for KSP-FF K=50 hops."""
    info: Dict[str, Any] = {}
    a = ksp_ff_highest_mod_action(obs_r)
    if a is None:
        return None, False, info
    return int(a), True, info


def _extract_r_action_meta(obs_r: Dict[str, Any], r_action: Tuple[int, int, int]) -> Dict[str, Any]:
    path_idx, mod_idx, block_idx = r_action
    path_feats = obs_r.get("path_features", [])
    meta = {
        "path_idx": path_idx,
        "mod_idx": mod_idx,
        "block_idx": block_idx,
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


def _spectrum_state(env) -> Tuple[float, float, float, float]:
    """Return (free_ratio, lfb, fragmentation, phi_spec)."""
    stats = env.net.get_global_spectrum_stats()
    free_ratio = float(stats.get("free_ratio", 0.0))
    lfb = float(stats.get("largest_free_block_ratio", 0.0))
    frag = float(stats.get("avg_frag_index", 0.0))
    # Simple phi_spec proxy: weighted combination.
    phi_spec = float(np.log1p(free_ratio * 100.0) + 0.3 * np.log1p(lfb * 100.0))
    return free_ratio, lfb, frag, phi_spec


def _generate_fixed_od_requests(
    num_nodes: int,
    fixed_src_node: int,
    rng: np.random.RandomState,
    num_requests: int,
    arrival_interval: float,
    holding_min: float,
    holding_max: float,
    deadline_min: float,
    deadline_max: float,
    size_min_mb: float,
    size_max_mb: float,
    edge_cost_min: float,
    edge_cost_max: float,
    num_splits: int,
    split_profile: str,
    poisson_arrivals: bool = False,
    exponential_holding: bool = False,
) -> List[DNNRequest]:
    """Generate requests with a fixed source node."""
    profile = SPLIT_PROFILES.get(split_profile, SPLIT_PROFILES["default3"])
    size_mults = profile["size_multipliers"]
    edge_ranges = profile["edge_ratio_ranges"]
    if num_splits != len(size_mults):
        raise ValueError(
            f"Profile '{split_profile}' expects {len(size_mults)} splits, but num_splits={num_splits}."
        )

    holding_mean = (holding_min + holding_max) / 2.0
    arrival_time = 0.0
    requests = []
    for i in range(num_requests):
        if poisson_arrivals:
            arrival_time += rng.exponential(arrival_interval)
        else:
            arrival_time += arrival_interval

        src_node = fixed_src_node

        if exponential_holding:
            holding_time = rng.exponential(holding_mean)
            holding_time = max(holding_time, 1e-3)
        else:
            holding_time = rng.uniform(holding_min, holding_max)

        base_size = rng.uniform(size_min_mb, size_max_mb)
        total_compute = rng.uniform(edge_cost_min + 0.5, edge_cost_max + 2.0)

        splits = []
        for split_id in range(num_splits):
            lo_size, hi_size = size_mults[split_id]
            lo_edge, hi_edge = edge_ranges[split_id]
            size_mb = base_size * rng.uniform(lo_size, hi_size)
            edge_ratio = rng.uniform(lo_edge, hi_edge)
            edge_cost = total_compute * edge_ratio
            local_cost = total_compute * (1.0 - edge_ratio)
            splits.append(SplitProfile(
                split_id=split_id,
                intermediate_size_mb=size_mb,
                local_compute_cost=local_cost,
                edge_compute_cost=edge_cost,
            ))

        req = DNNRequest(
            req_id=i,
            src_node=src_node,
            arrival_time=arrival_time,
            holding_time=holding_time,
            deadline_ms=rng.uniform(deadline_min, deadline_max),
            splits=splits,
        )
        requests.append(req)
    return requests


def _run_episode(
    env,
    requests: List[Any],
    agent_r,
    ranker: Optional[Dict[str, Any]],
    mode: str,  # "strict" or "ksp_ff"
    args: argparse.Namespace,
) -> Tuple[MethodSummary, List[FixedRequestRecord]]:
    """Run one episode under fixed-C / fixed-OD."""
    assert mode in ("strict", "ksp_ff")
    records: List[FixedRequestRecord] = []
    summary = MethodSummary(method=STRICT_V13_NAME if mode == "strict" else KSP_FF_NAME)
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks

    fixed_dst_node = int(env.mec.servers[args.fixed_server_id].node_id)

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < args.warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        free_ratio, lfb, frag, phi_spec = _spectrum_state(env)

        # R-side observation under fixed split/server.
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, args.fixed_split_id, args.fixed_server_id)
        agent_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        legal_count = int(agent_r_mask.sum())

        if not agent_r_mask.any():
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = FixedRequestRecord(
                    seed=args._current_seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    warmup=False,
                    fixed_src_node=args.fixed_src_node,
                    fixed_split_id=args.fixed_split_id,
                    fixed_server_id=args.fixed_server_id,
                    fixed_dst_node=fixed_dst_node,
                    r_idx=None,
                    r_valid=False,
                    path_idx=-1,
                    modulation_idx=-1,
                    block_idx=-1,
                    modulation="",
                    required_fs=0,
                    block_start=-1,
                    block_size=0,
                    block_waste=0.0,
                    path_hops=0,
                    path_length_km=0.0,
                    path_nodes=[],
                    free_ratio_before=free_ratio,
                    lfb_before=lfb,
                    fragmentation_before=frag,
                    phi_spec_before=phi_spec,
                    legal_r_action_count=legal_count,
                    strict_top30_contains_ksp_action=False,
                    strict_rank_of_ksp_action_if_contained=-1,
                    strict_selected_rank_in_ppo_r_top30=-1,
                    success=False,
                    failure_reason="r_no_valid_action",
                    delay_ms=0.0,
                    decision_ms=(time.perf_counter() - t0) * 1000.0,
                )
                records.append(rec)
                summary.add_record(rec)
            continue

        if mode == "strict":
            r_idx, r_valid, r_info = _select_r_action_strict(
                ranker, agent_r, env, req, obs_r, args.fixed_split_id, args.fixed_server_id
            )
            strict_top30_contains_ksp = r_info.get("strict_top30_contains_ksp_action", False)
            strict_rank_of_ksp = r_info.get("strict_rank_of_ksp_action_if_contained", -1)
            strict_selected_rank = r_info.get("strict_selected_rank_in_ppo_r_top30", -1)
        else:
            r_idx, r_valid, r_info = _select_r_action_ksp_ff(obs_r)
            strict_top30_contains_ksp = False
            strict_rank_of_ksp = -1
            strict_selected_rank = -1

        if not r_valid or r_idx is None:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = FixedRequestRecord(
                    seed=args._current_seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    warmup=False,
                    fixed_src_node=args.fixed_src_node,
                    fixed_split_id=args.fixed_split_id,
                    fixed_server_id=args.fixed_server_id,
                    fixed_dst_node=fixed_dst_node,
                    r_idx=None,
                    r_valid=False,
                    path_idx=-1,
                    modulation_idx=-1,
                    block_idx=-1,
                    modulation="",
                    required_fs=0,
                    block_start=-1,
                    block_size=0,
                    block_waste=0.0,
                    path_hops=0,
                    path_length_km=0.0,
                    path_nodes=[],
                    free_ratio_before=free_ratio,
                    lfb_before=lfb,
                    fragmentation_before=frag,
                    phi_spec_before=phi_spec,
                    legal_r_action_count=legal_count,
                    strict_top30_contains_ksp_action=strict_top30_contains_ksp,
                    strict_rank_of_ksp_action_if_contained=strict_rank_of_ksp,
                    strict_selected_rank_in_ppo_r_top30=strict_selected_rank,
                    success=False,
                    failure_reason="r_no_valid_action",
                    delay_ms=0.0,
                    decision_ms=(time.perf_counter() - t0) * 1000.0,
                )
                records.append(rec)
                summary.add_record(rec)
            continue

        r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
        _, _, _, info = env.step((args.fixed_split_id, args.fixed_server_id), r_action)

        if not is_warmup:
            meta = _extract_r_action_meta(obs_r, r_action)
            success = bool(info.get("success", False))
            rec = FixedRequestRecord(
                seed=args._current_seed,
                step_idx=step_idx,
                req_id=int(req.req_id),
                arrival_time=float(req.arrival_time),
                warmup=False,
                fixed_src_node=args.fixed_src_node,
                fixed_split_id=args.fixed_split_id,
                fixed_server_id=args.fixed_server_id,
                fixed_dst_node=fixed_dst_node,
                r_idx=int(r_idx),
                r_valid=True,
                path_idx=int(meta["path_idx"]),
                modulation_idx=int(meta["mod_idx"]),
                block_idx=int(meta["block_idx"]),
                modulation=str(meta["mod_name"]),
                required_fs=int(meta["required_fs"]),
                block_start=int(meta["block_start"]),
                block_size=int(meta["block_size"]),
                block_waste=float(meta["block_waste"]),
                path_hops=int(meta["hop_count"]),
                path_length_km=float(meta["path_length_km"]),
                path_nodes=[int(n) for n in meta["path_nodes"]],
                free_ratio_before=free_ratio,
                lfb_before=lfb,
                fragmentation_before=frag,
                phi_spec_before=phi_spec,
                legal_r_action_count=legal_count,
                strict_top30_contains_ksp_action=strict_top30_contains_ksp,
                strict_rank_of_ksp_action_if_contained=strict_rank_of_ksp,
                strict_selected_rank_in_ppo_r_top30=strict_selected_rank,
                success=success,
                failure_reason=str(info.get("reason", "")) if not success else "",
                delay_ms=float(info.get("delay_ms", 0.0)) if success else 0.0,
                decision_ms=(time.perf_counter() - t0) * 1000.0,
            )
            records.append(rec)
            summary.add_record(rec)

    return summary, records


# ---------------------------------------------------------------------------
# Paired analysis
# ---------------------------------------------------------------------------
def _build_paired_records(
    strict_records: List[FixedRequestRecord],
    ksp_records: List[FixedRequestRecord],
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
                warmup=sr.warmup,
                fixed_src_node=sr.fixed_src_node,
                fixed_split_id=sr.fixed_split_id,
                fixed_server_id=sr.fixed_server_id,
                fixed_dst_node=sr.fixed_dst_node,
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
                ksp_path_idx=kr.path_idx,
                ksp_modulation=kr.modulation,
                ksp_required_fs=kr.required_fs,
                ksp_block_start=kr.block_start,
                ksp_block_size=kr.block_size,
                ksp_block_waste=kr.block_waste,
                ksp_path_hops=kr.path_hops,
                ksp_path_length_km=kr.path_length_km,
                strict_success=sr.success,
                ksp_success=kr.success,
                strict_failure_reason=sr.failure_reason,
                ksp_failure_reason=kr.failure_reason,
                outcome_class=outcome_class,
                strict_top30_contains_ksp_action=sr.strict_top30_contains_ksp_action,
                strict_rank_of_ksp_action_if_contained=sr.strict_rank_of_ksp_action_if_contained,
                strict_selected_rank_in_ppo_r_top30=sr.strict_selected_rank_in_ppo_r_top30,
            )
        )
    return paired


def _analyze_paired(
    paired: List[PairedRecord],
    strict_records: List[FixedRequestRecord],
    ksp_records: List[FixedRequestRecord],
) -> Dict[str, Any]:
    strict_wins = [p for p in paired if p.outcome_class == "strict_win"]
    ksp_wins = [p for p in paired if p.outcome_class == "ksp_win"]

    def _failure_breakdown(items: List[PairedRecord], ksp_field: bool) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in items:
            reason = p.ksp_failure_reason if ksp_field else p.strict_failure_reason
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _last_divergence_distance(items: List[PairedRecord]) -> Dict[str, Any]:
        paired_index = {(p.seed, p.step_idx): i for i, p in enumerate(paired)}
        distances: List[int] = []
        for item in items:
            idx = paired_index.get((item.seed, item.step_idx))
            found = None
            if idx is not None:
                for j in range(idx - 1, -1, -1):
                    prev = paired[j]
                    if prev.seed != item.seed:
                        continue
                    if not prev.same_r_action:
                        found = item.step_idx - prev.step_idx
                        break
            distances.append(found if found is not None else -1)
        return {
            "mean": float(np.mean([d for d in distances if d >= 0])) if any(d >= 0 for d in distances) else None,
            "median": float(np.median([d for d in distances if d >= 0])) if any(d >= 0 for d in distances) else None,
        }

    strict_win_ksp_failure_breakdown = _failure_breakdown(strict_wins, ksp_field=True)
    ksp_win_strict_failure_breakdown = _failure_breakdown(ksp_wins, ksp_field=False)

    strict_win_last_r_divergence = _last_divergence_distance(strict_wins)
    ksp_win_last_r_divergence = _last_divergence_distance(ksp_wins)

    # First divergence positions.
    first_r_divergence_per_seed: Dict[int, int] = {}
    first_outcome_per_seed: Dict[int, int] = {}
    for p in paired:
        seed = p.seed
        if not p.same_r_action and seed not in first_r_divergence_per_seed:
            first_r_divergence_per_seed[seed] = p.step_idx
        if p.outcome_class in ("strict_win", "ksp_win") and seed not in first_outcome_per_seed:
            first_outcome_per_seed[seed] = p.step_idx

    # Top-30 coverage of KSP action.
    ksp_covered = sum(1 for p in paired if p.strict_top30_contains_ksp_action)
    ksp_covered_rate = ksp_covered / max(len(paired), 1)

    # Windowed preference analysis before strict_win requests.
    window_size = 20
    strict_win_preference: Dict[str, Any] = {}
    if strict_wins:
        window_records_strict: List[FixedRequestRecord] = []
        window_records_ksp: List[FixedRequestRecord] = []
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
        "strict_win_last_r_divergence_median": strict_win_last_r_divergence.get("median"),
        "ksp_win_last_r_divergence_mean": ksp_win_last_r_divergence.get("mean"),
        "ksp_win_last_r_divergence_median": ksp_win_last_r_divergence.get("median"),
        "first_r_divergence_per_seed": first_r_divergence_per_seed,
        "first_outcome_per_seed": first_outcome_per_seed,
        "ksp_action_in_strict_top30_rate": ksp_covered_rate,
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

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    ranker = _load_ranker_model(args.strict_ranker_checkpoint, args.device)

    total_requests = args.warmup_requests + args.requests_per_episode
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    per_seed_summaries: List[Dict[str, Any]] = []
    all_paired: List[PairedRecord] = []
    all_trace_rows: List[Dict[str, Any]] = []
    all_strict_records: List[FixedRequestRecord] = []
    all_ksp_records: List[FixedRequestRecord] = []

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
    fixed_dst_node = int(env_for_nodes.mec.servers[args.fixed_server_id].node_id)

    for seed in seeds:
        args._current_seed = seed
        rng = np.random.RandomState(seed)
        requests = _generate_fixed_od_requests(
            num_nodes=env_for_nodes.net.NUM_NODES,
            fixed_src_node=args.fixed_src_node,
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
            env_strict, requests, agent_r, ranker, "strict", args
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
            env_ksp, requests, agent_r, None, "ksp_ff", args
        )

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

    # Sample trace rows.
    sample_rate = max(1, len(all_paired) // args.max_trace_samples)
    sampled_paired = all_paired[::sample_rate]
    all_trace_rows = [p.to_dict() for p in sampled_paired]

    paired_analysis = _analyze_paired(all_paired, all_strict_records, all_ksp_records)

    payload = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "fixed_od": {
            "fixed_src_node": args.fixed_src_node,
            "fixed_split_id": args.fixed_split_id,
            "fixed_server_id": args.fixed_server_id,
            "fixed_dst_node": fixed_dst_node,
            "no_ppo_c_loaded": True,
            "no_ppo_c_action_selected": True,
        },
        "overall": {
            "strict": strict_total.to_dict(),
            "ksp_ff": ksp_total.to_dict(),
            "strict_minus_ksp_blocking_pp": (strict_total.blocking_rate() - ksp_total.blocking_rate()) * 100.0,
        },
        "per_seed": per_seed_summaries,
        "paired_analysis": paired_analysis,
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
            "strict_overload_rate", "strict_nsb_rate", "strict_r_no_valid_rate",
            "strict_avg_delay_ms", "strict_avg_fs", "strict_avg_path_km", "strict_avg_decision_ms",
            "ksp_total", "ksp_blocked", "ksp_blocking_rate",
            "ksp_overload_rate", "ksp_nsb_rate", "ksp_r_no_valid_rate",
            "ksp_avg_delay_ms", "ksp_avg_fs", "ksp_avg_path_km", "ksp_avg_decision_ms",
            "strict_minus_ksp_blocking_pp",
        ])
        for row in per_seed_summaries:
            s = row["strict"]
            k = row["ksp_ff"]
            writer.writerow([
                row["seed"],
                s["total"], s["blocked"], s["blocking_rate"],
                s["overload_rate"], s["nsb_rate"], s["r_no_valid_rate"],
                s["avg_delay_ms"], s["avg_fs"], s["avg_path_km"], s["avg_decision_ms"],
                k["total"], k["blocked"], k["blocking_rate"],
                k["overload_rate"], k["nsb_rate"], k["r_no_valid_rate"],
                k["avg_delay_ms"], k["avg_fs"], k["avg_path_km"], k["avg_decision_ms"],
                row["strict_minus_ksp_blocking_pp"],
            ])

    md_path = output_dir / "DIAGNOSIS.md"
    md_path.write_text(_build_markdown(payload), encoding="utf-8")

    decision_path = output_dir / "NEXT_STEP_DECISION.md"
    decision_path.write_text(_build_next_step_decision(payload), encoding="utf-8")

    print(f"[diagnose] Wrote {json_path}")
    print(f"[diagnose] Wrote {trace_path}")
    print(f"[diagnose] Wrote {csv_path}")
    print(f"[diagnose] Wrote {md_path}")
    print(f"[diagnose] Wrote {decision_path}")


def _build_markdown(payload: Dict[str, Any]) -> str:
    cfg = payload["config"]
    od = payload["fixed_od"]
    ov = payload["overall"]
    s = ov["strict"]
    k = ov["ksp_ff"]
    pa = payload["paired_analysis"]

    lines = [
        f"# Diagnosis: {STRICT_V13_NAME} vs {KSP_FF_NAME} (fixed-C / fixed-OD)",
        "",
        "## Protocol Lock",
        "",
        f"- **Comparison**: {STRICT_V13_NAME} vs {KSP_FF_NAME}",
        f"- Topology: `{cfg['topology']}`",
        f"- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)",
        f"- fixed_src_node: {od['fixed_src_node']}",
        f"- fixed_split_id: {od['fixed_split_id']}",
        f"- fixed_server_id: {od['fixed_server_id']}",
        f"- fixed_dst_node: {od['fixed_dst_node']} (env.mec.servers[{od['fixed_server_id']}].node_id)",
        f"- R-side K_path: {cfg['k_paths_r']}",
        f"- R-side path_sort_strategy: `{cfg['path_sort_strategy_r']}`",
        f"- R-side block_sort_strategy: `{cfg['block_sort_strategy_r']}`",
        f"- max_blocks: {cfg['max_blocks']}",
        f"- KSP-FF implementation: `ksp_ff_highest_mod_action` (distance-adaptive)",
        f"- Seeds: {cfg['seeds']}",
        f"- Warmup: {cfg['warmup_requests']}, Evaluated: {cfg['requests_per_episode']}",
        "",
        "**Strict v1.3 details**:",
        "",
        "- Candidate set: PPO-R legal Top-30 only (`ppo_r_topk_only`)",
        "- Feature dim: 25 pre-decision state-action features",
        "- Label: H=5, gamma=1.0 common-future counterfactual rollout",
        "- Model: MLP 25 -> 128 -> 64 -> 1, SiLU, dropout=0",
        "- Strict loss: listwise KL + SmoothL1, reg_weight=1.0, lambda_pair=0, lambda_hard=0",
        "- Checkpoint: selected by validation regret",
        "- Ranker gate: `all` (E=0/E=1 both invoke ranker)",
        "- Fallback: PPO-R top-1 only when no candidates",
        "- No KSP anchor, no heuristic filler, no diversity/random candidate, no all-legal candidate",
        "",
        "**KSP-FF K=50 hops details**:",
        "",
        "- Function: `ksp_ff_highest_mod_action`",
        "- Path ordering: hops (then km within same hop count)",
        "- Block ordering: start_asc",
        "- Modulation: 4 default formats (BPSK/QPSK/8QAM/16QAM)",
        "- `ksp_ff_action` (naive flat First-Fit / BPSK-first) is NOT used as the formal baseline",
        "",
        "## Main Result",
        "",
        "| Method | Blocking | Overload | NSB | R-no-valid | Deadline | Avg delay ms | Avg FS | Avg path km | Avg decision ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {STRICT_V13_NAME} | {s['blocking_rate']:.4%} | {s['overload_rate']:.4%} | {s['nsb_rate']:.4%} | "
        f"{s['r_no_valid_rate']:.4%} | {s['deadline_rate']:.4%} | {s['avg_delay_ms']:.2f} | {s['avg_fs']:.2f} | "
        f"{s['avg_path_km']:.2f} | {s['avg_decision_ms']:.2f} |",
        f"| {KSP_FF_NAME} | {k['blocking_rate']:.4%} | {k['overload_rate']:.4%} | {k['nsb_rate']:.4%} | "
        f"{k['r_no_valid_rate']:.4%} | {k['deadline_rate']:.4%} | {k['avg_delay_ms']:.2f} | {k['avg_fs']:.2f} | "
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
    lines.append(f"| Median R-divergence distance before strict_win | {_fmt(pa['strict_win_last_r_divergence_median'])} |")
    lines.append(f"| Mean R-divergence distance before ksp_win | {_fmt(pa['ksp_win_last_r_divergence_mean'])} |")
    lines.append(f"| Median R-divergence distance before ksp_win | {_fmt(pa['ksp_win_last_r_divergence_median'])} |")

    lines.extend(["", "### Action preference deltas (admitted requests)", "", "| Metric | Strict v1.3 | KSP-FF K=50 hops |", "|---|---|---|"])
    lines.append(f"| Avg path idx | {s['path_idx_distribution'] and _mean_key(s['path_idx_distribution']):.2f} | {k['path_idx_distribution'] and _mean_key(k['path_idx_distribution']):.2f} |")
    lines.append(f"| Avg required FS | {np.mean(s['required_fs_values']):.2f} | {np.mean(k['required_fs_values']):.2f} |")
    lines.append(f"| Avg block start | {s['avg_block_start']:.2f} | {k['avg_block_start']:.2f} |")
    lines.append(f"| Avg block waste | {s['avg_block_waste']:.4f} | {k['avg_block_waste']:.4f} |")
    lines.append(f"| Avg hops | {s['avg_hops']:.2f} | {k['avg_hops']:.2f} |")
    lines.append(f"| Avg path km | {s['avg_path_km']:.2f} | {k['avg_path_km']:.2f} |")

    lines.extend(["", "### Strict v1.3 candidate coverage", "", "| Metric | Value |", "|---|---:|"])
    lines.append(f"| KSP action in Strict Top-30 rate | {pa['ksp_action_in_strict_top30_rate']:.4%} |")

    lines.extend(["", "## Causal Caution", ""])
    lines.append(
        "Under fixed-C / fixed-OD, the R-side cannot change server selection. "
        "Any observed blocking difference is therefore attributable to optical spectrum "
        "trajectory management, not to server-choice coupling. "
        "If Strict v1.3 reduces server_overload relative to KSP-FF K=50 hops, this must reflect "
        "an indirect load imbalance caused by different success counts, not a direct server assignment."
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
        f"Across the evaluated seeds under fixed-C / fixed-OD, {STRICT_V13_NAME} achieves a blocking rate of "
        f"{s['blocking_rate']:.4%} while {KSP_FF_NAME} achieves {k['blocking_rate']:.4%}, "
        f"a difference of {blocking_diff_pp:.2f} percentage points."
    )

    if pa["strict_win_count"] > pa["ksp_win_count"]:
        paragraphs.append(
            f"Request-level decomposition shows {pa['strict_win_count']} strict_win requests versus "
            f"{pa['ksp_win_count']} ksp_win requests."
        )
    elif pa["ksp_win_count"] > pa["strict_win_count"]:
        paragraphs.append(
            f"Request-level decomposition shows {pa['ksp_win_count']} ksp_win requests versus "
            f"{pa['strict_win_count']} strict_win requests."
        )

    strict_failures = pa["strict_win_ksp_failure_breakdown"]
    if strict_failures:
        top_reason = max(strict_failures.items(), key=lambda x: x[1])
        paragraphs.append(
            f"Among strict_win requests, the dominant KSP-FF K=50 hops failure reason is "
            f"'{top_reason[0]}' ({top_reason[1]} occurrences)."
        )

    if pa["strict_win_last_r_divergence_mean"] is not None and pa["strict_win_last_r_divergence_mean"] > 0:
        paragraphs.append(
            f"The mean distance from the most recent R-action divergence to a strict_win event is "
            f"{pa['strict_win_last_r_divergence_mean']:.1f} requests."
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
            f"whereas KSP-FF K=50 hops uses {k['avg_fs']:.2f} FS."
        )

    if pa["ksp_action_in_strict_top30_rate"] < 1.0:
        paragraphs.append(
            f"KSP-FF K=50 hops action is present in Strict v1.3's PPO-R Top-30 candidate pool in "
            f"{pa['ksp_action_in_strict_top30_rate']:.2%} of evaluated requests."
        )

    return "\n\n".join(paragraphs)


def _build_next_step_decision(payload: Dict[str, Any]) -> str:
    s = payload["overall"]["strict"]
    k = payload["overall"]["ksp_ff"]
    pa = payload["paired_analysis"]
    blocking_diff_pp = payload["overall"]["strict_minus_ksp_blocking_pp"]
    od = payload["fixed_od"]

    lines = [
        "# Next Step Decision: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / fixed-OD)",
        "",
        "## Result Summary",
        "",
        f"- Strict v1.3 blocking: {s['blocking_rate']:.4%}",
        f"- KSP-FF K=50 hops blocking: {k['blocking_rate']:.4%}",
        f"- Difference: {blocking_diff_pp:.2f} pp",
        f"- strict_win: {pa['strict_win_count']}",
        f"- ksp_win: {pa['ksp_win_count']}",
        f"- KSP action in Strict Top-30 rate: {pa['ksp_action_in_strict_top30_rate']:.4%}",
        "",
        "## Decision",
        "",
    ]

    if blocking_diff_pp < -0.5:
        # Strict v1.3 clearly lower.
        lines.append("**Case 1: Strict v1.3 clearly lower than KSP-FF K=50 hops.**")
        lines.append("")
        lines.append(
            "The R-side spectrum trajectory management of Strict v1.3 provides a measurable advantage "
            "under fixed-C / fixed-OD. The next step should focus on identifying which component "
            "(path choice, modulation choice, or block placement) drives the blocking reduction."
        )
        lines.append("")
        lines.append("Recommended follow-ups:")
        lines.append("- Analyze LFB preservation and fragmentation across the trajectory.")
        lines.append("- Check whether Strict v1.3 avoids hot links or reduces block_start congestion.")
        lines.append("- Decompose the gain into path-level vs block-level contributions.")
    elif blocking_diff_pp > 0.5:
        # KSP-FF clearly lower.
        lines.append("**Case 2: KSP-FF K=50 hops clearly lower than Strict v1.3.**")
        lines.append("")
        lines.append(
            "Strict v1.3 underperforms under fixed-C / fixed-OD. This points to a mismatch between "
            "the Strict v1.3 candidate pool / label and the fixed-OD distribution."
        )
        lines.append("")
        lines.append("Recommended follow-ups:")
        lines.append("- Check whether the KSP-FF action is inside Strict v1.3's PPO-R Top-30 candidate pool.")
        lines.append("- Audit whether Strict v1.3 systematically prefers longer paths or higher-waste blocks.")
        lines.append("- Re-examine whether the H=5 common-future label is appropriate for fixed-OD long-horizon risk.")
    else:
        # Close.
        lines.append("**Case 3: Blocking rates are close.**")
        lines.append("")
        lines.append(
            "The fixed-OD setting may not create enough spectral pressure, or the chosen OD pair "
            "is not sensitive to the differences between the two policies."
        )
        lines.append("")
        lines.append("Recommended follow-ups:")
        lines.append("- Try a different fixed_src_node / fixed_server_id pair with higher spectral contention.")
        lines.append("- Increase load (smaller arrival_interval or longer holding times).")
        lines.append("- Verify that num_slots=320 is not overly generous for this OD pair.")

    lines.append("")
    lines.append("## Fixed-C Verification")
    lines.append("")
    lines.append(f"- no PPO-C loaded: {od['no_ppo_c_loaded']}")
    lines.append(f"- no PPO-C action selected: {od['no_ppo_c_action_selected']}")
    lines.append(f"- fixed_src_node: {od['fixed_src_node']}")
    lines.append(f"- fixed_split_id: {od['fixed_split_id']}")
    lines.append(f"- fixed_server_id: {od['fixed_server_id']}")
    lines.append(f"- fixed_dst_node: {od['fixed_dst_node']}")
    lines.append("")
    return "\n".join(lines)


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
    parser.add_argument("--k_paths_r", type=int, default=50)
    parser.add_argument("--path_sort_strategy_r", default="hops")
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
        default="sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_fixed_od_diagnosis",
    )
    parser.add_argument("--max_trace_samples", type=int, default=20000)
    parser.add_argument("--fixed_src_node", type=int, default=0)
    parser.add_argument("--fixed_split_id", type=int, default=0)
    parser.add_argument("--fixed_server_id", type=int, default=0)
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
