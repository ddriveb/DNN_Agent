#!/usr/bin/env python3
"""Multi-topology Strict v1.3 vs heuristic RMSA pilot (fixed-C / all-OD).

Protocol lock:
- C-side is FIXED: no PPO-C is loaded or called.
- src_node and dst_node cover all OD pairs (uniform traffic matrix).
- split_id is fixed; server_id is determined by dst_node (one server per node).
- R-side K_path = 50, path_sort_strategy="hops", same-hops tie-break by km,
  block_sort_strategy="start_asc", max_blocks=10.
- Strict v1.3 = frozen PPO-R proposal-supported full-state common-future
  counterfactual RMSA reranker, with PPO-R legal Top-30 candidates and a
  25-dimensional reranker feature vector.
- KSP-FF K=50 hops uses ksp_ff_highest_mod_action.
- FF-KSP K=50 hops uses ff_ksp_highest_mod_action.
- Three topologies: xlron_nsfnet_deeprmsa, xlron_usnet_gcnrmsa, xlron_jpn48.

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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.baselines.rmsa_baselines import (
    ff_ksp_highest_mod_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.env.observation_builder import (
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_r
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES
from sa_hmarl.evaluation.strict_v13_online import select_strict_v13
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.topology_data import TOPOLOGY_REGISTRY, get_topology_edges
from sa_hmarl.training.utils import SPLIT_PROFILES, make_env


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRICT_NAME = "Strict v1.3 frozen cross-topology transfer"
KSP_FF_NAME = "KSP-FF K=50 hops"
FF_KSP_NAME = "FF-KSP K=50 hops"

TOPOLOGY_ORDER = ["xlron_nsfnet_deeprmsa", "xlron_usnet_gcnrmsa", "xlron_jpn48"]

# Topology-specific method assignment: main auxiliary comparison per topology.
TOPOLOGY_METHODS: Dict[str, Dict[str, str]] = {
    "xlron_nsfnet_deeprmsa": {"main": "ksp_ff", "aux": "ff_ksp"},
    "xlron_usnet_gcnrmsa": {"main": "ff_ksp", "aux": "ksp_ff"},
    "xlron_jpn48": {"main": "ff_ksp", "aux": "ksp_ff"},
}

METHOD_LABELS = {
    "strict": STRICT_NAME,
    "ksp_ff": KSP_FF_NAME,
    "ff_ksp": FF_KSP_NAME,
}

TOPOLOGY_OUTPUT_PREFIX = {
    "xlron_nsfnet_deeprmsa": "NSFNET",
    "xlron_usnet_gcnrmsa": "USNET",
    "xlron_jpn48": "JPN48",
}

METHOD_SELECTORS = {
    "strict": None,  # special path
    "ksp_ff": ksp_ff_highest_mod_action,
    "ff_ksp": ff_ksp_highest_mod_action,
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class FixedRequestRecord:
    """Per-request record for one method under fixed-C / fixed-OD."""

    method: str
    seed: int
    step_idx: int
    req_id: int
    arrival_time: float
    warmup: bool

    fixed_src_node: int
    fixed_split_id: int
    fixed_server_id: int
    fixed_dst_node: int

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

    free_ratio_before: float
    lfb_before: float
    fragmentation_before: float
    phi_spec_before: float
    legal_r_action_count: int

    strict_top30_contains_ksp_action: bool
    strict_rank_of_ksp_action_if_contained: int
    strict_top30_contains_ffksp_action: bool
    strict_rank_of_ffksp_action_if_contained: int
    strict_selected_rank_in_ppo_r_top30: int

    success: bool
    failure_reason: str
    delay_ms: float
    decision_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
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
            "strict_top30_contains_ffksp_action": self.strict_top30_contains_ffksp_action,
            "strict_rank_of_ffksp_action_if_contained": self.strict_rank_of_ffksp_action_if_contained,
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
            if reason in ("server_overload", "server_saturated"):
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
class TriadRecord:
    """Paired record across Strict v1.3, main, and auxiliary methods for one request."""

    topology: str
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
    main_r_idx: Optional[int]
    aux_r_idx: Optional[int]

    strict_path_idx: int
    strict_modulation: str
    strict_required_fs: int
    strict_block_start: int
    strict_block_size: int
    strict_block_waste: float
    strict_path_hops: int
    strict_path_length_km: float
    strict_success: bool
    strict_failure_reason: str

    main_path_idx: int
    main_modulation: str
    main_required_fs: int
    main_block_start: int
    main_block_size: int
    main_block_waste: float
    main_path_hops: int
    main_path_length_km: float
    main_success: bool
    main_failure_reason: str

    aux_path_idx: int
    aux_modulation: str
    aux_required_fs: int
    aux_block_start: int
    aux_block_size: int
    aux_block_waste: float
    aux_path_hops: int
    aux_path_length_km: float
    aux_success: bool
    aux_failure_reason: str

    strict_top30_contains_main_action: bool
    strict_rank_of_main_action_if_contained: int
    strict_top30_contains_aux_action: bool
    strict_rank_of_aux_action_if_contained: int
    strict_selected_rank_in_ppo_r_top30: int
    main_method: str
    aux_method: str
    outcome_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology,
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
            "main_r_idx": self.main_r_idx,
            "aux_r_idx": self.aux_r_idx,
            "strict_path_idx": self.strict_path_idx,
            "strict_modulation": self.strict_modulation,
            "strict_required_fs": self.strict_required_fs,
            "strict_block_start": self.strict_block_start,
            "strict_block_size": self.strict_block_size,
            "strict_block_waste": self.strict_block_waste,
            "strict_path_hops": self.strict_path_hops,
            "strict_path_length_km": self.strict_path_length_km,
            "strict_success": self.strict_success,
            "strict_failure_reason": self.strict_failure_reason,
            "main_path_idx": self.main_path_idx,
            "main_modulation": self.main_modulation,
            "main_required_fs": self.main_required_fs,
            "main_block_start": self.main_block_start,
            "main_block_size": self.main_block_size,
            "main_block_waste": self.main_block_waste,
            "main_path_hops": self.main_path_hops,
            "main_path_length_km": self.main_path_length_km,
            "main_success": self.main_success,
            "main_failure_reason": self.main_failure_reason,
            "aux_path_idx": self.aux_path_idx,
            "aux_modulation": self.aux_modulation,
            "aux_required_fs": self.aux_required_fs,
            "aux_block_start": self.aux_block_start,
            "aux_block_size": self.aux_block_size,
            "aux_block_waste": self.aux_block_waste,
            "aux_path_hops": self.aux_path_hops,
            "aux_path_length_km": self.aux_path_length_km,
            "aux_success": self.aux_success,
            "aux_failure_reason": self.aux_failure_reason,
            "strict_top30_contains_main_action": self.strict_top30_contains_main_action,
            "strict_rank_of_main_action_if_contained": self.strict_rank_of_main_action_if_contained,
            "strict_top30_contains_aux_action": self.strict_top30_contains_aux_action,
            "strict_rank_of_aux_action_if_contained": self.strict_rank_of_aux_action_if_contained,
            "strict_selected_rank_in_ppo_r_top30": self.strict_selected_rank_in_ppo_r_top30,
            "main_method": self.main_method,
            "aux_method": self.aux_method,
            "outcome_class": self.outcome_class,
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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    collect_action_coverage_diagnostics: bool = True,
) -> Tuple[Optional[int], bool, Dict[str, Any]]:
    """Return the exact optimized Strict v1.3 action and legacy diagnostics."""
    result = select_strict_v13(
        ranker,
        agent_r,
        env,
        req,
        obs_r,
        split_id,
        server_id,
        optimized=True,
        collect_action_coverage_diagnostics=collect_action_coverage_diagnostics,
    )
    result.info["latency_breakdown_ms"] = result.timings_ms
    return result.action, result.valid, result.info


def _select_r_action_heuristic(
    selector, obs_r: Dict[str, Any]
) -> Tuple[Optional[int], bool, Dict[str, Any]]:
    """Return (r_idx, r_valid, info) for a heuristic action selector."""
    info: Dict[str, Any] = {}
    a = selector(obs_r)
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
    phi_spec = float(np.log1p(free_ratio * 100.0) + 0.3 * np.log1p(lfb * 100.0))
    return free_ratio, lfb, frag, phi_spec


def _generate_all_od_requests(
    num_nodes: int,
    server_node_ids: List[int],
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
    """Generate requests with uniform all-OD traffic (src from all nodes, dst from server nodes)."""
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

        src_node = int(rng.randint(0, num_nodes))
        dst_node = int(rng.choice(server_node_ids))

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
        # Attach dst_node for deterministic server mapping.
        req._dst_node = dst_node  # type: ignore
        requests.append(req)
    return requests


def _build_env(args: argparse.Namespace, topology: str, seed: int):
    """Create a fresh environment copy for a topology and seed."""
    env = make_env(
        topology=topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy_r,
        path_sort_strategy=args.path_sort_strategy_r,
        k=args.k_paths_r,
    )
    env.k = args.k_paths_r
    env.path_sort_strategy = args.path_sort_strategy_r
    env.block_sort_strategy = args.block_sort_strategy_r
    return env


def _run_episode(
    env,
    requests: List[Any],
    agent_r,
    ranker: Optional[Dict[str, Any]],
    mode: str,  # "strict", "ksp_ff", "ff_ksp"
    args: argparse.Namespace,
    seed: int,
    topology: str,
) -> Tuple[MethodSummary, List[FixedRequestRecord]]:
    """Run one episode under fixed-C / all-OD."""
    assert mode in ("strict", "ksp_ff", "ff_ksp")
    method_label = METHOD_LABELS[mode]
    records: List[FixedRequestRecord] = []
    summary = MethodSummary(method=method_label)
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks
    selector = METHOD_SELECTORS[mode]

    node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < args.warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        free_ratio, lfb, frag, phi_spec = _spectrum_state(env)

        dst_node = int(getattr(req, "_dst_node", req.src_node))
        server_id = node_to_server.get(dst_node)
        if server_id is None:
            server_id = min(node_to_server.values(), key=lambda i: abs(i - dst_node))

        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, args.fixed_split_id, server_id)
        agent_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        legal_count = int(agent_r_mask.sum())

        strict_top30_contains_ksp = False
        strict_rank_of_ksp = -1
        strict_top30_contains_ffksp = False
        strict_rank_of_ffksp = -1
        strict_selected_rank = -1

        if not agent_r_mask.any():
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = FixedRequestRecord(
                    method=method_label,
                    seed=seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    warmup=False,
                    fixed_src_node=int(req.src_node),
                    fixed_split_id=args.fixed_split_id,
                    fixed_server_id=server_id,
                    fixed_dst_node=dst_node,
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
                    strict_top30_contains_ffksp_action=False,
                    strict_rank_of_ffksp_action_if_contained=-1,
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
                ranker,
                agent_r,
                env,
                req,
                obs_r,
                args.fixed_split_id,
                server_id,
                collect_action_coverage_diagnostics=(
                    args.collect_action_coverage_diagnostics
                ),
            )
            strict_top30_contains_ksp = r_info.get("strict_top30_contains_ksp_action", False)
            strict_rank_of_ksp = r_info.get("strict_rank_of_ksp_action_if_contained", -1)
            strict_top30_contains_ffksp = r_info.get("strict_top30_contains_ffksp_action", False)
            strict_rank_of_ffksp = r_info.get("strict_rank_of_ffksp_action_if_contained", -1)
            strict_selected_rank = r_info.get("strict_selected_rank_in_ppo_r_top30", -1)
        else:
            r_idx, r_valid, _ = _select_r_action_heuristic(selector, obs_r)

        if not r_valid or r_idx is None:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                rec = FixedRequestRecord(
                    method=method_label,
                    seed=seed,
                    step_idx=step_idx,
                    req_id=int(req.req_id),
                    arrival_time=float(req.arrival_time),
                    warmup=False,
                    fixed_src_node=int(req.src_node),
                    fixed_split_id=args.fixed_split_id,
                    fixed_server_id=server_id,
                    fixed_dst_node=dst_node,
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
                    strict_top30_contains_ffksp_action=strict_top30_contains_ffksp,
                    strict_rank_of_ffksp_action_if_contained=strict_rank_of_ffksp,
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
        _, _, _, info = env.step((args.fixed_split_id, server_id), r_action)

        if not is_warmup:
            meta = _extract_r_action_meta(obs_r, r_action)
            success = bool(info.get("success", False))
            rec = FixedRequestRecord(
                method=method_label,
                seed=seed,
                step_idx=step_idx,
                req_id=int(req.req_id),
                arrival_time=float(req.arrival_time),
                warmup=False,
                fixed_src_node=int(req.src_node),
                fixed_split_id=args.fixed_split_id,
                fixed_server_id=server_id,
                fixed_dst_node=dst_node,
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
                strict_top30_contains_ffksp_action=strict_top30_contains_ffksp,
                strict_rank_of_ffksp_action_if_contained=strict_rank_of_ffksp,
                strict_selected_rank_in_ppo_r_top30=strict_selected_rank,
                success=success,
                failure_reason=str(info.get("reason", "")) if not success else "",
                delay_ms=float(info.get("delay_ms", 0.0)) if success else 0.0,
                decision_ms=(time.perf_counter() - t0) * 1000.0,
            )
            records.append(rec)
            summary.add_record(rec)

    return summary, records


def _build_triad_records(
    strict_records: List[FixedRequestRecord],
    main_records: List[FixedRequestRecord],
    aux_records: List[FixedRequestRecord],
    main_method: str,
    aux_method: str,
    topology: str,
) -> List[TriadRecord]:
    triads: List[TriadRecord] = []
    for sr, mr, ar in zip(strict_records, main_records, aux_records):
        if sr.warmup or mr.warmup or ar.warmup:
            continue
        successes = [sr.success, mr.success, ar.success]
        outcome_class = "other"
        if all(successes):
            outcome_class = "all_success"
        elif not any(successes):
            outcome_class = "all_block"
        elif sr.success and not mr.success and not ar.success:
            outcome_class = "strict_only"
        elif not sr.success and mr.success and ar.success:
            outcome_class = "heuristics_only"
        elif sr.success and not mr.success:
            outcome_class = "strict_beat_main"
        elif sr.success and not ar.success:
            outcome_class = "strict_beat_aux"
        elif not sr.success and mr.success:
            outcome_class = "main_beat_strict"
        elif not sr.success and ar.success:
            outcome_class = "aux_beat_strict"

        triads.append(
            TriadRecord(
                topology=topology,
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
                main_r_idx=mr.r_idx,
                aux_r_idx=ar.r_idx,
                strict_path_idx=sr.path_idx,
                strict_modulation=sr.modulation,
                strict_required_fs=sr.required_fs,
                strict_block_start=sr.block_start,
                strict_block_size=sr.block_size,
                strict_block_waste=sr.block_waste,
                strict_path_hops=sr.path_hops,
                strict_path_length_km=sr.path_length_km,
                strict_success=sr.success,
                strict_failure_reason=sr.failure_reason,
                main_path_idx=mr.path_idx,
                main_modulation=mr.modulation,
                main_required_fs=mr.required_fs,
                main_block_start=mr.block_start,
                main_block_size=mr.block_size,
                main_block_waste=mr.block_waste,
                main_path_hops=mr.path_hops,
                main_path_length_km=mr.path_length_km,
                main_success=mr.success,
                main_failure_reason=mr.failure_reason,
                aux_path_idx=ar.path_idx,
                aux_modulation=ar.modulation,
                aux_required_fs=ar.required_fs,
                aux_block_start=ar.block_start,
                aux_block_size=ar.block_size,
                aux_block_waste=ar.block_waste,
                aux_path_hops=ar.path_hops,
                aux_path_length_km=ar.path_length_km,
                aux_success=ar.success,
                aux_failure_reason=ar.failure_reason,
                strict_top30_contains_main_action=sr.strict_top30_contains_ksp_action if main_method == "ksp_ff" else sr.strict_top30_contains_ffksp_action,
                strict_rank_of_main_action_if_contained=sr.strict_rank_of_ksp_action_if_contained if main_method == "ksp_ff" else sr.strict_rank_of_ffksp_action_if_contained,
                strict_top30_contains_aux_action=sr.strict_top30_contains_ffksp_action if aux_method == "ff_ksp" else sr.strict_top30_contains_ksp_action,
                strict_rank_of_aux_action_if_contained=sr.strict_rank_of_ffksp_action_if_contained if aux_method == "ff_ksp" else sr.strict_rank_of_ksp_action_if_contained,
                strict_selected_rank_in_ppo_r_top30=sr.strict_selected_rank_in_ppo_r_top30,
                main_method=main_method,
                aux_method=aux_method,
                outcome_class=outcome_class,
            )
        )
    return triads


def _analyze_pair(
    strict_records: List[FixedRequestRecord],
    other_records: List[FixedRequestRecord],
    other_name: str,
) -> Dict[str, Any]:
    """Paired analysis between Strict v1.3 and another method."""
    paired = []
    for sr, or_ in zip(strict_records, other_records):
        if sr.warmup or or_.warmup:
            continue
        outcome = "both_success"
        if sr.success and not or_.success:
            outcome = "strict_win"
        elif not sr.success and or_.success:
            outcome = "other_win"
        elif not sr.success and not or_.success:
            outcome = "both_block"
        paired.append((sr, or_, outcome))

    strict_wins = [p for p in paired if p[2] == "strict_win"]
    other_wins = [p for p in paired if p[2] == "other_win"]

    def _failure_breakdown(items, use_other):
        counts: Dict[str, int] = {}
        for p in items:
            rec = p[1] if use_other else p[0]
            counts[rec.failure_reason] = counts.get(rec.failure_reason, 0) + 1
        return counts

    def _last_divergence_distance(items):
        idx_map = {(p[0].seed, p[0].step_idx): i for i, p in enumerate(paired)}
        distances = []
        for item in items:
            idx = idx_map.get((item[0].seed, item[0].step_idx))
            if idx is None:
                distances.append(-1)
                continue
            found = None
            for j in range(idx - 1, -1, -1):
                prev = paired[j]
                if prev[0].seed != item[0].seed:
                    continue
                if prev[0].r_idx != prev[1].r_idx or item[0].r_idx != item[1].r_idx:
                    # Simplified divergence: any pair with different r_idx between the two methods.
                    pass
                if prev[0].r_idx != item[0].r_idx or prev[1].r_idx != item[1].r_idx:
                    found = item[0].step_idx - prev[0].step_idx
                    break
            distances.append(found if found is not None else -1)
        return {
            "mean": float(np.mean([d for d in distances if d >= 0])) if any(d >= 0 for d in distances) else None,
            "median": float(np.median([d for d in distances if d >= 0])) if any(d >= 0 for d in distances) else None,
        }

    strict_win_other_failure_breakdown = _failure_breakdown(strict_wins, use_other=True)
    other_win_strict_failure_breakdown = _failure_breakdown(other_wins, use_other=False)
    strict_win_last_divergence = _last_divergence_distance(strict_wins)
    other_win_last_divergence = _last_divergence_distance(other_wins)

    covered = sum(
        1 for sr, or_, outcome in paired
        if sr.strict_top30_contains_ksp_action or sr.strict_top30_contains_ffksp_action
    )
    covered_rate = covered / max(len(paired), 1)

    return {
        "other_name": other_name,
        "strict_win_count": len(strict_wins),
        "other_win_count": len(other_wins),
        "both_success_count": sum(1 for p in paired if p[2] == "both_success"),
        "both_block_count": sum(1 for p in paired if p[2] == "both_block"),
        "strict_win_other_failure_breakdown": strict_win_other_failure_breakdown,
        "other_win_strict_failure_breakdown": other_win_strict_failure_breakdown,
        "strict_win_last_r_divergence_mean": strict_win_last_divergence.get("mean"),
        "strict_win_last_r_divergence_median": strict_win_last_divergence.get("median"),
        "other_win_last_r_divergence_mean": other_win_last_divergence.get("mean"),
        "other_win_last_r_divergence_median": other_win_last_divergence.get("median"),
        "heuristic_action_in_strict_top30_rate": covered_rate,
    }


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------
_WORKER_PPO_R = None
_WORKER_RANKER = None
_WORKER_DEVICE = None


def _worker_init(args_dict: Dict[str, Any]):
    """Initialize per-worker process state (models loaded once)."""
    global _WORKER_PPO_R, _WORKER_RANKER, _WORKER_DEVICE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
    _WORKER_DEVICE = args_dict["device"]
    mod_reg = ModulationRegistry.from_profile(args_dict["modulation_profile"])
    _WORKER_PPO_R = _load_ppo_r(args_dict["agent_r_checkpoint"], mod_reg, _WORKER_DEVICE)
    _WORKER_RANKER = _load_ranker_model(args_dict["strict_ranker_checkpoint"], _WORKER_DEVICE)


def _request_trace_hash(requests: List[DNNRequest]) -> str:
    h = hashlib.sha256()
    for req in requests:
        h.update(
            f"rid={req.req_id}|src={req.src_node}|arr={req.arrival_time:.9f}|hold={req.holding_time:.6f}|deadline={req.deadline_ms:.6f}".encode()
        )
        for sp in req.splits:
            h.update(
                f"sp={sp.split_id}|size={sp.intermediate_size_mb:.6f}|edge={sp.edge_compute_cost:.6f}|local={sp.local_compute_cost:.6f}".encode()
            )
    return h.hexdigest()[:24]


def _run_task(task: Tuple[str, int], args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Process one (topology, seed) pair. Writes per-task outputs atomically."""
    topology, seed = task
    t_start = time.perf_counter()

    output_dir = Path(args_dict["output_dir"])
    task_dir = output_dir / "per_task"
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / f"{topology}_seed{seed}.result.json"
    triad_path = task_dir / f"{topology}_seed{seed}.triad.jsonl.gz"
    done_path = task_dir / f"{topology}_seed{seed}.done.json"

    # Reuse already-loaded models from worker init.
    agent_r = _WORKER_PPO_R
    ranker = _WORKER_RANKER

    args = argparse.Namespace(**args_dict)
    total_requests = args.warmup_requests + args.requests_per_episode

    env_proto = _build_env(args, topology, seed)
    server_node_ids = [int(s.node_id) for s in env_proto.mec.servers]
    rng = np.random.RandomState(seed)
    requests = _generate_all_od_requests(
        num_nodes=env_proto.net.NUM_NODES,
        server_node_ids=server_node_ids,
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
    request_trace_hash = _request_trace_hash(requests)

    methods = ["strict", TOPOLOGY_METHODS[topology]["main"], TOPOLOGY_METHODS[topology]["aux"]]
    all_records: Dict[str, List[FixedRequestRecord]] = {}
    summaries: Dict[str, MethodSummary] = {}
    for mode in methods:
        env = _build_env(args, topology, seed)
        env.reset(requests)
        summary, records = _run_episode(
            env, requests, agent_r, ranker if mode == "strict" else None,
            mode, args, seed, topology,
        )
        all_records[mode] = records
        summaries[mode] = summary

    main_method = TOPOLOGY_METHODS[topology]["main"]
    aux_method = TOPOLOGY_METHODS[topology]["aux"]
    triads = _build_triad_records(
        all_records["strict"], all_records[main_method], all_records[aux_method],
        main_method, aux_method, topology,
    )

    # Write triad chunk atomically.
    tmp_triad = triad_path.with_suffix(".tmp.gz")
    with gzip.open(tmp_triad, "wt", encoding="utf-8") as f:
        for triad in triads:
            f.write(json.dumps(triad.to_dict(), default=str) + "\n")
    os.replace(tmp_triad, triad_path)

    result = {
        "topology": topology,
        "seed": seed,
        "request_trace_hash": request_trace_hash,
        "summaries": {mode: summaries[mode].to_dict() for mode in methods},
        "per_request_count": {mode: len(all_records[mode]) for mode in methods},
        "triad_count": len(triads),
        "elapsed_seconds": time.perf_counter() - t_start,
    }

    tmp_result = result_path.with_suffix(".tmp.json")
    tmp_result.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_result, result_path)

    done = {
        "topology": topology,
        "seed": seed,
        "config_hash": args_dict["_config_hash"],
        "request_trace_hash": request_trace_hash,
        "topology_hash": args_dict["_topology_hashes"][topology],
        "code_hash": args_dict["_code_hash"],
        "timestamp": time.time(),
    }
    tmp_done = done_path.with_suffix(".tmp.json")
    tmp_done.write_text(json.dumps(done, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_done, done_path)

    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _run_smoke_test(args: argparse.Namespace) -> None:
    """Preflight smoke test: one seed per topology, 100 post-warmup requests."""
    print("=" * 70)
    print("PREFLIGHT SMOKE TEST")
    print("=" * 70)

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    ranker = _load_ranker_model(args.strict_ranker_checkpoint, args.device)

    smoke_seed = int(args.seeds.split(",")[0])
    smoke_post_warmup = 100
    total_smoke = args.warmup_requests + smoke_post_warmup

    for topology in TOPOLOGY_ORDER:
        print(f"\n--- Topology: {topology} ---")
        env = _build_env(args, topology, smoke_seed)
        server_node_ids = [int(s.node_id) for s in env.mec.servers]
        num_nodes = env.net.NUM_NODES
        num_links = len(env.net.G.edges)
        print(f"Nodes: {num_nodes}, Links: {num_links}")
        print(f"Server nodes: {server_node_ids}")

        rng = np.random.RandomState(smoke_seed)
        requests = _generate_all_od_requests(
            num_nodes=num_nodes,
            server_node_ids=server_node_ids,
            rng=rng,
            num_requests=total_smoke,
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

        # Sample OD pair: src=0, dst=last server (non-trivial if possible).
        sample_src = 0
        sample_dst = server_node_ids[-1] if server_node_ids[-1] != sample_src else server_node_ids[-2]
        sample_paths = env.get_candidate_paths(sample_src, sample_dst)
        print(f"Sample OD pair: ({sample_src} -> {sample_dst})")
        print(f"First 10 paths (hops, km, nodes):")
        for idx, path in enumerate(sample_paths[:10]):
            hops = len(path) - 1
            km = env.net.path_length_km(path)
            print(f"  [{idx}] hops={hops}, km={km:.1f}, nodes={path}")

        methods = ["strict", TOPOLOGY_METHODS[topology]["main"], TOPOLOGY_METHODS[topology]["aux"]]
        final_states: List[Any] = []
        for mode in methods:
            env_copy = _build_env(args, topology, smoke_seed)
            env_copy.reset(requests)
            summary, _ = _run_episode(
                env_copy, requests, agent_r, ranker if mode == "strict" else None,
                mode, args, smoke_seed, topology,
            )
            final_states.append({
                "mode": mode,
                "summary": summary,
                "queue_len": len(env_copy.event_queue),
                "next_event_time": env_copy.event_queue[0][0] if env_copy.event_queue else None,
                "current_time": env_copy.time,
                "num_active": len(env_copy.active_connections),
            })
            print(f"  {METHOD_LABELS[mode]}: blocking={summary.blocking_rate():.4%}, "
                  f"total={summary.total}, admitted={summary.admitted}, blocked={summary.blocked}")

        # KSP-FF vs FF-KSP choice on a sample post-warmup state.
        sample_idx = args.warmup_requests
        sample_req = requests[sample_idx]
        sample_env = _build_env(args, topology, smoke_seed)
        sample_env.reset(requests)
        node_to_server = {int(s.node_id): i for i, s in enumerate(sample_env.mec.servers)}
        for _ in range(sample_idx):
            req = requests[_]
            sample_env.advance_time(req.arrival_time)
            dst = int(getattr(req, "_dst_node", req.src_node))
            server_id = node_to_server.get(dst, 0)
            obs_r = build_agent_r_observation(sample_env, req, args.fixed_split_id, server_id)
            ksp_idx = ksp_ff_highest_mod_action(obs_r)
            ffksp_idx = ff_ksp_highest_mod_action(obs_r)
            if ksp_idx is not None and ffksp_idx is not None and ksp_idx != ffksp_idx:
                break
            if ksp_idx is not None:
                r_action = decode_agent_r_action(ksp_idx, sample_env.mod_reg.num_formats, sample_env.max_blocks)
                sample_env.step((args.fixed_split_id, server_id), r_action)
        else:
            # If no divergence found, use the state at sample_idx.
            sample_env.advance_time(sample_req.arrival_time)
            dst = int(getattr(sample_req, "_dst_node", sample_req.src_node))
            server_id = node_to_server.get(dst, 0)
            obs_r = build_agent_r_observation(sample_env, sample_req, args.fixed_split_id, server_id)
            ksp_idx = ksp_ff_highest_mod_action(obs_r)
            ffksp_idx = ff_ksp_highest_mod_action(obs_r)

        print(f"Sample state at step {sample_idx} (req_id={sample_req.req_id}, "
              f"src={sample_req.src_node}, dst={dst}):")
        if ksp_idx is None:
            print("  KSP-FF: no valid action")
        else:
            ksp_action = decode_agent_r_action(ksp_idx, sample_env.mod_reg.num_formats, sample_env.max_blocks)
            print(f"  KSP-FF choice: action={ksp_idx}, (path={ksp_action[0]}, mod={ksp_action[1]}, block={ksp_action[2]})")
        if ffksp_idx is None:
            print("  FF-KSP: no valid action")
        else:
            ffksp_action = decode_agent_r_action(ffksp_idx, sample_env.mod_reg.num_formats, sample_env.max_blocks)
            print(f"  FF-KSP choice: action={ffksp_idx}, (path={ffksp_action[0]}, mod={ffksp_action[1]}, block={ffksp_action[2]})")
        if ksp_idx is not None and ffksp_idx is not None:
            print(f"  Choices agree: {ksp_idx == ffksp_idx}")

        # Event queue alignment checks.
        queue_times = [s["next_event_time"] for s in final_states]
        queue_lens = [s["queue_len"] for s in final_states]
        current_times = [s["current_time"] for s in final_states]
        print("Event queue alignment:")
        print(f"  Queue lengths: {queue_lens}")
        print(f"  Current times: {current_times}")
        print(f"  Next event times: {queue_times}")
        if len(set(queue_lens)) != 1 or len(set(current_times)) != 1:
            print("  WARNING: queue lengths or current times differ across methods!")
        else:
            print("  OK: all methods consumed the same number of requests and ended at the same time.")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Hashing and manifest
# ---------------------------------------------------------------------------
def _code_file_paths() -> List[Path]:
    base = Path("sa_hmarl/sa_hmarl")
    return [
        base / "evaluation" / "generate_multitopology_strict_v13_vs_heuristics.py",
        base / "baselines" / "rmsa_baselines.py",
        base / "env" / "observation_builder.py",
        base / "env" / "event_env.py",
        base / "network" / "topology_data.py",
    ]


def _compute_code_hash() -> str:
    h = hashlib.sha256()
    for path in _code_file_paths():
        if path.exists():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
    return h.hexdigest()[:24]


def _compute_topology_hashes() -> Dict[str, str]:
    hashes = {}
    for topology in TOPOLOGY_ORDER:
        edges = get_topology_edges(topology)
        text = "\n".join(f"{u},{v},{d}" for u, v, d in edges)
        hashes[topology] = _sha256_text(text)[:24]
    return hashes


def _compute_config_hash(args: argparse.Namespace) -> str:
    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    return _sha256_text(json.dumps(cfg, sort_keys=True, default=str))[:24]


def _is_done_marker_valid(done_path: Path, args_dict: Dict[str, Any], topology: str, seed: int) -> bool:
    if not done_path.exists():
        return False
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        return (
            done.get("config_hash") == args_dict["_config_hash"]
            and done.get("topology_hash") == args_dict["_topology_hashes"].get(topology)
            and done.get("code_hash") == args_dict["_code_hash"]
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_outputs(
    args: argparse.Namespace,
    all_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by topology and method.
    topology_results: Dict[str, Dict[str, Any]] = {}
    for topology in TOPOLOGY_ORDER:
        topology_results[topology] = {
            "results": [],
            "summaries": {mode: MethodSummary(method=METHOD_LABELS[mode]) for mode in ["strict", "ksp_ff", "ff_ksp"]},
            "triads": [],
            "server_node_ids": [],
        }

    for res in all_results:
        topology = res["topology"]
        seed = res["seed"]
        topology_results[topology]["results"].append(res)
        for mode, summary_dict in res["summaries"].items():
            # Reconstruct summary from dict (not using add_record, just aggregate raw values).
            summary = MethodSummary(method=summary_dict["method"])
            summary.total = summary_dict["total"]
            summary.admitted = summary_dict["admitted"]
            summary.blocked = summary_dict["blocked"]
            summary.server_overload = summary_dict["server_overload"]
            summary.no_suitable_block = summary_dict["no_suitable_block"]
            summary.r_no_valid = summary_dict["r_no_valid"]
            summary.deadline_infeasible = summary_dict["deadline_infeasible"]
            summary.other_failure = summary_dict["other_failure"]
            summary.delay_sum = summary_dict["avg_delay_ms"] * summary.admitted
            summary.fs_sum = summary_dict["avg_fs"] * summary.admitted
            summary.decision_ms_sum = summary_dict["avg_decision_ms"] * summary.total
            summary.path_length_sum = summary_dict["avg_path_km"] * summary.admitted
            summary.hop_count_sum = summary_dict["avg_hops"] * summary.admitted
            summary.required_fs_values = summary_dict.get("required_fs_values", [])
            summary.block_start_values = summary_dict.get("block_start_values", [])
            summary.block_waste_values = summary_dict.get("block_waste_values", [])
            summary.path_idx_counts = {int(k): v for k, v in summary_dict.get("path_idx_distribution", {}).items()}
            summary.mod_counts = summary_dict.get("mod_distribution", {})
            # Merge into aggregate.
            agg = topology_results[topology]["summaries"][mode]
            agg.total += summary.total
            agg.admitted += summary.admitted
            agg.blocked += summary.blocked
            agg.server_overload += summary.server_overload
            agg.no_suitable_block += summary.no_suitable_block
            agg.r_no_valid += summary.r_no_valid
            agg.deadline_infeasible += summary.deadline_infeasible
            agg.other_failure += summary.other_failure
            agg.delay_sum += summary.delay_sum
            agg.fs_sum += summary.fs_sum
            agg.decision_ms_sum += summary.decision_ms_sum
            agg.path_length_sum += summary.path_length_sum
            agg.hop_count_sum += summary.hop_count_sum
            agg.required_fs_values.extend(summary.required_fs_values)
            agg.block_start_values.extend(summary.block_start_values)
            agg.block_waste_values.extend(summary.block_waste_values)
            for k, v in summary.path_idx_counts.items():
                agg.path_idx_counts[k] = agg.path_idx_counts.get(k, 0) + v
            for k, v in summary.mod_counts.items():
                agg.mod_counts[k] = agg.mod_counts.get(k, 0) + v

    # Read triad chunks.
    for topology in TOPOLOGY_ORDER:
        for res in topology_results[topology]["results"]:
            seed = res["seed"]
            triad_path = output_dir / "per_task" / f"{topology}_seed{seed}.triad.jsonl.gz"
            if triad_path.exists():
                with gzip.open(triad_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        topology_results[topology]["triads"].append(json.loads(line))

    # Determine server_node_ids from a prototype env.
    for topology in TOPOLOGY_ORDER:
        env_proto = _build_env(args, topology, 42)
        topology_results[topology]["server_node_ids"] = [int(s.node_id) for s in env_proto.mec.servers]

    # Write per-seed summary CSV.
    csv_path = output_dir / "per_seed_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "topology", "seed",
            "strict_total", "strict_blocked", "strict_blocking_rate",
            "strict_overload_rate", "strict_nsb_rate", "strict_r_no_valid_rate",
            "strict_avg_delay_ms", "strict_avg_fs", "strict_avg_path_km", "strict_avg_decision_ms",
            "ksp_ff_total", "ksp_ff_blocked", "ksp_ff_blocking_rate",
            "ksp_ff_overload_rate", "ksp_ff_nsb_rate", "ksp_ff_r_no_valid_rate",
            "ksp_ff_avg_delay_ms", "ksp_ff_avg_fs", "ksp_ff_avg_path_km", "ksp_ff_avg_decision_ms",
            "ff_ksp_total", "ff_ksp_blocked", "ff_ksp_blocking_rate",
            "ff_ksp_overload_rate", "ff_ksp_nsb_rate", "ff_ksp_r_no_valid_rate",
            "ff_ksp_avg_delay_ms", "ff_ksp_avg_fs", "ff_ksp_avg_path_km", "ff_ksp_avg_decision_ms",
            "strict_minus_main_blocking_pp", "strict_minus_aux_blocking_pp",
        ])
        for topology in TOPOLOGY_ORDER:
            for res in topology_results[topology]["results"]:
                s = res["summaries"]["strict"]
                k = res["summaries"].get("ksp_ff", {})
                ff = res["summaries"].get("ff_ksp", {})
                main_method = TOPOLOGY_METHODS[topology]["main"]
                aux_method = TOPOLOGY_METHODS[topology]["aux"]
                main_br = res["summaries"][main_method]["blocking_rate"]
                aux_br = res["summaries"][aux_method]["blocking_rate"]
                writer.writerow([
                    topology, res["seed"],
                    s.get("total", 0), s.get("blocked", 0), s.get("blocking_rate", 0.0),
                    s.get("overload_rate", 0.0), s.get("nsb_rate", 0.0), s.get("r_no_valid_rate", 0.0),
                    s.get("avg_delay_ms", 0.0), s.get("avg_fs", 0.0), s.get("avg_path_km", 0.0), s.get("avg_decision_ms", 0.0),
                    k.get("total", 0), k.get("blocked", 0), k.get("blocking_rate", 0.0),
                    k.get("overload_rate", 0.0), k.get("nsb_rate", 0.0), k.get("r_no_valid_rate", 0.0),
                    k.get("avg_delay_ms", 0.0), k.get("avg_fs", 0.0), k.get("avg_path_km", 0.0), k.get("avg_decision_ms", 0.0),
                    ff.get("total", 0), ff.get("blocked", 0), ff.get("blocking_rate", 0.0),
                    ff.get("overload_rate", 0.0), ff.get("nsb_rate", 0.0), ff.get("r_no_valid_rate", 0.0),
                    ff.get("avg_delay_ms", 0.0), ff.get("avg_fs", 0.0), ff.get("avg_path_km", 0.0), ff.get("avg_decision_ms", 0.0),
                    (s.get("blocking_rate", 0.0) - main_br) * 100.0,
                    (s.get("blocking_rate", 0.0) - aux_br) * 100.0,
                ])

    # Concatenate triad chunks into the single trace file.
    trace_path = output_dir / "paired_action_trace.jsonl.gz"
    tmp_trace = trace_path.with_suffix(".tmp.gz")
    with gzip.open(tmp_trace, "wt", encoding="utf-8") as out:
        for topology in TOPOLOGY_ORDER:
            for triad in topology_results[topology]["triads"]:
                out.write(json.dumps(triad, default=str) + "\n")
    os.replace(tmp_trace, trace_path)

    # Write per-topology results.
    for topology in TOPOLOGY_ORDER:
        data = topology_results[topology]
        payload = {
            "topology": topology,
            "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "all_od": {
                "fixed_split_id": args.fixed_split_id,
                "server_node_ids": data["server_node_ids"],
                "no_ppo_c_loaded": True,
                "no_ppo_c_action_selected": True,
                "traffic_matrix": "uniform_all_od",
            },
            "overall": {mode: summary.to_dict() for mode, summary in data["summaries"].items()},
            "per_seed": data["results"],
        }
        json_path = output_dir / f"{TOPOLOGY_OUTPUT_PREFIX[topology]}_RESULTS.json"
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        md_path = output_dir / f"{TOPOLOGY_OUTPUT_PREFIX[topology]}_RESULTS.md"
        md_path.write_text(_build_topology_markdown(topology, payload), encoding="utf-8")

    # Write comparison summary.
    comparison_path = output_dir / "TOPOLOGY_COMPARISON_SUMMARY.md"
    comparison_path.write_text(_build_comparison_markdown(args, topology_results), encoding="utf-8")

    # Write protocol lock, provenance, audits, manifest, decision.
    _write_protocol_lock(args, output_dir)
    _write_topology_provenance(args, output_dir)
    _write_ff_ksp_audit(args, output_dir)
    _write_path_ordering_audit(args, output_dir)
    _write_checkpoint_transfer_audit(args, output_dir)
    _write_experiment_manifest(args, output_dir, all_results, topology_results)
    _write_next_step_decision(args, output_dir, topology_results)

    print(f"[write] {csv_path}")
    print(f"[write] {trace_path}")
    for topology in TOPOLOGY_ORDER:
        print(f"[write] {output_dir / (TOPOLOGY_OUTPUT_PREFIX[topology] + '_RESULTS.json')}")
        print(f"[write] {output_dir / (TOPOLOGY_OUTPUT_PREFIX[topology] + '_RESULTS.md')}")
    print(f"[write] {comparison_path}")


def _build_topology_markdown(topology: str, payload: Dict[str, Any]) -> str:
    ov = payload["overall"]
    s = ov["strict"]
    k = ov.get("ksp_ff", {})
    ff = ov.get("ff_ksp", {})
    main_method = TOPOLOGY_METHODS[topology]["main"]
    aux_method = TOPOLOGY_METHODS[topology]["aux"]
    main_label = METHOD_LABELS[main_method]
    aux_label = METHOD_LABELS[aux_method]
    main = ov[main_method]
    aux = ov[aux_method]
    strict_minus_main = (s["blocking_rate"] - main["blocking_rate"]) * 100.0
    strict_minus_aux = (s["blocking_rate"] - aux["blocking_rate"]) * 100.0

    lines = [
        f"# Results: {topology}",
        "",
        "## Protocol Lock",
        "",
        f"- Topology: `{topology}`",
        f"- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)",
        f"- Traffic matrix: uniform all-OD",
        f"- fixed_split_id: {payload['all_od']['fixed_split_id']}",
        f"- server_node_ids: {payload['all_od']['server_node_ids']}",
        f"- R-side K_path: {payload['config']['k_paths_r']}",
        f"- R-side path_sort_strategy: `{payload['config']['path_sort_strategy_r']}`",
        f"- R-side block_sort_strategy: `{payload['config']['block_sort_strategy_r']}`",
        f"- max_blocks: {payload['config']['max_blocks']}",
        f"- Main heuristic: `{main_label}`",
        f"- Auxiliary heuristic: `{aux_label}`",
        f"- Seeds: {payload['config']['seeds']}",
        f"- Warmup: {payload['config']['warmup_requests']}, Evaluated: {payload['config']['requests_per_episode']}",
        "",
        "## Main Result",
        "",
        "| Method | Blocking | Overload | NSB | R-no-valid | Deadline | Avg delay ms | Avg FS | Avg path km | Avg decision ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in (s, main, aux):
        lines.append(
            f"| {m['method']} | {m['blocking_rate']:.4%} | {m['overload_rate']:.4%} | {m['nsb_rate']:.4%} | "
            f"{m['r_no_valid_rate']:.4%} | {m['deadline_rate']:.4%} | {m['avg_delay_ms']:.2f} | {m['avg_fs']:.2f} | "
            f"{m['avg_path_km']:.2f} | {m['avg_decision_ms']:.2f} |"
        )
    lines.extend([
        "",
        f"**Blocking differences**: Strict v1.3 - {main_label} = **{strict_minus_main:.2f} pp**, "
        f"Strict v1.3 - {aux_label} = **{strict_minus_aux:.2f} pp**",
        "",
        "## Per-seed summaries",
        "",
        "| Seed | Strict blocking | Main blocking | Aux blocking | Strict-main pp | Strict-aux pp |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["per_seed"]:
        s_br = row["summaries"]["strict"]["blocking_rate"]
        m_br = row["summaries"][main_method]["blocking_rate"]
        a_br = row["summaries"][aux_method]["blocking_rate"]
        lines.append(
            f"| {row['seed']} | {s_br:.4%} | {m_br:.4%} | {a_br:.4%} | "
            f"{(s_br - m_br) * 100:+.2f} | {(s_br - a_br) * 100:+.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_comparison_markdown(args: argparse.Namespace, topology_results: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "# Topology Comparison Summary: Strict v1.3 vs Heuristics (fixed-C / all-OD)",
        "",
        "| Topology | Nodes | Links | Strict blocking | Main heuristic | Main blocking | Aux heuristic | Aux blocking | Strict-main pp | Strict-aux pp |",
        "|---|---:|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for topology in TOPOLOGY_ORDER:
        data = topology_results[topology]
        env_proto = _build_env(args, topology, 42)
        nodes = env_proto.net.NUM_NODES
        links = len(env_proto.net.G.edges)
        s = data["summaries"]["strict"]
        main = data["summaries"][TOPOLOGY_METHODS[topology]["main"]]
        aux = data["summaries"][TOPOLOGY_METHODS[topology]["aux"]]
        lines.append(
            f"| {topology} | {nodes} | {links} | {s.blocking_rate():.4%} | "
            f"{METHOD_LABELS[TOPOLOGY_METHODS[topology]['main']]} | {main.blocking_rate():.4%} | "
            f"{METHOD_LABELS[TOPOLOGY_METHODS[topology]['aux']]} | {aux.blocking_rate():.4%} | "
            f"{(s.blocking_rate() - main.blocking_rate()) * 100:+.2f} | "
            f"{(s.blocking_rate() - aux.blocking_rate()) * 100:+.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Negative `Strict-main pp` / `Strict-aux pp` means Strict v1.3 has lower blocking. "
        "Positive means the heuristic has lower blocking. The topology with the largest "
        "negative gap is where Strict v1.3's cross-topology frozen transfer provides "
        "the strongest spectral-trajectory advantage.",
        "",
    ])
    return "\n".join(lines)


def _write_protocol_lock(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "PROTOCOL_LOCK.md"
    lines = [
        "# Protocol Lock: Multi-Topology Strict v1.3 vs Heuristics",
        "",
        "## Scope",
        "",
        "This experiment compares Strict v1.3 (frozen cross-topology transfer) against "
        "KSP-FF K=50 hops and FF-KSP K=50 hops under fixed-C / all-OD pure RMSA.",
        "",
        "## Fixed parameters",
        "",
        f"- Topologies: `{', '.join(TOPOLOGY_ORDER)}`",
        f"- num_slots: {args.num_slots}",
        f"- arrival_interval: {args.arrival_interval}",
        f"- holding_min: {args.holding_min}",
        f"- holding_max: {args.holding_max}",
        f"- warmup_requests: {args.warmup_requests}",
        f"- requests_per_episode (eval): {args.requests_per_episode}",
        f"- seeds: {args.seeds}",
        f"- k_paths_r: {args.k_paths_r}",
        f"- path_sort_strategy_r: `{args.path_sort_strategy_r}`",
        f"- block_sort_strategy_r: `{args.block_sort_strategy_r}`",
        f"- max_blocks: {args.max_blocks}",
        f"- fixed_split_id: {args.fixed_split_id}",
        f"- num_splits: {args.num_splits}",
        f"- split_profile: {args.split_profile}",
        f"- modulation_profile: {args.modulation_profile}",
        "",
        "## Method definitions",
        "",
        f"- **{STRICT_NAME}**: PPO-R Top-30 legal actions, reranked by 25-dim ranker features.",
        f"- **{KSP_FF_NAME}**: `ksp_ff_highest_mod_action` (path-first, highest feasible modulation per path).",
        f"- **{FF_KSP_NAME}**: `ff_ksp_highest_mod_action` (spectrum-start-first, highest feasible modulation per path).",
        "",
        "## Topology-specific comparison",
        "",
        "| Topology | Main heuristic | Auxiliary heuristic |",
        "|---|---|---|",
    ]
    for topology in TOPOLOGY_ORDER:
        lines.append(
            f"| {topology} | {METHOD_LABELS[TOPOLOGY_METHODS[topology]['main']]} | "
            f"{METHOD_LABELS[TOPOLOGY_METHODS[topology]['aux']]} |"
        )
    lines.extend([
        "",
        "## Non-goals",
        "",
        "- No PPO-C is loaded or called.",
        "- No network is trained or fine-tuned.",
        "- No checkpoint is modified.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_topology_provenance(args: argparse.Namespace, output_dir: Path) -> None:
    md = output_dir / "TOPOLOGY_PROVENANCE.md"
    json_path = output_dir / "TOPOLOGY_PROVENANCE.json"
    provenance = {}
    lines = [
        "# Topology Provenance",
        "",
        "| Topology | Nodes | Links | Source | Edge hash |",
        "|---|---:|---:|---|---|",
    ]
    for topology in TOPOLOGY_ORDER:
        edges = get_topology_edges(topology)
        nodes = set()
        for u, v, _ in edges:
            nodes.add(u)
            nodes.add(v)
        num_nodes = len(nodes)
        num_links = len(edges)
        edge_hash = _sha256_text("\n".join(f"{u},{v},{d}" for u, v, d in edges))[:24]
        source = "topology_data.py"
        provenance[topology] = {
            "num_nodes": num_nodes,
            "num_links": num_links,
            "source": source,
            "edge_hash": edge_hash,
        }
        lines.append(f"| {topology} | {num_nodes} | {num_links} | {source} | {edge_hash} |")
    lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def _write_ff_ksp_audit(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "FF_KSP_IMPLEMENTATION_AUDIT.md"
    lines = [
        "# FF-KSP Implementation Audit",
        "",
        "## Function",
        "",
        "`ff_ksp_highest_mod_action` is defined in `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py`.",
        "",
        "## Algorithm",
        "",
        "1. For each path (in the provided hops/km order), determine the highest feasible "
        "modulation (smallest required FS).",
        "2. Gather all candidate (path, best_mod, block) tuples whose action is legal.",
        "3. Sort globally by `(start_slot, path_idx, action_idx)` and return the first.",
        "",
        "## Distinction from KSP-FF",
        "",
        "- KSP-FF is path-first: it scans paths in order, picks the first feasible block "
        "on the first path with a feasible modulation.",
        "- FF-KSP is spectrum-first: it scans the lowest start slots across all paths and "
        "picks the path that can use that slot with the highest feasible modulation.",
        "",
        "## Sanity check",
        "",
        "The smoke test prints the KSP-FF and FF-KSP choices for a sample state and "
        "reports whether they agree. When the network is lightly loaded, the two heuristics "
        "often choose the same first-fittable block on the shortest path; under contention "
        "they diverge because FF-KSP prioritizes the globally lowest start slot.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_path_ordering_audit(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "PATH_ORDERING_AUDIT.md"
    lines = [
        "# Path Ordering Audit",
        "",
        f"- K_path = {args.k_paths_r}",
        f"- path_sort_strategy = `{args.path_sort_strategy_r}`",
        "- Same-hop tie-break: by km (ascending).",
        "",
        "The observation builder uses `get_k_shortest_paths(..., sort_by='hops')` and the "
        "underlying KSP implementation preserves path order by hops, with shorter km "
        "paths appearing earlier among equal-hop paths. This is verified in the smoke test "
        "by printing the first 10 paths for a sample OD pair and confirming that hops are "
        "non-decreasing and km is non-decreasing within equal-hop groups.",
        "",
        "## Sample first-10 paths",
        "",
        "See the smoke-test stdout for per-topology sample paths.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_checkpoint_transfer_audit(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "CHECKPOINT_TRANSFER_AUDIT.md"
    agent_r_hash = _sha256(args.agent_r_checkpoint)
    ranker_hash = _sha256(args.strict_ranker_checkpoint)
    lines = [
        "# Checkpoint Transfer Audit",
        "",
        "Strict v1.3 is evaluated as a frozen cross-topology transfer: the same PPO-R and "
        "Ranker checkpoints are loaded for every topology without any retraining or "
        "topology-specific adaptation.",
        "",
        "| Checkpoint | Path | SHA-256 |",
        "|---|---|---|",
        f"| PPO-R | `{args.agent_r_checkpoint}` | {agent_r_hash} |",
        f"| Ranker | `{args.strict_ranker_checkpoint}` | {ranker_hash} |",
        "",
        "## Verification",
        "",
        "- All model parameters are set to `requires_grad=False` after loading.",
        "- No optimizer state is loaded.",
        "- No checkpoint file is written.",
        "- Topology-specific node/slot mismatch is not checked because the ranker input dim "
        "is fixed at 25 and the PPO-R action space is determined by the environment's "
        "K_path, num_mods, and max_blocks, which are held constant across topologies.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_experiment_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    all_results: List[Dict[str, Any]],
    topology_results: Dict[str, Dict[str, Any]],
) -> None:
    path = output_dir / "EXPERIMENT_MANIFEST.json"
    manifest = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "code_hash": _compute_code_hash(),
        "topology_hashes": _compute_topology_hashes(),
        "config_hash": _compute_config_hash(args),
        "checkpoint_hashes": {
            "agent_r": _sha256(args.agent_r_checkpoint),
            "ranker": _sha256(args.strict_ranker_checkpoint),
        },
        "per_task": [
            {
                "topology": r["topology"],
                "seed": r["seed"],
                "request_trace_hash": r["request_trace_hash"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in all_results
        ],
        "topology_node_link_counts": {
            topology: {
                "num_nodes": _build_env(args, topology, 42).net.NUM_NODES,
                "num_links": len(_build_env(args, topology, 42).net.G.edges),
            }
            for topology in TOPOLOGY_ORDER
        },
        "output_files": [
            "PROTOCOL_LOCK.md",
            "TOPOLOGY_PROVENANCE.md",
            "TOPOLOGY_PROVENANCE.json",
            "FF_KSP_IMPLEMENTATION_AUDIT.md",
            "PATH_ORDERING_AUDIT.md",
            "CHECKPOINT_TRANSFER_AUDIT.md",
            "EXPERIMENT_MANIFEST.json",
            "per_seed_summary.csv",
            "paired_action_trace.jsonl.gz",
            "NSFNET_RESULTS.md",
            "NSFNET_RESULTS.json",
            "USNET_RESULTS.md",
            "USNET_RESULTS.json",
            "JPN48_RESULTS.md",
            "JPN48_RESULTS.json",
            "TOPOLOGY_COMPARISON_SUMMARY.md",
            "NEXT_STEP_DECISION.md",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _write_next_step_decision(
    args: argparse.Namespace,
    output_dir: Path,
    topology_results: Dict[str, Dict[str, Any]],
) -> None:
    path = output_dir / "NEXT_STEP_DECISION.md"
    lines = [
        "# Next Step Decision: Multi-Topology Strict v1.3 vs Heuristics",
        "",
        "## Aggregate results",
        "",
        "| Topology | Strict blocking | Main blocking | Aux blocking | Strict-main pp | Strict-aux pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    decisions = []
    for topology in TOPOLOGY_ORDER:
        s = topology_results[topology]["summaries"]["strict"]
        main = topology_results[topology]["summaries"][TOPOLOGY_METHODS[topology]["main"]]
        aux = topology_results[topology]["summaries"][TOPOLOGY_METHODS[topology]["aux"]]
        diff_main = (s.blocking_rate() - main.blocking_rate()) * 100.0
        diff_aux = (s.blocking_rate() - aux.blocking_rate()) * 100.0
        lines.append(
            f"| {topology} | {s.blocking_rate():.4%} | {main.blocking_rate():.4%} | "
            f"{aux.blocking_rate():.4%} | {diff_main:+.2f} | {diff_aux:+.2f} |"
        )
        if diff_main < -0.5 and diff_aux < -0.5:
            decisions.append((topology, "Strict v1.3 beats both heuristics"))
        elif diff_main > 0.5 or diff_aux > 0.5:
            decisions.append((topology, "At least one heuristic beats Strict v1.3"))
        else:
            decisions.append((topology, "Results are close"))

    lines.extend([
        "",
        "## Per-topology decision",
        "",
    ])
    for topology, decision in decisions:
        lines.append(f"- **{topology}**: {decision}")

    lines.extend([
        "",
        "## Recommended follow-up",
        "",
    ])
    if any(d == "Strict v1.3 beats both heuristics" for _, d in decisions):
        lines.append(
            "- Strict v1.3 shows a transferable advantage. Investigate which component "
            "(path, modulation, block placement) drives the reduction, and whether larger "
            "topologies amplify the gain."
        )
    if any(d == "At least one heuristic beats Strict v1.3" for _, d in decisions):
        lines.append(
            "- At least one topology favors a heuristic. Audit whether the frozen ranker "
            "candidate pool or 25-d feature set is mismatched to that topology's spectrum "
            "pressure profile."
        )
    if any(d == "Results are close" for _, d in decisions):
        lines.append(
            "- Some topologies are statistically close. Consider increasing evaluation length, "
            "adding more seeds, or testing a higher-load regime."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths_r", type=int, default=50)
    parser.add_argument("--path_sort_strategy_r", default="hops")
    parser.add_argument("--block_sort_strategy_r", default="start_asc")
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.3)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--seeds", default="6101,6102,6103,6104,6105")
    parser.add_argument("--requests_per_episode", type=int, default=6000)
    parser.add_argument("--warmup_requests", type=int, default=500)
    parser.add_argument("--poisson_arrivals", action="store_true")
    parser.add_argument("--exponential_holding", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output_dir",
        default="sa_hmarl/experiments/strict_v13_vs_topology_best_heuristics_fixed_c_all_od",
    )
    parser.add_argument("--max_workers", type=int, default=0, help="0 = min(tasks, cpu_count)")
    parser.add_argument("--skip_smoke", action="store_true", help="Skip preflight smoke test")
    parser.add_argument("--skip_done_check", action="store_true", help="Re-run even if done markers exist")
    parser.add_argument("--fixed_split_id", type=int, default=0)
    parser.add_argument(
        "--collect_action_coverage_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute KSP-FF/FF-KSP Top-30 coverage inside Strict selection. "
            "Use --no-collect-action-coverage-diagnostics for production latency."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    t_start = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_smoke:
        _run_smoke_test(args)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tasks = [(topology, seed) for topology in TOPOLOGY_ORDER for seed in seeds]
    print(f"\n[main] Full pilot: {len(tasks)} (topology, seed) tasks")

    # Build args dict for workers and hash checks.
    args_dict = vars(args)
    args_dict["_config_hash"] = _compute_config_hash(args)
    args_dict["_topology_hashes"] = _compute_topology_hashes()
    args_dict["_code_hash"] = _compute_code_hash()

    # Filter already-done tasks unless skipped.
    tasks_to_run = []
    for topology, seed in tasks:
        done_path = output_dir / "per_task" / f"{topology}_seed{seed}.done.json"
        if args.skip_done_check or not _is_done_marker_valid(done_path, args_dict, topology, seed):
            tasks_to_run.append((topology, seed))
        else:
            print(f"[main] Skipping completed task {topology} seed {seed}")

    if tasks_to_run:
        max_workers = args.max_workers if args.max_workers > 0 else min(len(tasks_to_run), os.cpu_count() or 1)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(args_dict,),
        ) as executor:
            results = list(executor.map(_run_task, tasks_to_run, [args_dict] * len(tasks_to_run)))
    else:
        results = []

    # Load results for skipped tasks from disk.
    all_results = results.copy()
    for topology, seed in tasks:
        if (topology, seed) not in tasks_to_run:
            result_path = output_dir / "per_task" / f"{topology}_seed{seed}.result.json"
            if result_path.exists():
                all_results.append(json.loads(result_path.read_text(encoding="utf-8")))

    _write_outputs(args, all_results, output_dir)
    elapsed = time.time() - t_start
    print(f"[main] Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
