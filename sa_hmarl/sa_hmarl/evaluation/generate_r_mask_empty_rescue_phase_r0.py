#!/usr/bin/env python3
"""Phase-R0: Fixed-C / All-OD R-mask-empty Expanded-Path Rescue Ceiling.

Protocol lock:
- C-side is FIXED: no PPO-C is loaded or called.
- src_node and dst_node cover all OD pairs (uniform traffic matrix).
- split_id is fixed; server_id is determined by dst_node (one server per node).
- R-side K_path = 50, path_sort_strategy="hops", same-hops tie-break by km
  then lexicographic node tuple, block_sort_strategy="start_asc", max_blocks=10.
- Strict v1.3 = frozen PPO-R proposal-supported full-state common-future
  counterfactual RMSA reranker, with PPO-R legal Top-30 candidates and a
  25-dimensional reranker feature vector.
- Rescue methods expand paths lazily to K=100/200/500 when the K=50 R-mask is
  empty and execute a deterministic expanded KSP-FF action.
- Four methods are compared: strict, preferred_heuristic, strict_k500_rescue,
  preferred_heuristic_k500_rescue.

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

# Import the base multi-topology evaluation script as a module.
from sa_hmarl.evaluation import generate_multitopology_strict_v13_vs_heuristics as _base
from sa_hmarl.baselines.rmsa_baselines import (
    expanded_ksp_ff_rescue_action,
    ff_ksp_highest_mod_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.env.observation_builder import (
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.topology_data import get_topology_edges


# ---------------------------------------------------------------------------
# Phase-R0 constants (override the base module globals)
# ---------------------------------------------------------------------------
STRICT_NAME = "Strict v1.3 frozen cross-topology transfer"
KSP_FF_NAME = "KSP-FF K=50 hops"
FF_KSP_NAME = "FF-KSP K=50 hops"
STRICT_RESCUE_NAME = "Strict v1.3 + K500 rescue"
HEURISTIC_RESCUE_NAME = "{heuristic} + K500 rescue"

TOPOLOGY_ORDER = [
    "xlron_cost239_ptrnet_real",
    "xlron_nsfnet_deeprmsa",
    "xlron_usnet_gcnrmsa",
    "xlron_jpn48",
]

TOPOLOGY_SEEDS = {
    "xlron_cost239_ptrnet_real": [5001, 5002, 5003, 5004, 5005],
    "xlron_nsfnet_deeprmsa": [6101, 6102, 6103, 6104, 6105],
    "xlron_usnet_gcnrmsa": [6101, 6102, 6103, 6104, 6105],
    "xlron_jpn48": [6101, 6102, 6103, 6104, 6105],
}

TOPOLOGY_METHODS: Dict[str, Dict[str, str]] = {
    "xlron_cost239_ptrnet_real": {"preferred": "ksp_ff"},
    "xlron_nsfnet_deeprmsa": {"preferred": "ksp_ff"},
    "xlron_usnet_gcnrmsa": {"preferred": "ff_ksp"},
    "xlron_jpn48": {"preferred": "ff_ksp"},
}

TOPOLOGY_OUTPUT_PREFIX = {
    "xlron_cost239_ptrnet_real": "COST239",
    "xlron_nsfnet_deeprmsa": "NSFNET",
    "xlron_usnet_gcnrmsa": "USNET",
    "xlron_jpn48": "JPN48",
}

METHOD_LABELS = {
    "strict": STRICT_NAME,
    "preferred_heuristic": "Preferred heuristic",
    "ksp_ff": KSP_FF_NAME,
    "ff_ksp": FF_KSP_NAME,
    "strict_k500_rescue": STRICT_RESCUE_NAME,
    "preferred_heuristic_k500_rescue": HEURISTIC_RESCUE_NAME,
}

METHOD_SELECTORS = {
    "strict": None,
    "ksp_ff": ksp_ff_highest_mod_action,
    "ff_ksp": ff_ksp_highest_mod_action,
}

RESCUE_KS = [100, 200, 500]

OUTPUT_DIR = "sa_hmarl/experiments/r_mask_empty_expanded_path_rescue_phase_r0"

# Patch base module globals so that any base functions that read them see the
# Phase-R0 configuration.
_base.TOPOLOGY_ORDER = TOPOLOGY_ORDER
_base.TOPOLOGY_METHODS = TOPOLOGY_METHODS
_base.TOPOLOGY_OUTPUT_PREFIX = TOPOLOGY_OUTPUT_PREFIX
_base.STRICT_NAME = STRICT_NAME
_base.KSP_FF_NAME = KSP_FF_NAME
_base.FF_KSP_NAME = FF_KSP_NAME
_base.METHOD_LABELS = METHOD_LABELS
_base.METHOD_SELECTORS = METHOD_SELECTORS


# ---------------------------------------------------------------------------
# Data containers (override the base module versions)
# ---------------------------------------------------------------------------
@dataclass
class FixedRequestRecord:
    """Per-request record for one method under fixed-C / all-OD."""

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

    rescue_attempted: bool = False
    rescue_action_rank: Optional[int] = None
    rescue_path_idx: Optional[int] = None
    rescue_path_rank: Optional[int] = None
    rescue_success: bool = False
    rescue_reason: str = ""

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
            "rescue_attempted": self.rescue_attempted,
            "rescue_action_rank": self.rescue_action_rank,
            "rescue_path_idx": self.rescue_path_idx,
            "rescue_path_rank": self.rescue_path_rank,
            "rescue_success": self.rescue_success,
            "rescue_reason": self.rescue_reason,
        }


@dataclass
class RescueProbeRecord:
    """Instantaneous rescue-ceiling probe collected during baseline runs."""

    topology: str
    seed: int
    method: str
    step_idx: int
    req_id: int
    src_node: int
    dst_node: int
    free_ratio_before: float
    lfb_before: float
    fragmentation_before: float
    phi_spec_before: float
    k100_available: bool
    k200_available: bool
    k500_available: bool
    first_rescue_rank: Optional[int] = None
    first_rescue_k: Optional[int] = None
    path_hops: Optional[int] = None
    path_length_km: Optional[float] = None
    modulation: Optional[str] = None
    required_fs: Optional[int] = None
    block_start: Optional[int] = None
    block_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology": self.topology,
            "seed": self.seed,
            "method": self.method,
            "step_idx": self.step_idx,
            "req_id": self.req_id,
            "src_node": self.src_node,
            "dst_node": self.dst_node,
            "free_ratio_before": self.free_ratio_before,
            "lfb_before": self.lfb_before,
            "fragmentation_before": self.fragmentation_before,
            "phi_spec_before": self.phi_spec_before,
            "k100_available": self.k100_available,
            "k200_available": self.k200_available,
            "k500_available": self.k500_available,
            "first_rescue_rank": self.first_rescue_rank,
            "first_rescue_k": self.first_rescue_k,
            "path_hops": self.path_hops,
            "path_length_km": self.path_length_km,
            "modulation": self.modulation,
            "required_fs": self.required_fs,
            "block_start": self.block_start,
            "block_size": self.block_size,
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

    rescue_attempted: int = 0
    rescue_success: int = 0
    rescue_failed: int = 0

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
        if rec.rescue_attempted:
            self.rescue_attempted += 1
            if rec.rescue_success:
                self.rescue_success += 1
            else:
                self.rescue_failed += 1
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

    def rescue_attempted_rate(self) -> float:
        return float(self.rescue_attempted / max(self.total, 1))

    def rescue_success_rate(self) -> float:
        return float(self.rescue_success / max(self.total, 1))

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
            "rescue_attempted": self.rescue_attempted,
            "rescue_success": self.rescue_success,
            "rescue_failed": self.rescue_failed,
            "rescue_attempted_rate": self.rescue_attempted_rate(),
            "rescue_success_rate": self.rescue_success_rate(),
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
class BlockIdentityRecord:
    """Per-request paired record across the four Phase-R0 methods."""

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
    k50_empty: bool
    strict: Dict[str, Any]
    preferred: Dict[str, Any]
    strict_rescue: Dict[str, Any]
    preferred_rescue: Dict[str, Any]

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
            "k50_empty": self.k50_empty,
            "strict": self.strict,
            "preferred": self.preferred,
            "strict_rescue": self.strict_rescue,
            "preferred_rescue": self.preferred_rescue,
        }


# Override base module data classes so that base worker/process pickling uses
# the new definitions.
_base.FixedRequestRecord = FixedRequestRecord
_base.RescueProbeRecord = RescueProbeRecord
_base.MethodSummary = MethodSummary
_base.BlockIdentityRecord = BlockIdentityRecord


# ---------------------------------------------------------------------------
# Rescue helpers
# ---------------------------------------------------------------------------
def _method_label_for_mode(mode: str, topology: str) -> str:
    if mode == "strict":
        return STRICT_NAME
    if mode == "preferred_heuristic":
        return METHOD_LABELS[TOPOLOGY_METHODS[topology]["preferred"]]
    if mode == "strict_k500_rescue":
        return STRICT_RESCUE_NAME
    if mode == "preferred_heuristic_k500_rescue":
        heuristic = TOPOLOGY_METHODS[topology]["preferred"]
        return HEURISTIC_RESCUE_NAME.format(heuristic=METHOD_LABELS[heuristic])
    raise ValueError(f"Unknown mode: {mode}")


def _probe_rescue_availability(
    env,
    req,
    args: argparse.Namespace,
    server_id: int,
    topology: str,
    seed: int,
    method_label: str,
    step_idx: int,
    free_ratio: float,
    lfb: float,
    frag: float,
    phi_spec: float,
) -> RescueProbeRecord:
    """Probe K=100/200/500 availability without mutating environment state."""
    original_k = env.k
    original_path_sort = env.path_sort_strategy
    original_block_sort = env.block_sort_strategy
    dst_node = int(getattr(req, "_dst_node", req.src_node))
    k100_available = False
    k200_available = False
    k500_available = False
    first_rescue_rank: Optional[int] = None
    first_rescue_k: Optional[int] = None
    path_hops: Optional[int] = None
    path_length_km: Optional[float] = None
    modulation: Optional[str] = None
    required_fs: Optional[int] = None
    block_start: Optional[int] = None
    block_size: Optional[int] = None
    try:
        for k in RESCUE_KS:
            env.k = k
            env.path_sort_strategy = "hops"
            obs = build_agent_r_observation(env, req, args.fixed_split_id, server_id)
            action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=k - 1)
            available = action is not None
            if k == 100:
                k100_available = available
            elif k == 200:
                k200_available = available
            elif k == 500:
                k500_available = available
            if action is not None and first_rescue_k is None:
                path_idx, mod_idx, block_idx = decode_agent_r_action(
                    action, len(obs["mod_names"]), env.max_blocks
                )
                meta = _base._extract_r_action_meta(obs, (path_idx, mod_idx, block_idx))
                first_rescue_k = k
                first_rescue_rank = path_idx
                path_hops = meta["hop_count"]
                path_length_km = meta["path_length_km"]
                modulation = meta["mod_name"]
                required_fs = meta["required_fs"]
                block_start = meta["block_start"]
                block_size = meta["block_size"]
    finally:
        env.k = original_k
        env.path_sort_strategy = original_path_sort
        env.block_sort_strategy = original_block_sort
    return RescueProbeRecord(
        topology=topology,
        seed=seed,
        method=method_label,
        step_idx=step_idx,
        req_id=int(req.req_id),
        src_node=int(req.src_node),
        dst_node=dst_node,
        free_ratio_before=free_ratio,
        lfb_before=lfb,
        fragmentation_before=frag,
        phi_spec_before=phi_spec,
        k100_available=k100_available,
        k200_available=k200_available,
        k500_available=k500_available,
        first_rescue_rank=first_rescue_rank,
        first_rescue_k=first_rescue_k,
        path_hops=path_hops,
        path_length_km=path_length_km,
        modulation=modulation,
        required_fs=required_fs,
        block_start=block_start,
        block_size=block_size,
    )


def _attempt_rescue(
    env,
    req,
    args: argparse.Namespace,
    server_id: int,
    agent_r,
    ranker,
    base_method: str,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Lazily expand to K=100/200/500 and return the first rescue action."""
    for k in RESCUE_KS:
        env.k = k
        env.path_sort_strategy = "hops"
        obs = build_agent_r_observation(env, req, args.fixed_split_id, server_id)
        action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=k - 1)
        if action is not None:
            path_idx, mod_idx, block_idx = decode_agent_r_action(
                action, len(obs["mod_names"]), env.max_blocks
            )
            meta = _base._extract_r_action_meta(obs, (path_idx, mod_idx, block_idx))
            return int(action), {
                "found_k": k,
                "action_rank": int(action),
                "path_idx": path_idx,
                "path_rank": path_idx,
                "meta": meta,
            }
    # No rescue action found; restore environment settings.
    env.k = args.k_paths_r
    env.path_sort_strategy = args.path_sort_strategy_r
    env.block_sort_strategy = args.block_sort_strategy_r
    return None, {}


# ---------------------------------------------------------------------------
# Episode runner (override base version)
# ---------------------------------------------------------------------------
def _run_episode(
    env,
    requests: List[Any],
    agent_r,
    ranker: Optional[Dict[str, Any]],
    mode: str,
    args: argparse.Namespace,
    seed: int,
    topology: str,
) -> Tuple[MethodSummary, List[FixedRequestRecord], List[RescueProbeRecord]]:
    """Run one episode under fixed-C / all-OD with optional rescue expansion."""
    assert mode in (
        "strict",
        "preferred_heuristic",
        "strict_k500_rescue",
        "preferred_heuristic_k500_rescue",
    )
    is_rescue = mode.endswith("_k500_rescue")
    base_method = mode.replace("_k500_rescue", "")
    if base_method == "preferred_heuristic":
        base_method = TOPOLOGY_METHODS[topology]["preferred"]
    method_label = _method_label_for_mode(mode, topology)
    records: List[FixedRequestRecord] = []
    probes: List[RescueProbeRecord] = []
    summary = MethodSummary(method=method_label)
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks

    node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < args.warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        free_ratio, lfb, frag, phi_spec = _base._spectrum_state(env)

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
        k50_empty = not agent_r_mask.any()

        strict_top30_contains_ksp = False
        strict_rank_of_ksp = -1
        strict_top30_contains_ffksp = False
        strict_rank_of_ffksp = -1
        strict_selected_rank = -1

        if k50_empty:
            if not is_rescue:
                probe = _probe_rescue_availability(
                    env, req, args, server_id, topology, seed, method_label, step_idx,
                    free_ratio, lfb, frag, phi_spec,
                )
                probes.append(probe)
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
                        rescue_attempted=False,
                    )
                    records.append(rec)
                    summary.add_record(rec)
                continue

            # Rescue mode: attempt expanded KSP-FF.
            rescue_action, rescue_info = _attempt_rescue(
                env, req, args, server_id, agent_r, ranker, base_method
            )
            if rescue_action is not None:
                r_idx = int(rescue_action)
                r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
                _, _, _, info = env.step((args.fixed_split_id, server_id), r_action)
                # Restore K=50 after the rescue step.
                env.k = args.k_paths_r
                env.path_sort_strategy = args.path_sort_strategy_r
                env.block_sort_strategy = args.block_sort_strategy_r
                if not is_warmup:
                    meta = rescue_info["meta"]
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
                        success=True,
                        failure_reason="",
                        delay_ms=float(info.get("delay_ms", 0.0)),
                        decision_ms=(time.perf_counter() - t0) * 1000.0,
                        rescue_attempted=True,
                        rescue_action_rank=rescue_info["action_rank"],
                        rescue_path_idx=rescue_info["path_idx"],
                        rescue_path_rank=rescue_info["path_rank"],
                        rescue_success=True,
                        rescue_reason="",
                    )
                    records.append(rec)
                    summary.add_record(rec)
            else:
                env.reject_next_request(req.req_id, "r_no_valid_action_after_k500_rescue")
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
                        failure_reason="r_no_valid_action_after_k500_rescue",
                        delay_ms=0.0,
                        decision_ms=(time.perf_counter() - t0) * 1000.0,
                        rescue_attempted=True,
                        rescue_action_rank=None,
                        rescue_path_idx=None,
                        rescue_path_rank=None,
                        rescue_success=False,
                        rescue_reason="r_no_valid_action_after_k500_rescue",
                    )
                    records.append(rec)
                    summary.add_record(rec)
            continue

        # K=50 mask is non-empty: run the baseline selector for this mode.
        if base_method == "strict":
            r_idx, r_valid, r_info = _base._select_r_action_strict(
                ranker, agent_r, env, req, obs_r, args.fixed_split_id, server_id
            )
            strict_top30_contains_ksp = r_info.get("strict_top30_contains_ksp_action", False)
            strict_rank_of_ksp = r_info.get("strict_rank_of_ksp_action_if_contained", -1)
            strict_top30_contains_ffksp = r_info.get("strict_top30_contains_ffksp_action", False)
            strict_rank_of_ffksp = r_info.get("strict_rank_of_ffksp_action_if_contained", -1)
            strict_selected_rank = r_info.get("strict_selected_rank_in_ppo_r_top30", -1)
        else:
            selector = METHOD_SELECTORS[base_method]
            r_idx, r_valid, _ = _base._select_r_action_heuristic(selector, obs_r)

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
                    rescue_attempted=False,
                )
                records.append(rec)
                summary.add_record(rec)
            continue

        r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
        _, _, _, info = env.step((args.fixed_split_id, server_id), r_action)

        if not is_warmup:
            meta = _base._extract_r_action_meta(obs_r, r_action)
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
                rescue_attempted=False,
            )
            records.append(rec)
            summary.add_record(rec)

    return summary, records, probes


# ---------------------------------------------------------------------------
# Worker process (override base version)
# ---------------------------------------------------------------------------
def _run_task(task: Tuple[str, int], args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Process one (topology, seed) pair for all four Phase-R0 methods."""
    topology, seed = task
    t_start = time.perf_counter()

    output_dir = Path(args_dict["output_dir"])
    task_dir = output_dir / "per_task"
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / f"{topology}_seed{seed}.result.json"
    triad_path = task_dir / f"{topology}_seed{seed}.triad.jsonl.gz"
    probe_path = task_dir / f"{topology}_seed{seed}.probes.jsonl.gz"
    done_path = task_dir / f"{topology}_seed{seed}.done.json"

    agent_r = _base._WORKER_PPO_R
    ranker = _base._WORKER_RANKER

    args = argparse.Namespace(**args_dict)
    total_requests = args.warmup_requests + args.requests_per_episode

    env_proto = _base._build_env(args, topology, seed)
    server_node_ids = [int(s.node_id) for s in env_proto.mec.servers]
    rng = np.random.RandomState(seed)
    requests = _base._generate_all_od_requests(
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
    request_trace_hash = _base._request_trace_hash(requests)

    modes = [
        "strict",
        "preferred_heuristic",
        "strict_k500_rescue",
        "preferred_heuristic_k500_rescue",
    ]
    all_records: Dict[str, List[FixedRequestRecord]] = {}
    summaries: Dict[str, MethodSummary] = {}
    all_probes: List[RescueProbeRecord] = []
    for mode in modes:
        env = _base._build_env(args, topology, seed)
        env.reset(requests)
        summary, records, probes = _run_episode(
            env, requests, agent_r, ranker, mode, args, seed, topology
        )
        all_records[mode] = records
        summaries[mode] = summary
        all_probes.extend(probes)

    preferred = TOPOLOGY_METHODS[topology]["preferred"]
    block_identities = _build_block_identity_records(
        all_records["strict"],
        all_records["preferred_heuristic"],
        all_records["strict_k500_rescue"],
        all_records["preferred_heuristic_k500_rescue"],
        preferred,
        topology,
    )

    # Write per-task block identity trace (named .triad.jsonl.gz for compatibility).
    tmp_triad = triad_path.with_suffix(".tmp.gz")
    with gzip.open(tmp_triad, "wt", encoding="utf-8") as f:
        for ident in block_identities:
            f.write(json.dumps(ident.to_dict(), default=str) + "\n")
    os.replace(tmp_triad, triad_path)

    # Write per-task probe records.
    tmp_probe = probe_path.with_suffix(".tmp.gz")
    with gzip.open(tmp_probe, "wt", encoding="utf-8") as f:
        for p in all_probes:
            f.write(json.dumps(p.to_dict(), default=str) + "\n")
    os.replace(tmp_probe, probe_path)

    result = {
        "topology": topology,
        "seed": seed,
        "request_trace_hash": request_trace_hash,
        "summaries": {mode: summaries[mode].to_dict() for mode in modes},
        "per_request_count": {mode: len(all_records[mode]) for mode in modes},
        "block_identity_count": len(block_identities),
        "probe_count": len(all_probes),
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
# Block identity builder
# ---------------------------------------------------------------------------
def _build_block_identity_records(
    strict_records: List[FixedRequestRecord],
    preferred_records: List[FixedRequestRecord],
    strict_rescue_records: List[FixedRequestRecord],
    preferred_rescue_records: List[FixedRequestRecord],
    preferred_method: str,
    topology: str,
) -> List[BlockIdentityRecord]:
    """Build paired per-request records across the four methods."""
    identities: List[BlockIdentityRecord] = []
    for sr, pr, srr, prr in zip(
        strict_records, preferred_records, strict_rescue_records, preferred_rescue_records
    ):
        if sr.warmup:
            continue
        k50_empty = not sr.success and sr.failure_reason == "r_no_valid_action"

        def _method_record(rec: FixedRequestRecord) -> Dict[str, Any]:
            return {
                "method": rec.method,
                "r_idx": rec.r_idx,
                "path_idx": rec.path_idx,
                "modulation": rec.modulation,
                "required_fs": rec.required_fs,
                "block_start": rec.block_start,
                "block_size": rec.block_size,
                "success": rec.success,
                "failure_reason": rec.failure_reason,
                "rescue_attempted": rec.rescue_attempted,
                "rescue_success": rec.rescue_success,
                "rescue_path_idx": rec.rescue_path_idx,
            }

        identities.append(
            BlockIdentityRecord(
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
                k50_empty=k50_empty,
                strict=_method_record(sr),
                preferred=_method_record(pr),
                strict_rescue=_method_record(srr),
                preferred_rescue=_method_record(prr),
            )
        )
    return identities


# ---------------------------------------------------------------------------
# Smoke test (override base version)
# ---------------------------------------------------------------------------
def _run_smoke_test(args: argparse.Namespace) -> None:
    """Preflight smoke test: one seed per topology, four methods."""
    print("=" * 70)
    print("PREFLIGHT SMOKE TEST: Phase-R0 Expanded-Path Rescue Ceiling")
    print("=" * 70)

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _base._load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    ranker = _base._load_ranker_model(args.strict_ranker_checkpoint, args.device)

    smoke_post = getattr(args, "smoke_requests", 2000)
    smoke_results: List[Dict[str, Any]] = []

    for topology in TOPOLOGY_ORDER:
        seed = TOPOLOGY_SEEDS[topology][0]
        print(f"\n--- Topology: {topology} (seed {seed}) ---")
        env = _base._build_env(args, topology, seed)
        server_node_ids = [int(s.node_id) for s in env.mec.servers]
        num_nodes = env.net.NUM_NODES
        num_links = len(env.net.G.edges)
        print(f"Nodes: {num_nodes}, Links: {num_links}")
        print(f"Server nodes: {server_node_ids}")

        rng = np.random.RandomState(seed)
        total_smoke = args.warmup_requests + smoke_post
        requests = _base._generate_all_od_requests(
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

        modes = [
            "strict",
            "preferred_heuristic",
            "strict_k500_rescue",
            "preferred_heuristic_k500_rescue",
        ]
        per_mode: Dict[str, Dict[str, Any]] = {}
        for mode in modes:
            env_copy = _base._build_env(args, topology, seed)
            env_copy.reset(requests)
            summary, records, probes = _run_episode(
                env_copy, requests, agent_r, ranker, mode, args, seed, topology
            )
            per_mode[mode] = {
                "summary": summary,
                "records": records,
                "probes": probes,
            }
            print(
                f"  {summary.method}: blocking={summary.blocking_rate():.4%}, "
                f"total={summary.total}, admitted={summary.admitted}, blocked={summary.blocked}, "
                f"rescue_attempted={summary.rescue_attempted}, rescue_success={summary.rescue_success}"
            )

        baseline_records = (
            per_mode["strict"]["records"] + per_mode["preferred_heuristic"]["records"]
        )
        k50_empty_events = sum(
            1 for r in baseline_records
            if not r.warmup and r.failure_reason == "r_no_valid_action"
        )
        strict_rescue_success = sum(
            1 for r in per_mode["strict_k500_rescue"]["records"] if not r.warmup and r.rescue_success
        )
        preferred_rescue_success = sum(
            1 for r in per_mode["preferred_heuristic_k500_rescue"]["records"]
            if not r.warmup and r.rescue_success
        )
        print(f"  K50-empty events (baseline): {k50_empty_events}")
        print(f"  Rescue successes: strict={strict_rescue_success}, preferred={preferred_rescue_success}")

        smoke_results.append(
            {
                "topology": topology,
                "seed": seed,
                "smoke_post_warmup": smoke_post,
                "k50_empty_events": k50_empty_events,
                "strict_rescue_success": strict_rescue_success,
                "preferred_rescue_success": preferred_rescue_success,
                "per_mode": {
                    mode: {
                        "method": per_mode[mode]["summary"].method,
                        "blocking_rate": per_mode[mode]["summary"].blocking_rate(),
                        "total": per_mode[mode]["summary"].total,
                        "admitted": per_mode[mode]["summary"].admitted,
                        "blocked": per_mode[mode]["summary"].blocked,
                        "rescue_attempted": per_mode[mode]["summary"].rescue_attempted,
                        "rescue_success": per_mode[mode]["summary"].rescue_success,
                    }
                    for mode in modes
                },
            }
        )

        if k50_empty_events < 20 and smoke_post < args.requests_per_episode:
            print(
                f"  WARNING: only {k50_empty_events} K50-empty events observed. "
                f"Consider increasing --smoke_requests to {args.requests_per_episode} for full coverage."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_json = output_dir / "SMOKE_RESULTS.json"
    smoke_md = output_dir / "SMOKE_RESULTS.md"
    smoke_json.write_text(json.dumps(smoke_results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Smoke Test Results: Phase-R0 Expanded-Path Rescue Ceiling",
        "",
        "| Topology | Seed | Evaluated | K50-empty events | Strict rescue success | Preferred rescue success |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in smoke_results:
        lines.append(
            f"| {r['topology']} | {r['seed']} | {r['smoke_post_warmup']} | "
            f"{r['k50_empty_events']} | {r['strict_rescue_success']} | {r['preferred_rescue_success']} |"
        )
    lines.extend(["", "## Per-method blocking rates", ""])
    for r in smoke_results:
        lines.append(f"### {r['topology']}")
        lines.append("| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for mode, m in r["per_mode"].items():
            lines.append(
                f"| {m['method']} | {m['blocking_rate']:.4%} | {m['total']} | "
                f"{m['admitted']} | {m['blocked']} | {m['rescue_attempted']} | {m['rescue_success']} |"
            )
        lines.append("")
    smoke_md.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"SMOKE TEST COMPLETE. Results written to {smoke_json} and {smoke_md}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Output writers (override base versions)
# ---------------------------------------------------------------------------
def _write_outputs(
    args: argparse.Namespace,
    all_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write all Phase-R0 aggregate outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [
        "strict",
        "preferred_heuristic",
        "strict_k500_rescue",
        "preferred_heuristic_k500_rescue",
    ]

    topology_results: Dict[str, Dict[str, Any]] = {}
    for topology in TOPOLOGY_ORDER:
        topology_results[topology] = {
            "results": [],
            "summaries": {mode: MethodSummary(method=_method_label_for_mode(mode, topology)) for mode in modes},
            "block_identities": [],
            "probes": [],
            "server_node_ids": [],
        }

    for res in all_results:
        topology = res["topology"]
        seed = res["seed"]
        topology_results[topology]["results"].append(res)
        for mode, summary_dict in res["summaries"].items():
            agg = topology_results[topology]["summaries"][mode]
            agg.total += summary_dict["total"]
            agg.admitted += summary_dict["admitted"]
            agg.blocked += summary_dict["blocked"]
            agg.server_overload += summary_dict["server_overload"]
            agg.no_suitable_block += summary_dict["no_suitable_block"]
            agg.r_no_valid += summary_dict["r_no_valid"]
            agg.deadline_infeasible += summary_dict["deadline_infeasible"]
            agg.other_failure += summary_dict["other_failure"]
            agg.rescue_attempted += summary_dict["rescue_attempted"]
            agg.rescue_success += summary_dict["rescue_success"]
            agg.rescue_failed += summary_dict["rescue_failed"]
            agg.delay_sum += summary_dict["avg_delay_ms"] * summary_dict["admitted"]
            agg.fs_sum += summary_dict["avg_fs"] * summary_dict["admitted"]
            agg.decision_ms_sum += summary_dict["avg_decision_ms"] * summary_dict["total"]
            agg.path_length_sum += summary_dict["avg_path_km"] * summary_dict["admitted"]
            agg.hop_count_sum += summary_dict["avg_hops"] * summary_dict["admitted"]
            agg.required_fs_values.extend(summary_dict.get("required_fs_values", []))
            agg.block_start_values.extend(summary_dict.get("block_start_values", []))
            agg.block_waste_values.extend(summary_dict.get("block_waste_values", []))
            for k, v in summary_dict.get("path_idx_distribution", {}).items():
                agg.path_idx_counts[int(k)] = agg.path_idx_counts.get(int(k), 0) + int(v * summary_dict["admitted"])
            for k, v in summary_dict.get("mod_distribution", {}).items():
                agg.mod_counts[k] = agg.mod_counts.get(k, 0) + int(v * summary_dict["admitted"])

    # Read per-task block identity and probe files.
    for topology in TOPOLOGY_ORDER:
        for res in topology_results[topology]["results"]:
            seed = res["seed"]
            triad_path = output_dir / "per_task" / f"{topology}_seed{seed}.triad.jsonl.gz"
            if triad_path.exists():
                with gzip.open(triad_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        topology_results[topology]["block_identities"].append(json.loads(line))
            probe_path = output_dir / "per_task" / f"{topology}_seed{seed}.probes.jsonl.gz"
            if probe_path.exists():
                with gzip.open(probe_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        topology_results[topology]["probes"].append(json.loads(line))

    for topology in TOPOLOGY_ORDER:
        env_proto = _base._build_env(args, topology, 42)
        topology_results[topology]["server_node_ids"] = [int(s.node_id) for s in env_proto.mec.servers]

    # Per-seed summary CSV.
    csv_path = output_dir / "per_seed_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["topology", "seed"]
        for mode in modes:
            prefix = mode
            header.extend([
                f"{prefix}_total", f"{prefix}_blocked", f"{prefix}_blocking_rate",
                f"{prefix}_r_no_valid_rate", f"{prefix}_rescue_attempted", f"{prefix}_rescue_success",
                f"{prefix}_avg_path_km", f"{prefix}_avg_decision_ms",
            ])
        writer.writerow(header)
        for topology in TOPOLOGY_ORDER:
            for res in topology_results[topology]["results"]:
                row = [topology, res["seed"]]
                for mode in modes:
                    s = res["summaries"][mode]
                    row.extend([
                        s.get("total", 0), s.get("blocked", 0), s.get("blocking_rate", 0.0),
                        s.get("r_no_valid_rate", 0.0), s.get("rescue_attempted", 0), s.get("rescue_success", 0),
                        s.get("avg_path_km", 0.0), s.get("avg_decision_ms", 0.0),
                    ])
                writer.writerow(row)

    # Global block identity trace.
    trace_path = output_dir / "paired_block_identity.jsonl.gz"
    tmp_trace = trace_path.with_suffix(".tmp.gz")
    with gzip.open(tmp_trace, "wt", encoding="utf-8") as out:
        for topology in TOPOLOGY_ORDER:
            for ident in topology_results[topology]["block_identities"]:
                out.write(json.dumps(ident, default=str) + "\n")
    os.replace(tmp_trace, trace_path)

    # Global K50-empty rescue probe events.
    probe_out_path = output_dir / "mask_empty_rescue_events.jsonl.gz"
    tmp_probe = probe_out_path.with_suffix(".tmp.gz")
    with gzip.open(tmp_probe, "wt", encoding="utf-8") as out:
        for topology in TOPOLOGY_ORDER:
            for probe in topology_results[topology]["probes"]:
                out.write(json.dumps(probe, default=str) + "\n")
    os.replace(tmp_probe, probe_out_path)

    # Per-topology results.
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

    # Comparison / rescue ceiling summary.
    _write_rescue_ceiling_summary(args, output_dir, topology_results)

    # Protocol lock and audits.
    _write_protocol_lock(args, output_dir)
    _write_implementation_audit(args, output_dir)
    _write_path_expansion_audit(args, output_dir)
    _write_request_trace_audit(args, output_dir)
    _write_experiment_manifest(args, output_dir, all_results, topology_results)
    _write_next_step_decision(args, output_dir, topology_results)

    print(f"[write] {csv_path}")
    print(f"[write] {trace_path}")
    print(f"[write] {probe_out_path}")
    for topology in TOPOLOGY_ORDER:
        print(f"[write] {output_dir / (TOPOLOGY_OUTPUT_PREFIX[topology] + '_RESULTS.json')}")
        print(f"[write] {output_dir / (TOPOLOGY_OUTPUT_PREFIX[topology] + '_RESULTS.md')}")


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------
def _build_topology_markdown(topology: str, payload: Dict[str, Any]) -> str:
    ov = payload["overall"]
    modes = ["strict", "preferred_heuristic", "strict_k500_rescue", "preferred_heuristic_k500_rescue"]
    lines = [
        f"# Results: {topology}",
        "",
        "## Protocol Lock",
        "",
        f"- Topology: `{topology}`",
        "- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)",
        "- Traffic matrix: uniform all-OD",
        f"- fixed_split_id: {payload['all_od']['fixed_split_id']}",
        f"- server_node_ids: {payload['all_od']['server_node_ids']}",
        f"- R-side K_path: {payload['config']['k_paths_r']}",
        f"- R-side path_sort_strategy: `{payload['config']['path_sort_strategy_r']}`",
        f"- R-side block_sort_strategy: `{payload['config']['block_sort_strategy_r']}`",
        f"- max_blocks: {payload['config']['max_blocks']}",
        f"- Preferred heuristic: `{METHOD_LABELS[TOPOLOGY_METHODS[topology]['preferred']]}`, rescue enabled at K=100/200/500",
        f"- Seeds: {payload['config']['seeds']}",
        f"- Warmup: {payload['config']['warmup_requests']}, Evaluated: {payload['config']['requests_per_episode']}",
        "",
        "## Main Result",
        "",
        "| Method | Blocking | R-no-valid | Rescue attempted | Rescue success | Avg path km | Avg decision ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        m = ov[mode]
        lines.append(
            f"| {m['method']} | {m['blocking_rate']:.4%} | {m['r_no_valid_rate']:.4%} | "
            f"{m['rescue_attempted']} | {m['rescue_success']} | {m['avg_path_km']:.2f} | {m['avg_decision_ms']:.2f} |"
        )
    lines.extend(["", "## Per-seed summaries", "", "| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |", "|---|---:|---:|---:|---:|"])
    for row in payload["per_seed"]:
        lines.append(
            f"| {row['seed']} | "
            f"{row['summaries']['strict']['blocking_rate']:.4%} | "
            f"{row['summaries']['preferred_heuristic']['blocking_rate']:.4%} | "
            f"{row['summaries']['strict_k500_rescue']['blocking_rate']:.4%} | "
            f"{row['summaries']['preferred_heuristic_k500_rescue']['blocking_rate']:.4%} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_rescue_ceiling_summary(
    args: argparse.Namespace,
    output_dir: Path,
    topology_results: Dict[str, Dict[str, Any]],
) -> None:
    """Aggregate rescue ceiling statistics across all topologies."""
    total_k50_empty = 0
    total_k100_available = 0
    total_k200_available = 0
    total_k500_available = 0
    strict_rescue_success = 0
    preferred_rescue_success = 0

    per_topology: List[Dict[str, Any]] = []
    for topology in TOPOLOGY_ORDER:
        probes = topology_results[topology]["probes"]
        k50_empty = len(probes)
        k100 = sum(1 for p in probes if p["k100_available"])
        k200 = sum(1 for p in probes if p["k200_available"])
        k500 = sum(1 for p in probes if p["k500_available"])
        total_k50_empty += k50_empty
        total_k100_available += k100
        total_k200_available += k200
        total_k500_available += k500

        # Rescue successes are recorded in the rescue-mode summaries.
        s_summary = topology_results[topology]["summaries"]["strict_k500_rescue"]
        p_summary = topology_results[topology]["summaries"]["preferred_heuristic_k500_rescue"]
        strict_rescue_success += s_summary.rescue_success
        preferred_rescue_success += p_summary.rescue_success

        per_topology.append({
            "topology": topology,
            "k50_empty_events": k50_empty,
            "k100_available": k100,
            "k200_available": k200,
            "k500_available": k500,
            "k500_ceiling_rate": k500 / max(k50_empty, 1),
            "strict_rescue_success": s_summary.rescue_success,
            "preferred_rescue_success": p_summary.rescue_success,
        })

    payload = {
        "aggregate": {
            "k50_empty_events": total_k50_empty,
            "k100_available": total_k100_available,
            "k200_available": total_k200_available,
            "k500_available": total_k500_available,
            "k500_ceiling_rate": total_k500_available / max(total_k50_empty, 1),
            "strict_rescue_success": strict_rescue_success,
            "preferred_rescue_success": preferred_rescue_success,
            "remaining_blocked_after_k500_rescue": total_k50_empty - max(strict_rescue_success, preferred_rescue_success),
        },
        "per_topology": per_topology,
    }

    json_path = output_dir / "RESCUE_CEILING_SUMMARY.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Rescue Ceiling Summary: Fixed-C / All-OD K50-empty Expanded-Path Rescue",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| K50-empty events (baseline) | {total_k50_empty} |",
        f"| K100 available | {total_k100_available} |",
        f"| K200 available | {total_k200_available} |",
        f"| K500 available | {total_k500_available} |",
        f"| K500 ceiling rate | {payload['aggregate']['k500_ceiling_rate']:.4%} |",
        f"| Strict rescue success | {strict_rescue_success} |",
        f"| Preferred rescue success | {preferred_rescue_success} |",
        "",
        "## Per-topology",
        "",
        "| Topology | K50-empty | K100 | K200 | K500 | K500 ceiling | Strict rescue | Preferred rescue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_topology:
        lines.append(
            f"| {row['topology']} | {row['k50_empty_events']} | {row['k100_available']} | "
            f"{row['k200_available']} | {row['k500_available']} | "
            f"{row['k500_ceiling_rate']:.4%} | {row['strict_rescue_success']} | {row['preferred_rescue_success']} |"
        )
    lines.append("")
    md_path = output_dir / "RESCUE_CEILING_SUMMARY.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Audit / protocol lock writers
# ---------------------------------------------------------------------------
def _write_protocol_lock(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "PROTOCOL_LOCK.md"
    lines = [
        "# Protocol Lock: Phase-R0 Fixed-C / All-OD R-mask-empty Expanded-Path Rescue Ceiling",
        "",
        "## Scope",
        "",
        "This experiment compares Strict v1.3 (frozen cross-topology transfer), the preferred topology heuristic, and their lazy K=100/200/500 expanded-path rescue variants under fixed-C / all-OD pure RMSA.",
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
        "- k_paths_r: 50",
        "- path_sort_strategy_r: `hops`",
        "- block_sort_strategy_r: `start_asc`",
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
        f"- **{STRICT_RESCUE_NAME}**: Strict v1.3 when K=50 mask is non-empty; otherwise lazy K=100/200/500 expanded KSP-FF rescue.",
        f"- **{HEURISTIC_RESCUE_NAME}**: Preferred topology heuristic when K=50 mask is non-empty; otherwise lazy K=100/200/500 expanded KSP-FF rescue.",
        "",
        "## Topology-specific preferred heuristic",
        "",
        "| Topology | Preferred heuristic |",
        "|---|---|",
    ]
    for topology in TOPOLOGY_ORDER:
        lines.append(
            f"| {topology} | {METHOD_LABELS[TOPOLOGY_METHODS[topology]['preferred']]} |"
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


def _write_implementation_audit(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "IMPLEMENTATION_AUDIT.md"
    lines = [
        "# Implementation Audit: Phase-R0 Expanded-Path Rescue",
        "",
        "## Rescue action selector",
        "",
        "`expanded_ksp_ff_rescue_action` is defined in `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py`. It scans expanded candidate paths in hops/km/lexicographic node-tuple order, selects the feasible modulation with the smallest required FS (highest spectral efficiency on ties, then smallest mod index), and selects the lowest start-slot feasible block.",
        "",
        "## Rescue trigger",
        "",
        "Rescue is only triggered when the K=50 R-mask is empty. If K=100, K=200, and K=500 all fail, the request is rejected with reason `r_no_valid_action_after_k500_rescue`.",
        "",
        "## Environment K restoration",
        "",
        "During rescue, `env.k` is temporarily set to the expansion K, the rescue action is executed via `env.step`, and `env.k` is restored to 50 immediately after the step.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_path_expansion_audit(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "PATH_EXPANSION_AUDIT.md"
    lines = [
        "# Path Expansion Audit",
        "",
        f"- Base K_path: {args.k_paths_r}",
        "- Expansion K values: 100, 200, 500",
        f"- Path ordering: `{args.path_sort_strategy_r}` with lexicographic node-tuple tie-break.",
        "- The first 50 paths of any expansion are identical to the K=50 path set because the KSP implementation is deterministic.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_request_trace_audit(args: argparse.Namespace, output_dir: Path) -> None:
    path = output_dir / "REQUEST_TRACE_AUDIT.md"
    lines = [
        "# Request Trace Audit",
        "",
        "Each task uses the same generated request trace for all four methods, ensuring a paired comparison. The per-task `.triad.jsonl.gz` files contain block identity records linking the four method outcomes per request.",
        "",
        f"- Output trace: `paired_block_identity.jsonl.gz`",
        f"- Probe events: `mask_empty_rescue_events.jsonl.gz`",
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
            "agent_r": _base._sha256(args.agent_r_checkpoint),
            "ranker": _base._sha256(args.strict_ranker_checkpoint),
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
                "num_nodes": _base._build_env(args, topology, 42).net.NUM_NODES,
                "num_links": len(_base._build_env(args, topology, 42).net.G.edges),
            }
            for topology in TOPOLOGY_ORDER
        },
        "output_files": [
            "PROTOCOL_LOCK.md",
            "IMPLEMENTATION_AUDIT.md",
            "PATH_EXPANSION_AUDIT.md",
            "REQUEST_TRACE_AUDIT.md",
            "EXPERIMENT_MANIFEST.json",
            "SMOKE_RESULTS.json",
            "SMOKE_RESULTS.md",
            "per_seed_summary.csv",
            "mask_empty_rescue_events.jsonl.gz",
            "paired_block_identity.jsonl.gz",
            "COST239_RESULTS.md",
            "COST239_RESULTS.json",
            "NSFNET_RESULTS.md",
            "NSFNET_RESULTS.json",
            "USNET_RESULTS.md",
            "USNET_RESULTS.json",
            "JPN48_RESULTS.md",
            "JPN48_RESULTS.json",
            "RESCUE_CEILING_SUMMARY.md",
            "RESCUE_CEILING_SUMMARY.json",
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
        "# Next Step Decision: Phase-R0 Expanded-Path Rescue Ceiling",
        "",
        "## Aggregate results",
        "",
        "| Topology | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |",
        "|---|---:|---:|---:|---:|",
    ]
    for topology in TOPOLOGY_ORDER:
        s = topology_results[topology]["summaries"]["strict"]
        p = topology_results[topology]["summaries"]["preferred_heuristic"]
        sr = topology_results[topology]["summaries"]["strict_k500_rescue"]
        pr = topology_results[topology]["summaries"]["preferred_heuristic_k500_rescue"]
        lines.append(
            f"| {topology} | {s.blocking_rate():.4%} | {p.blocking_rate():.4%} | "
            f"{sr.blocking_rate():.4%} | {pr.blocking_rate():.4%} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "If rescue variants substantially lower blocking compared to their baselines, the K50 path pool is the binding constraint and Phase-R1 should investigate path pool expansion. "
        "If rescue provides little benefit, the bottleneck is not path coverage but spectrum/contention, and the next step should focus on spectrum management or load regime."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hashing helpers (override base to include the new script)
# ---------------------------------------------------------------------------
def _code_file_paths() -> List[Path]:
    base = Path("sa_hmarl/sa_hmarl")
    return [
        base / "evaluation" / "generate_r_mask_empty_rescue_phase_r0.py",
        base / "evaluation" / "generate_multitopology_strict_v13_vs_heuristics.py",
        base / "baselines" / "rmsa_baselines.py",
        base / "env" / "observation_builder.py",
        base / "env" / "event_env.py",
        base / "network" / "topology_data.py",
        base / "network" / "ksp.py",
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
        hashes[topology] = _base._sha256_text(text)[:24]
    return hashes


def _compute_config_hash(args: argparse.Namespace) -> str:
    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    return _base._sha256_text(json.dumps(cfg, sort_keys=True, default=str))[:24]


# Override base hashing functions.
_base._code_file_paths = _code_file_paths
_base._compute_code_hash = _compute_code_hash
_base._compute_topology_hashes = _compute_topology_hashes
_base._compute_config_hash = _compute_config_hash


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = _base.build_parser()
    parser.set_defaults(
        output_dir=OUTPUT_DIR,
        seeds=None,
    )
    parser.add_argument(
        "--smoke_requests",
        type=int,
        default=2000,
        help="Number of post-warmup requests to evaluate in the smoke test (default: 2000).",
    )
    parser.add_argument(
        "--smoke_only",
        action="store_true",
        help="Run the smoke test and exit without launching the full pilot.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.skip_smoke or args.smoke_only:
        _run_smoke_test(args)
        if args.smoke_only:
            return 0

    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use per-topology seeds unless --seeds is explicitly provided.
    seeds_by_topology: Dict[str, List[int]] = {}
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        for topology in TOPOLOGY_ORDER:
            seeds_by_topology[topology] = seeds
    else:
        for topology in TOPOLOGY_ORDER:
            seeds_by_topology[topology] = TOPOLOGY_SEEDS[topology]

    tasks = [
        (topology, seed)
        for topology in TOPOLOGY_ORDER
        for seed in seeds_by_topology[topology]
    ]
    print(f"\n[main] Full pilot: {len(tasks)} (topology, seed) tasks")

    args_dict = vars(args)
    args_dict["_config_hash"] = _compute_config_hash(args)
    args_dict["_topology_hashes"] = _compute_topology_hashes()
    args_dict["_code_hash"] = _compute_code_hash()

    tasks_to_run = []
    for topology, seed in tasks:
        done_path = output_dir / "per_task" / f"{topology}_seed{seed}.done.json"
        if args.skip_done_check or not _base._is_done_marker_valid(done_path, args_dict, topology, seed):
            tasks_to_run.append((topology, seed))
        else:
            print(f"[main] Skipping completed task {topology} seed {seed}")

    if tasks_to_run:
        max_workers = args.max_workers if args.max_workers > 0 else min(len(tasks_to_run), os.cpu_count() or 1)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_base._worker_init,
            initargs=(args_dict,),
        ) as executor:
            results = list(executor.map(_run_task, tasks_to_run, [args_dict] * len(tasks_to_run)))
    else:
        results = []

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


# Override base episode/task functions so the base worker initializer and any
# direct callers use the Phase-R0 logic.
_base._run_episode = _run_episode
_base._run_task = _run_task
_base._run_smoke_test = _run_smoke_test
_base._write_outputs = _write_outputs
_base._build_topology_markdown = _build_topology_markdown
_base._write_protocol_lock = _write_protocol_lock
_base._write_experiment_manifest = _write_experiment_manifest
_base._write_next_step_decision = _write_next_step_decision

if __name__ == "__main__":
    raise SystemExit(main())
