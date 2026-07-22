"""Strict v1.3 multi-topology, multi-C-side fair closed-loop evaluation (verified).

Implements the full hard-gating, latency-decomposition, hash-conservation and
action-distribution requirements of the v1.3-verified protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    build_poststate_v1_feature_batch,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


SCHEMA_VERSION = "v1.3-verified-2026-07-14"

# Source files tracked in the reproducibility manifest.
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

RANKER_MODES = ("strict_v13", "old_v13", "v135_afterstate", "v135_afterstate_explicit")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    topology: str
    seed: int
    c_mode: str
    r_mode: str
    method_name: str

    # Request accounting
    evaluated_requests: int = 0
    admitted: int = 0
    blocked: int = 0
    r_reached: int = 0
    r_decisions: int = 0
    c_no_valid_action: int = 0
    r_no_valid_action: int = 0

    # C-side mask diagnostics
    c_raw_mask_empty: int = 0
    c_effective_mask_empty: int = 0
    c_action_not_in_effective_mask: int = 0

    # R-side mask diagnostics
    r_mask_empty: int = 0

    # Failure decomposition (mutually exclusive)
    no_suitable_block: int = 0
    server_overload: int = 0          # server_saturated + server_overload
    deadline_failure: int = 0
    other_failure: int = 0
    failure_reason_counts: Dict[str, int] = field(default_factory=dict)

    # Performance
    delay_sum: float = 0.0
    fs_sum: float = 0.0
    delay_ms_list: List[float] = field(default_factory=list)

    # Latency decomposition (milliseconds)
    c_policy_ms_sum: float = 0.0
    r_proposer_ms_sum: float = 0.0
    ranker_or_ksp_ms_sum: float = 0.0
    audit_ms_sum: float = 0.0
    total_policy_ms_sum: float = 0.0
    decision_ms_sum: float = 0.0
    decision_ms_list: List[float] = field(default_factory=list)
    c_policy_ms_list: List[float] = field(default_factory=list)
    r_proposer_ms_list: List[float] = field(default_factory=list)
    ranker_or_ksp_ms_list: List[float] = field(default_factory=list)

    # PPO-R agreement
    same_as_ppo_r: int = 0

    # Action distributions -- admitted only
    path_length_list: List[float] = field(default_factory=list)
    hop_count_list: List[float] = field(default_factory=list)
    mod_name_list: List[str] = field(default_factory=list)
    path_idx_list: List[int] = field(default_factory=list)
    block_start_list: List[int] = field(default_factory=list)
    required_fs_list: List[int] = field(default_factory=list)

    # Action distributions -- all selected (r_decisions)
    selected_path_idx_list: List[int] = field(default_factory=list)
    selected_mod_name_list: List[str] = field(default_factory=list)
    selected_block_start_list: List[int] = field(default_factory=list)
    selected_required_fs_list: List[int] = field(default_factory=list)
    selected_hop_count_list: List[float] = field(default_factory=list)
    selected_path_length_list: List[float] = field(default_factory=list)

    # C-side distributions (evaluated requests)
    split_ids: List[int] = field(default_factory=list)
    server_ids: List[int] = field(default_factory=list)
    split_server_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # E-stratum diagnostics (E == number of edges traversed == split_id for default3)
    e_stratum_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Traceability
    selected_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Strict v1.3 extras
    candidate_counts: List[int] = field(default_factory=list)
    fallback_count: int = 0
    ranker_top1_in_candidates_count: int = 0
    score_margins: List[float] = field(default_factory=list)
    feature_finite_flags: List[bool] = field(default_factory=list)
    nan_inf_feature_count: int = 0
    nan_inf_score_count: int = 0

    # Request-sync / action-conservation diagnostics
    initial_event_queue_length: int = 0
    consumed_request_count: int = 0
    final_event_queue_length: int = 0
    request_id_mismatch_count: int = 0

    selected_mask_false_count: int = 0
    invalid_path_after_legal_action_count: int = 0
    modulation_reach_after_legal_action_count: int = 0
    no_suitable_block_after_legal_action_count: int = 0
    action_observation_consistency_error_count: int = 0

    # Depth stratum diagnostics (fixed thresholds)
    depth_stratum_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.evaluated_requests, 1))

    def admitted_rate(self) -> float:
        return float(self.admitted / max(self.evaluated_requests, 1))

    def overload_rate(self) -> float:
        return float(self.server_overload / max(self.evaluated_requests, 1))

    def nsb_rate(self) -> float:
        return float(self.no_suitable_block / max(self.evaluated_requests, 1))

    def deadline_failure_rate(self) -> float:
        return float(self.deadline_failure / max(self.evaluated_requests, 1))

    def other_rate(self) -> float:
        return float(self.other_failure / max(self.evaluated_requests, 1))

    def c_no_valid_action_rate(self) -> float:
        return float(self.c_no_valid_action / max(self.evaluated_requests, 1))

    def r_no_valid_action_rate(self) -> float:
        return float(self.r_no_valid_action / max(self.evaluated_requests, 1))

    def avg_delay_ms(self) -> float:
        return float(np.mean(self.delay_ms_list)) if self.delay_ms_list else 0.0

    def delay_p50_ms(self) -> float:
        return float(np.percentile(self.delay_ms_list, 50)) if self.delay_ms_list else 0.0

    def delay_p90_ms(self) -> float:
        return float(np.percentile(self.delay_ms_list, 90)) if self.delay_ms_list else 0.0

    def delay_p95_ms(self) -> float:
        return float(np.percentile(self.delay_ms_list, 95)) if self.delay_ms_list else 0.0

    def avg_fs(self) -> float:
        return float(self.fs_sum / max(self.admitted, 1))

    def avg_decision_ms(self) -> float:
        return float(self.decision_ms_sum / max(self.evaluated_requests, 1))

    def avg_c_policy_ms(self) -> float:
        return float(self.c_policy_ms_sum / max(self.evaluated_requests, 1))

    def avg_r_proposer_ms(self) -> float:
        return float(self.r_proposer_ms_sum / max(self.evaluated_requests, 1))

    def avg_ranker_or_ksp_ms(self) -> float:
        return float(self.ranker_or_ksp_ms_sum / max(self.evaluated_requests, 1))

    def avg_audit_ms(self) -> float:
        return float(self.audit_ms_sum / max(self.evaluated_requests, 1))

    def avg_total_policy_ms(self) -> float:
        return float(self.total_policy_ms_sum / max(self.evaluated_requests, 1))

    def decision_p95_ms(self) -> float:
        return float(np.percentile(self.decision_ms_list, 95)) if self.decision_ms_list else 0.0

    def avg_path_length_km(self) -> float:
        return float(np.mean(self.path_length_list)) if self.path_length_list else 0.0

    def avg_hop_count(self) -> float:
        return float(np.mean(self.hop_count_list)) if self.hop_count_list else 0.0

    def ppo_agreement_rate(self) -> Optional[float]:
        if self.r_decisions == 0:
            return None
        return float(self.same_as_ppo_r / self.r_decisions)

    def fallback_rate(self) -> float:
        return float(self.fallback_count / max(self.r_reached, 1))

    def ranker_top1_in_candidates_rate(self) -> float:
        return float(self.ranker_top1_in_candidates_count / max(self.r_reached, 1))

    def avg_candidate_count(self) -> float:
        return float(np.mean(self.candidate_counts)) if self.candidate_counts else 0.0

    def candidate_count_p95(self) -> float:
        return float(np.percentile(self.candidate_counts, 95)) if self.candidate_counts else 0.0

    def avg_score_margin(self) -> float:
        return float(np.mean(self.score_margins)) if self.score_margins else 0.0

    def score_margin_p95(self) -> float:
        return float(np.percentile(self.score_margins, 95)) if self.score_margins else 0.0

    def feature_finite_rate(self) -> float:
        if not self.feature_finite_flags:
            return 1.0
        return float(np.mean(self.feature_finite_flags))

    def consistency_error_rate(self) -> float:
        return float(self.action_observation_consistency_error_count / max(self.evaluated_requests, 1))

    def _count(self, items: List[Any]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for x in items:
            k = str(x)
            c[k] = c.get(k, 0) + 1
        return c

    def _normalized_dist(self, counts: Dict[str, int], denom: int) -> Dict[str, Any]:
        if denom <= 0:
            return {"_reason": "denominator_zero", "normalized": {}}
        total = sum(counts.values())
        if total == 0:
            return {"_reason": "count_zero", "normalized": {}}
        return {
            "raw_counts": counts,
            "normalized": {k: v / denom for k, v in counts.items()},
            "sum_normalized": sum(v / denom for v in counts.values()),
        }

    def to_summary_dict(
        self,
        request_trace_hash: str,
        selected_action_hash: str,
        config_hash: str,
        code_hash: str,
        runner_code_hash: str,
        analyzer_code_hash: str,
        source_manifest: Dict[str, str],
        checkpoint_sha256s: Dict[str, str],
    ) -> Dict[str, Any]:
        ev = self.evaluated_requests
        ad = self.admitted
        rd = self.r_decisions

        split_counts = self._count(self.split_ids)
        server_counts = self._count(self.server_ids)
        joint_counts: Dict[str, int] = {}
        for pair in self.split_server_pairs:
            joint_counts[str(pair)] = joint_counts.get(str(pair), 0) + 1

        # Admitted-only distributions
        mod_counts = self._count(self.mod_name_list)
        path_idx_counts = self._count(self.path_idx_list)
        block_start_counts = self._count(self.block_start_list)
        required_fs_counts = self._count(self.required_fs_list)
        hop_count_counts = self._count(self.hop_count_list)
        path_length_counts = self._count([round(pl, 2) for pl in self.path_length_list])

        # Selected distributions
        sel_path_idx_counts = self._count(self.selected_path_idx_list)
        sel_mod_counts = self._count(self.selected_mod_name_list)
        sel_block_start_counts = self._count(self.selected_block_start_list)
        sel_required_fs_counts = self._count(self.selected_required_fs_list)
        sel_hop_counts = self._count([round(h, 2) for h in self.selected_hop_count_list])
        sel_path_lengths = self._count([round(pl, 2) for pl in self.selected_path_length_list])

        out: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "topology": self.topology,
            "seed": self.seed,
            "c_mode": self.c_mode,
            "r_mode": self.r_mode,
            "method_name": self.method_name,
            "evaluated_requests": ev,
            "admitted": ad,
            "blocked": self.blocked,
            "blocking_rate": self.blocking_rate(),
            "admitted_rate": self.admitted_rate(),
            "r_reached": self.r_reached,
            "r_decisions": rd,
            "r_no_valid_action": self.r_no_valid_action,
            "c_no_valid_action": self.c_no_valid_action,
            "c_no_valid_action_rate": self.c_no_valid_action_rate(),
            "r_no_valid_action_rate": self.r_no_valid_action_rate(),
            "c_raw_mask_empty": self.c_raw_mask_empty,
            "c_effective_mask_empty": self.c_effective_mask_empty,
            "c_action_not_in_effective_mask": self.c_action_not_in_effective_mask,
            "r_mask_empty": self.r_mask_empty,
            "server_overload": self.server_overload,
            "overload_rate": self.overload_rate(),
            "no_suitable_block": self.no_suitable_block,
            "nsb_rate": self.nsb_rate(),
            "deadline_failure": self.deadline_failure,
            "deadline_failure_rate": self.deadline_failure_rate(),
            "other_failure": self.other_failure,
            "other_rate": self.other_rate(),
            "failure_reason_distribution": dict(self.failure_reason_counts),
            "avg_delay_ms": self.avg_delay_ms(),
            "delay_p50_ms": self.delay_p50_ms(),
            "delay_p90_ms": self.delay_p90_ms(),
            "delay_p95_ms": self.delay_p95_ms(),
            "avg_fs": self.avg_fs(),
            "avg_decision_ms": self.avg_decision_ms(),
            "decision_p95_ms": self.decision_p95_ms(),
            "avg_c_policy_ms": self.avg_c_policy_ms(),
            "avg_r_proposer_ms": self.avg_r_proposer_ms(),
            "avg_ranker_or_ksp_ms": self.avg_ranker_or_ksp_ms(),
            "avg_total_policy_ms": self.avg_total_policy_ms(),
            "avg_audit_ms": self.avg_audit_ms(),
            "c_policy_ms_p95": float(np.percentile(self.c_policy_ms_list, 95)) if self.c_policy_ms_list else 0.0,
            "r_proposer_ms_p95": float(np.percentile(self.r_proposer_ms_list, 95)) if self.r_proposer_ms_list else 0.0,
            "ranker_or_ksp_ms_p95": float(np.percentile(self.ranker_or_ksp_ms_list, 95)) if self.ranker_or_ksp_ms_list else 0.0,
            "avg_path_length_km": self.avg_path_length_km(),
            "avg_hop_count": self.avg_hop_count(),
            "ppo_agreement_rate": self.ppo_agreement_rate(),
            "ppo_agreement_numerator": self.same_as_ppo_r,
            "ppo_agreement_denominator": rd,
            "admitted_path_idx_distribution": self._normalized_dist(path_idx_counts, ad),
            "admitted_block_start_distribution": self._normalized_dist(block_start_counts, ad),
            "admitted_required_fs_distribution": self._normalized_dist(required_fs_counts, ad),
            "admitted_hop_count_distribution": self._normalized_dist(hop_count_counts, ad),
            "admitted_path_length_distribution": self._normalized_dist(path_length_counts, ad),
            "admitted_mod_distribution": self._normalized_dist(mod_counts, ad),
            "selected_path_idx_distribution": self._normalized_dist(sel_path_idx_counts, rd),
            "selected_mod_distribution": self._normalized_dist(sel_mod_counts, rd),
            "selected_block_start_distribution": self._normalized_dist(sel_block_start_counts, rd),
            "selected_required_fs_distribution": self._normalized_dist(sel_required_fs_counts, rd),
            "selected_hop_count_distribution": self._normalized_dist(sel_hop_counts, rd),
            "selected_path_length_distribution": self._normalized_dist(sel_path_lengths, rd),
            "split_distribution": self._normalized_dist(split_counts, ev),
            "server_distribution": self._normalized_dist(server_counts, ev),
            "split_server_joint_distribution": self._normalized_dist(joint_counts, ev),
            "e_stratum_counts": self.e_stratum_counts,
            "depth_stratum_counts": self.depth_stratum_counts,
            "initial_event_queue_length": self.initial_event_queue_length,
            "consumed_request_count": self.consumed_request_count,
            "final_event_queue_length": self.final_event_queue_length,
            "request_id_mismatch_count": self.request_id_mismatch_count,
            "selected_mask_false_count": self.selected_mask_false_count,
            "invalid_path_after_legal_action_count": self.invalid_path_after_legal_action_count,
            "modulation_reach_after_legal_action_count": self.modulation_reach_after_legal_action_count,
            "no_suitable_block_after_legal_action_count": self.no_suitable_block_after_legal_action_count,
            "action_observation_consistency_error_count": self.action_observation_consistency_error_count,
            "request_trace_hash": request_trace_hash,
            "selected_action_hash": selected_action_hash,
            "config_hash": config_hash,
            "evaluator_code_hash": code_hash,
            "runner_code_hash": runner_code_hash,
            "analyzer_code_hash": analyzer_code_hash,
            "source_manifest": source_manifest,
            "checkpoint_sha256s": checkpoint_sha256s,
        }
        if self.r_mode in ("strict_v13", "old_v13"):
            out.update({
                "avg_candidate_count": self.avg_candidate_count(),
                "candidate_count_p95": self.candidate_count_p95(),
                "fallback_rate": self.fallback_rate(),
                "ranker_top1_in_candidates_rate": self.ranker_top1_in_candidates_rate(),
                "avg_score_margin": self.avg_score_margin(),
                "score_margin_p95": self.score_margin_p95(),
                "feature_finite_rate": self.feature_finite_rate(),
                "nan_inf_feature_count": self.nan_inf_feature_count,
                "nan_inf_score_count": self.nan_inf_score_count,
            })
        return out


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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_manifest(root: Path) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for rel in SOURCE_MANIFEST_FILES:
        path = root / rel
        if path.exists():
            manifest[rel] = _sha256(str(path))
        else:
            manifest[rel] = "missing"
    return manifest


def _hash_requests(requests: List[Any]) -> str:
    payload = []
    for req in requests:
        payload.append({
            "src": req.src_node,
            "arrival": req.arrival_time,
            "holding": req.holding_time,
            "deadline": req.deadline_ms,
            "splits": [
                {
                    "split_id": sp.split_id,
                    "intermediate_size_mb": sp.intermediate_size_mb,
                    "local_compute_cost": sp.local_compute_cost,
                    "edge_compute_cost": sp.edge_compute_cost,
                }
                for sp in req.splits
            ],
        })
    return _hash_text(json.dumps(payload, sort_keys=True, default=str))


def _code_hash() -> str:
    try:
        return _sha256(str(Path(__file__).resolve()))
    except Exception:
        return "N/A"


def _env_hash(env) -> str:
    cfg = {
        "topology": env.net.topology_name,
        "num_slots": env.net.num_slots,
        "num_servers": len(env.mec.servers),
        "modulation_profile": [m.name for m in env.mod_reg.mod_table],
        "max_blocks": env.max_blocks,
        "block_sort_strategy": env.block_sort_strategy,
        "path_sort_strategy": env.path_sort_strategy,
    }
    return _hash_text(json.dumps(cfg, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# Request-sync / action-consistency errors
# ---------------------------------------------------------------------------

class ActionObservationConsistencyError(Exception):
    """Raised when a masked legal action produces an impossible env failure."""


# ---------------------------------------------------------------------------
# E-stratum and depth-stratum definitions (fixed for v1.3)
# ---------------------------------------------------------------------------

def _depth_stratum(max_path_idx: int) -> int:
    """Stratify by deepest path index among PPO-R Top-30 legal candidates.

    0: max_path_idx < 5
    1: 5 <= max_path_idx < 10
    2: 10 <= max_path_idx < 20
    3: max_path_idx >= 20
    """
    if max_path_idx < 5:
        return 0
    if max_path_idx < 10:
        return 1
    if max_path_idx < 20:
        return 2
    return 3


def _e_gate_from_candidates(
    candidate_actions: List[int], obs_r: Dict[str, Any], max_blocks: int
) -> Tuple[bool, int, int]:
    """Return (is_deep_path/E=1, max_path_idx, depth_stratum)."""
    if not candidate_actions:
        return False, -1, -1
    num_mods = len(obs_r["mod_names"])
    path_indices = [
        decode_agent_r_action(a, num_mods, max_blocks)[0] for a in candidate_actions
    ]
    max_path_idx = int(max(path_indices))
    return max_path_idx >= 5, max_path_idx, _depth_stratum(max_path_idx)


def _e_label(max_path_idx: int, has_candidates: bool) -> str:
    if not has_candidates:
        return "E=NA/no_candidate"
    return "E=0" if max_path_idx < 5 else "E=1"


def _validate_selected_r_action(
    obs_r: Dict[str, Any],
    r_idx: int,
    r_action: Tuple[int, int, int],
    mod_reg: ModulationRegistry,
) -> None:
    """Hard-validate that a selected R action is consistent with its obs.

    Raises ActionObservationConsistencyError on any violation.
    """
    path_idx, mod_idx, block_idx = r_action
    mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

    if not (0 <= r_idx < len(mask) and mask[r_idx]):
        raise ActionObservationConsistencyError(
            f"selected action {r_idx} not in agent_r_mask (len={len(mask)})"
        )

    paths = obs_r["candidate_paths"]
    mod_names = obs_r["mod_names"]
    if not (0 <= path_idx < len(paths)):
        raise ActionObservationConsistencyError(
            f"decoded path_idx {path_idx} out of range [0, {len(paths)})"
        )
    if not (0 <= mod_idx < len(mod_names)):
        raise ActionObservationConsistencyError(
            f"decoded mod_idx {mod_idx} out of range [0, {len(mod_names)})"
        )

    required_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    if required_fs is None or required_fs <= 0:
        raise ActionObservationConsistencyError(
            f"required_fs {required_fs} invalid for path {path_idx} mod {mod_idx}"
        )

    path_length_km = obs_r["path_features"][path_idx]["path_length_km"]
    mod = mod_reg[mod_idx]
    if path_length_km > mod.reach_km:
        raise ActionObservationConsistencyError(
            f"modulation reach violation: path_length {path_length_km} > "
            f"{mod.name} reach {mod.reach_km}"
        )

    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    if not (0 <= block_idx < len(blocks)):
        raise ActionObservationConsistencyError(
            f"block_idx {block_idx} out of range [0, {len(blocks)})"
        )
    block_size = blocks[block_idx][1]
    if block_size < required_fs:
        raise ActionObservationConsistencyError(
            f"block_size {block_size} < required_fs {required_fs}"
        )


# ---------------------------------------------------------------------------
# Ranker loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# C-side selection
# ---------------------------------------------------------------------------

def _select_c_action(
    c_mode: str,
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
    baseline_map = {"df_c": "df", "rf_c": "rf", "wo_c": "wo", "greedy_c": "greedy", "iwd_c": "iwd"}
    baseline_name = baseline_map.get(c_mode, c_mode)

    if baseline_name in ("df", "rf", "wo", "greedy", "iwd"):
        raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        info["raw_mask_empty"] = not raw_mask.any()
        if info["raw_mask_empty"]:
            return None, None, None, False, info
        t0 = time.perf_counter()
        action_idx = select_offloading_action(baseline_name, env, req, obs_c, raw_mask)
        info["c_policy_ms"] = (time.perf_counter() - t0) * 1000.0
        if action_idx is None or not (0 <= int(action_idx) < len(raw_mask) and raw_mask[int(action_idx)]):
            info["action_in_effective_mask"] = False
            return None, None, None, False, info
        info["action_in_effective_mask"] = True
    else:
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


# ---------------------------------------------------------------------------
# R-side selection
# ---------------------------------------------------------------------------

def _select_r_action(
    r_mode: str,
    agent_r,
    ranker: Optional[Dict[str, Any]],
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    ppo_idx: int,
    split_id: int,
    server_id: int,
) -> Tuple[int, bool, Optional[Dict[str, Any]]]:
    """Return (r_idx, r_valid, info). info contains latency and ranker diagnostics.

    Preconditions (enforced by the caller):
      - obs_r["agent_r_mask"] has at least one True entry.
      - ppo_idx is a legal action under that mask.
    """
    info: Dict[str, Any] = {
        "r_mask_empty": False,
        "r_proposer_ms": 0.0,
        "ranker_or_ksp_ms": 0.0,
        "candidate_count": 0,
        "in_candidates": False,
        "fallback": False,
        "score_margin": 0.0,
        "feature_finite": True,
        "nan_inf_feature_count": 0,
        "nan_inf_score_count": 0,
        "selected_path_idx": 0,
    }

    if obs_r is not None:
        mask = np.asarray(obs_r.get("agent_r_mask", []), dtype=bool)
        if not mask.any():
            info["r_mask_empty"] = True
            return 0, False, info

    if r_mode == "ppo_r_top1":
        return int(ppo_idx), True, info

    if r_mode == "ksp_ff_highest":
        t0 = time.perf_counter()
        a = ksp_ff_highest_mod_action(obs_r)
        info["ranker_or_ksp_ms"] = (time.perf_counter() - t0) * 1000.0
        if a is None:
            return 0, False, info
        return int(a), True, info

    # Ranker-based selection (strict_v13, old_v13, v135_afterstate)
    t0 = time.perf_counter()
    candidates = _ppo_r_topk_actions(agent_r, obs_r, 30)
    info["r_proposer_ms"] = (time.perf_counter() - t0) * 1000.0
    candidates = np.asarray(candidates, dtype=np.int64)
    has_candidates = candidates.size > 0
    info["candidate_count"] = int(candidates.size)
    info["fallback"] = not has_candidates
    if not has_candidates:
        return int(ppo_idx), False, info

    t0 = time.perf_counter()
    r_features, _ = agent_r.build_action_features(obs_r)
    if r_mode in ("v135_afterstate", "v135_afterstate_explicit"):
        x = build_poststate_v1_feature_batch(
            env,
            req,
            obs_c,
            obs_r,
            r_features,
            candidates.tolist(),
            split_id,
            server_id,
            feature_names=POSTSTATE_V1_FEATURE_NAMES,
        ).astype(np.float32)
        # Support explicit-only checkpoints that keep only the 16 afterstate+delta dims.
        if ranker["input_dim"] == len(POSTSTATE_V1_FEATURE_NAMES) - 25:
            x = x[:, 25:]
    else:
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
    if not finite_all:
        info["nan_inf_feature_count"] = int(np.sum(~np.isfinite(x)))
    x = (x - ranker["feature_mean"]) / ranker["feature_std"]
    x[~np.isfinite(x)] = 0.0
    with torch.no_grad():
        scores = (
            ranker["model"](torch.as_tensor(x, dtype=torch.float32, device=ranker["device"]))
            .cpu()
            .numpy()
        )
    if not np.all(np.isfinite(scores)):
        info["nan_inf_score_count"] = int(np.sum(~np.isfinite(scores)))
    best_local = int(np.argmax(scores))
    r_idx = int(candidates[best_local])
    info["in_candidates"] = True
    sorted_scores = np.sort(scores)[::-1]
    info["score_margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
    path_idx, _, _ = decode_agent_r_action(r_idx, len(obs_r.get("mod_names", [])), env.max_blocks)
    info["selected_path_idx"] = int(path_idx)
    info["ranker_or_ksp_ms"] = (time.perf_counter() - t0) * 1000.0
    return r_idx, True, info


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _record_failure(res: MethodResult, reason: str) -> None:
    res.failure_reason_counts[reason] = res.failure_reason_counts.get(reason, 0) + 1
    if reason in ("server_saturated", "server_overload"):
        res.server_overload += 1
    elif reason == "no_suitable_block":
        res.no_suitable_block += 1
    elif reason in ("deadline_infeasible", "deadline_failure"):
        res.deadline_failure += 1
    elif reason in ("c_no_valid_action", "r_no_valid_action"):
        # These are tracked in dedicated counters and should not spill into other_failure.
        pass
    else:
        res.other_failure += 1


def _update_e_stratum(res: MethodResult, stratum: str, admitted: bool, blocked: bool) -> None:
    if stratum not in res.e_stratum_counts:
        res.e_stratum_counts[stratum] = {"evaluated": 0, "admitted": 0, "blocked": 0}
    res.e_stratum_counts[stratum]["evaluated"] += 1
    if admitted:
        res.e_stratum_counts[stratum]["admitted"] += 1
    if blocked:
        res.e_stratum_counts[stratum]["blocked"] += 1


def _extract_r_action_meta(obs_r: Dict[str, Any], r_action: Tuple[int, int, int]) -> Dict[str, Any]:
    path_idx, mod_idx, block_idx = r_action
    path_feats = obs_r.get("path_features", [])
    meta = {
        "path_idx": path_idx,
        "mod_name": "",
        "path_length_km": 0.0,
        "hop_count": 0.0,
        "block_start": -1,
        "required_fs": 0,
    }
    if path_feats and 0 <= path_idx < len(path_feats):
        meta["path_length_km"] = float(path_feats[path_idx].get("path_length_km", 0.0))
        meta["hop_count"] = float(path_feats[path_idx].get("hop_count", 0.0))
    mod_names = obs_r.get("mod_names", [])
    if mod_names and 0 <= mod_idx < len(mod_names):
        meta["mod_name"] = str(mod_names[mod_idx])
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    if block_idx < len(blocks):
        meta["block_start"] = int(blocks[block_idx][0])
    req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    meta["required_fs"] = int(req_fs) if req_fs is not None else 0
    return meta


def _run_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    ranker: Optional[Dict[str, Any]],
    c_mode: str,
    r_mode: str,
    args,
) -> MethodResult:
    res = MethodResult(
        topology=args.topology,
        seed=args.seed,
        c_mode=c_mode,
        r_mode=r_mode,
        method_name=f"{c_mode}+{r_mode}",
    )
    num_servers = len(env.mec.servers)
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks
    res.initial_event_queue_length = len(env.event_queue)

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < args.warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        # Pre-loop request-sync invariants.
        if not env.event_queue:
            raise RuntimeError(f"event queue empty before processing req {req.req_id}")
        head_arrival, _, head_req = env.event_queue[0]
        if int(head_req.req_id) != int(req.req_id):
            res.request_id_mismatch_count += 1
            raise RuntimeError(
                f"request id mismatch at step {step_idx}: loop req {req.req_id}, "
                f"queue head {head_req.req_id}"
            )
        if float(head_arrival) != float(req.arrival_time):
            res.request_id_mismatch_count += 1
            raise RuntimeError(
                f"arrival_time mismatch at step {step_idx}: loop {req.arrival_time}, "
                f"queue head {head_arrival}"
            )
        queue_len_before = len(env.event_queue)

        def _account_common(elapsed_ms: float, audit_ms: float = 0.0) -> None:
            res.decision_ms_sum += elapsed_ms
            res.decision_ms_list.append(elapsed_ms)
            res.c_policy_ms_sum += c_policy_ms
            res.audit_ms_sum += audit_ms
            res.c_policy_ms_list.append(c_policy_ms)

        # C-side under K_C
        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        c_idx, split_id, server_id, c_valid, c_info = _select_c_action(
            c_mode, agent_c, env, req, obs_c, num_servers
        )
        c_policy_ms = c_info.get("c_policy_ms", 0.0)

        if not c_valid:
            env.reject_next_request(req.req_id, "c_no_valid_action")
            res.consumed_request_count += 1
            assert len(env.event_queue) == queue_len_before - 1, (
                f"C-no-valid did not consume request {req.req_id}"
            )
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.c_no_valid_action += 1
                if c_info["raw_mask_empty"]:
                    res.c_raw_mask_empty += 1
                elif c_info["effective_mask_empty"]:
                    res.c_effective_mask_empty += 1
                else:
                    res.c_action_not_in_effective_mask += 1
                _record_failure(res, "c_no_valid_action")
                res.split_ids.append(-1)
                res.server_ids.append(-1)
                res.split_server_pairs.append((-1, -1))
                res.selected_actions.append({"c_idx": None, "split_id": -1, "server_id": -1, "r_idx": None, "c_valid": False, "r_valid": False})
                _account_common((time.perf_counter() - t0) * 1000.0)
                res.total_policy_ms_sum += c_policy_ms
            continue

        # R-side under K_R
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        agent_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        if not agent_r_mask.any():
            env.reject_next_request(req.req_id, "r_no_valid_action")
            res.consumed_request_count += 1
            assert len(env.event_queue) == queue_len_before - 1
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.r_reached += 1
                res.r_no_valid_action += 1
                res.r_mask_empty += 1
                _record_failure(res, "r_no_valid_action")
                res.split_ids.append(split_id)
                res.server_ids.append(server_id)
                res.split_server_pairs.append((split_id, server_id))
                res.selected_actions.append({"c_idx": int(c_idx), "split_id": int(split_id), "server_id": int(server_id), "r_idx": None, "c_valid": True, "r_valid": False})
                _account_common((time.perf_counter() - t0) * 1000.0)
                res.total_policy_ms_sum += c_policy_ms
            continue

        # Audit: PPO-R top-1 for agreement measurement (only when R mask is non-empty).
        audit_t0 = time.perf_counter()
        ppo_idx, ppo_raw_mask = _select_r_action_from_obs(agent_r, obs_r, max_blocks)
        audit_elapsed = (time.perf_counter() - audit_t0) * 1000.0

        # If the frozen PPO-R top-1 is itself illegal, treat as R-no-valid.
        if not (0 <= int(ppo_idx) < len(agent_r_mask) and agent_r_mask[int(ppo_idx)]):
            env.reject_next_request(req.req_id, "r_no_valid_action")
            res.consumed_request_count += 1
            assert len(env.event_queue) == queue_len_before - 1
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.r_reached += 1
                res.r_no_valid_action += 1
                _record_failure(res, "r_no_valid_action")
                res.split_ids.append(split_id)
                res.server_ids.append(server_id)
                res.split_server_pairs.append((split_id, server_id))
                res.selected_actions.append({"c_idx": int(c_idx), "split_id": int(split_id), "server_id": int(server_id), "r_idx": None, "c_valid": True, "r_valid": False})
                _account_common((time.perf_counter() - t0) * 1000.0, audit_elapsed)
                res.total_policy_ms_sum += c_policy_ms
            continue

        # Compute E-gate and depth stratum from PPO-R Top-30 legal candidates.
        candidates = _ppo_r_topk_actions(agent_r, obs_r, 30)
        is_deep, max_path_idx, depth_stratum = _e_gate_from_candidates(
            candidates, obs_r, max_blocks
        )
        e_label = _e_label(max_path_idx, len(candidates) > 0)
        depth_label = f"D={depth_stratum}"

        # Policy decision.
        policy_t0 = time.perf_counter()
        r_idx, r_valid, r_info = _select_r_action(
            r_mode, agent_r, ranker, env, req, obs_c, obs_r, ppo_idx, split_id, server_id
        )
        policy_elapsed = (time.perf_counter() - policy_t0) * 1000.0
        r_proposer_ms = r_info.get("r_proposer_ms", 0.0) if r_info else 0.0
        ranker_or_ksp_ms = r_info.get("ranker_or_ksp_ms", 0.0) if r_info else 0.0

        if not r_valid:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            res.consumed_request_count += 1
            assert len(env.event_queue) == queue_len_before - 1
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.r_reached += 1
                res.r_no_valid_action += 1
                _record_failure(res, "r_no_valid_action")
                res.split_ids.append(split_id)
                res.server_ids.append(server_id)
                res.split_server_pairs.append((split_id, server_id))
                res.selected_actions.append({"c_idx": int(c_idx), "split_id": int(split_id), "server_id": int(server_id), "r_idx": None, "c_valid": True, "r_valid": False})
                if r_mode in RANKER_MODES and r_info is not None:
                    res.candidate_counts.append(r_info["candidate_count"])
                    if r_info["fallback"]:
                        res.fallback_count += 1
                    else:
                        if r_info["in_candidates"]:
                            res.ranker_top1_in_candidates_count += 1
                        res.score_margins.append(r_info["score_margin"])
                        res.feature_finite_flags.append(r_info["feature_finite"])
                        res.nan_inf_feature_count += r_info["nan_inf_feature_count"]
                        res.nan_inf_score_count += r_info["nan_inf_score_count"]
                _account_common((time.perf_counter() - t0) * 1000.0, audit_elapsed)
                res.r_proposer_ms_sum += r_proposer_ms
                res.ranker_or_ksp_ms_sum += ranker_or_ksp_ms
                res.total_policy_ms_sum += c_policy_ms + r_proposer_ms + ranker_or_ksp_ms
                res.r_proposer_ms_list.append(r_proposer_ms)
                res.ranker_or_ksp_ms_list.append(ranker_or_ksp_ms)
                _update_e_stratum(res, e_label, admitted=False, blocked=True)
                _update_e_stratum(res, depth_label, admitted=False, blocked=True)
            continue

        # Validate selected action against the mask and observation.
        r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
        try:
            _validate_selected_r_action(obs_r, r_idx, r_action, env.mod_reg)
        except ActionObservationConsistencyError as exc:
            res.selected_mask_false_count += 1
            res.action_observation_consistency_error_count += 1
            raise

        same_as_ppo = False
        if ppo_idx is not None:
            same_as_ppo = int(r_idx) == int(ppo_idx)

        _, _, _, info = env.step((split_id, server_id), r_action)
        res.consumed_request_count += 1
        assert len(env.event_queue) == queue_len_before - 1, (
            f"env.step did not consume request {req.req_id}"
        )
        if info.get("req_id") != req.req_id:
            res.request_id_mismatch_count += 1
            raise RuntimeError(
                f"env.step consumed wrong request: expected {req.req_id}, got {info.get('req_id')}"
            )

        # Any impossible failure from a legal masked action is a consistency error.
        if not info.get("success", False):
            reason = info.get("reason", "unknown")
            if reason == "invalid_path":
                res.invalid_path_after_legal_action_count += 1
                res.action_observation_consistency_error_count += 1
                raise ActionObservationConsistencyError(
                    f"invalid_path after legal action {r_idx}"
                )
            if reason == "modulation_reach":
                res.modulation_reach_after_legal_action_count += 1
                res.action_observation_consistency_error_count += 1
                raise ActionObservationConsistencyError(
                    f"modulation_reach after legal action {r_idx}"
                )
            if reason == "no_suitable_block":
                res.no_suitable_block_after_legal_action_count += 1
                res.action_observation_consistency_error_count += 1
                raise ActionObservationConsistencyError(
                    f"no_suitable_block after legal action {r_idx}"
                )

        meta = _extract_r_action_meta(obs_r, r_action)

        if not is_warmup:
            res.evaluated_requests += 1
            res.r_reached += 1
            res.r_decisions += 1
            res.split_ids.append(split_id)
            res.server_ids.append(server_id)
            res.split_server_pairs.append((split_id, server_id))
            res.selected_actions.append({"c_idx": int(c_idx), "split_id": int(split_id), "server_id": int(server_id), "r_idx": int(r_idx), "c_valid": True, "r_valid": True})
            decision_ms = (time.perf_counter() - t0) * 1000.0
            _account_common(decision_ms, audit_elapsed)
            res.r_proposer_ms_sum += r_proposer_ms
            res.ranker_or_ksp_ms_sum += ranker_or_ksp_ms
            res.total_policy_ms_sum += c_policy_ms + r_proposer_ms + ranker_or_ksp_ms
            res.r_proposer_ms_list.append(r_proposer_ms)
            res.ranker_or_ksp_ms_list.append(ranker_or_ksp_ms)

            # selected distribution
            res.selected_path_idx_list.append(meta["path_idx"])
            res.selected_mod_name_list.append(meta["mod_name"])
            res.selected_block_start_list.append(meta["block_start"])
            res.selected_required_fs_list.append(meta["required_fs"])
            res.selected_hop_count_list.append(meta["hop_count"])
            res.selected_path_length_list.append(meta["path_length_km"])

            if same_as_ppo:
                res.same_as_ppo_r += 1

            admitted = bool(info.get("success", False))
            if admitted:
                res.admitted += 1
                res.delay_sum += float(info.get("delay_ms", 0.0))
                res.fs_sum += float(info.get("num_slots", 0.0))
                res.delay_ms_list.append(float(info.get("delay_ms", 0.0)))
                res.path_length_list.append(meta["path_length_km"])
                res.hop_count_list.append(meta["hop_count"])
                res.mod_name_list.append(meta["mod_name"])
                res.path_idx_list.append(meta["path_idx"])
                res.block_start_list.append(meta["block_start"])
                res.required_fs_list.append(meta["required_fs"])
            else:
                res.blocked += 1
                reason = info.get("reason", "unknown")
                _record_failure(res, reason)

            _update_e_stratum(res, e_label, admitted=admitted, blocked=not admitted)
            _update_e_stratum(res, depth_label, admitted=admitted, blocked=not admitted)

            if r_mode in ("strict_v13", "old_v13") and r_info is not None:
                res.candidate_counts.append(r_info["candidate_count"])
                if r_info["fallback"]:
                    res.fallback_count += 1
                else:
                    if r_info["in_candidates"]:
                        res.ranker_top1_in_candidates_count += 1
                    res.score_margins.append(r_info["score_margin"])
                    res.feature_finite_flags.append(r_info["feature_finite"])
                    res.nan_inf_feature_count += r_info["nan_inf_feature_count"]
                    res.nan_inf_score_count += r_info["nan_inf_score_count"]

    if len(env.event_queue) != 0:
        raise RuntimeError(
            f"episode ended with {len(env.event_queue)} events remaining"
        )
    res.final_event_queue_length = len(env.event_queue)
    if res.consumed_request_count != res.initial_event_queue_length:
        raise RuntimeError(
            f"consumed_request_count ({res.consumed_request_count}) != "
            f"initial_event_queue_length ({res.initial_event_queue_length})"
        )
    return res


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _validate_result_row(row: Dict[str, Any], strict: bool = True) -> None:
    """Validate a per-method summary row. Raises AssertionError on violation."""
    required = [
        "schema_version", "topology", "seed", "c_mode", "r_mode", "method_name",
        "evaluated_requests", "admitted", "blocked", "r_reached", "r_decisions",
        "c_no_valid_action", "r_no_valid_action", "failure_reason_distribution",
        "ppo_agreement_rate", "ppo_agreement_numerator", "ppo_agreement_denominator",
        "request_trace_hash", "selected_action_hash", "config_hash",
        "evaluator_code_hash", "runner_code_hash", "analyzer_code_hash",
        "source_manifest", "checkpoint_sha256s",
    ]
    for key in required:
        if key not in row:
            raise AssertionError(f"Missing required field: {key}")

    ev = int(row["evaluated_requests"])
    ad = int(row["admitted"])
    blocked = int(row["blocked"])
    c_no = int(row["c_no_valid_action"])
    r_no = int(row["r_no_valid_action"])
    r_reached = int(row["r_reached"])
    r_decisions = int(row["r_decisions"])
    fr = row["failure_reason_distribution"]

    assert ev == ad + blocked, (
        f"evaluated_requests ({ev}) != admitted ({ad}) + blocked ({blocked})"
    )
    env_failures = sum(
        v for k, v in fr.items()
        if k not in ("c_no_valid_action", "r_no_valid_action")
    )
    assert blocked == c_no + r_no + env_failures, (
        f"blocked ({blocked}) != c_no ({c_no}) + r_no ({r_no}) + env_failures ({env_failures})"
    )
    assert r_reached == r_no + r_decisions, (
        f"r_reached ({r_reached}) != r_no ({r_no}) + r_decisions ({r_decisions})"
    )

    if r_decisions > 0:
        expected_agree = row["ppo_agreement_numerator"] / r_decisions
        assert abs(row["ppo_agreement_rate"] - expected_agree) < 1e-12, (
            "ppo_agreement_rate does not match numerator/denominator"
        )
    else:
        assert row["ppo_agreement_rate"] is None, "ppo_agreement_rate must be None when r_decisions=0"

    if row["r_mode"] == "ppo_r_top1":
        if r_decisions > 0:
            assert row["ppo_agreement_rate"] == 1.0, (
                f"ppo_r_top1 must have agreement 1.0, got {row['ppo_agreement_rate']}"
            )

    # Conservation of selected distributions
    if r_decisions > 0:
        for dist_key in ["selected_path_idx_distribution", "selected_mod_distribution"]:
            dist = row.get(dist_key, {})
            norm = dist.get("normalized", {})
            if norm:
                s = sum(norm.values())
                assert abs(s - 1.0) < 1e-6, f"{dist_key} normalized sums to {s}"

    if ad > 0:
        for dist_key in ["admitted_path_idx_distribution", "admitted_mod_distribution"]:
            dist = row.get(dist_key, {})
            norm = dist.get("normalized", {})
            if norm:
                s = sum(norm.values())
                assert abs(s - 1.0) < 1e-6, f"{dist_key} normalized sums to {s}"

    # Action-observation consistency: legal masked actions must never produce
    # these env-level failures.
    for key in (
        "selected_mask_false_count",
        "invalid_path_after_legal_action_count",
        "modulation_reach_after_legal_action_count",
        "no_suitable_block_after_legal_action_count",
        "action_observation_consistency_error_count",
    ):
        if row.get(key, 0) != 0:
            raise AssertionError(f"{key} must be 0, got {row.get(key)}")

    if strict and row["r_mode"] in RANKER_MODES:
        cc = row.get("avg_candidate_count", 0.0)
        if r_reached > 0:
            assert 0.0 <= cc <= 30.0, f"candidate_count {cc} out of [0,30]"
        assert row.get("feature_finite_rate", 1.0) == 1.0, "non-finite ranker features/scores detected"


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace, *, runner_code_hash: str = "", analyzer_code_hash: str = "") -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    root = Path(__file__).resolve().parents[3]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    agent_r_sha256 = _sha256(args.agent_r_checkpoint)

    c_modes = [m.strip() for m in args.c_modes.split(",") if m.strip()]
    c_checkpoints = [p.strip() for p in args.c_checkpoints.split(",") if p.strip()]
    if len(c_checkpoints) < len(c_modes):
        c_checkpoints.extend(["_none_"] * (len(c_modes) - len(c_checkpoints)))
    agent_c_map: Dict[str, Any] = {}
    c_checkpoint_sha256s: Dict[str, str] = {}
    for mode, ckpt in zip(c_modes, c_checkpoints):
        if mode.startswith("df") or mode.startswith("rf") or mode in ("wo", "greedy", "iwd"):
            agent_c_map[mode] = None
            c_checkpoint_sha256s[mode] = "heuristic"
        else:
            agent_c_map[mode] = _load_ppo_c(ckpt, args.device)
            c_checkpoint_sha256s[mode] = _sha256(ckpt)

    r_modes = [m.strip() for m in args.r_modes.split(",") if m.strip()]
    rankers: Dict[str, Dict[str, Any]] = {}
    ranker_sha256s: Dict[str, str] = {}
    for spec in args.ranker_specs or []:
        if "=" not in spec:
            raise ValueError(f"Ranker spec must be name=path, got {spec}")
        name, path = spec.split("=", 1)
        rankers[name] = _load_ranker_model(path, args.device)
        ranker_sha256s[name] = _sha256(path)

    total_requests = args.warmup_requests + args.requests_per_episode
    rng = np.random.RandomState(args.seed)
    env_for_nodes = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        args.seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy_r,
        path_sort_strategy=args.path_sort_strategy_r,
        k=args.k_paths_r,
    )
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
    request_trace_hash = _hash_requests(requests)
    code_hash = _code_hash()
    config_hash = _hash_text(json.dumps({k: v for k, v in vars(args).items() if not k.startswith("_")}, sort_keys=True, default=str))
    source_manifest = _source_manifest(root)

    checkpoint_sha256s = {
        "agent_r": agent_r_sha256,
        **c_checkpoint_sha256s,
        **ranker_sha256s,
    }

    results: List[Dict[str, Any]] = []
    logs: List[str] = []
    for c_mode in c_modes:
        agent_c = agent_c_map[c_mode]
        for r_mode in r_modes:
            ranker = rankers.get(r_mode) if r_mode in rankers else None
            env = make_env(
                args.topology,
                args.num_slots,
                args.num_servers,
                args.seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy_r,
                path_sort_strategy=args.path_sort_strategy_r,
                k=args.k_paths_r,
            )
            env.reset(requests)
            try:
                res = _run_episode(env, requests, agent_c, agent_r, ranker, c_mode, r_mode, args)
                selected_action_hash = _hash_text(json.dumps(res.selected_actions, sort_keys=True, default=str))
                row = res.to_summary_dict(
                    request_trace_hash,
                    selected_action_hash,
                    config_hash,
                    code_hash,
                    runner_code_hash,
                    analyzer_code_hash,
                    source_manifest,
                    checkpoint_sha256s,
                )
                _validate_result_row(row)
                results.append(row)
                logs.append(
                    f"[{args.topology}][seed={args.seed}][{res.method_name}] "
                    f"blocking={res.blocking_rate():.4%} "
                    f"c_no={res.c_no_valid_action} r_no={res.r_no_valid_action} "
                    f"r_reached={res.r_reached} r_decisions={res.r_decisions} "
                    f"policy_ms={res.avg_total_policy_ms():.3f} "
                    f"audit_ms={res.avg_audit_ms():.3f} "
                    f"ppo_agree={res.ppo_agreement_rate()}"
                )
            except Exception as exc:
                logs.append(
                    f"[{args.topology}][seed={args.seed}][{c_mode}+{r_mode}] FAILED: {exc}\n{traceback.format_exc()}"
                )
                raise

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "request_trace_hash": request_trace_hash,
        "config_hash": config_hash,
        "evaluator_code_hash": code_hash,
        "runner_code_hash": runner_code_hash,
        "analyzer_code_hash": analyzer_code_hash,
        "source_manifest": source_manifest,
        "checkpoint_sha256s": checkpoint_sha256s,
        "results": results,
        "logs": logs,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.output_json
    out_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log_file = out_dir / args.output_log
    log_file.write_text("\n".join(logs), encoding="utf-8")
    done_file = out_dir / "done.marker"
    done_file.write_text("done", encoding="utf-8")
    print(f"[eval] Saved {out_file} and {log_file}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--c_modes", default="ppo_c,df_c")
    parser.add_argument("--c_checkpoints", default="_none_,_none_")
    parser.add_argument("--r_modes", default="ppo_r_top1,ksp_ff_highest,strict_v13")
    parser.add_argument("--ranker_specs", action="append", default=[])
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
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
    parser.add_argument("--warmup_requests", type=int, default=1000)
    parser.add_argument("--requests_per_episode", type=int, default=5000)
    parser.add_argument("--poisson_arrivals", action="store_true")
    parser.add_argument("--exponential_holding", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v13_strict_multitopology_cside_verified/per_task")
    parser.add_argument("--output_json", default="results.json")
    parser.add_argument("--output_log", default="eval.log")
    parser.add_argument("--runner_code_hash", default="")
    parser.add_argument("--analyzer_code_hash", default="")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args, runner_code_hash=args.runner_code_hash, analyzer_code_hash=args.analyzer_code_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
