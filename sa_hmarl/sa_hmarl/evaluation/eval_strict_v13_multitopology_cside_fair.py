"""Strict v1.3 multi-topology fair closed-loop evaluation (verified).

Each topology-seed worker runs a fixed request trace through:
  C-side: configured C modes (e.g. ppo_c, df_c, ppo_c_snap24)
  R-side: ppo_r_top1, ksp_ff_highest (formal KSP-FF K=50 hops), strict_v13, (optional old_v13)

All methods share the same exogenous request sequence for a given topology-seed,
but each method gets an independent environment copy.

Verified v1.3 semantics implemented in this version:
  - Hard R-side gating: empty ``agent_r_mask`` records ``r_no_valid_action`` and
    skips all R-side inference (including audit PPO-R), env.step, and action 0.
  - Hard C-side gating: empty raw/effective mask or out-of-mask PPO-C action
    records ``c_no_valid_action`` and blocks without R inference.
  - ``r_reached = r_no_valid_action + r_decisions`` is conserved.
  - ``ppo_agreement_rate = same_as_ppo_r / r_decisions`` (null when denominator=0).
  - ``ppo_r_top1`` agreement is exactly 1.0 (or null).
  - Failure reasons are fully decomposed (c_no_valid_action, r_no_valid_action,
    no_suitable_block, server_overload, deadline_failure, other_failure) and
    conserved; server_saturated is merged into server_overload.
  - Latency is decomposed into c_policy, r_proposer, ranker_or_ksp, total_policy,
    and audit components with averages and p95.
  - Action distributions are reported for both selected (all r_decisions) and
    admitted actions.
  - E=0/E=1 split-depth diagnostics are reported per method.
  - Output schema includes code/config/trace hashes, checkpoint SHA-256s, and a
    source manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
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
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


SCHEMA_VERSION = "v1.3-verified-2026-07-14"


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
    total: int = 0
    admitted: int = 0
    blocked: int = 0
    r_reached: int = 0              # C-valid requests that reached R-side
    r_decisions: int = 0            # legal R action selected
    c_no_valid_action: int = 0
    r_no_valid_action: int = 0

    # C-side mask diagnostics (per evaluated request counts)
    c_raw_mask_empty: int = 0
    c_effective_mask_empty: int = 0
    c_action_in_effective_mask: int = 0
    r_mask_empty: int = 0           # C-valid requests with empty R mask

    # Failure decomposition
    server_overload: int = 0        # server_saturated + server_overload
    no_suitable_block: int = 0
    deadline_failure: int = 0
    other_failure: int = 0
    failure_reason_counts: Dict[str, int] = field(default_factory=dict)

    # Performance
    delay_sum: float = 0.0
    fs_sum: float = 0.0
    same_as_ppo_r: int = 0

    # Total decision latency (end-to-end per request)
    decision_ms_sum: float = 0.0
    decision_ms_list: List[float] = field(default_factory=list)

    # Decomposed policy latency sums (ms)
    c_policy_ms_sum: float = 0.0
    r_proposer_ms_sum: float = 0.0
    ranker_or_ksp_ms_sum: float = 0.0
    total_policy_ms_sum: float = 0.0
    audit_ms_sum: float = 0.0

    # Decomposed latency lists for p95 (ms)
    c_policy_ms_list: List[float] = field(default_factory=list)
    r_proposer_ms_list: List[float] = field(default_factory=list)
    ranker_or_ksp_ms_list: List[float] = field(default_factory=list)
    total_policy_ms_list: List[float] = field(default_factory=list)
    audit_ms_list: List[float] = field(default_factory=list)

    # Legacy aliases for backward-compatible consumption
    policy_decision_ms_sum: float = 0.0
    audit_decision_ms_sum: float = 0.0

    delay_ms_list: List[float] = field(default_factory=list)

    # Action distributions (admitted requests only; kept for compatibility)
    path_length_list: List[float] = field(default_factory=list)
    hop_count_list: List[float] = field(default_factory=list)
    mod_name_list: List[str] = field(default_factory=list)
    path_idx_list: List[int] = field(default_factory=list)
    block_start_list: List[int] = field(default_factory=list)
    required_fs_list: List[int] = field(default_factory=list)

    # C-side distributions (all evaluated requests)
    split_ids: List[int] = field(default_factory=list)
    server_ids: List[int] = field(default_factory=list)
    split_server_pairs: List[Tuple[int, int]] = field(default_factory=list)

    # Traceability
    selected_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Strict v1.3 extras
    candidate_counts: List[int] = field(default_factory=list)
    fallback_count: int = 0
    ranker_top1_in_candidates_count: int = 0
    score_margins: List[float] = field(default_factory=list)
    selected_path_indices: List[int] = field(default_factory=list)
    feature_finite_flags: List[bool] = field(default_factory=list)
    nan_inf_feature_count: int = 0
    nan_inf_score_count: int = 0

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.total, 1))

    def overload_rate(self) -> float:
        return float(self.server_overload / max(self.total, 1))

    def nsb_rate(self) -> float:
        return float(self.no_suitable_block / max(self.total, 1))

    def deadline_rate(self) -> float:
        return float(self.deadline_failure / max(self.total, 1))

    def other_rate(self) -> float:
        return float(self.other_failure / max(self.total, 1))

    def avg_delay_ms(self) -> float:
        return float(np.mean(self.delay_ms_list)) if self.delay_ms_list else 0.0

    def delay_p95_ms(self) -> float:
        return float(np.percentile(self.delay_ms_list, 95)) if self.delay_ms_list else 0.0

    def avg_fs(self) -> float:
        return float(self.fs_sum / max(self.admitted, 1))

    def avg_decision_ms(self) -> float:
        return float(self.decision_ms_sum / max(self.total, 1))

    def avg_c_policy_ms(self) -> float:
        return float(self.c_policy_ms_sum / max(self.total, 1))

    def c_policy_p95_ms(self) -> float:
        return float(np.percentile(self.c_policy_ms_list, 95)) if self.c_policy_ms_list else 0.0

    def avg_r_proposer_ms(self) -> float:
        return float(self.r_proposer_ms_sum / max(self.total, 1))

    def r_proposer_p95_ms(self) -> float:
        return float(np.percentile(self.r_proposer_ms_list, 95)) if self.r_proposer_ms_list else 0.0

    def avg_ranker_or_ksp_ms(self) -> float:
        return float(self.ranker_or_ksp_ms_sum / max(self.total, 1))

    def ranker_or_ksp_p95_ms(self) -> float:
        return float(np.percentile(self.ranker_or_ksp_ms_list, 95)) if self.ranker_or_ksp_ms_list else 0.0

    def avg_total_policy_ms(self) -> float:
        return float(self.total_policy_ms_sum / max(self.total, 1))

    def total_policy_p95_ms(self) -> float:
        return float(np.percentile(self.total_policy_ms_list, 95)) if self.total_policy_ms_list else 0.0

    def avg_audit_ms(self) -> float:
        return float(self.audit_ms_sum / max(self.total, 1))

    def audit_p95_ms(self) -> float:
        return float(np.percentile(self.audit_ms_list, 95)) if self.audit_ms_list else 0.0

    def avg_policy_decision_ms(self) -> float:
        return float(self.policy_decision_ms_sum / max(self.total, 1))

    def avg_audit_decision_ms(self) -> float:
        return float(self.audit_decision_ms_sum / max(self.total, 1))

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
        return float(self.fallback_count / max(self.r_decisions, 1))

    def ranker_top1_in_candidates_rate(self) -> float:
        return float(self.ranker_top1_in_candidates_count / max(self.r_decisions, 1))

    def avg_candidate_count(self) -> float:
        return float(np.mean(self.candidate_counts)) if self.candidate_counts else 0.0

    def candidate_count_p95(self) -> float:
        return float(np.percentile(self.candidate_counts, 95)) if self.candidate_counts else 0.0

    def avg_score_margin(self) -> float:
        return float(np.mean(self.score_margins)) if self.score_margins else 0.0

    def score_margin_p95(self) -> float:
        return float(np.percentile(self.score_margins, 95)) if self.score_margins else 0.0

    def avg_selected_path_idx(self) -> float:
        return float(np.mean(self.selected_path_indices)) if self.selected_path_indices else 0.0

    def selected_path_idx_p95(self) -> float:
        return float(np.percentile(self.selected_path_indices, 95)) if self.selected_path_indices else 0.0

    def feature_finite_rate(self) -> float:
        if not self.feature_finite_flags:
            return 1.0
        return float(np.mean(self.feature_finite_flags))

    def _count(self, items: List[Any]) -> Dict[str, int]:
        return _count_items(items)

    def _distribution(self, items: List[Any], denominator: int) -> Dict[str, Any]:
        """Return raw counts and normalized distribution.

        Normalized values sum to 1 when denominator > 0.  If denominator is 0
        the distribution is empty and a reason is recorded.
        """
        counts = self._count(items)
        out: Dict[str, Any] = {"raw_counts": counts}
        if denominator > 0:
            out["normalized"] = {k: v / denominator for k, v in counts.items()}
        else:
            out["normalized"] = {}
            out["reason"] = "zero_denominator"
        return out

    def to_summary_dict(self, request_trace_hash: str, config_hash: str,
                        code_hash: str, checkpoint_sha256s: Dict[str, str]) -> Dict[str, Any]:
        total = self.total
        admitted = self.admitted
        r_decisions = self.r_decisions
        r_reached = self.r_reached

        split_counts = self._count(self.split_ids)
        server_counts = self._count(self.server_ids)
        joint_counts: Dict[str, int] = {}
        for pair in self.split_server_pairs:
            joint_counts[str(pair)] = joint_counts.get(str(pair), 0) + 1

        # Admitted-only distributions (legacy lists)
        mod_counts = self._count(self.mod_name_list)
        path_idx_counts = self._count(self.path_idx_list)
        block_start_counts = self._count(self.block_start_list)
        required_fs_counts = self._count(self.required_fs_list)
        hop_count_counts = self._count(self.hop_count_list)
        path_length_counts = self._count([round(pl, 2) for pl in self.path_length_list])

        # Selected-action distributions from selected_actions
        sel_path_idx = []
        sel_mod = []
        sel_block_start = []
        sel_required_fs = []
        sel_hop_count = []
        sel_path_length = []
        sel_split = []
        sel_server = []
        for a in self.selected_actions:
            if not a.get("r_valid"):
                continue
            sel_path_idx.append(a.get("path_idx", -1))
            sel_mod.append(a.get("mod_name", "unknown"))
            sel_block_start.append(a.get("block_start", -1))
            sel_required_fs.append(a.get("required_fs", -1))
            sel_hop_count.append(a.get("hop_count", 0.0))
            sel_path_length.append(round(a.get("path_length_km", 0.0), 2))
            sel_split.append(a.get("split_id", -1))
            sel_server.append(a.get("server_id", -1))

        e_diagnostics = _compute_e_diagnostics(self.selected_actions)

        out: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "topology": self.topology,
            "seed": self.seed,
            "c_mode": self.c_mode,
            "r_mode": self.r_mode,
            "method_name": self.method_name,
            "total": total,
            "admitted": admitted,
            "blocked": self.blocked,
            "r_reached": r_reached,
            "r_decisions": r_decisions,
            "c_no_valid_action": self.c_no_valid_action,
            "r_no_valid_action": self.r_no_valid_action,
            "c_raw_mask_empty": self.c_raw_mask_empty,
            "c_effective_mask_empty": self.c_effective_mask_empty,
            "c_action_in_effective_mask": self.c_action_in_effective_mask,
            "r_mask_empty": self.r_mask_empty,
            "blocking_rate": self.blocking_rate(),
            "server_overload": self.server_overload,
            "overload_rate": self.overload_rate(),
            "no_suitable_block": self.no_suitable_block,
            "nsb_rate": self.nsb_rate(),
            "deadline_failure": self.deadline_failure,
            "deadline_rate": self.deadline_rate(),
            "other_failure": self.other_failure,
            "other_rate": self.other_rate(),
            "failure_reason_distribution": dict(self.failure_reason_counts),
            "avg_delay_ms": self.avg_delay_ms(),
            "delay_p95_ms": self.delay_p95_ms(),
            "avg_fs": self.avg_fs(),
            "avg_decision_ms": self.avg_decision_ms(),
            "decision_p95_ms": self.decision_p95_ms(),
            "avg_c_policy_ms": self.avg_c_policy_ms(),
            "c_policy_p95_ms": self.c_policy_p95_ms(),
            "avg_r_proposer_ms": self.avg_r_proposer_ms(),
            "r_proposer_p95_ms": self.r_proposer_p95_ms(),
            "avg_ranker_or_ksp_ms": self.avg_ranker_or_ksp_ms(),
            "ranker_or_ksp_p95_ms": self.ranker_or_ksp_p95_ms(),
            "avg_total_policy_ms": self.avg_total_policy_ms(),
            "total_policy_p95_ms": self.total_policy_p95_ms(),
            "avg_audit_ms": self.avg_audit_ms(),
            "audit_p95_ms": self.audit_p95_ms(),
            "avg_policy_decision_ms": self.avg_policy_decision_ms(),
            "avg_audit_decision_ms": self.avg_audit_decision_ms(),
            "ppo_agreement_rate": self.ppo_agreement_rate(),
            "ppo_agreement_numerator": self.same_as_ppo_r,
            "ppo_agreement_denominator": r_decisions,
            "mod_distribution": self._distribution(self.mod_name_list, admitted),
            "path_idx_distribution": self._distribution(self.path_idx_list, admitted),
            "block_start_distribution": self._distribution(self.block_start_list, admitted),
            "required_fs_distribution": self._distribution(self.required_fs_list, admitted),
            "hop_count_distribution": self._distribution(self.hop_count_list, admitted),
            "path_length_km_distribution": self._distribution([round(pl, 2) for pl in self.path_length_list], admitted),
            "selected_path_idx_distribution": self._distribution(sel_path_idx, r_decisions),
            "selected_mod_distribution": self._distribution(sel_mod, r_decisions),
            "selected_block_start_distribution": self._distribution(sel_block_start, r_decisions),
            "selected_required_fs_distribution": self._distribution(sel_required_fs, r_decisions),
            "selected_hop_count_distribution": self._distribution(sel_hop_count, r_decisions),
            "selected_path_length_km_distribution": self._distribution(sel_path_length, r_decisions),
            "selected_split_distribution": self._distribution(sel_split, r_decisions),
            "selected_server_distribution": self._distribution(sel_server, r_decisions),
            "split_distribution": {k: v / max(total, 1) for k, v in split_counts.items()},
            "server_distribution": {k: v / max(total, 1) for k, v in server_counts.items()},
            "split_server_joint_distribution": {k: v / max(total, 1) for k, v in joint_counts.items()},
            "e_stratum_diagnostics": e_diagnostics,
            "request_trace_hash": request_trace_hash,
            "selected_action_hash": _hash_text(json.dumps(self.selected_actions, sort_keys=True, default=str)),
            "config_hash": config_hash,
            "code_hash": code_hash,
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
                "avg_selected_path_idx": self.avg_selected_path_idx(),
                "selected_path_idx_p95": self.selected_path_idx_p95(),
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


def _hash_requests(requests: List[Any]) -> str:
    """Stable hash of a request trace."""
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
    """Hash of this source file for reproducibility checks."""
    try:
        return _sha256(str(Path(__file__).resolve()))
    except Exception:
        return "N/A"


def _env_hash(env) -> str:
    """Hash of environment static configuration."""
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


def _count_items(items: List[Any]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for x in items:
        k = str(x)
        c[k] = c.get(k, 0) + 1
    return c


def _e_stratum(split_id: int, num_splits: int) -> str:
    """E=1 is the deepest (last) split; all earlier splits are E=0."""
    if num_splits <= 1:
        return "E=0"
    return "E=1" if split_id == num_splits - 1 else "E=0"


def _source_manifest(root: Path) -> Dict[str, str]:
    """SHA-256 manifest of key source files for reproducibility audits."""
    rel_paths = [
        "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_fair.py",
        "sa_hmarl/sa_hmarl/evaluation/run_strict_v13_multitopology_cside.py",
        "sa_hmarl/sa_hmarl/evaluation/analyze_strict_v13_multitopology_cside.py",
        "sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py",
        "sa_hmarl/sa_hmarl/network/ksp.py",
        "sa_hmarl/sa_hmarl/env/action_mask.py",
        "sa_hmarl/sa_hmarl/env/observation_builder.py",
        "sa_hmarl/sa_hmarl/agents/r_agent.py",
        "sa_hmarl/sa_hmarl/agents/ppo_agents.py",
        "sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py",
        "sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py",
    ]
    manifest: Dict[str, str] = {}
    for rel in rel_paths:
        p = root / rel
        manifest[rel] = _sha256(str(p)) if p.exists() else "N/A"
    return manifest


def _compute_e_diagnostics(selected_actions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-E-stratum blocking and action distributions."""
    strata: Dict[str, Dict[str, Any]] = {}

    def _ensure(key: str) -> None:
        if key not in strata:
            strata[key] = {
                "total": 0,
                "admitted": 0,
                "blocked": 0,
                "blocking_rate": 0.0,
                "ppo_r_agreement_numerator": 0,
                "ppo_r_agreement_denominator": 0,
                "ppo_r_agreement_rate": None,
                "improvement_over_ppo_r_pp": None,
            }

    path_idx_items: Dict[str, List[Any]] = {}
    mod_items: Dict[str, List[Any]] = {}
    block_start_items: Dict[str, List[Any]] = {}
    required_fs_items: Dict[str, List[Any]] = {}
    hop_count_items: Dict[str, List[Any]] = {}
    path_length_items: Dict[str, List[Any]] = {}
    split_items: Dict[str, List[Any]] = {}
    server_items: Dict[str, List[Any]] = {}

    for a in selected_actions:
        e = a.get("e_stratum", "NA")
        _ensure(e)
        strata[e]["total"] += 1
        if a.get("admitted"):
            strata[e]["admitted"] += 1
        else:
            strata[e]["blocked"] += 1

        if a.get("r_valid") and a.get("ppo_r_idx") is not None:
            strata[e]["ppo_r_agreement_denominator"] += 1
            if a.get("r_idx") == a.get("ppo_r_idx"):
                strata[e]["ppo_r_agreement_numerator"] += 1

        if a.get("r_valid"):
            path_idx_items.setdefault(e, []).append(a.get("path_idx", -1))
            mod_items.setdefault(e, []).append(a.get("mod_name", "unknown"))
            block_start_items.setdefault(e, []).append(a.get("block_start", -1))
            required_fs_items.setdefault(e, []).append(a.get("required_fs", -1))
            hop_count_items.setdefault(e, []).append(a.get("hop_count", 0.0))
            path_length_items.setdefault(e, []).append(round(a.get("path_length_km", 0.0), 2))
            split_items.setdefault(e, []).append(a.get("split_id", -1))
            server_items.setdefault(e, []).append(a.get("server_id", -1))

    for e, s in strata.items():
        total = s["total"]
        s["blocking_rate"] = float(s["blocked"] / max(total, 1))
        denom = s["ppo_r_agreement_denominator"]
        if denom > 0:
            s["ppo_r_agreement_rate"] = float(s["ppo_r_agreement_numerator"] / denom)

        # Build per-stratum action distributions (selected actions)
        decisions = len(path_idx_items.get(e, []))
        s["selected_path_idx_distribution"] = {"raw_counts": _count_items(path_idx_items.get(e, [])), "normalized": {}}
        s["selected_mod_distribution"] = {"raw_counts": _count_items(mod_items.get(e, [])), "normalized": {}}
        s["selected_block_start_distribution"] = {"raw_counts": _count_items(block_start_items.get(e, [])), "normalized": {}}
        s["selected_required_fs_distribution"] = {"raw_counts": _count_items(required_fs_items.get(e, [])), "normalized": {}}
        s["selected_hop_count_distribution"] = {"raw_counts": _count_items(hop_count_items.get(e, [])), "normalized": {}}
        s["selected_path_length_km_distribution"] = {"raw_counts": _count_items(path_length_items.get(e, [])), "normalized": {}}
        s["selected_split_distribution"] = {"raw_counts": _count_items(split_items.get(e, [])), "normalized": {}}
        s["selected_server_distribution"] = {"raw_counts": _count_items(server_items.get(e, [])), "normalized": {}}
        if decisions > 0:
            for key, items in [
                ("selected_path_idx_distribution", path_idx_items.get(e, [])),
                ("selected_mod_distribution", mod_items.get(e, [])),
                ("selected_block_start_distribution", block_start_items.get(e, [])),
                ("selected_required_fs_distribution", required_fs_items.get(e, [])),
                ("selected_hop_count_distribution", hop_count_items.get(e, [])),
                ("selected_path_length_km_distribution", path_length_items.get(e, [])),
                ("selected_split_distribution", split_items.get(e, [])),
                ("selected_server_distribution", server_items.get(e, [])),
            ]:
                counts = _count_items(items)
                s[key]["normalized"] = {k: v / decisions for k, v in counts.items()}
        else:
            for key in [
                "selected_path_idx_distribution", "selected_mod_distribution",
                "selected_block_start_distribution", "selected_required_fs_distribution",
                "selected_hop_count_distribution", "selected_path_length_km_distribution",
                "selected_split_distribution", "selected_server_distribution",
            ]:
                s[key]["reason"] = "zero_denominator"

    return strata


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
    """Return (flat_c_idx, split_id, server_id, c_valid, c_info).

    c_info contains:
      - raw_mask_empty (bool)
      - effective_mask_empty (bool)
      - action_in_effective_mask (bool)

    If no legal C action exists, returns (None, None, None, False, c_info).
    """
    c_info: Dict[str, Any] = {
        "raw_mask_empty": False,
        "effective_mask_empty": False,
        "action_in_effective_mask": False,
    }
    baseline_map = {"df_c": "df", "rf_c": "rf", "wo_c": "wo", "greedy_c": "greedy", "iwd_c": "iwd"}
    baseline_name = baseline_map.get(c_mode, c_mode)
    if baseline_name in ("df", "rf", "wo", "greedy", "iwd"):
        mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        c_info["raw_mask_empty"] = not mask.any()
        c_info["effective_mask_empty"] = c_info["raw_mask_empty"]
        if not mask.any():
            return None, None, None, False, c_info
        action_idx = select_offloading_action(baseline_name, env, req, obs_c, mask)
        if action_idx is None:
            return None, None, None, False, c_info
        c_info["action_in_effective_mask"] = True
    else:
        action_idx, raw_mask, risk_mask = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        raw = np.asarray(raw_mask, dtype=bool)
        effective = np.asarray(risk_mask, dtype=bool)
        c_info["raw_mask_empty"] = not raw.any()
        c_info["effective_mask_empty"] = not effective.any()
        if not effective.any():
            return None, None, None, False, c_info
        if not (0 <= int(action_idx) < len(effective) and effective[int(action_idx)]):
            return None, None, None, False, c_info
        c_info["action_in_effective_mask"] = True
    split_id, server_id = decode_agent_c_action(action_idx, num_servers)
    return int(action_idx), int(split_id), int(server_id), True, c_info


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
) -> Tuple[int, bool, Optional[Dict[str, Any]], Dict[str, float]]:
    """Return (r_idx, r_valid, ranker_info, timing_ms).

    The caller must already have verified ``agent_r_mask`` is non-empty; this
    function still defensively returns invalid if the mask is empty.

    timing_ms keys: proposer_ms, ranker_or_ksp_ms.
    """
    timing: Dict[str, float] = {"proposer_ms": 0.0, "ranker_or_ksp_ms": 0.0}

    if r_mode == "ppo_r_top1":
        return int(ppo_idx), True, None, timing

    if obs_r is None:
        return 0, False, None, timing

    mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    if not mask.any():
        return 0, False, None, timing

    if r_mode == "ksp_ff_highest":
        t0 = time.perf_counter()
        a = ksp_ff_highest_mod_action(obs_r)
        timing["ranker_or_ksp_ms"] = (time.perf_counter() - t0) * 1000.0
        if a is None:
            return 0, False, None, timing
        return int(a), True, None, timing

    # strict_v13 or old_v13
    t0 = time.perf_counter()
    candidates = _ppo_r_topk_actions(agent_r, obs_r, 30)
    timing["proposer_ms"] = (time.perf_counter() - t0) * 1000.0
    candidates = np.asarray(candidates, dtype=np.int64)
    has_candidates = candidates.size > 0
    info: Dict[str, Any] = {
        "candidate_count": int(candidates.size),
        "in_candidates": False,
        "fallback": not has_candidates,
        "score_margin": 0.0,
        "feature_finite": True,
        "nan_inf_feature_count": 0,
        "nan_inf_score_count": 0,
        "selected_path_idx": 0,
    }
    if not has_candidates:
        # Fallback to PPO-R top-1: still a legal R action decision.
        return int(ppo_idx), True, info, timing

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
    if not finite_all:
        info["nan_inf_feature_count"] = int(np.sum(~np.isfinite(x)))
    x = (x - ranker["feature_mean"]) / ranker["feature_std"]
    x[~np.isfinite(x)] = 0.0

    t0 = time.perf_counter()
    with torch.no_grad():
        scores = (
            ranker["model"](torch.as_tensor(x, dtype=torch.float32, device=ranker["device"]))
            .cpu()
            .numpy()
        )
    timing["ranker_or_ksp_ms"] = (time.perf_counter() - t0) * 1000.0
    if not np.all(np.isfinite(scores)):
        info["nan_inf_score_count"] = int(np.sum(~np.isfinite(scores)))
    best_local = int(np.argmax(scores))
    r_idx = int(candidates[best_local])
    info["in_candidates"] = True
    sorted_scores = np.sort(scores)[::-1]
    info["score_margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
    path_idx, _, _ = decode_agent_r_action(r_idx, len(obs_r.get("mod_names", [])), env.max_blocks)
    info["selected_path_idx"] = int(path_idx)
    return r_idx, True, info, timing


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _record_failure(res: MethodResult, reason: str) -> None:
    """Increment failure counters.  c_no_valid_action/r_no_valid_action are
    tracked separately and do not contribute to other_failure."""
    res.failure_reason_counts[reason] = res.failure_reason_counts.get(reason, 0) + 1
    if reason in ("server_saturated", "server_overload"):
        res.server_overload += 1
    elif reason == "no_suitable_block":
        res.no_suitable_block += 1
    elif reason == "deadline_infeasible":
        res.deadline_failure += 1
    elif reason in ("c_no_valid_action", "r_no_valid_action"):
        # These are accounted for in their own dedicated counters.
        pass
    else:
        res.other_failure += 1


def _append_latency(
    res: MethodResult,
    c_policy_ms: float,
    r_proposer_ms: float,
    ranker_or_ksp_ms: float,
    audit_ms: float,
) -> None:
    """Update all latency sums and lists."""
    total_policy_ms = c_policy_ms + r_proposer_ms + ranker_or_ksp_ms
    res.c_policy_ms_sum += c_policy_ms
    res.r_proposer_ms_sum += r_proposer_ms
    res.ranker_or_ksp_ms_sum += ranker_or_ksp_ms
    res.total_policy_ms_sum += total_policy_ms
    res.audit_ms_sum += audit_ms
    res.c_policy_ms_list.append(c_policy_ms)
    res.r_proposer_ms_list.append(r_proposer_ms)
    res.ranker_or_ksp_ms_list.append(ranker_or_ksp_ms)
    res.total_policy_ms_list.append(total_policy_ms)
    res.audit_ms_list.append(audit_ms)
    res.policy_decision_ms_sum += total_policy_ms
    res.audit_decision_ms_sum += audit_ms


def _record_action_common(
    res: MethodResult,
    c_idx: Optional[int],
    split_id: int,
    server_id: int,
    r_idx: Optional[int],
    ppo_r_idx: Optional[int],
    c_valid: bool,
    r_valid: bool,
    r_mask_empty: bool,
    admitted: bool,
    reason: str,
    path_idx: int,
    mod_idx: int,
    block_idx: int,
    path_len: float,
    hop_cnt: float,
    mod_name: str,
    block_start: int,
    req_fs: Optional[int],
    num_splits: int,
) -> None:
    """Append a traceable action record."""
    e_stratum = _e_stratum(split_id, num_splits) if split_id >= 0 else "NA"
    res.selected_actions.append({
        "c_idx": None if c_idx is None else int(c_idx),
        "split_id": int(split_id),
        "server_id": int(server_id),
        "r_idx": None if r_idx is None else int(r_idx),
        "ppo_r_idx": None if ppo_r_idx is None else int(ppo_r_idx),
        "c_valid": c_valid,
        "r_valid": r_valid,
        "r_mask_empty": r_mask_empty,
        "admitted": admitted,
        "reason": reason,
        "path_idx": int(path_idx),
        "mod_idx": int(mod_idx),
        "block_idx": int(block_idx),
        "path_length_km": float(path_len),
        "hop_count": float(hop_cnt),
        "mod_name": str(mod_name),
        "block_start": int(block_start),
        "required_fs": None if req_fs is None else int(req_fs),
        "e_stratum": e_stratum,
    })


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

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < args.warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        # C-side under K_C
        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        c_t0 = time.perf_counter()
        c_idx, split_id, server_id, c_valid, c_info = _select_c_action(
            c_mode, agent_c, env, req, obs_c, num_servers
        )
        c_elapsed = time.perf_counter() - c_t0
        c_policy_ms = c_elapsed * 1000.0

        if not is_warmup:
            if c_info["raw_mask_empty"]:
                res.c_raw_mask_empty += 1
            if c_info["effective_mask_empty"]:
                res.c_effective_mask_empty += 1
            if c_info["action_in_effective_mask"]:
                res.c_action_in_effective_mask += 1

        if not c_valid:
            if not is_warmup:
                res.total += 1
                res.blocked += 1
                res.c_no_valid_action += 1
                _record_failure(res, "c_no_valid_action")
                res.split_ids.append(-1)
                res.server_ids.append(-1)
                res.split_server_pairs.append((-1, -1))
                _record_action_common(
                    res, None, -1, -1, None, None, False, False, False, False,
                    "c_no_valid_action", -1, -1, -1, 0.0, 0.0, "", -1, None, len(req.splits),
                )
                decision_ms = (time.perf_counter() - t0) * 1000.0
                res.decision_ms_sum += decision_ms
                res.decision_ms_list.append(decision_ms)
                _append_latency(res, c_policy_ms, 0.0, 0.0, 0.0)
            continue

        # R-side under K_R
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        if not r_mask.any():
            if not is_warmup:
                res.total += 1
                res.blocked += 1
                res.r_reached += 1
                res.r_no_valid_action += 1
                res.r_mask_empty += 1
                _record_failure(res, "r_no_valid_action")
                res.split_ids.append(split_id)
                res.server_ids.append(server_id)
                res.split_server_pairs.append((split_id, server_id))
                _record_action_common(
                    res, c_idx, split_id, server_id, None, None, True, False, True, False,
                    "r_no_valid_action", -1, -1, -1, 0.0, 0.0, "", -1, None, len(req.splits),
                )
                decision_ms = (time.perf_counter() - t0) * 1000.0
                res.decision_ms_sum += decision_ms
                res.decision_ms_list.append(decision_ms)
                _append_latency(res, c_policy_ms, 0.0, 0.0, 0.0)
            continue

        # Audit / policy timing depends on R mode.
        audit_ms = 0.0
        r_proposer_ms = 0.0
        ranker_or_ksp_ms = 0.0
        ppo_idx: Optional[int] = None
        r_idx: int = 0
        r_valid: bool = False
        ranker_info: Optional[Dict[str, Any]] = None

        if r_mode == "ppo_r_top1":
            t_r = time.perf_counter()
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, max_blocks)
            r_proposer_ms = (time.perf_counter() - t_r) * 1000.0
            r_idx = int(ppo_idx)
            r_valid = True
        else:
            t_audit = time.perf_counter()
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, max_blocks)
            audit_ms = (time.perf_counter() - t_audit) * 1000.0

            r_idx, r_valid, ranker_info, timing = _select_r_action(
                r_mode, agent_r, ranker, env, req, obs_c, obs_r, ppo_idx, split_id, server_id
            )
            r_proposer_ms = timing.get("proposer_ms", 0.0)
            ranker_or_ksp_ms = timing.get("ranker_or_ksp_ms", 0.0)

        if not r_valid:
            if not is_warmup:
                res.total += 1
                res.blocked += 1
                res.r_reached += 1
                res.r_no_valid_action += 1
                _record_failure(res, "r_no_valid_action")
                res.split_ids.append(split_id)
                res.server_ids.append(server_id)
                res.split_server_pairs.append((split_id, server_id))
                _record_action_common(
                    res, c_idx, split_id, server_id, r_idx, ppo_idx, True, False, False, False,
                    "r_no_valid_action", -1, -1, -1, 0.0, 0.0, "", -1, None, len(req.splits),
                )
                if r_mode in ("strict_v13", "old_v13") and ranker_info is not None:
                    res.candidate_counts.append(ranker_info["candidate_count"])
                    if ranker_info["fallback"]:
                        res.fallback_count += 1
                    else:
                        if ranker_info["in_candidates"]:
                            res.ranker_top1_in_candidates_count += 1
                        res.score_margins.append(ranker_info["score_margin"])
                        res.feature_finite_flags.append(ranker_info["feature_finite"])
                        res.nan_inf_feature_count += ranker_info["nan_inf_feature_count"]
                        res.nan_inf_score_count += ranker_info["nan_inf_score_count"]
                    res.selected_path_indices.append(ranker_info.get("selected_path_idx", 0))
                decision_ms = (time.perf_counter() - t0) * 1000.0
                res.decision_ms_sum += decision_ms
                res.decision_ms_list.append(decision_ms)
                _append_latency(res, c_policy_ms, r_proposer_ms, ranker_or_ksp_ms, audit_ms)
            continue

        same_as_ppo = False
        if ppo_idx is not None:
            same_as_ppo = int(r_idx) == int(ppo_idx)

        r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)

        path_idx, mod_idx, block_idx = r_action
        path_feats = obs_r.get("path_features", [])
        path_len = 0.0
        hop_cnt = 0.0
        mod_name = ""
        if path_feats and 0 <= path_idx < len(path_feats):
            path_len = float(path_feats[path_idx].get("path_length_km", 0.0))
            hop_cnt = float(path_feats[path_idx].get("hop_count", 0.0))
        mod_names = obs_r.get("mod_names", [])
        if mod_names and 0 <= mod_idx < len(mod_names):
            mod_name = str(mod_names[mod_idx])
        blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        block_start = int(blocks[block_idx][0]) if block_idx < len(blocks) else -1
        req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]

        if not is_warmup:
            res.total += 1
            res.r_reached += 1
            res.r_decisions += 1
            res.split_ids.append(split_id)
            res.server_ids.append(server_id)
            res.split_server_pairs.append((split_id, server_id))

            success = bool(info.get("success", False))
            reason = info.get("reason", "unknown") if not success else ""

            _record_action_common(
                res, c_idx, split_id, server_id, r_idx, ppo_idx, True, True, False,
                success, reason, path_idx, mod_idx, block_idx, path_len, hop_cnt,
                mod_name, block_start, req_fs, len(req.splits),
            )

            decision_ms = (time.perf_counter() - t0) * 1000.0
            res.decision_ms_sum += decision_ms
            res.decision_ms_list.append(decision_ms)
            _append_latency(res, c_policy_ms, r_proposer_ms, ranker_or_ksp_ms, audit_ms)

            if same_as_ppo:
                res.same_as_ppo_r += 1

            if success:
                res.admitted += 1
                res.delay_sum += float(info.get("delay_ms", 0.0))
                res.fs_sum += float(info.get("num_slots", 0.0))
                res.delay_ms_list.append(float(info.get("delay_ms", 0.0)))
                res.path_length_list.append(path_len)
                res.hop_count_list.append(hop_cnt)
                res.mod_name_list.append(mod_name)
                res.path_idx_list.append(path_idx)
                res.block_start_list.append(block_start)
                res.required_fs_list.append(int(req_fs) if req_fs is not None else 0)
            else:
                res.blocked += 1
                _record_failure(res, reason)

            if r_mode in ("strict_v13", "old_v13") and ranker_info is not None:
                res.candidate_counts.append(ranker_info["candidate_count"])
                if ranker_info["fallback"]:
                    res.fallback_count += 1
                else:
                    if ranker_info["in_candidates"]:
                        res.ranker_top1_in_candidates_count += 1
                    res.score_margins.append(ranker_info["score_margin"])
                    res.feature_finite_flags.append(ranker_info["feature_finite"])
                    res.nan_inf_feature_count += ranker_info["nan_inf_feature_count"]
                    res.nan_inf_score_count += ranker_info["nan_inf_score_count"]
                res.selected_path_indices.append(ranker_info.get("selected_path_idx", path_idx))
    return res


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _validate_result_row(row: Dict[str, Any], strict: bool = True) -> None:
    """Validate a per-method summary row. Raises AssertionError on violation."""
    required = [
        "schema_version", "topology", "seed", "c_mode", "r_mode", "method_name",
        "total", "admitted", "blocked", "r_reached", "r_decisions",
        "c_no_valid_action", "r_no_valid_action",
        "c_raw_mask_empty", "c_effective_mask_empty", "c_action_in_effective_mask",
        "r_mask_empty", "deadline_failure",
        "failure_reason_distribution", "ppo_agreement_rate",
        "ppo_agreement_numerator", "ppo_agreement_denominator",
    ]
    for key in required:
        if key not in row:
            raise AssertionError(f"Missing required field: {key}")

    total = int(row["total"])
    admitted = int(row["admitted"])
    blocked = int(row["blocked"])
    r_reached = int(row["r_reached"])
    r_decisions = int(row["r_decisions"])
    c_no = int(row["c_no_valid_action"])
    r_no = int(row["r_no_valid_action"])

    # Conservation: evaluated requests = admitted + blocked
    assert total == admitted + blocked, (
        f"total ({total}) != admitted ({admitted}) + blocked ({blocked})"
    )

    # R-side conservation: C-valid requests = no-valid-R + legal R decisions
    assert r_reached == r_no + r_decisions, (
        f"r_reached ({r_reached}) != r_no_valid_action ({r_no}) + r_decisions ({r_decisions})"
    )
    assert r_reached == total - c_no, (
        f"r_reached ({r_reached}) != total ({total}) - c_no_valid_action ({c_no})"
    )

    # Failure decomposition conservation
    expected_blocked = (
        c_no + r_no + int(row["server_overload"])
        + int(row["no_suitable_block"]) + int(row["deadline_failure"])
        + int(row["other_failure"])
    )
    assert blocked == expected_blocked, (
        f"blocked ({blocked}) != c_no ({c_no}) + r_no ({r_no}) + "
        f"server_overload ({row['server_overload']}) + "
        f"no_suitable_block ({row['no_suitable_block']}) + "
        f"deadline_failure ({row['deadline_failure']}) + "
        f"other_failure ({row['other_failure']})"
    )

    # PPO agreement consistency
    agree = row["ppo_agreement_rate"]
    if agree is not None:
        assert isinstance(agree, (int, float)), "ppo_agreement_rate must be numeric or null"
        expected_agree = row["ppo_agreement_numerator"] / max(row["ppo_agreement_denominator"], 1)
        assert abs(float(agree) - expected_agree) < 1e-9, (
            "ppo_agreement_rate does not match numerator/denominator"
        )

    if row["r_mode"] == "ppo_r_top1":
        if r_decisions > 0:
            assert agree == 1.0, (
                f"ppo_r_top1 must have agreement 1.0, got {agree}"
            )

    # Distribution conservation (admitted-only distributions sum to 1 over admitted)
    admitted_only_dists = [
        "path_idx_distribution", "block_start_distribution",
        "required_fs_distribution", "hop_count_distribution", "path_length_km_distribution",
    ]
    for dist_key in admitted_only_dists:
        dist = row.get(dist_key, {})
        norm = dist.get("normalized", {})
        if norm:
            s = sum(norm.values())
            assert abs(s - 1.0) < 1e-6 or admitted == 0, (
                f"{dist_key} normalized sums to {s}, expected 1.0 (admitted={admitted})"
            )
        raw = dist.get("raw_counts", {})
        if raw:
            assert sum(raw.values()) == admitted, (
                f"{dist_key} raw count sum != admitted ({admitted})"
            )

    # Selected distributions sum to 1 over r_decisions
    selected_dists = [
        "selected_path_idx_distribution", "selected_mod_distribution",
        "selected_block_start_distribution", "selected_required_fs_distribution",
        "selected_hop_count_distribution", "selected_path_length_km_distribution",
        "selected_split_distribution", "selected_server_distribution",
    ]
    for dist_key in selected_dists:
        dist = row.get(dist_key, {})
        norm = dist.get("normalized", {})
        if norm:
            s = sum(norm.values())
            assert abs(s - 1.0) < 1e-6 or r_decisions == 0, (
                f"{dist_key} normalized sums to {s}, expected 1.0 (r_decisions={r_decisions})"
            )
        raw = dist.get("raw_counts", {})
        if raw:
            assert sum(raw.values()) == r_decisions, (
                f"{dist_key} raw count sum != r_decisions ({r_decisions})"
            )

    # Latency non-negativity
    for key in [
        "avg_c_policy_ms", "avg_r_proposer_ms", "avg_ranker_or_ksp_ms",
        "avg_total_policy_ms", "avg_audit_ms",
    ]:
        assert float(row.get(key, 0.0)) >= 0.0, f"{key} must be non-negative"

    if strict and row["r_mode"] in ("strict_v13", "old_v13"):
        cc = row.get("avg_candidate_count", 0.0)
        if r_decisions > 0:
            assert 1.0 <= cc <= 30.0, f"candidate_count {cc} out of [1,30]"
        assert row.get("feature_finite_rate", 1.0) == 1.0, "non-finite ranker features/scores detected"


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

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

    checkpoint_sha256s = {
        "agent_r": agent_r_sha256,
        **c_checkpoint_sha256s,
        **ranker_sha256s,
    }

    root = Path(__file__).resolve().parents[3]
    source_manifest = _source_manifest(root)

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
                row = res.to_summary_dict(request_trace_hash, config_hash, code_hash, checkpoint_sha256s)
                results.append(row)
                logs.append(
                    f"[{args.topology}][seed={args.seed}][{res.method_name}] "
                    f"blocking={res.blocking_rate():.4%} "
                    f"overload={res.overload_rate():.4%} "
                    f"nsb={res.nsb_rate():.4%} "
                    f"c_no={res.c_no_valid_action} r_no={res.r_no_valid_action} "
                    f"r_decisions={res.r_decisions} "
                    f"policy_ms={res.avg_total_policy_ms():.3f} "
                    f"audit_ms={res.avg_audit_ms():.3f} "
                    f"ppo_agree={res.ppo_agreement_rate()}"
                )
            except Exception as exc:
                logs.append(
                    f"[{args.topology}][seed={args.seed}][{c_mode}+{r_mode}] FAILED: {exc}\n{traceback.format_exc()}"
                )
                raise

    # Cross-method E-diagnostics: improvement over ppo_r_top1 within each stratum.
    ppo_rows = {
        (row["topology"], row["seed"], row["c_mode"]): row
        for row in results
        if row["r_mode"] == "ppo_r_top1"
    }
    for row in results:
        if row["r_mode"] == "ppo_r_top1":
            continue
        ppo_row = ppo_rows.get((row["topology"], row["seed"], row["c_mode"]))
        if ppo_row is None:
            continue
        for e, diag in row.get("e_stratum_diagnostics", {}).items():
            ppo_diag = ppo_row.get("e_stratum_diagnostics", {}).get(e, {})
            ppo_block = ppo_diag.get("blocking_rate", 0.0)
            this_block = diag.get("blocking_rate", 0.0)
            diag["improvement_over_ppo_r_pp"] = float((ppo_block - this_block) * 100.0)

    # Validate after cross-method enrichment.
    for row in results:
        _validate_result_row(row)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "request_trace_hash": request_trace_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "evaluator_code_hash": code_hash,
        "runner_code_hash": None,
        "analyzer_code_hash": None,
        "checkpoint_sha256s": checkpoint_sha256s,
        "source_manifest": source_manifest,
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
    parser.add_argument("--r_modes", default="ppo_r_top1,ksp_ff_highest,strict_v13,old_v13")
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
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v13_strict_multitopology_cside_corrected/per_task")
    parser.add_argument("--output_json", default="results.json")
    parser.add_argument("--output_log", default="eval.log")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
