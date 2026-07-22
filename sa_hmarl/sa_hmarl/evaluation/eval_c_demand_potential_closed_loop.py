"""Closed-loop evaluation of demand-aware potential for Agent-C decision reranking.

Compares three methods on identical request traces:
  1. ppo_c — baseline r_feasibility PPO-C (no reranking)
  2. potential_only — argmax Phi_after among successful raw-mask-legal candidates
  3. ppo_potential_rerank — z-score blend of PPO policy logits + Phi_after

Each method runs its own independent closed-loop environment to ensure fair comparison.

Usage:
    # Smoke test
    PYTHONPATH=sa_hmarl python sa_hmarl/sa_hmarl/evaluation/eval_c_demand_potential_closed_loop.py \
        --seeds 42 --episodes 2 --requests_per_episode 6

    # Full evaluation
    PYTHONPATH=sa_hmarl python sa_hmarl/sa_hmarl/evaluation/eval_c_demand_potential_closed_loop.py \
        --seeds 42,123,456 --episodes 20 --requests_per_episode 80 \
        --arrival_interval 0.15
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_c_action_horizon_oracle import (
    _demand_aware_potential,
    _execute_c_with_frozen_r,
    _snapshot_before_c_decision,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


# ---------------------------------------------------------------------------
# Safe z-score
# ---------------------------------------------------------------------------
def _zscore(values: np.ndarray) -> np.ndarray:
    """Safe z-score: returns all zeros when std == 0 or only one value."""
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return np.zeros_like(arr)
    std = float(np.std(arr))
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


# ---------------------------------------------------------------------------
# Spectrum/compute-only field computations
# ---------------------------------------------------------------------------
def _compute_spectrum_field_spectrum_only(
    obs_c: Dict[str, Any],
) -> Tuple[int, int]:
    """Compute K_C and K_R using ONLY spectrum feasibility (n_r > 0).

    Does NOT use server compute/util checks.  The ``n_r > 0`` condition
    captures path/mod/block availability, which is purely spectrum-continuity.
    """
    num_servers = len(obs_c["server_utilizations"])
    feasible_counts = obs_c["feasible_counts"]

    k_c_spec = 0
    k_r_total = 0

    for split_id, server_list in enumerate(feasible_counts):
        for server_id, count in enumerate(server_list):
            if count > 0:
                k_c_spec += 1
            k_r_total += count

    return k_c_spec, k_r_total


def _compute_spectrum_field_compute_only(
    obs_c: Dict[str, Any],
    util_threshold: float = 0.95,
) -> int:
    """Compute K_C using ONLY compute feasibility (edge_cost + utilization).

    Ignores spectrum feasibility (n_r).  Counts a (split, server) candidate
    as valid when server_available_compute >= edge_cost AND util <= threshold.
    """
    num_servers = len(obs_c["server_utilizations"])
    candidate_features = obs_c["candidate_features"]

    k_c_comp = 0

    for split_id in range(len(obs_c.get("feasible_counts", []))):
        for server_id in range(num_servers):
            cand_idx = split_id * num_servers + server_id
            if cand_idx >= len(candidate_features):
                continue
            feat_dict = candidate_features[cand_idx]

            available = float(feat_dict.get("server_available_compute", 0.0))
            edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
            edge_ms = feat_dict.get("edge_compute_ms", 0.0)
            compute_ok = (
                (edge_cost <= available)
                and np.isfinite(edge_ms)
                and edge_ms != float('inf')
            )

            util = float(obs_c["server_utilizations"][server_id])
            util_ok = util <= util_threshold

            if compute_ok and util_ok:
                k_c_comp += 1

    return k_c_comp


def _demand_aware_potential_spectrum_only(
    env,
    request_history: List[Any],
    window: int,
    probe_limit: int,
    alpha: float,
) -> Optional[float]:
    """Estimate Phi_spectrum(X, mu_t) using only spectrum-feasibility counts."""
    if probe_limit <= 0 or not request_history:
        return None
    recent = request_history[-max(window, 1):]
    if len(recent) > probe_limit:
        indices = np.linspace(0, len(recent) - 1, probe_limit, dtype=int)
        probes = [recent[int(i)] for i in indices]
    else:
        probes = recent

    values = []
    for historical_req in probes:
        probe = copy.deepcopy(historical_req)
        probe.arrival_time = env.time
        obs_probe = build_agent_c_observation(env, probe)
        k_q, n_q = _compute_spectrum_field_spectrum_only(obs_probe)
        values.append(_phi_spec(k_q, n_q, alpha))
    return float(np.mean(values)) if values else None


def _demand_aware_potential_compute_only(
    env,
    request_history: List[Any],
    window: int,
    probe_limit: int,
    util_threshold: float,
) -> Optional[float]:
    """Estimate Phi_compute(X, mu_t) using only compute-feasibility counts."""
    if probe_limit <= 0 or not request_history:
        return None
    recent = request_history[-max(window, 1):]
    if len(recent) > probe_limit:
        indices = np.linspace(0, len(recent) - 1, probe_limit, dtype=int)
        probes = [recent[int(i)] for i in indices]
    else:
        probes = recent

    values = []
    for historical_req in probes:
        probe = copy.deepcopy(historical_req)
        probe.arrival_time = env.time
        obs_probe = build_agent_c_observation(env, probe)
        k_c = _compute_spectrum_field_compute_only(obs_probe, util_threshold)
        values.append(float(np.log1p(k_c)))
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    """Result of evaluating one C candidate from a snapshot."""
    c_action_idx: int
    split_id: int
    server_id: int
    success: bool
    reason: str
    demand_potential: Optional[float]  # None if failed or potential disabled
    demand_potential_spec: Optional[float] = None  # spectrum-only Phi
    demand_potential_comp: Optional[float] = None  # compute-only Phi
    block_waste: Optional[float] = None
    r_action_idx: int = 0


@dataclass
class PerMethodMetrics:
    """Accumulated metrics for one evaluation method."""
    total: int = 0
    blocked: int = 0
    raw_empty: int = 0
    no_suitable_block: int = 0
    server_overload: int = 0
    deadline_failure: int = 0
    reason_counts: Dict[str, int] = field(default_factory=dict)
    delays: List[float] = field(default_factory=list)
    fses: List[float] = field(default_factory=list)
    wastes: List[float] = field(default_factory=list)
    path_kms: List[float] = field(default_factory=list)
    hop_counts: List[float] = field(default_factory=list)
    block_starts: List[float] = field(default_factory=list)
    block_sizes: List[float] = field(default_factory=list)
    mod_counts: Dict[str, int] = field(default_factory=dict)
    server_counts: Dict[int, int] = field(default_factory=dict)
    active_connections: List[int] = field(default_factory=list)
    valid_c_actions: List[int] = field(default_factory=list)
    total_valid_r_actions: List[int] = field(default_factory=list)
    selected_valid_r_actions: List[int] = field(default_factory=list)
    # Agreement / change tracking (relative to ppo_c)
    actions_same: List[bool] = field(default_factory=list)
    changed_blocked_count: int = 0  # blocked requests among those with changed action
    decision_times_ms: List[float] = field(default_factory=list)
    # Lyapunov rerank diagnostics (only populated for lyapunov_rank_only)
    lyap_H_spec: List[float] = field(default_factory=list)
    lyap_adjustment_abs: List[float] = field(default_factory=list)
    lyap_changed_vs_v12: List[bool] = field(default_factory=list)
    # Split choice distribution (for fixed-split / no-partition-style ablations)
    split_counts: Dict[int, int] = field(default_factory=dict)
    # Optional ranker profiling breakdown (only populated when enable_profile=True)
    profile_r_feature_build_ms: List[float] = field(default_factory=list)
    profile_legal_extract_ms: List[float] = field(default_factory=list)
    profile_candidate_select_ms: List[float] = field(default_factory=list)
    profile_feature_batch_ms: List[float] = field(default_factory=list)
    profile_normalize_ms: List[float] = field(default_factory=list)
    profile_ranker_forward_ms: List[float] = field(default_factory=list)
    profile_total_ranker_policy_ms: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PPO-C logit extraction
# ---------------------------------------------------------------------------
def _get_ppo_c_logits(
    agent_c, obs_c: Dict[str, Any], raw_mask: np.ndarray,
) -> np.ndarray:
    """Extract raw policy logits for raw-mask-legal C candidates only.

    Returns logits of shape (num_valid,).
    """
    features, _ = agent_c.build_action_features(obs_c)
    if features.size == 0:
        return np.array([], dtype=float)
    x = torch.tensor(features, dtype=torch.float32, device=agent_c.device).unsqueeze(0)
    with torch.no_grad():
        raw_logits = agent_c.policy_net(x).squeeze(0).cpu().numpy()
    return raw_logits[raw_mask]


# ---------------------------------------------------------------------------
# Standard PPO-C action selection (replicates eval_c_closed_loop logic)
# ---------------------------------------------------------------------------
def _select_ppo_c_action(
    agent_c, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[int, np.ndarray]:
    """Select PPO-C action with risk mask, return (action_idx, raw_mask)."""
    features, mask = agent_c.build_action_features(obs_c)
    raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    if action is None:
        action = 0
    return int(action), raw_mask


# ---------------------------------------------------------------------------
# Standard PPO-R action selection
# ---------------------------------------------------------------------------
def _select_r_action(agent_r, obs_r: Dict[str, Any], max_blocks: int) -> Tuple[int, int, int]:
    """Select PPO-R action deterministically, return (path, mod, block)."""
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return 0, 0, 0
    return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), max_blocks)


# ---------------------------------------------------------------------------
# Candidate evaluation from snapshot
# ---------------------------------------------------------------------------
def _evaluate_candidates(
    snapshot_env,
    req,
    legal_c_indices: List[int],
    agent_r,
    num_servers: int,
    request_history: List,
    pot_window: int,
    pot_probe_limit: int,
    util_threshold: float,
    alpha: float,
    potential_components: Optional[set[str]] = None,
) -> List[CandidateRecord]:
    """Evaluate all raw-mask-legal C candidates from the same snapshot.

    For each candidate:
      1. copy.deepcopy(snapshot)
      2. Execute C+R on the copy with frozen PPO-R
      3. If successful, compute Phi_after via _demand_aware_potential
      4. Return CandidateRecord

    IMPORTANT: Does NOT mutate the snapshot.
    """
    candidates = []
    components = (
        {"full", "spectrum", "compute"}
        if potential_components is None else potential_components
    )
    for c_idx in legal_c_indices:
        env_c = copy.deepcopy(snapshot_env)
        r_idx, info = _execute_c_with_frozen_r(
            env_c, req, int(c_idx), agent_r, num_servers
        )
        success = bool(info.get("success", False))
        reason = info.get("reason", "unknown")

        phi = None
        phi_spec = None
        phi_comp = None
        if success and pot_probe_limit > 0:
            if "full" in components:
                phi = _demand_aware_potential(
                    env_c, request_history,
                    pot_window, pot_probe_limit, util_threshold, alpha,
                )
            if "spectrum" in components:
                phi_spec = _demand_aware_potential_spectrum_only(
                    env_c, request_history,
                    pot_window, pot_probe_limit, alpha,
                )
            if "compute" in components:
                phi_comp = _demand_aware_potential_compute_only(
                    env_c, request_history,
                    pot_window, pot_probe_limit, util_threshold,
                )

        split_id, server_id = decode_agent_c_action(int(c_idx), num_servers)

        candidates.append(CandidateRecord(
            c_action_idx=int(c_idx),
            split_id=split_id,
            server_id=server_id,
            success=success,
            reason=reason,
            demand_potential=phi,
            demand_potential_spec=phi_spec,
            demand_potential_comp=phi_comp,
            block_waste=info.get("block_waste"),
            r_action_idx=r_idx,
        ))
    return candidates


# ---------------------------------------------------------------------------
# Selection functions
# ---------------------------------------------------------------------------
def _select_potential_only(
    candidates: List[CandidateRecord],
    ppo_c_valid_logits: np.ndarray,
    ppo_c_default_action: int,
    legal_c_indices: List[int],
) -> int:
    """Select successful candidate with highest Phi_after.

    Tiebreak: higher policy logit → smaller block_waste → smaller action index.
    Falls back to ppo_c_default_action if no successful candidates.
    """
    successful = [
        c for c in candidates
        if c.success and c.demand_potential is not None
    ]
    if not successful:
        return ppo_c_default_action

    # Build logit lookup
    logit_map = {idx: float(logit) for idx, logit in zip(legal_c_indices, ppo_c_valid_logits)}

    def _sort_key(c: CandidateRecord):
        logit = logit_map.get(c.c_action_idx, -1e9)
        waste = c.block_waste if c.block_waste is not None else float('inf')
        # Higher potential better, higher logit better, smaller waste better, smaller idx better
        return (-c.demand_potential, -logit, waste, c.c_action_idx)

    return min(successful, key=_sort_key).c_action_idx


def _select_ppo_potential_rerank(
    candidates: List[CandidateRecord],
    ppo_c_valid_logits: np.ndarray,
    ppo_c_default_action: int,
    legal_c_indices: List[int],
    lambda_phi: float,
) -> int:
    """Select using z-score blend: score = z_policy + lambda * z_potential.

    If all potentials equal → degenerate to PPO-C argmax logit.
    Falls back to ppo_c_default_action if no successful candidates.
    """
    successful = [
        c for c in candidates
        if c.success and c.demand_potential is not None
    ]
    if not successful:
        return ppo_c_default_action

    # Build logit lookup
    logit_map = {idx: float(logit) for idx, logit in zip(legal_c_indices, ppo_c_valid_logits)}

    # Check if all potentials are equal → degenerate to PPO-C
    phi_values = np.array([c.demand_potential for c in successful])
    if len(phi_values) <= 1 or np.std(phi_values) == 0.0:
        # All potentials equal: pick highest PPO logit among successful
        best_c = max(successful, key=lambda c: logit_map.get(c.c_action_idx, -1e9))
        return best_c.c_action_idx

    # Extract logits for successful candidates
    success_logits = np.array([logit_map.get(c.c_action_idx, -1e9) for c in successful])

    # z-score
    z_policy = _zscore(success_logits)
    z_potential = _zscore(phi_values)

    # Combined score
    scores = z_policy + lambda_phi * z_potential

    # Select best, with tiebreak
    best_idx = 0
    best_score = -float('inf')
    for i, (c, score, logit) in enumerate(zip(successful, scores, success_logits)):
        waste = c.block_waste if c.block_waste is not None else float('inf')
        # Composite comparison
        better = False
        if score > best_score:
            better = True
        elif score == best_score:
            if logit > success_logits[best_idx]:
                better = True
            elif logit == success_logits[best_idx]:
                best_waste = successful[best_idx].block_waste
                if best_waste is None:
                    best_waste = float('inf')
                if waste < best_waste:
                    better = True
                elif waste == best_waste:
                    if c.c_action_idx < successful[best_idx].c_action_idx:
                        better = True
        if better:
            best_score = score
            best_idx = i

    return successful[best_idx].c_action_idx


def _select_success_filter_only(
    candidates: List[CandidateRecord],
    ppo_c_valid_logits: np.ndarray,
    ppo_c_default_action: int,
    legal_c_indices: List[int],
) -> int:
    """Select argmax PPO logit among successful candidates (no potential).

    This is the ablation that isolates the benefit of excluding
    currently-failing candidates from the benefit of demand-aware potential.
    """
    successful = [
        c for c in candidates if c.success
    ]
    if not successful:
        return ppo_c_default_action

    logit_map = {idx: float(logit) for idx, logit in zip(legal_c_indices, ppo_c_valid_logits)}

    def _sort_key(c: CandidateRecord):
        logit = logit_map.get(c.c_action_idx, -1e9)
        waste = c.block_waste if c.block_waste is not None else float('inf')
        return (-logit, waste, c.c_action_idx)

    return min(successful, key=_sort_key).c_action_idx


def _select_spectrum_only(
    candidates: List[CandidateRecord],
    ppo_c_valid_logits: np.ndarray,
    ppo_c_default_action: int,
    legal_c_indices: List[int],
) -> int:
    """Select argmax Phi_spectrum among successful candidates (spectrum-only ablation)."""
    successful = [
        c for c in candidates
        if c.success and c.demand_potential_spec is not None
    ]
    if not successful:
        return ppo_c_default_action

    logit_map = {idx: float(logit) for idx, logit in zip(legal_c_indices, ppo_c_valid_logits)}

    def _sort_key(c: CandidateRecord):
        logit = logit_map.get(c.c_action_idx, -1e9)
        waste = c.block_waste if c.block_waste is not None else float('inf')
        return (-c.demand_potential_spec, -logit, waste, c.c_action_idx)

    return min(successful, key=_sort_key).c_action_idx


def _select_compute_only(
    candidates: List[CandidateRecord],
    ppo_c_valid_logits: np.ndarray,
    ppo_c_default_action: int,
    legal_c_indices: List[int],
) -> int:
    """Select argmax Phi_compute among successful candidates (compute-only ablation)."""
    successful = [
        c for c in candidates
        if c.success and c.demand_potential_comp is not None
    ]
    if not successful:
        return ppo_c_default_action

    logit_map = {idx: float(logit) for idx, logit in zip(legal_c_indices, ppo_c_valid_logits)}

    def _sort_key(c: CandidateRecord):
        logit = logit_map.get(c.c_action_idx, -1e9)
        waste = c.block_waste if c.block_waste is not None else float('inf')
        return (-c.demand_potential_comp, -logit, waste, c.c_action_idx)

    return min(successful, key=_sort_key).c_action_idx


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def _aggregate_metrics(m: PerMethodMetrics) -> Dict[str, Any]:
    """Compute derived metrics from raw accumulators."""
    n = max(m.total, 1)
    changed_n = sum(1 for same in m.actions_same if not same)
    changed_safe = max(changed_n, 1)
    changed_blocked_rate = m.changed_blocked_count / changed_safe if changed_n > 0 else None
    categorized_blocked = (
        m.no_suitable_block
        + m.server_overload
        + m.deadline_failure
    )
    other_failure = max(m.blocked - categorized_blocked, 0)

    return {
        "total": m.total,
        "blocking_rate": m.blocked / n,
        "raw_mask_empty_rate": m.raw_empty / n,
        "no_suitable_block_rate": m.no_suitable_block / n,
        "server_overload_rate": m.server_overload / n,
        "deadline_failure_rate": m.deadline_failure / n,
        "other_failure_rate": other_failure / n,
        "reason_counts": dict(m.reason_counts),
        "mean_delay_ms": float(np.mean(m.delays)) if m.delays else 0.0,
        "p50_delay_ms": float(np.percentile(m.delays, 50)) if m.delays else 0.0,
        "p95_delay_ms": float(np.percentile(m.delays, 95)) if m.delays else 0.0,
        "avg_fs": float(np.mean(m.fses)) if m.fses else 0.0,
        "avg_waste": float(np.mean(m.wastes)) if m.wastes else 0.0,
        "avg_path_km": float(np.mean(m.path_kms)) if m.path_kms else 0.0,
        "avg_hop_count": float(np.mean(m.hop_counts)) if m.hop_counts else 0.0,
        "avg_block_start": float(np.mean(m.block_starts)) if m.block_starts else 0.0,
        "avg_block_size": float(np.mean(m.block_sizes)) if m.block_sizes else 0.0,
        "mod_dist": {str(k): v / max(sum(m.mod_counts.values()), 1) for k, v in sorted(m.mod_counts.items())} if m.mod_counts else {},
        "server_dist": {str(k): v / n for k, v in sorted(m.server_counts.items())} if m.server_counts else {},
        "mean_active_connections": float(np.mean(m.active_connections)) if m.active_connections else 0.0,
        "p95_active_connections": float(np.percentile(m.active_connections, 95)) if m.active_connections else 0.0,
        "max_active_connections": max(m.active_connections) if m.active_connections else 0,
        "mean_valid_c_actions": float(np.mean(m.valid_c_actions)) if m.valid_c_actions else 0.0,
        "mean_total_valid_r_actions": float(np.mean(m.total_valid_r_actions)) if m.total_valid_r_actions else 0.0,
        "mean_selected_valid_r_actions": float(np.mean(m.selected_valid_r_actions)) if m.selected_valid_r_actions else 0.0,
        "agreement_with_ppo_c": 1.0 - changed_n / n,
        "changed_action_count": changed_n,
        "changed_action_blocking_rate": changed_blocked_rate,
        "mean_decision_time_ms": float(np.mean(m.decision_times_ms)) if m.decision_times_ms else 0.0,
        "p95_decision_time_ms": float(np.percentile(m.decision_times_ms, 95)) if m.decision_times_ms else 0.0,
        # Lyapunov diagnostics
        "mean_H_spec": float(np.mean(m.lyap_H_spec)) if m.lyap_H_spec else None,
        "p95_H_spec": float(np.percentile(m.lyap_H_spec, 95)) if m.lyap_H_spec else None,
        "mean_adjustment_abs": float(np.mean(m.lyap_adjustment_abs)) if m.lyap_adjustment_abs else None,
        "changed_action_rate_vs_v12": (
            float(np.mean([float(x) for x in m.lyap_changed_vs_v12]))
            if m.lyap_changed_vs_v12 else None
        ),
        # Split choice distribution
        "split_counts": dict(m.split_counts),
        "split_dist": {
            str(split_id): count / n
            for split_id, count in sorted(m.split_counts.items())
        },
        # Ranker profiling breakdown (only meaningful when profile samples exist)
        "profile_stats": _aggregate_profile_lists(m),
    }


def _aggregate_profile_lists(m: PerMethodMetrics) -> Dict[str, Any]:
    """Return mean/P95 profile timings if any samples were collected."""
    keys = [
        "profile_r_feature_build_ms",
        "profile_legal_extract_ms",
        "profile_candidate_select_ms",
        "profile_feature_batch_ms",
        "profile_normalize_ms",
        "profile_ranker_forward_ms",
        "profile_total_ranker_policy_ms",
    ]
    out: Dict[str, Any] = {}
    has_any = False
    for key in keys:
        samples = getattr(m, key, [])
        if samples:
            has_any = True
            short = key.replace("profile_", "").replace("_ms", "")
            out[f"{short}_mean_ms"] = float(np.mean(samples))
            out[f"{short}_p95_ms"] = float(np.percentile(samples, 95))
    if not has_any:
        return {}
    return out


# ---------------------------------------------------------------------------
# Per-method episode runner
# ---------------------------------------------------------------------------
def _run_episode_ppo_c(
    env,
    requests: List,
    agent_c,
    agent_r,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
) -> List[int]:
    """Run a standard PPO-C closed-loop episode (baseline).

    Returns list of selected C-action indices for agreement tracking.
    """
    actions = []
    for step_idx, req in enumerate(requests):
        t_start = time.time()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)

        # raw_mask_empty BEFORE risk mask
        raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        raw_empty = int(raw_mask.sum()) == 0

        action_idx, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
        actions.append(int(action_idx))
        split_id, server_id = decode_agent_c_action(action_idx, args.num_servers)

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_action = _select_r_action(agent_r, obs_r, env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)

        success = bool(info.get("success", False))
        reason = info.get("reason", "")
        total_valid_r = sum(
            sum(slist) for slist in obs_c.get("feasible_counts", [])
        )
        sel_valid_r = int(
            obs_c.get("feasible_counts", [[0]])[split_id][server_id]
        )

        metrics.total += 1
        if not success:
            metrics.blocked += 1
        if raw_empty:
            metrics.raw_empty += 1
        if reason == "no_suitable_block":
            metrics.no_suitable_block += 1
        if reason == "server_overload":
            metrics.server_overload += 1
        if reason == "deadline_infeasible":
            metrics.deadline_failure += 1
        if success:
            metrics.delays.append(float(info.get("delay_ms", 0.0)))
            metrics.fses.append(float(info.get("num_slots", 0.0)))
            metrics.wastes.append(float(info.get("block_waste", 0.0) or 0.0))
            metrics.path_kms.append(float(info.get("path_dist_km", 0.0) or 0.0))
        metrics.active_connections.append(len(env.active_connections))
        metrics.valid_c_actions.append(int(raw_mask.sum()))
        metrics.total_valid_r_actions.append(total_valid_r)
        metrics.selected_valid_r_actions.append(sel_valid_r)
        metrics.actions_same.append(True)  # PPO-C is itself
        metrics.decision_times_ms.append((time.time() - t_start) * 1000.0)

    return actions


def _run_episode_potential_method(
    env,
    requests: List,
    agent_c,
    agent_r,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    method_name: str,
    lambda_phi: Optional[float] = None,
    ppo_c_actions: Optional[List[int]] = None,
):
    """Run a closed-loop episode with potential-based C-action selection.

    Args:
        method_name: "potential_only" or "ppo_potential_rerank"
        lambda_phi: only used for ppo_potential_rerank
        ppo_c_actions: baseline PPO-C actions for agreement tracking
    """
    for step_idx, req in enumerate(requests):
        t_start = time.time()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)

        # raw_mask_empty BEFORE risk mask
        raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        legal_c = np.flatnonzero(raw_mask).tolist()
        raw_empty = len(legal_c) == 0

        # Default PPO-C action (for fallback)
        ppo_c_default_action, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)

        if raw_empty:
            # No valid C candidates: use PPO-C fallback
            selected_c_idx = ppo_c_default_action
            decision_ms = (time.time() - t_start) * 1000.0
        elif len(legal_c) == 1:
            # Single candidate: just use it
            selected_c_idx = legal_c[0]
            decision_ms = (time.time() - t_start) * 1000.0
        else:
            # Get PPO logits for valid candidates
            valid_logits = _get_ppo_c_logits(agent_c, obs_c, raw_mask)

            # Snapshot pre-decision state
            snapshot = _snapshot_before_c_decision(env, req.req_id)

            # Build request history (completed + current only)
            request_history = requests[:step_idx + 1]

            # Evaluate all candidates from snapshot
            candidates = _evaluate_candidates(
                snapshot, req, legal_c, agent_r, args.num_servers,
                request_history,
                args.potential_history_window,
                args.potential_probe_limit,
                args.util_threshold,
                args.alpha,
                potential_components=(
                    {"spectrum"} if method_name == "spectrum_only"
                    else {"compute"} if method_name == "compute_only"
                    else set() if method_name == "success_filter_only"
                    else {"full"}
                ),
            )

            # Select action based on method
            if method_name == "potential_only":
                selected_c_idx = _select_potential_only(
                    candidates, valid_logits, ppo_c_default_action, legal_c,
                )
            elif method_name == "success_filter_only":
                selected_c_idx = _select_success_filter_only(
                    candidates, valid_logits, ppo_c_default_action, legal_c,
                )
            elif method_name == "spectrum_only":
                selected_c_idx = _select_spectrum_only(
                    candidates, valid_logits, ppo_c_default_action, legal_c,
                )
            elif method_name == "compute_only":
                selected_c_idx = _select_compute_only(
                    candidates, valid_logits, ppo_c_default_action, legal_c,
                )
            else:  # ppo_potential_rerank
                selected_c_idx = _select_ppo_potential_rerank(
                    candidates, valid_logits, ppo_c_default_action, legal_c, lambda_phi,
                )

            decision_ms = (time.time() - t_start) * 1000.0

        # Execute selected action
        split_id, server_id = decode_agent_c_action(selected_c_idx, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_action = _select_r_action(agent_r, obs_r, env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)

        success = bool(info.get("success", False))
        reason = info.get("reason", "")
        total_valid_r = sum(
            sum(slist) for slist in obs_c.get("feasible_counts", [])
        )
        sel_valid_r = int(
            obs_c.get("feasible_counts", [[0]])[split_id][server_id]
        )

        metrics.total += 1
        if not success:
            metrics.blocked += 1
        if raw_empty:
            metrics.raw_empty += 1
        if reason == "no_suitable_block":
            metrics.no_suitable_block += 1
        if reason == "server_overload":
            metrics.server_overload += 1
        if reason == "deadline_infeasible":
            metrics.deadline_failure += 1
        if success:
            metrics.delays.append(float(info.get("delay_ms", 0.0)))
            metrics.fses.append(float(info.get("num_slots", 0.0)))
            metrics.wastes.append(float(info.get("block_waste", 0.0) or 0.0))
            metrics.path_kms.append(float(info.get("path_dist_km", 0.0) or 0.0))
        metrics.active_connections.append(len(env.active_connections))
        metrics.valid_c_actions.append(int(raw_mask.sum()))
        metrics.total_valid_r_actions.append(total_valid_r)
        metrics.selected_valid_r_actions.append(sel_valid_r)
        metrics.decision_times_ms.append(decision_ms)

        # Agreement tracking
        changed = False
        if ppo_c_actions is not None and step_idx < len(ppo_c_actions):
            changed = (selected_c_idx != ppo_c_actions[step_idx])
            metrics.actions_same.append(not changed)
        else:
            metrics.actions_same.append(True)
        if changed and not success:
            metrics.changed_blocked_count += 1


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def _compute_verdict(
    ppo_c_agg: Dict[str, Any],
    method_name: str,
    method_agg: Dict[str, Any],
    ppo_c_per_seed: List[Dict[str, Any]],
    method_per_seed: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply PASS / MARGINAL / FAIL criteria."""
    blk_improve = ppo_c_agg["blocking_rate"] - method_agg["blocking_rate"]
    empty_improve = ppo_c_agg["raw_mask_empty_rate"] - method_agg["raw_mask_empty_rate"]
    overload_change = method_agg["server_overload_rate"] - ppo_c_agg["server_overload_rate"]

    seeds_improved = sum(
        1 for p, s in zip(ppo_c_per_seed, method_per_seed)
        if p["blocking_rate"] > s["blocking_rate"]
    )
    total_seeds = len(ppo_c_per_seed)

    if (
        blk_improve >= 0.01
        and empty_improve > -0.005  # at least not worse
        and seeds_improved >= max(2, total_seeds - 1)  # at least 2/3
        and overload_change <= 0.005
    ):
        status = "PASS"
    elif blk_improve >= 0.003 or (blk_improve > 0 and seeds_improved >= 2):
        status = "MARGINAL"
    else:
        status = "FAIL"

    return {
        "method": method_name,
        "status": status,
        "blocking_improvement_pp": blk_improve * 100.0,
        "raw_mask_empty_improvement_pp": empty_improve * 100.0,
        "seeds_improved": f"{seeds_improved}/{total_seeds}",
        "server_overload_change_pp": overload_change * 100.0,
    }


# ---------------------------------------------------------------------------
# Partial save / resume helpers
# ---------------------------------------------------------------------------
import os as _os
import signal as _signal

_PARTIAL_STATE = {"report": None, "output_json": None}


def _partial_path(output_json: str) -> str:
    return output_json + ".partial.json"


def _save_partial(report: Dict[str, Any], output_json: str):
    """Atomically write partial results."""
    tmp = _partial_path(output_json) + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        _os.replace(tmp, _partial_path(output_json))
    except Exception:
        pass


def _load_partial(output_json: str) -> Optional[Dict[str, Any]]:
    """Load partial results if they exist."""
    path = _partial_path(output_json)
    if _os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _signal_handler(signum, frame):
    """Save partial state on SIGINT/SIGTERM."""
    if _PARTIAL_STATE["report"] is not None and _PARTIAL_STATE["output_json"] is not None:
        print(f"\n[signal={signum}] Saving partial results...", flush=True)
        _save_partial(_PARTIAL_STATE["report"], _PARTIAL_STATE["output_json"])
        print("Partial results saved.", flush=True)
    _os._exit(1)


_signal.signal(_signal.SIGINT, _signal_handler)
_signal.signal(_signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Main evaluation (with partial save, resume, methods filter, progress)
# ---------------------------------------------------------------------------
def evaluate_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    lambda_values = [
        float(x.strip()) for x in args.lambda_phi_values.split(",") if x.strip()
    ] if args.lambda_phi_values.strip() else []

    # Methods filter
    method_names_raw = [
        m.strip() for m in getattr(args, "methods", "").split(",") if m.strip()
    ]
    ALL_METHODS = [
        "ppo_c", "success_filter_only", "spectrum_only",
        "compute_only", "potential_only",
    ]
    method_names = [m for m in ALL_METHODS if m in method_names_raw] if method_names_raw else ALL_METHODS

    # Resume support
    resume = getattr(args, "resume", False)
    progress_every = getattr(args, "progress_every", 1)
    output_json = getattr(args, "output_json", None)

    # Load partial from previous run
    completed_set: set = set()  # (seed, method, episode)
    all_method_metrics: Dict[str, Dict[int, List[PerMethodMetrics]]] = {
        m: {s: [] for s in seeds} for m in method_names
    }
    partial_elapsed = 0.0

    if resume and output_json:
        prev = _load_partial(output_json)
        if prev:
            for entry in prev.get("completed", []):
                completed_set.add((entry["seed"], entry["method"], entry["episode"]))
            partial_elapsed = prev.get("elapsed_seconds", 0.0)
            for method, seed_dict in prev.get("per_method_seed", {}).items():
                if method not in all_method_metrics:
                    continue
                for seed_str, eps_data in seed_dict.items():
                    seed = int(seed_str)
                    if seed not in all_method_metrics[method]:
                        continue
                    mm_list = all_method_metrics[method][seed]
                    for ep_m in eps_data:
                        mt = PerMethodMetrics()
                        mt.total = ep_m["total"]
                        mt.blocked = ep_m["blocked"]
                        mt.raw_empty = ep_m["raw_empty"]
                        mt.no_suitable_block = ep_m["no_suitable_block"]
                        mt.server_overload = ep_m["server_overload"]
                        mt.deadline_failure = ep_m["deadline_failure"]
                        mt.delays = ep_m.get("delays", [])
                        mt.fses = ep_m.get("fses", [])
                        mt.wastes = ep_m.get("wastes", [])
                        mt.path_kms = ep_m.get("path_kms", [])
                        mt.active_connections = ep_m.get("active_connections", [])
                        mt.valid_c_actions = ep_m.get("valid_c_actions", [])
                        mt.total_valid_r_actions = ep_m.get("total_valid_r_actions", [])
                        mt.selected_valid_r_actions = ep_m.get("selected_valid_r_actions", [])
                        mt.actions_same = ep_m.get("actions_same", [])
                        mt.changed_blocked_count = ep_m.get("changed_blocked_count", 0)
                        mt.decision_times_ms = ep_m.get("decision_times_ms", [])
                        mm_list.append(mt)

    # Prototype environment
    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots, num_servers=args.num_servers,
        seed=42, slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    r_state_ref = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}

    # Pre-generate episodes
    episodes_by_seed: Dict[int, List[List]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            episodes.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
                traffic_mode=args.traffic_mode, regime_stay_prob=args.regime_stay_prob,
            ))
        episodes_by_seed[seed] = episodes

    t_start = time.time()

    # Build ordered work list: methods × seeds × episodes
    work_specs = []
    for method in method_names:
        for seed in seeds:
            for ep_idx in range(args.episodes):
                work_specs.append((method, seed, ep_idx))

    # --- Main loop with resume and partial save ---
    for work_idx, (method, seed, ep_idx) in enumerate(work_specs):
        if (seed, method, ep_idx) in completed_set:
            continue

        requests = episodes_by_seed[seed][ep_idx]
        mt = PerMethodMetrics()
        ep_t0 = time.time()

        if method == "ppo_c":
            env = make_env(
                topology=args.topology, num_slots=args.num_slots,
                num_servers=args.num_servers, seed=42,
                slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            ppo_c_actions = _run_episode_ppo_c(env, requests, agent_c, agent_r, args, mt)
            # Store ppo_c_actions for future episodes if needed
        else:
            # Need ppo_c actions for agreement tracking — rerun ppo_c for this episode
            env_ppo = make_env(
                topology=args.topology, num_slots=args.num_slots,
                num_servers=args.num_servers, seed=42,
                slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env_ppo.reset(requests)
            ppo_c_mt = PerMethodMetrics()
            ppo_c_actions = _run_episode_ppo_c(env_ppo, requests, agent_c, agent_r, args, ppo_c_mt)

            env = make_env(
                topology=args.topology, num_slots=args.num_slots,
                num_servers=args.num_servers, seed=42,
                slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            if method == "success_filter_only":
                _run_episode_potential_method(env, requests, agent_c, agent_r, args, mt, "success_filter_only", ppo_c_actions=ppo_c_actions)
            elif method == "spectrum_only":
                _run_episode_potential_method(env, requests, agent_c, agent_r, args, mt, "spectrum_only", ppo_c_actions=ppo_c_actions)
            elif method == "compute_only":
                _run_episode_potential_method(env, requests, agent_c, agent_r, args, mt, "compute_only", ppo_c_actions=ppo_c_actions)
            elif method == "potential_only":
                _run_episode_potential_method(env, requests, agent_c, agent_r, args, mt, "potential_only", ppo_c_actions=ppo_c_actions)

        all_method_metrics[method][seed].append(mt)
        completed_set.add((seed, method, ep_idx))
        ep_elapsed = time.time() - ep_t0

        # Progress logging
        total_done = len(completed_set)
        total_work = len(work_specs)
        if progress_every > 0 and (total_done % progress_every == 0 or total_done == total_work):
            elapsed = time.time() - t_start + partial_elapsed
            print(
                f"[seed={seed}][method={method}][episode={ep_idx+1}/{args.episodes}]"
                f"[done={total_done}/{total_work}][elapsed={elapsed:.0f}s]",
                flush=True,
            )

        # Atomic partial save
        if output_json:
            ser = {}
            for m in method_names:
                ser[m] = {}
                for s in seeds:
                    eps = all_method_metrics[m][s]
                    ser[m][str(s)] = [
                        {
                            "total": e.total, "blocked": e.blocked,
                            "raw_empty": e.raw_empty,
                            "no_suitable_block": e.no_suitable_block,
                            "server_overload": e.server_overload,
                            "deadline_failure": e.deadline_failure,
                            "delays": e.delays, "fses": e.fses,
                            "wastes": e.wastes, "path_kms": e.path_kms,
                            "active_connections": e.active_connections,
                            "valid_c_actions": e.valid_c_actions,
                            "total_valid_r_actions": e.total_valid_r_actions,
                            "selected_valid_r_actions": e.selected_valid_r_actions,
                            "actions_same": e.actions_same,
                            "changed_blocked_count": e.changed_blocked_count,
                            "decision_times_ms": e.decision_times_ms,
                        }
                        for e in eps
                    ]
            partial_report = {
                "config": {
                    "seeds": seeds, "episodes": args.episodes,
                    "methods": method_names,
                    "requests_per_episode": args.requests_per_episode,
                    "agent_c_checkpoint": args.agent_c_checkpoint,
                    "agent_r_checkpoint": args.agent_r_checkpoint,
                },
                "completed": [
                    {"seed": s, "method": m, "episode": e}
                    for s, m, e in sorted(completed_set)
                ],
                "per_method_seed": ser,
                "elapsed_seconds": time.time() - t_start + partial_elapsed,
            }
            _PARTIAL_STATE["report"] = partial_report
            _PARTIAL_STATE["output_json"] = output_json
            _save_partial(partial_report, output_json)

    elapsed = time.time() - t_start + partial_elapsed

    # --- Aggregate from per-episode metrics ---
    def _agg_from_eps(eps: List[PerMethodMetrics]) -> Dict[str, Any]:
        mt = PerMethodMetrics()
        for e in eps:
            mt.total += e.total
            mt.blocked += e.blocked
            mt.raw_empty += e.raw_empty
            mt.no_suitable_block += e.no_suitable_block
            mt.server_overload += e.server_overload
            mt.deadline_failure += e.deadline_failure
            mt.delays.extend(e.delays)
            mt.fses.extend(e.fses)
            mt.wastes.extend(e.wastes)
            mt.path_kms.extend(e.path_kms)
            mt.active_connections.extend(e.active_connections)
            mt.valid_c_actions.extend(e.valid_c_actions)
            mt.total_valid_r_actions.extend(e.total_valid_r_actions)
            mt.selected_valid_r_actions.extend(e.selected_valid_r_actions)
            mt.actions_same.extend(e.actions_same)
            mt.changed_blocked_count += e.changed_blocked_count
            mt.decision_times_ms.extend(e.decision_times_ms)
        return _aggregate_metrics(mt)

    def _per_seed_from_eps(method: str) -> List[Dict[str, Any]]:
        result = []
        for i, seed in enumerate(seeds):
            result.append(dict(seed=seed, **_agg_from_eps(all_method_metrics[method][seed])))
        return result

    def _per_method_agg(method: str) -> Dict[str, Any]:
        all_eps = []
        for s in seeds:
            all_eps.extend(all_method_metrics[method][s])
        return _agg_from_eps(all_eps)

    ppo_c_agg = _per_method_agg("ppo_c")
    ppo_c_ps = _per_seed_from_eps("ppo_c")

    # Bootstrap CI — collect per-episode blocking rates
    def _ep_rates(method: str) -> List[float]:
        rates = []
        for s in seeds:
            for e in all_method_metrics[method][s]:
                rates.append(e.blocked / max(e.total, 1))
        return rates

    ppo_c_ep_rates = _ep_rates("ppo_c")

    def _bootstrap_ci(rates_a: List[float], rates_b: List[float], n_boot: int = 10000):
        if len(rates_a) != len(rates_b):
            n_b = min(len(rates_a), len(rates_b))
            rates_a = rates_a[:n_b]; rates_b = rates_b[:n_b]
        diffs = np.array(rates_a) - np.array(rates_b)
        mean_delta = float(np.mean(diffs))
        rng = np.random.RandomState(42)
        boot_means = [float(np.mean(diffs[rng.randint(0, len(diffs), size=len(diffs))])) for _ in range(n_boot)]
        return mean_delta, float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    methods_output = {"ppo_c": {"aggregate": ppo_c_agg, "per_seed": ppo_c_ps}}
    for method in method_names:
        if method == "ppo_c":
            continue
        agg = _per_method_agg(method)
        ps = _per_seed_from_eps(method)
        verdict = _compute_verdict(ppo_c_agg, method, agg, ppo_c_ps, ps)

        ci_info = {}
        if method in ("spectrum_only", "compute_only", "potential_only"):
            m_rates = _ep_rates(method)
            delta, ci_lo, ci_hi = _bootstrap_ci(m_rates, ppo_c_ep_rates)
            ci_info = {"bootstrap_ci": {"delta_vs_ppo_c": delta, "ci_95_lo": ci_lo, "ci_95_hi": ci_hi}}

        methods_output[method] = {"aggregate": agg, "per_seed": ps, "verdict": verdict, **ci_info}

    r_frozen = all(
        torch.equal(agent_r.policy_net.state_dict()[k], v)
        for k, v in r_state_ref.items()
    )

    return {
        "config": {
            "agent_c_checkpoint": args.agent_c_checkpoint,
            "agent_r_checkpoint": args.agent_r_checkpoint,
            "topology": args.topology, "num_slots": args.num_slots,
            "num_servers": args.num_servers, "k_paths": args.k_paths,
            "max_blocks": args.max_blocks, "block_sort_strategy": args.block_sort_strategy,
            "split_profile": args.split_profile, "num_splits": args.num_splits,
            "seeds": seeds, "episodes": args.episodes,
            "requests_per_episode": args.requests_per_episode,
            "arrival_interval": args.arrival_interval,
            "holding_min": args.holding_min, "holding_max": args.holding_max,
            "potential_history_window": args.potential_history_window,
            "potential_probe_limit": args.potential_probe_limit,
            "alpha": args.alpha, "util_threshold": args.util_threshold,
            "lambda_phi_values": lambda_values,
            "methods": method_names,
            "oracle_semantics": "closed-loop with frozen PPO-R, no future request access",
        },
        "methods": methods_output,
        "frozen_params_unchanged": {"agent_r": r_frozen},
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _build_markdown(report: Dict[str, Any]) -> str:
    cfg = report["config"]

    def _f(v):
        if v is None:
            return "N/A"
        return f"{v:.4f}"

    def _pct(v):
        if v is None:
            return "N/A"
        return f"{v * 100:.2f}%"

    lines = [
        "# Closed-Loop Demand-Aware Potential Evaluation",
        "",
        f"- **Agent-C:** `{cfg['agent_c_checkpoint']}`",
        f"- **PPO-R:** `{cfg['agent_r_checkpoint']}`",
        f"- **Topology:** {cfg['topology']}",
        f"- **Slots:** {cfg['num_slots']}  **Servers:** {cfg['num_servers']}  **K:** {cfg['k_paths']}  **Max blocks:** {cfg['max_blocks']}",
        f"- **Arrival interval:** {cfg['arrival_interval']}s  **Holding:** [{cfg['holding_min']}, {cfg['holding_max']}]s",
        f"- **Seeds:** {cfg['seeds']}  **Eps/seed:** {cfg['episodes']}  **Req/ep:** {cfg['requests_per_episode']}",
        f"- **Potential:** window={cfg['potential_history_window']}, probes={cfg['potential_probe_limit']}, α={cfg['alpha']}",
        f"- **Lambda sweep:** {cfg['lambda_phi_values']}",
        f"- **Frozen R:** {report['frozen_params_unchanged']['agent_r']}",
        "",
    ]

    # Aggregate comparison table — dynamic
    ppo_c_agg = report["methods"]["ppo_c"]["aggregate"]

    def _delta_pp(base_val, method_val):
        return (base_val - method_val) * 100.0

    # Read bootstrap CI from report methods (robust to missing methods)
    def _ci(key):
        c = report["methods"].get(key, {}).get("bootstrap_ci", {})
        return c.get("ci_95_lo", 0), c.get("ci_95_hi", 0), c.get("delta_vs_ppo_c", 0)
    spec_ci_lo, spec_ci_hi, spec_delta = _ci("spectrum_only")
    comp_ci_lo, comp_ci_hi, comp_delta = _ci("compute_only")
    full_ci_lo, full_ci_hi, full_delta = _ci("potential_only")
    has_sf = "success_filter_only" in report["methods"]
    has_spec = "spectrum_only" in report["methods"]
    has_comp = "compute_only" in report["methods"]
    has_pot = "potential_only" in report["methods"]

    # Build dynamic aggregate table
    all_method_names = [k for k in report["methods"].keys() if not k.startswith("ppo_potential_rerank")]
    agg_cols = " | ".join(m.replace("_", " ").title() for m in all_method_names)
    lines.extend([
        "## Aggregate Comparison",
        "",
        f"| Metric | {' | '.join(m.replace('_',' ').title() for m in all_method_names)} |",
        f"|---|---{'|' * (len(all_method_names)-1)}",
    ])
    metric_keys = [
        ("Total requests", "total", False),
        ("Blocking rate", "blocking_rate", True),
        ("Raw-mask-empty rate", "raw_mask_empty_rate", True),
        ("No-suitable-block rate", "no_suitable_block_rate", True),
        ("Server-overload rate", "server_overload_rate", True),
        ("Deadline-infeasible rate", "deadline_failure_rate", True),
        ("Mean delay (success) ms", "mean_delay_ms", False),
        ("Avg FS", "avg_fs", False),
    ]
    for label, key, is_pct in metric_keys:
        vals = []
        for m in all_method_names:
            v = report["methods"][m]["aggregate"][key]
            vals.append(_pct(v) if is_pct else _f(v))
        lines.append(f"| {label} | {' | '.join(vals)} |")
    lines.extend(["", "### Resource-Component Decomposition", "",
        "| Component | Blocking | vs PPO-C | Bootstrap 95% CI |",
        "|---|---:|---:|:---:|",
        f"| PPO-C baseline | {_pct(ppo_c_agg['blocking_rate'])} | — | — |"])
    for m_key, m_label in [("success_filter_only","Success-filter only"),
                            ("spectrum_only","Spectrum only"),
                            ("compute_only","Compute only"),
                            ("potential_only","Full potential")]:
        if m_key not in report["methods"]:
            continue
        m_agg = report["methods"][m_key]["aggregate"]
        ci = report["methods"][m_key].get("bootstrap_ci", {})
        ci_lo, ci_hi = ci.get("ci_95_lo", 0), ci.get("ci_95_hi", 0)
        ci_str = f"[{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] pp" if ci else "—"
        lines.append(
            f"| {m_label} | {_pct(m_agg['blocking_rate'])} | "
            f"{_delta_pp(ppo_c_agg['blocking_rate'], m_agg['blocking_rate']):+.2f}pp | {ci_str} |")
    lines.append("")

    # Lambda sweep table
    lines.extend([
        "## PPO-Potential Rerank (Lambda Sweep)",
        "",
        "| λ | Blocking | vs PPO-C | Raw-Empty | vs PPO-C | SvrOver | vs PPO-C | Agreement | Changed | Chg-Blk | Mean/P95 Dec ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for lam_str, method_data in report["methods"].items():
        if not lam_str.startswith("ppo_potential_rerank_lambda_"):
            continue
        agg = method_data["aggregate"]
        lam_val = method_data.get("lambda_phi", 0)
        blk_delta = (ppo_c_agg["blocking_rate"] - agg["blocking_rate"]) * 100
        empty_delta = (ppo_c_agg["raw_mask_empty_rate"] - agg["raw_mask_empty_rate"]) * 100
        sovr_delta = (agg["server_overload_rate"] - ppo_c_agg["server_overload_rate"]) * 100
        chg_blk = _pct(agg["changed_action_blocking_rate"])
        lines.append(
            f"| {lam_val:.2f} | {_pct(agg['blocking_rate'])} | {blk_delta:+.2f}pp | "
            f"{_pct(agg['raw_mask_empty_rate'])} | {empty_delta:+.2f}pp | "
            f"{_pct(agg['server_overload_rate'])} | {sovr_delta:+.2f}pp | "
            f"{_pct(agg['agreement_with_ppo_c'])} | {agg['changed_action_count']} | "
            f"{chg_blk} | {_f(agg['mean_decision_time_ms'])} / {_f(agg['p95_decision_time_ms'])} |"
        )
    lines.append("")

    # Per-seed detail
    # Per-seed blocking
    lines.extend(["## Per-Seed Breakdown", "", "### Blocking Rate", "",
        "| Seed | " + " | ".join(m.replace("_"," ").title() for m in all_method_names) + " |",
        "|---|" + "|".join("---" for _ in all_method_names) + "|"])
    for i, seed in enumerate(cfg["seeds"]):
        vals = [_pct(report["methods"][m]["per_seed"][i]["blocking_rate"]) for m in all_method_names]
        lines.append(f"| {seed} | " + " | ".join(vals) + " |")
    lines.append("")

    # Verdicts
    lines.extend(["## Verdicts", ""])
    for m_key in all_method_names:
        if m_key == "ppo_c":
            continue
        v = report["methods"][m_key].get("verdict", {})
        ci = report["methods"][m_key].get("bootstrap_ci", {})
        lines.append(f"### {m_key.replace('_',' ').title()}: **{v.get('status','N/A')}**")
        lines.append(f"- Blocking improvement: {v.get('blocking_improvement_pp', 0):.2f} pp")
        if ci:
            lines.append(f"- Bootstrap 95% CI: [{ci.get('ci_95_lo',0)*100:+.2f}, {ci.get('ci_95_hi',0)*100:+.2f}] pp")
        lines.append("")

    # Resource-component verdict
    spec_sig = has_spec and spec_ci_lo < 0
    comp_sig = has_comp and comp_ci_lo < 0
    full_sig = has_pot and full_ci_lo < 0

    if has_spec and has_comp:
        spec_agg = report["methods"]["spectrum_only"]["aggregate"]
        comp_agg = report["methods"]["compute_only"]["aggregate"]
        spec_nsb = spec_agg["no_suitable_block_rate"] - ppo_c_agg["no_suitable_block_rate"]
        comp_svr = comp_agg["server_overload_rate"] - ppo_c_agg["server_overload_rate"]

        best_single = min(spec_delta, comp_delta)
        full_beats = full_delta < best_single if has_pot else False

        if full_sig and full_beats and spec_sig and comp_sig:
            resource_verdict = "MULTI_RESOURCE_SYNERGY_SUPPORTED"
        elif spec_sig and spec_nsb < -0.001:
            resource_verdict = "SPECTRUM_SUPPORTED"
        elif comp_sig and comp_svr < -0.001:
            resource_verdict = "COMPUTE_SUPPORTED"
        elif full_sig and not (spec_sig or comp_sig):
            resource_verdict = "NOT_SUPPORTED"
        else:
            resource_verdict = "INCONCLUSIVE"
    else:
        resource_verdict = "INCONCLUSIVE"

    # Summary
    best_name = "ppo_c"
    best_blk = ppo_c_agg["blocking_rate"]
    for method_key, method_data in report["methods"].items():
        if method_key == "ppo_c" or "rerank" in method_key:
            continue
        agg = method_data["aggregate"]
        if agg["blocking_rate"] < best_blk:
            best_blk = agg["blocking_rate"]
            best_name = method_key

    lines.extend([
        "## Summary",
        "",
        f"Best method: **{best_name}** ({_pct(best_blk)})",
        f"Resource-component verdict: **{resource_verdict}**",
        "",
        "---",
        f"*Elapsed: {report['elapsed_seconds']:.1f}s*",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop demand-aware potential evaluation for Agent-C"
    )
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.15)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--traffic_mode", type=str, default="iid",
                        choices=["iid", "markov_regime"])
    parser.add_argument("--regime_stay_prob", type=float, default=0.9)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--potential_history_window", type=int, default=12)
    parser.add_argument("--potential_probe_limit", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--lambda_phi_values", type=str, default="0.0,0.25,0.5,1.0")
    parser.add_argument("--methods", type=str, default="",
                        help="Comma-separated methods: ppo_c,success_filter_only,spectrum_only,compute_only,potential_only")
    parser.add_argument("--progress_every", type=int, default=1,
                        help="Print progress every N completed episodes")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial JSON if exists")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = "sa_hmarl/experiments/c_demand_potential_closed_loop.json"
    if args.output_md is None:
        args.output_md = "sa_hmarl/experiments/c_demand_potential_closed_loop.md"

    report = evaluate_checkpoint(args)

    # Write outputs
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    Path(args.output_md).write_text(_build_markdown(report), encoding="utf-8")

    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")

    # Print key verdicts
    pot_v = report["methods"]["potential_only"]["verdict"]
    print(f"\nPotential Only: {pot_v['status']} (blocking Δ: {pot_v['blocking_improvement_pp']:.3f}pp)")
    for lam_str, method_data in report["methods"].items():
        if lam_str.startswith("ppo_potential_rerank_lambda_"):
            v = method_data["verdict"]
            print(f"Rerank λ={method_data.get('lambda_phi', '?'):.2f}: {v['status']} (blocking Δ: {v['blocking_improvement_pp']:.3f}pp)")
    print(f"R frozen: {report['frozen_params_unchanged']['agent_r']}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
