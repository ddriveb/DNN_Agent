"""Diagnose whether candidate-level resource/spectrum impact is useful for Agent-C.

For each request, the script:
  1. Computes the pre-decision spectrum feasibility field (K_C_valid, K_R_total, Phi_spec).
  2. For every raw-mask-valid (split, server) candidate, simulates selecting that candidate
     (with frozen Agent-R picking the R action) and measures the post-step spectrum
     feasibility field for the *next* request.
  3. Compares the simulated candidates to the actual reference policy's choice.

This answers: "If I pick candidate c, how much future spectrum feasibility do I leave?"
It is *not* another instant-safety feature; it is a forward-looking impact signal.

Outputs:
    * JSON: ``experiments/candidate_resource_impact_diagnostic.json``
    * Markdown: ``experiments/candidate_resource_impact_diagnostic.md``
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


@dataclass
class CandidateRecord:
    seed: int
    episode: int
    step: int
    req_id: int
    checkpoint: str

    k_c_valid_before: int
    k_r_total_before: int
    phi_spec_before: float
    raw_mask_empty_before: bool

    candidate_split: int
    candidate_server: int
    candidate_index: int
    raw_mask_valid: bool

    simulated_success: bool
    simulated_reason: str
    k_c_valid_next: Optional[int]
    k_r_total_next: Optional[int]
    phi_spec_next: Optional[float]

    # Non-leaky proxy features (computed from current obs only)
    proxy_lfb_max_norm: float
    proxy_lfb_mean_norm: float
    proxy_free_mean: float
    proxy_frag_mean: float
    proxy_spectrum_congestion: float
    proxy_lfb_pressure: float
    proxy_path_diversity: float
    proxy_num_feas_paths_norm: float
    proxy_feasible_count: float
    proxy_n_feas_path_mod: float
    proxy_server_utilization: float
    proxy_server_available_ratio: float
    proxy_edge_compute_cost: float
    proxy_edge_compute_ms: float
    proxy_local_compute_ms: float
    proxy_path_km: float
    proxy_best_fs_estimate: float
    proxy_safe_fs_estimate: float
    proxy_composite_congestion: float
    proxy_composite_resource: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "step": self.step,
            "req_id": self.req_id,
            "checkpoint": self.checkpoint,
            "k_c_valid_before": self.k_c_valid_before,
            "k_r_total_before": self.k_r_total_before,
            "phi_spec_before": self.phi_spec_before,
            "raw_mask_empty_before": self.raw_mask_empty_before,
            "candidate_split": self.candidate_split,
            "candidate_server": self.candidate_server,
            "candidate_index": self.candidate_index,
            "raw_mask_valid": self.raw_mask_valid,
            "simulated_success": self.simulated_success,
            "simulated_reason": self.simulated_reason,
            "k_c_valid_next": self.k_c_valid_next,
            "k_r_total_next": self.k_r_total_next,
            "phi_spec_next": self.phi_spec_next,
            "proxy_lfb_max_norm": self.proxy_lfb_max_norm,
            "proxy_lfb_mean_norm": self.proxy_lfb_mean_norm,
            "proxy_free_mean": self.proxy_free_mean,
            "proxy_frag_mean": self.proxy_frag_mean,
            "proxy_spectrum_congestion": self.proxy_spectrum_congestion,
            "proxy_lfb_pressure": self.proxy_lfb_pressure,
            "proxy_path_diversity": self.proxy_path_diversity,
            "proxy_num_feas_paths_norm": self.proxy_num_feas_paths_norm,
            "proxy_feasible_count": self.proxy_feasible_count,
            "proxy_n_feas_path_mod": self.proxy_n_feas_path_mod,
            "proxy_server_utilization": self.proxy_server_utilization,
            "proxy_server_available_ratio": self.proxy_server_available_ratio,
            "proxy_edge_compute_cost": self.proxy_edge_compute_cost,
            "proxy_edge_compute_ms": self.proxy_edge_compute_ms,
            "proxy_local_compute_ms": self.proxy_local_compute_ms,
            "proxy_path_km": self.proxy_path_km,
            "proxy_best_fs_estimate": self.proxy_best_fs_estimate,
            "proxy_safe_fs_estimate": self.proxy_safe_fs_estimate,
            "proxy_composite_congestion": self.proxy_composite_congestion,
            "proxy_composite_resource": self.proxy_composite_resource,
        }


@dataclass
class PolicyRecord:
    seed: int
    episode: int
    step: int
    req_id: int
    checkpoint: str

    k_c_valid_before: int
    phi_spec_before: float
    raw_mask_empty_before: bool

    selected_split: int
    selected_server: int
    selected_candidate_index: int

    actual_success: bool
    actual_reason: str
    phi_spec_next_of_selected: Optional[float]
    best_phi_spec_next: Optional[float]
    worst_phi_spec_next: Optional[float]
    mean_phi_spec_next: Optional[float]
    rank_of_selected_by_phi: Optional[int]
    n_simulated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "step": self.step,
            "req_id": self.req_id,
            "checkpoint": self.checkpoint,
            "k_c_valid_before": self.k_c_valid_before,
            "phi_spec_before": self.phi_spec_before,
            "raw_mask_empty_before": self.raw_mask_empty_before,
            "selected_split": self.selected_split,
            "selected_server": self.selected_server,
            "selected_candidate_index": self.selected_candidate_index,
            "actual_success": self.actual_success,
            "actual_reason": self.actual_reason,
            "phi_spec_next_of_selected": self.phi_spec_next_of_selected,
            "best_phi_spec_next": self.best_phi_spec_next,
            "worst_phi_spec_next": self.worst_phi_spec_next,
            "mean_phi_spec_next": self.mean_phi_spec_next,
            "rank_of_selected_by_phi": self.rank_of_selected_by_phi,
            "n_simulated": self.n_simulated,
        }


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"),
    )
    num_servers = ckpt_args.get("num_servers")
    if num_servers is None:
        num_servers = ckpt.get("num_servers")
    agent_c_kwargs = dict(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    if feature_mode in (
        "mean_field", "typed_mean_field", "gated_typed_mean_field",
        "fixed_blend_typed_mean_field", "candidate_mean_field",
        "candidate_mean_field_count_only",
    ):
        agent_c_kwargs["num_servers"] = num_servers
    if feature_mode == "fixed_blend_typed_mean_field":
        agent_c_kwargs["fixed_blend_alpha"] = ckpt_args.get("fixed_blend_alpha", 0.5)
    agent = PPOAgentC(**agent_c_kwargs)
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=ckpt.get("agent_r_feature_mode", "default"),
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _select_c_action(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
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
    return action, raw_mask, mask


def _select_r_action(agent_r: PPOAgentR, obs_r: Dict[str, Any], max_blocks: int) -> Tuple[int, int, int]:
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return 0, 0, 0
    return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), max_blocks)


def _compute_proxy_features(obs_c: Dict[str, Any], feat_dict: Dict[str, Any], num_slots: int, k_paths: int) -> Dict[str, float]:
    """Compute non-leaky proxy features for a candidate from current obs only."""
    spec_vec = np.asarray(feat_dict.get("spectrum_summary", []), dtype=float)

    def _vec(i, default=0.0):
        return float(spec_vec[i]) if len(spec_vec) > i else default

    lfb_max = _vec(0, 24.0)
    lfb_mean = _vec(1, 24.0)
    free_mean = _vec(6, 1.0)
    frag_mean = _vec(4, 0.0)
    num_feas_paths = _vec(7, 0.0)
    path_diversity = _vec(9, 0.0)

    lfb_max_norm = lfb_max / max(num_slots, 1)
    lfb_mean_norm = lfb_mean / max(num_slots, 1)
    spectrum_congestion = 1.0 - free_mean

    safe_fs = feat_dict.get("safe_fs_estimate")
    safe_fs = float(safe_fs) if safe_fs is not None else 0.0
    lfb_pressure = safe_fs / max(lfb_max, 1.0)

    num_feas_paths_norm = num_feas_paths / max(k_paths, 1)

    available = float(feat_dict.get("server_available_compute", 0.0))
    capacity = max(float(feat_dict.get("server_capacity", 1.0)), 1e-6)
    server_available_ratio = available / capacity

    composite_congestion = spectrum_congestion * lfb_pressure
    composite_resource = safe_fs / max(lfb_max, 1.0)

    return {
        "proxy_lfb_max_norm": lfb_max_norm,
        "proxy_lfb_mean_norm": lfb_mean_norm,
        "proxy_free_mean": free_mean,
        "proxy_frag_mean": frag_mean,
        "proxy_spectrum_congestion": spectrum_congestion,
        "proxy_lfb_pressure": lfb_pressure,
        "proxy_path_diversity": path_diversity,
        "proxy_num_feas_paths_norm": num_feas_paths_norm,
        "proxy_feasible_count": float(feat_dict.get("feasible_count", 0)),
        "proxy_n_feas_path_mod": float(feat_dict.get("n_feas_path_mod", 0)),
        "proxy_server_utilization": float(feat_dict.get("server_utilization", 0.0)),
        "proxy_server_available_ratio": server_available_ratio,
        "proxy_edge_compute_cost": float(feat_dict.get("edge_compute_cost", 0.0)),
        "proxy_edge_compute_ms": float(feat_dict.get("edge_compute_ms", 0.0)),
        "proxy_local_compute_ms": float(feat_dict.get("local_compute_ms", 0.0)),
        "proxy_path_km": float(feat_dict.get("server_min_path_km", 0.0)),
        "proxy_best_fs_estimate": float(feat_dict.get("best_fs_estimate", 0) or 0),
        "proxy_safe_fs_estimate": safe_fs,
        "proxy_composite_congestion": composite_congestion,
        "proxy_composite_resource": composite_resource,
    }


def _compute_spectrum_field(obs_c: Dict[str, Any], util_threshold: float = 0.95) -> Tuple[int, int]:
    """Compute K_C_valid and K_R_total for one request observation."""
    num_servers = len(obs_c["server_utilizations"])
    feasible_counts = obs_c["feasible_counts"]

    k_c_valid = 0
    k_r_total = 0

    for split_id, server_list in enumerate(feasible_counts):
        for server_id, count in enumerate(server_list):
            cand_idx = split_id * num_servers + server_id
            feat_dict = obs_c["candidate_features"][cand_idx]

            available = float(feat_dict.get("server_available_compute", 0.0))
            edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
            edge_ms = feat_dict.get("edge_compute_ms", 0.0)
            compute_ok = (edge_cost <= available) and np.isfinite(edge_ms) and edge_ms != float('inf')

            util = float(obs_c["server_utilizations"][server_id])
            util_ok = util <= util_threshold

            diag = AgentC.compute_r_feasibility_diagnostics(obs_c, feat_dict)
            n_r = int(diag.get("valid_r_actions", 0))

            if compute_ok and util_ok and n_r > 0:
                k_c_valid += 1
            k_r_total += n_r

    return k_c_valid, k_r_total


def _simulate_candidate(
    env,
    req,
    next_req,
    agent_r: PPOAgentR,
    split_id: int,
    server_id: int,
    args: argparse.Namespace,
) -> Tuple[bool, str, Optional[int], Optional[int], Optional[float]]:
    """Simulate selecting one candidate and return next-request spectrum field."""
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
    _, _, _, info = env.step((split_id, server_id), action_r)
    success = bool(info.get("success", False))
    reason = info.get("reason", "unknown")

    if next_req is None:
        return success, reason, None, None, None

    obs_c_next = build_agent_c_observation(env, next_req)
    k_c_next, k_r_next = _compute_spectrum_field(obs_c_next, args.util_threshold)
    phi_next = np.log1p(k_c_next) + args.alpha * np.log1p(k_r_next)
    return success, reason, k_c_next, k_r_next, float(phi_next)


def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg

    print("Loading frozen R backend...")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    print(f"Loading reference C policy: {args.rollout_c_checkpoint}")
    agent_c = _load_ppo_c(args.rollout_c_checkpoint, args.device)

    # Generate requests once
    all_episodes: Dict[int, List[List[Any]]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps

    cand_records: List[CandidateRecord] = []
    policy_records: List[PolicyRecord] = []
    t_start = time.time()

    for seed in sorted(all_episodes.keys()):
        for ep_idx, requests in enumerate(all_episodes[seed]):
            env = make_env(
                topology=args.topology, num_slots=args.num_slots,
                num_servers=args.num_servers, seed=42,
                slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)

            for step_idx, req in enumerate(requests):
                next_req = requests[step_idx + 1] if step_idx + 1 < len(requests) else None

                obs_c = build_agent_c_observation(env, req)
                k_c_before, k_r_before = _compute_spectrum_field(obs_c, args.util_threshold)
                phi_before = np.log1p(k_c_before) + args.alpha * np.log1p(k_r_before)
                raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                raw_empty = int(raw_mask.sum()) == 0

                # Save state before simulating candidates
                snapshot = copy.deepcopy(env)

                # Actual policy selection
                action_idx_c, raw_mask_arr, _ = _select_c_action(agent_c, obs_c, env.net.num_slots)
                selected_split, selected_server = decode_agent_c_action(action_idx_c, args.num_servers)
                selected_index = selected_split * args.num_servers + selected_server

                # Simulate all raw-mask-valid candidates
                sim_results: List[Tuple[int, int, int, bool, bool, str, Optional[float]]] = []
                num_splits = len(obs_c["feasible_counts"])
                num_servers = len(obs_c["server_utilizations"])
                for split_id in range(num_splits):
                    for server_id in range(num_servers):
                        idx = split_id * num_servers + server_id
                        valid = bool(raw_mask[idx])
                        feat_dict = obs_c["candidate_features"][idx]
                        proxies = _compute_proxy_features(
                            obs_c, feat_dict, env.net.num_slots, args.k_paths
                        )

                        # Restore state
                        env = copy.deepcopy(snapshot)
                        success, reason, k_c_next, k_r_next, phi_next = _simulate_candidate(
                            env, req, next_req, agent_r, split_id, server_id, args
                        )

                        cand_records.append(CandidateRecord(
                            seed=seed, episode=ep_idx, step=step_idx,
                            req_id=int(req.req_id),
                            checkpoint=args.checkpoint_label,
                            k_c_valid_before=k_c_before,
                            k_r_total_before=k_r_before,
                            phi_spec_before=float(phi_before),
                            raw_mask_empty_before=raw_empty,
                            candidate_split=split_id,
                            candidate_server=server_id,
                            candidate_index=idx,
                            raw_mask_valid=valid,
                            simulated_success=success,
                            simulated_reason=reason,
                            k_c_valid_next=k_c_next,
                            k_r_total_next=k_r_next,
                            phi_spec_next=phi_next,
                            **proxies,
                        ))

                        if phi_next is not None:
                            sim_results.append((
                                split_id, server_id, idx, valid, success, reason, phi_next
                            ))

                # Restore state and run actual policy step
                env = copy.deepcopy(snapshot)
                obs_r = build_agent_r_observation(env, req, selected_split, selected_server)
                action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
                _, _, _, info = env.step((selected_split, selected_server), action_r)
                actual_success = bool(info.get("success", False))
                actual_reason = info.get("reason", "unknown")

                # Rank selected candidate by phi_spec_next
                phi_of_selected = None
                rank_of_selected = None
                best_phi = None
                worst_phi = None
                mean_phi = None
                if sim_results:
                    phi_values = [r[6] for r in sim_results]
                    best_phi = max(phi_values)
                    worst_phi = min(phi_values)
                    mean_phi = float(np.mean(phi_values))

                    # Find phi of selected candidate
                    for r in sim_results:
                        if r[0] == selected_split and r[1] == selected_server:
                            phi_of_selected = r[6]
                            break

                    if phi_of_selected is not None:
                        # Rank: higher phi = better = lower rank
                        sorted_phis = sorted(phi_values, reverse=True)
                        rank_of_selected = sorted_phis.index(phi_of_selected) + 1

                policy_records.append(PolicyRecord(
                    seed=seed, episode=ep_idx, step=step_idx,
                    req_id=int(req.req_id),
                    checkpoint=args.checkpoint_label,
                    k_c_valid_before=k_c_before,
                    phi_spec_before=float(phi_before),
                    raw_mask_empty_before=raw_empty,
                    selected_split=selected_split,
                    selected_server=selected_server,
                    selected_candidate_index=selected_index,
                    actual_success=actual_success,
                    actual_reason=actual_reason,
                    phi_spec_next_of_selected=phi_of_selected,
                    best_phi_spec_next=best_phi,
                    worst_phi_spec_next=worst_phi,
                    mean_phi_spec_next=mean_phi,
                    rank_of_selected_by_phi=rank_of_selected,
                    n_simulated=len(sim_results),
                ))

    # Build DataFrames
    df_cand = pd.DataFrame([r.to_dict() for r in cand_records])
    df_policy = pd.DataFrame([r.to_dict() for r in policy_records])

    # Overall candidate impact stats
    valid_cands = df_cand[df_cand["raw_mask_valid"] == True]
    successful_cands = df_cand[(df_cand["raw_mask_valid"] == True) & (df_cand["simulated_success"] == True)]
    failed_cands = df_cand[(df_cand["raw_mask_valid"] == True) & (df_cand["simulated_success"] == False)]

    def _safe_mean(series):
        return float(series.mean()) if len(series) > 0 else None

    def _safe_std(series):
        return float(series.std()) if len(series) > 0 else None

    candidate_impact_summary = {
        "total_candidate_evaluations": len(df_cand),
        "raw_mask_valid_evaluations": len(valid_cands),
        "simulated_successful": len(successful_cands),
        "simulated_failed": len(failed_cands),
        "phi_spec_next": {
            "all_valid_mean": _safe_mean(valid_cands["phi_spec_next"]),
            "all_valid_std": _safe_std(valid_cands["phi_spec_next"]),
            "successful_mean": _safe_mean(successful_cands["phi_spec_next"]),
            "successful_std": _safe_std(successful_cands["phi_spec_next"]),
            "failed_mean": _safe_mean(failed_cands["phi_spec_next"]),
            "failed_std": _safe_std(failed_cands["phi_spec_next"]),
        },
        "k_c_valid_next": {
            "all_valid_mean": _safe_mean(valid_cands["k_c_valid_next"]),
            "successful_mean": _safe_mean(successful_cands["k_c_valid_next"]),
            "failed_mean": _safe_mean(failed_cands["k_c_valid_next"]),
        },
    }

    # Per-request impact variance
    valid_only = df_cand[df_cand["raw_mask_valid"] == True]
    per_request = valid_only.groupby(["seed", "episode", "step"]).agg(
        phi_spec_next_min=("phi_spec_next", "min"),
        phi_spec_next_max=("phi_spec_next", "max"),
        phi_spec_next_mean=("phi_spec_next", "mean"),
        phi_spec_next_std=("phi_spec_next", "std"),
        k_c_valid_next_min=("k_c_valid_next", "min"),
        k_c_valid_next_max=("k_c_valid_next", "max"),
        n_valid=("phi_spec_next", "size"),
    ).reset_index()
    per_request["phi_spec_range"] = per_request["phi_spec_next_max"] - per_request["phi_spec_next_min"]

    impact_variance = {
        "requests_with_multiple_valid_candidates": int((per_request["n_valid"] > 1).sum()),
        "mean_phi_spec_range": _safe_mean(per_request["phi_spec_range"]),
        "std_phi_spec_range": _safe_std(per_request["phi_spec_range"]),
        "mean_k_c_valid_range": _safe_mean(per_request["k_c_valid_next_max"] - per_request["k_c_valid_next_min"]),
    }

    # Policy selection quality
    df_policy_non_empty = df_policy[df_policy["raw_mask_empty_before"] == False]
    rank_dist = df_policy_non_empty["rank_of_selected_by_phi"].value_counts().sort_index().to_dict()
    rank_dist = {int(k): int(v) for k, v in rank_dist.items()}

    # Oracle analysis: what if we always pick the candidate with highest phi_spec_next?
    # Build per-request best candidate outcome (only for requests with at least one valid candidate)
    best_outcomes = []
    for (seed, ep, step), group in valid_only.groupby(["seed", "episode", "step"]):
        if group["phi_spec_next"].notna().sum() == 0:
            continue
        best_row = group.loc[group["phi_spec_next"].idxmax()]
        best_outcomes.append({
            "seed": seed, "episode": ep, "step": step,
            "raw_mask_empty_before": bool(group["raw_mask_empty_before"].iloc[0]),
            "best_phi": float(best_row["phi_spec_next"]),
            "best_success": bool(best_row["simulated_success"]),
            "best_reason": best_row["simulated_reason"],
        })
    df_best = pd.DataFrame(best_outcomes)

    # Oracle denominator-aware statistics
    n_total = len(df_policy)
    n_empty = int(df_policy["raw_mask_empty_before"].sum())
    n_valid = n_total - n_empty
    n_best_eval = len(df_best)

    oracle_summary = {}
    if len(df_best) > 0:
        oracle_valid_success_rate = float(df_best["best_success"].mean())
        oracle_valid_blocking_rate = float((~df_best["best_success"]).mean())
        oracle_all_success_rate = oracle_valid_success_rate * (n_valid / max(n_total, 1))
        oracle_all_blocking_rate = 1.0 - oracle_all_success_rate
        oracle_empty_success_rate = 0.0  # oracle cannot help when no valid candidate exists

        oracle_summary = {
            "requests_evaluated_for_best_candidate": n_best_eval,
            "denominators": {
                "all_requests": n_total,
                "raw_mask_valid_requests": n_valid,
                "raw_mask_empty_requests": n_empty,
            },
            "oracle_success_rate": {
                "all_requests": oracle_all_success_rate,
                "raw_mask_valid_subset": oracle_valid_success_rate,
                "raw_mask_empty_subset": oracle_empty_success_rate,
            },
            "oracle_blocking_rate": {
                "all_requests": oracle_all_blocking_rate,
                "raw_mask_valid_subset": oracle_valid_blocking_rate,
                "raw_mask_empty_subset": 1.0 - oracle_empty_success_rate,
            },
            "oracle_no_suitable_block_rate": {
                "raw_mask_valid_subset": float((df_best["best_reason"] == "no_suitable_block").mean()),
            },
            "policy_blocking_rate_all_requests": float((~df_policy["actual_success"]).mean()),
            "policy_blocking_rate_raw_mask_valid": float((~df_policy[~df_policy["raw_mask_empty_before"]]["actual_success"]).mean()) if n_valid > 0 else None,
            "policy_blocking_rate_raw_mask_empty": float((~df_policy[df_policy["raw_mask_empty_before"]]["actual_success"]).mean()) if n_empty > 0 else None,
        }

    # Correlations
    def _corr(x, y):
        try:
            with np.errstate(invalid="ignore"):
                r, p = stats.pearsonr(x, y)
                return float(r), float(p)
        except Exception:
            return float("nan"), float("nan")

    correlations = {}
    # 1. Does higher future Phi predict simulated success across all valid candidates?
    vc_drop = valid_cands.dropna(subset=["phi_spec_next"])
    if len(vc_drop) > 1 and vc_drop["simulated_success"].nunique() > 1:
        correlations["phi_spec_next_vs_simulated_success"] = _corr(
            vc_drop["phi_spec_next"].astype(float),
            vc_drop["simulated_success"].astype(float)
        )

    # 2. Does higher future Phi predict being the best candidate per request?
    if len(vc_drop) > 1 and vc_drop["phi_spec_next"].nunique() > 1:
        vc_drop = vc_drop.copy()
        vc_drop["is_best"] = vc_drop.groupby(["seed", "episode", "step"])["phi_spec_next"].transform(lambda x: (x == x.max()).astype(int))
        if vc_drop["is_best"].nunique() > 1:
            correlations["phi_spec_next_vs_is_best"] = _corr(
                vc_drop["phi_spec_next"].astype(float),
                vc_drop["is_best"].astype(float)
            )

    # 3. Policy-level: across all requests, does selected rank / future Phi predict success?
    df_policy_rank = df_policy[df_policy["rank_of_selected_by_phi"].notna()].copy()
    if len(df_policy_rank) > 1 and df_policy_rank["actual_success"].nunique() > 1:
        correlations["rank_by_phi_vs_actual_success"] = _corr(
            df_policy_rank["rank_of_selected_by_phi"].astype(float),
            df_policy_rank["actual_success"].astype(float)
        )
    if len(df_policy_rank) > 1 and df_policy_rank["actual_success"].nunique() > 1:
        correlations["phi_spec_next_of_selected_vs_actual_success"] = _corr(
            df_policy_rank["phi_spec_next_of_selected"].astype(float),
            df_policy_rank["actual_success"].astype(float)
        )

    # 4. Non-leaky proxy features: can current-state features predict true Phi_spec_next?
    proxy_cols = [c for c in df_cand.columns if c.startswith("proxy_")]
    proxy_correlations = {}
    valid_cands_drop = valid_cands.dropna(subset=["phi_spec_next"])
    if len(valid_cands_drop) > 1:
        for col in proxy_cols:
            if valid_cands_drop[col].nunique() > 1 and valid_cands_drop["phi_spec_next"].nunique() > 1:
                proxy_correlations[col] = {
                    "with_phi_spec_next": _corr(
                        valid_cands_drop[col].astype(float),
                        valid_cands_drop["phi_spec_next"].astype(float)
                    ),
                    "with_simulated_success": _corr(
                        valid_cands_drop[col].astype(float),
                        valid_cands_drop["simulated_success"].astype(float)
                    ) if valid_cands_drop["simulated_success"].nunique() > 1 else (None, None),
                }

    # 5. Per-request rank correlation: proxy ranking vs true Phi ranking
    proxy_rank_correlations = {}
    for col in proxy_cols:
        if col not in valid_cands_drop.columns:
            continue
        per_request_rank_corr = []
        for (seed, ep, step), group in valid_cands_drop.groupby(["seed", "episode", "step"]):
            if len(group) < 2:
                continue
            sub = group.dropna(subset=[col, "phi_spec_next"])
            if len(sub) < 2:
                continue
            try:
                if sub[col].nunique() <= 1 or sub["phi_spec_next"].nunique() <= 1:
                    continue
                proxy_rank = sub[col].rank(method="average")
                phi_rank = sub["phi_spec_next"].rank(method="average")
                r, p = stats.spearmanr(proxy_rank, phi_rank)
                if np.isfinite(r):
                    per_request_rank_corr.append(float(r))
            except Exception:
                continue
        if per_request_rank_corr:
            proxy_rank_correlations[col] = {
                "mean_spearman_r": float(np.nanmean(per_request_rank_corr)),
                "std_spearman_r": float(np.nanstd(per_request_rank_corr)),
                "n_requests": len(per_request_rank_corr),
            }

    # Conditional on K_C_valid_before
    k_c_conditional = []
    for k_c_val in sorted(df_policy["k_c_valid_before"].unique()):
        sub = df_policy[df_policy["k_c_valid_before"] == k_c_val]
        sub_non_empty = sub[sub["raw_mask_empty_before"] == False]
        row = {
            "k_c_valid_before": int(k_c_val),
            "count": len(sub),
            "raw_empty_rate": float(sub["raw_mask_empty_before"].mean()),
            "actual_blocking_rate": float((~sub["actual_success"]).mean()),
            "mean_n_simulated": float(sub["n_simulated"].mean()) if len(sub) else None,
        }
        if len(sub_non_empty) > 0:
            row.update({
                "mean_rank_of_selected": float(sub_non_empty["rank_of_selected_by_phi"].mean()),
                "mean_best_phi": float(sub_non_empty["best_phi_spec_next"].mean()),
                "mean_selected_phi": float(sub_non_empty["phi_spec_next_of_selected"].mean()),
            })
        k_c_conditional.append(row)

    report = {
        "config": {
            "topology": args.topology,
            "num_slots": args.num_slots,
            "num_servers": args.num_servers,
            "k_paths": args.k_paths,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "split_profile": args.split_profile,
            "seeds": seeds,
            "episodes": args.episodes,
            "requests_per_episode": args.requests_per_episode,
            "rollout_c_checkpoint": args.rollout_c_checkpoint,
            "checkpoint_label": args.checkpoint_label,
            "agent_r_checkpoint": args.agent_r_checkpoint,
            "alpha": args.alpha,
            "util_threshold": args.util_threshold,
        },
        "total_requests": len(df_policy),
        "total_candidate_evaluations": len(df_cand),
        "reference_policy": {
            "blocking_rate": float((~df_policy["actual_success"]).mean()),
            "raw_mask_empty_rate": float(df_policy["raw_mask_empty_before"].mean()),
        },
        "candidate_impact_summary": candidate_impact_summary,
        "impact_variance": impact_variance,
        "policy_selection_quality": {
            "rank_distribution": rank_dist,
            "mean_rank_of_selected": _safe_mean(df_policy_non_empty["rank_of_selected_by_phi"]),
            "mean_phi_spec_next_selected": _safe_mean(df_policy_non_empty["phi_spec_next_of_selected"]),
            "mean_best_phi_spec_next": _safe_mean(df_policy_non_empty["best_phi_spec_next"]),
        },
        "oracle_summary": oracle_summary,
        "correlations": {k: {"r": v[0], "pvalue": v[1]} for k, v in correlations.items()},
        "proxy_correlations": {
            col: {
                "with_phi_spec_next": {"r": v["with_phi_spec_next"][0], "pvalue": v["with_phi_spec_next"][1]},
                "with_simulated_success": {"r": v["with_simulated_success"][0], "pvalue": v["with_simulated_success"][1]} if v["with_simulated_success"][0] is not None else {"r": None, "pvalue": None},
            }
            for col, v in proxy_correlations.items()
        },
        "proxy_rank_correlations": proxy_rank_correlations,
        "k_c_valid_conditional": k_c_conditional,
        "elapsed_seconds": time.time() - t_start,
    }
    return report


def _build_markdown(report: Dict[str, Any]) -> str:
    def _fmt(x, fmt=".4f"):
        if x is None:
            return "N/A"
        try:
            return f"{x:{fmt}}"
        except Exception:
            return str(x)

    lines: List[str] = []
    lines.append("# Candidate Resource / Spectrum Impact Diagnostic")
    lines.append("")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    cfg = report["config"]
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Topology:** {cfg['topology']}")
    lines.append(f"- **Slots:** {cfg['num_slots']}, **Servers:** {cfg['num_servers']}, **k_paths:** {cfg['k_paths']}")
    lines.append(f"- **max_blocks:** {cfg['max_blocks']}, **block_sort:** {cfg['block_sort_strategy']}")
    lines.append(f"- **split_profile:** {cfg['split_profile']}")
    lines.append(f"- **Seeds:** {cfg['seeds']}, **Episodes/seed:** {cfg['episodes']}, **Requests/episode:** {cfg['requests_per_episode']}")
    lines.append(f"- **Alpha:** {cfg['alpha']}")
    lines.append(f"- **Reference C policy:** `{cfg['rollout_c_checkpoint']}`")
    lines.append("")

    lines.append("## Reference Policy Performance")
    lines.append("")
    rp = report["reference_policy"]
    lines.append(f"- Total requests: **{report['total_requests']}**")
    lines.append(f"- raw_mask_empty_rate: **{_fmt(rp['raw_mask_empty_rate'])}**")
    lines.append(f"- blocking_rate: **{_fmt(rp['blocking_rate'])}**")
    lines.append("")

    lines.append("## Candidate Impact Summary")
    lines.append("")
    cis = report["candidate_impact_summary"]
    lines.append(f"- Total candidate evaluations: {cis['total_candidate_evaluations']}")
    lines.append(f"- Raw-mask-valid evaluations: {cis['raw_mask_valid_evaluations']}")
    lines.append(f"- Simulated successful: {cis['simulated_successful']}, failed: {cis['simulated_failed']}")
    lines.append("")
    lines.append("### Phi_spec_next Distribution")
    lines.append("")
    lines.append("| Group | Mean | Std |")
    lines.append("|---|---:|---:|")
    for grp in ["all_valid", "successful", "failed"]:
        d = cis["phi_spec_next"]
        lines.append(f"| {grp} | {_fmt(d[f'{grp}_mean'])} | {_fmt(d[f'{grp}_std'])} |")
    lines.append("")

    lines.append("### K_C_valid_next Distribution")
    lines.append("")
    lines.append("| Group | Mean |")
    lines.append("|---|---:|")
    for grp in ["all_valid", "successful", "failed"]:
        d = cis["k_c_valid_next"]
        lines.append(f"| {grp} | {_fmt(d[f'{grp}_mean'])} |")
    lines.append("")

    lines.append("## Per-Request Impact Variance")
    lines.append("")
    iv = report["impact_variance"]
    lines.append(f"- Requests with multiple valid candidates: {iv['requests_with_multiple_valid_candidates']}")
    lines.append(f"- Mean Phi_spec range across candidates: **{_fmt(iv['mean_phi_spec_range'])}**")
    lines.append(f"- Std Phi_spec range: {_fmt(iv['std_phi_spec_range'])}")
    lines.append(f"- Mean K_C_valid range: {_fmt(iv['mean_k_c_valid_range'])}")
    lines.append("")

    lines.append("## Policy Selection Quality")
    lines.append("")
    psq = report["policy_selection_quality"]
    lines.append(f"- Mean rank of selected candidate by future Phi: **{_fmt(psq['mean_rank_of_selected'], '.2f')}**")
    lines.append(f"- Mean Phi_spec_next of selected: {_fmt(psq['mean_phi_spec_next_selected'])}")
    lines.append(f"- Mean best Phi_spec_next available: {_fmt(psq['mean_best_phi_spec_next'])}")
    lines.append("")
    lines.append("### Rank Distribution (1 = best future Phi)")
    lines.append("")
    lines.append("| Rank | Count |")
    lines.append("|---|---:|")
    for rank, count in sorted(psq["rank_distribution"].items()):
        lines.append(f"| {rank} | {count} |")
    lines.append("")

    lines.append("## Oracle Analysis")
    lines.append("")
    if report["oracle_summary"]:
        os = report["oracle_summary"]
        lines.append(f"- Requests with valid candidates (oracle denominator): {os['requests_evaluated_for_best_candidate']}")
        lines.append("")
        lines.append("### Oracle Success Rate by Subset")
        lines.append("")
        lines.append("| Subset | Success Rate | Blocking Rate | Notes |")
        lines.append("|---|---:|---:|:---|")
        lines.append(f"| all_requests | {_fmt(os['oracle_success_rate']['all_requests'])} | {_fmt(os['oracle_blocking_rate']['all_requests'])} | Assumes 0 success on raw-empty requests |")
        lines.append(f"| raw_mask_valid_subset | {_fmt(os['oracle_success_rate']['raw_mask_valid_subset'])} | {_fmt(os['oracle_blocking_rate']['raw_mask_valid_subset'])} | True oracle denominator |")
        lines.append(f"| raw_mask_empty_subset | {_fmt(os['oracle_success_rate']['raw_mask_empty_subset'])} | {_fmt(os['oracle_blocking_rate']['raw_mask_empty_subset'])} | Oracle cannot help here |")
        lines.append("")
        lines.append("### Policy Blocking Rate by Subset")
        lines.append("")
        lines.append("| Subset | Blocking Rate |")
        lines.append("|---|---:|")
        lines.append(f"| all_requests | {_fmt(os['policy_blocking_rate_all_requests'])} |")
        lines.append(f"| raw_mask_valid_subset | {_fmt(os['policy_blocking_rate_raw_mask_valid'])} |")
        lines.append(f"| raw_mask_empty_subset | {_fmt(os['policy_blocking_rate_raw_mask_empty'])} |")
        lines.append("")
        lines.append(f"- Oracle no_suitable_block rate (valid subset): {_fmt(os['oracle_no_suitable_block_rate']['raw_mask_valid_subset'])}")
    else:
        lines.append("- No oracle data available.")
    lines.append("")

    lines.append("## Correlations")
    lines.append("")
    lines.append("| Pair | r | p-value |")
    lines.append("|---|---:|---:|")
    for name, vals in report["correlations"].items():
        r = vals['r'] if vals['r'] is not None else float('nan')
        p = vals['pvalue'] if vals['pvalue'] is not None else float('nan')
        lines.append(f"| {name} | {_fmt(r)} | {_fmt(p, '.2e')} |")
    lines.append("")

    lines.append("## Non-Leaky Proxy Feature Correlations")
    lines.append("")
    lines.append("Can current-state-only features predict the true future `Phi_spec_next`?")
    lines.append("")
    lines.append("| Proxy Feature | r vs Phi_next | p-value | r vs Success | p-value | Mean Spearman Rank Corr |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    proxy_rank = report.get("proxy_rank_correlations", {})
    for name, vals in report.get("proxy_correlations", {}).items():
        short = name.replace("proxy_", "")
        r_phi = vals["with_phi_spec_next"]["r"]
        p_phi = vals["with_phi_spec_next"]["pvalue"]
        r_succ = vals["with_simulated_success"]["r"]
        p_succ = vals["with_simulated_success"]["pvalue"]
        rank_info = proxy_rank.get(name, {})
        mean_rank = rank_info.get("mean_spearman_r")
        lines.append(
            f"| {short} | {_fmt(r_phi)} | {_fmt(p_phi, '.2e')} | {_fmt(r_succ)} | "
            f"{_fmt(p_succ, '.2e')} | {_fmt(mean_rank)} |"
        )
    lines.append("")

    lines.append("## Conditional on K_C_valid_before")
    lines.append("")
    lines.append("| K_C_valid | Count | RawEmpty | ActualBlock | Mean N Sim | Mean Rank Sel | Mean Best Phi | Mean Sel Phi |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["k_c_valid_conditional"]:
        lines.append(
            f"| {row['k_c_valid_before']} | {row['count']} | {_fmt(row['raw_empty_rate'])} | "
            f"{_fmt(row['actual_blocking_rate'])} | {_fmt(row['mean_n_simulated'], '.2f')} | "
            f"{_fmt(row.get('mean_rank_of_selected'), '.2f')} | "
            f"{_fmt(row.get('mean_best_phi'))} | "
            f"{_fmt(row.get('mean_selected_phi'))} |"
        )
    lines.append("")

    lines.append("---")
    lines.append(f"*Elapsed: {report['elapsed_seconds']:.1f}s*")
    return "\n".join(lines)


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--checkpoint_label", type=str, default="r_feasibility")
    parser.add_argument("--agent_r_checkpoint", type=str, default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="sa_hmarl/experiments")
    args = parser.parse_args()

    report = _run_diagnostic(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "candidate_resource_impact_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(report), f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    md_path = output_dir / "candidate_resource_impact_diagnostic.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))
    print(f"Markdown saved: {md_path}")

    print(f"\nElapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
