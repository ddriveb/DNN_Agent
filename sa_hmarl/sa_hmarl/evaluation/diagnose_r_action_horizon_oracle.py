"""R-action H-step Oracle diagnostic (phase 1).

Freezes a trained Agent-C split/server policy and an independent PPO-R backend.
For each request with at least two legal R actions, the script:
  1. Aligns time semantics by calling env.advance_time(req.arrival_time) before
     building observations.
  2. Uses Agent-C to pick a split/server.
  3. Enumerates every raw-mask-legal R action.
  4. For each legal R action, restores the pre-step env snapshot, executes that
     R action, and rolls the same env forward for H future requests using the
     frozen Agent-C and PPO-R policies.
  5. Records current and future metrics and defines an Oracle-R-H that ranks
     legal R actions by their H-step consequences.

This is a single-decision, counterfactual H-step oracle, not a recursive oracle.

Outputs:
    JSON: sa_hmarl/experiments/r_action_horizon_oracle_h{H}.json
    Markdown: sa_hmarl/experiments/r_action_horizon_oracle_h{H}.md
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

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RActionRecord:
    seed: int
    episode: int
    request: int
    req_id: int

    split_id: int
    server_id: int

    r_action_idx: int
    path_idx: int
    mod_idx: int
    block_idx: int
    path_length_km: Optional[float]
    modulation: Optional[str]
    block_start: Optional[int]
    block_size: Optional[int]

    current_success: bool
    current_reason: str
    current_fs: Optional[int]
    current_block_waste: Optional[float]

    frag_index_before: float
    frag_index_after: float
    lfb_ratio_before: float
    lfb_ratio_after: float

    k_c_valid_before: int
    k_r_total_before: int
    phi_spec_before: float
    k_c_valid_after: Optional[int]
    k_r_total_after: Optional[int]
    phi_spec_after: Optional[float]

    future_blocked_count: int
    future_raw_mask_empty_count: int
    future_no_suitable_block_count: int
    future_server_overload_count: int
    future_phi_spec_mean: Optional[float]
    future_phi_spec_min: Optional[float]
    future_phi_spec_end: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "request": self.request,
            "req_id": self.req_id,
            "split_id": self.split_id,
            "server_id": self.server_id,
            "r_action_idx": self.r_action_idx,
            "path_idx": self.path_idx,
            "mod_idx": self.mod_idx,
            "block_idx": self.block_idx,
            "path_length_km": self.path_length_km,
            "modulation": self.modulation,
            "block_start": self.block_start,
            "block_size": self.block_size,
            "current_success": self.current_success,
            "current_reason": self.current_reason,
            "current_fs": self.current_fs,
            "current_block_waste": self.current_block_waste,
            "frag_index_before": self.frag_index_before,
            "frag_index_after": self.frag_index_after,
            "lfb_ratio_before": self.lfb_ratio_before,
            "lfb_ratio_after": self.lfb_ratio_after,
            "k_c_valid_before": self.k_c_valid_before,
            "k_r_total_before": self.k_r_total_before,
            "phi_spec_before": self.phi_spec_before,
            "k_c_valid_after": self.k_c_valid_after,
            "k_r_total_after": self.k_r_total_after,
            "phi_spec_after": self.phi_spec_after,
            "future_blocked_count": self.future_blocked_count,
            "future_raw_mask_empty_count": self.future_raw_mask_empty_count,
            "future_no_suitable_block_count": self.future_no_suitable_block_count,
            "future_server_overload_count": self.future_server_overload_count,
            "future_phi_spec_mean": self.future_phi_spec_mean,
            "future_phi_spec_min": self.future_phi_spec_min,
            "future_phi_spec_end": self.future_phi_spec_end,
        }


@dataclass
class StateRecord:
    seed: int
    episode: int
    request: int
    req_id: int

    c_mask_empty: bool
    r_mask_empty: bool
    r_mask_count: int
    multi_action: bool
    mask_empty_but_success: bool

    selected_split: int
    selected_server: int
    ppo_r_action_idx: int
    ppo_r_current_success: bool
    ppo_r_future_blocked: Optional[int]
    ppo_r_future_raw_empty: Optional[int]
    ppo_r_rank_among_successful: Optional[int]

    oracle_r_action_idx: Optional[int]
    oracle_current_success: Optional[bool]
    oracle_future_blocked: Optional[int]
    oracle_future_raw_empty: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "request": self.request,
            "req_id": self.req_id,
            "c_mask_empty": self.c_mask_empty,
            "r_mask_empty": self.r_mask_empty,
            "r_mask_count": self.r_mask_count,
            "multi_action": self.multi_action,
            "mask_empty_but_success": self.mask_empty_but_success,
            "selected_split": self.selected_split,
            "selected_server": self.selected_server,
            "ppo_r_action_idx": self.ppo_r_action_idx,
            "ppo_r_current_success": self.ppo_r_current_success,
            "ppo_r_future_blocked": self.ppo_r_future_blocked,
            "ppo_r_future_raw_empty": self.ppo_r_future_raw_empty,
            "ppo_r_rank_among_successful": self.ppo_r_rank_among_successful,
            "oracle_r_action_idx": self.oracle_r_action_idx,
            "oracle_current_success": self.oracle_current_success,
            "oracle_future_blocked": self.oracle_future_blocked,
            "oracle_future_raw_empty": self.oracle_future_raw_empty,
        }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
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


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    for p in agent.policy_net.parameters():
        p.requires_grad = False
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Spectrum field
# ---------------------------------------------------------------------------
def _phi_spec(k_c_valid: int, k_r_total: int, alpha: float) -> float:
    return float(np.log1p(k_c_valid) + alpha * np.log1p(k_r_total))


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


def _spectrum_stats(env) -> Dict[str, float]:
    stats = env.net.get_global_spectrum_stats()
    return {
        "frag_index": float(stats.get("avg_frag_index", 0.0)),
        "lfb_ratio": float(stats.get("largest_free_block_ratio", 0.0)),
    }


def _snapshot_before_r_decision(env, expected_req_id: int):
    """Copy a state whose event-queue head is still the current request."""
    if not env.event_queue:
        raise RuntimeError("Cannot snapshot an empty event queue")
    queued_req = env.event_queue[0][2]
    if int(queued_req.req_id) != int(expected_req_id):
        raise RuntimeError(
            "R counterfactual snapshot is not pre-decision: "
            f"expected request {expected_req_id}, queue head is {queued_req.req_id}"
        )
    return copy.deepcopy(env)


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------
def _select_c_action_from_obs(
    agent_c: PPOAgentC,
    obs_c: Dict[str, Any],
    num_slots: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Return action, raw mask, risk mask using the trained C policy."""
    features, raw_mask = agent_c.build_action_features(obs_c)
    risk_mask = raw_mask.copy()
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        risk_mask = apply_agent_c_risk_mask(
            obs_c, risk_mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, risk_mask, deterministic=True)
    if action is None:
        action = 0
    return int(action), raw_mask, risk_mask


def _select_r_action_from_obs(
    agent_r: PPOAgentR,
    obs_r: Dict[str, Any],
    max_blocks: int,
) -> Tuple[int, np.ndarray]:
    """Return action and raw R-mask from the frozen PPO-R."""
    raw_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    action = agent_r.select_action(obs_r, deterministic=True)
    if action is None:
        action = 0
    return int(action), raw_mask


# ---------------------------------------------------------------------------
# Future rollout
# ---------------------------------------------------------------------------
def _rollout_future(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    num_servers: int,
    util_threshold: float,
    alpha: float,
) -> Dict[str, Any]:
    """Roll out H future requests from env starting at requests[start_idx]."""
    blocked = 0
    raw_empty = 0
    no_suitable_block = 0
    server_overload = 0
    phi_specs: List[float] = []

    for offset in range(horizon):
        t = start_idx + offset
        if t >= len(requests):
            break
        req = requests[t]
        env.advance_time(req.arrival_time)

        obs_c = build_agent_c_observation(env, req)
        k_c, k_r = _compute_spectrum_field(obs_c, util_threshold)
        phi_specs.append(_phi_spec(k_c, k_r, alpha))

        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if int(raw_c_mask.sum()) == 0:
            raw_empty += 1
            blocked += 1
            env.step((0, 0), (0, 0, 0))
            continue

        action_c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split, server = decode_agent_c_action(action_c_idx, num_servers)

        obs_r = build_agent_r_observation(env, req, split, server)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if int(raw_r_mask.sum()) == 0:
            raw_empty += 1
            blocked += 1
            env.step((split, server), (0, 0, 0))
            continue

        action_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        action_r = decode_agent_r_action(action_r_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split, server), action_r)

        if not info.get("success", False):
            blocked += 1
            reason = info.get("reason", "")
            if reason == "no_suitable_block":
                no_suitable_block += 1
            elif reason == "server_overload":
                server_overload += 1
        # If success, no failure counted.

    result = {
        "blocked": blocked,
        "raw_mask_empty": raw_empty,
        "no_suitable_block": no_suitable_block,
        "server_overload": server_overload,
        "phi_spec_mean": float(np.mean(phi_specs)) if phi_specs else None,
        "phi_spec_min": float(np.min(phi_specs)) if phi_specs else None,
    }
    # End Phi_spec: use the last computed pre-request value if any.
    result["phi_spec_end"] = phi_specs[-1] if phi_specs else None
    return result


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------
def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg

    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading PPO-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    # Snapshot parameters to verify they are unchanged.
    c_state_ref = {k: v.clone() for k, v in agent_c.policy_net.state_dict().items()}
    r_state_ref = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}

    # Generate all requests up front.
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

    action_records: List[RActionRecord] = []
    state_records: List[StateRecord] = []
    t_start = time.time()

    for seed in sorted(all_episodes.keys()):
        for ep_idx, requests in enumerate(all_episodes[seed]):
            env = make_env(
                topology=args.topology,
                num_slots=args.num_slots,
                num_servers=args.num_servers,
                seed=42,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)

            for step_idx, req in enumerate(requests):
                env.advance_time(req.arrival_time)

                obs_c = build_agent_c_observation(env, req)
                k_c_before, k_r_before = _compute_spectrum_field(obs_c, args.util_threshold)
                phi_before = _phi_spec(k_c_before, k_r_before, args.alpha)

                raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                c_mask_empty = int(raw_c_mask.sum()) == 0

                # Agent-C split/server selection (risk mask as at deployment).
                action_c_idx, raw_c_mask_ret, _ = _select_c_action_from_obs(
                    agent_c, obs_c, env.net.num_slots
                )
                selected_split, selected_server = decode_agent_c_action(
                    action_c_idx, args.num_servers
                )

                obs_r = build_agent_r_observation(env, req, selected_split, selected_server)
                raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
                r_mask_count = int(raw_r_mask.sum())
                r_mask_empty = r_mask_count == 0
                multi_action = r_mask_count >= 2

                # Baseline PPO-R action and its current outcome.
                ppo_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

                # Every counterfactual must start before the current R decision.
                # The event queue still contains the current request here.
                snapshot = _snapshot_before_r_decision(env, req.req_id)

                # Step the real environment with the PPO-R action to advance time.
                ppo_action_r = decode_agent_r_action(
                    ppo_r_idx, len(obs_r["mod_names"]), env.max_blocks
                )
                _, _, _, ppo_info = env.step(
                    (selected_split, selected_server), ppo_action_r
                )
                ppo_current_success = bool(ppo_info.get("success", False))
                ppo_current_reason = ppo_info.get("reason", "unknown")

                mask_empty_but_success = (c_mask_empty or r_mask_empty) and ppo_current_success

                state_rec = StateRecord(
                    seed=seed,
                    episode=ep_idx,
                    request=step_idx,
                    req_id=int(req.req_id),
                    c_mask_empty=c_mask_empty,
                    r_mask_empty=r_mask_empty,
                    r_mask_count=r_mask_count,
                    multi_action=multi_action,
                    mask_empty_but_success=mask_empty_but_success,
                    selected_split=selected_split,
                    selected_server=selected_server,
                    ppo_r_action_idx=ppo_r_idx,
                    ppo_r_current_success=ppo_current_success,
                    ppo_r_future_blocked=None,
                    ppo_r_future_raw_empty=None,
                    ppo_r_rank_among_successful=None,
                    oracle_r_action_idx=None,
                    oracle_current_success=None,
                    oracle_future_blocked=None,
                    oracle_future_raw_empty=None,
                )

                if not multi_action:
                    state_records.append(state_rec)
                    continue

                legal_r_indices = np.where(raw_r_mask)[0].tolist()
                step_action_records: List[RActionRecord] = []

                for r_idx in legal_r_indices:
                    env_r = copy.deepcopy(snapshot)
                    stats_before = _spectrum_stats(env_r)

                    action_r = decode_agent_r_action(
                        r_idx, len(obs_r["mod_names"]), env_r.max_blocks
                    )
                    _, _, _, info = env_r.step(
                        (selected_split, selected_server), action_r
                    )
                    success = bool(info.get("success", False))
                    reason = info.get("reason", "unknown")

                    stats_after = _spectrum_stats(env_r)

                    next_req = requests[step_idx + 1] if step_idx + 1 < len(requests) else None
                    if next_req is not None:
                        # Match deployment semantics: release resources expiring
                        # before the next arrival, then build its observation.
                        env_r.advance_time(next_req.arrival_time)
                        obs_c_after = build_agent_c_observation(env_r, next_req)
                        k_c_after, k_r_after = _compute_spectrum_field(
                            obs_c_after, args.util_threshold
                        )
                        phi_after = _phi_spec(k_c_after, k_r_after, args.alpha)
                    else:
                        k_c_after = None
                        k_r_after = None
                        phi_after = None

                    future = _rollout_future(
                        env_r, requests, step_idx + 1, args.horizon,
                        agent_c, agent_r, args.num_servers,
                        args.util_threshold, args.alpha,
                    )

                    path_idx, mod_idx, block_idx = action_r
                    path = obs_r["candidate_paths"][path_idx] if path_idx < len(obs_r["candidate_paths"]) else None
                    path_length_km = env_r.net.path_length_km(path) if path is not None else None
                    modulation = obs_r["mod_names"][mod_idx] if mod_idx < len(obs_r["mod_names"]) else None
                    block_tuples = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
                    block_start, block_size = (
                        block_tuples[block_idx] if block_idx < len(block_tuples) else (None, None)
                    )

                    rec = RActionRecord(
                        seed=seed,
                        episode=ep_idx,
                        request=step_idx,
                        req_id=int(req.req_id),
                        split_id=selected_split,
                        server_id=selected_server,
                        r_action_idx=r_idx,
                        path_idx=path_idx,
                        mod_idx=mod_idx,
                        block_idx=block_idx,
                        path_length_km=path_length_km,
                        modulation=modulation,
                        block_start=block_start,
                        block_size=block_size,
                        current_success=success,
                        current_reason=reason,
                        current_fs=info.get("num_slots"),
                        current_block_waste=info.get("block_waste"),
                        frag_index_before=stats_before["frag_index"],
                        frag_index_after=stats_after["frag_index"],
                        lfb_ratio_before=stats_before["lfb_ratio"],
                        lfb_ratio_after=stats_after["lfb_ratio"],
                        k_c_valid_before=k_c_before,
                        k_r_total_before=k_r_before,
                        phi_spec_before=phi_before,
                        k_c_valid_after=k_c_after,
                        k_r_total_after=k_r_after,
                        phi_spec_after=phi_after,
                        future_blocked_count=future["blocked"],
                        future_raw_mask_empty_count=future["raw_mask_empty"],
                        future_no_suitable_block_count=future["no_suitable_block"],
                        future_server_overload_count=future["server_overload"],
                        future_phi_spec_mean=future["phi_spec_mean"],
                        future_phi_spec_min=future["phi_spec_min"],
                        future_phi_spec_end=future["phi_spec_end"],
                    )
                    step_action_records.append(rec)

                action_records.extend(step_action_records)

                # Oracle: rank only current-successful legal R actions.
                successful_recs = [r for r in step_action_records if r.current_success]
                if successful_recs:
                    oracle_sorted = sorted(
                        successful_recs,
                        key=lambda r: (
                            r.future_blocked_count,
                            r.future_raw_mask_empty_count,
                            r.future_no_suitable_block_count,
                            r.current_block_waste if r.current_block_waste is not None else float('inf'),
                            -(r.future_phi_spec_mean if r.future_phi_spec_mean is not None else -float('inf')),
                        ),
                    )
                    oracle_rec = oracle_sorted[0]
                    state_rec.oracle_r_action_idx = oracle_rec.r_action_idx
                    state_rec.oracle_current_success = oracle_rec.current_success
                    state_rec.oracle_future_blocked = oracle_rec.future_blocked_count
                    state_rec.oracle_future_raw_empty = oracle_rec.future_raw_mask_empty_count

                    # Rank of PPO-R action among successful oracle-sorted actions.
                    ppo_rec = next(
                        (r for r in oracle_sorted if r.r_action_idx == ppo_r_idx), None
                    )
                    if ppo_rec is not None:
                        state_rec.ppo_r_rank_among_successful = (
                            oracle_sorted.index(ppo_rec) + 1
                        )
                        state_rec.ppo_r_future_blocked = ppo_rec.future_blocked_count
                        state_rec.ppo_r_future_raw_empty = ppo_rec.future_raw_mask_empty_count

                state_records.append(state_rec)

    # -----------------------------------------------------------------------
    # Aggregate summaries
    # -----------------------------------------------------------------------
    total_requests = len(state_records)
    r_mask_non_empty = sum(1 for s in state_records if not s.r_mask_empty)
    multi_action_states = [s for s in state_records if s.multi_action]
    multi_action_count = len(multi_action_states)
    mask_empty_but_success_count = sum(s.mask_empty_but_success for s in state_records)

    multi_with_diff_block = 0
    multi_with_diff_raw_empty = 0
    for s in multi_action_states:
        recs = [r for r in action_records
                if r.seed == s.seed and r.episode == s.episode and r.request == s.request]
        if len(set(r.future_blocked_count for r in recs)) > 1:
            multi_with_diff_block += 1
        if len(set(r.future_raw_mask_empty_count for r in recs)) > 1:
            multi_with_diff_raw_empty += 1

    # Oracle denominator: multi-action states where at least one current-successful
    # legal R action exists.
    oracle_states = [s for s in multi_action_states if s.oracle_r_action_idx is not None]
    oracle_state_count = len(oracle_states)
    # Common denominator for PPO-vs-Oracle comparison: states where PPO-R's chosen
    # action is also current-successful, so both policies are choosing from the
    # same feasible set.
    common_states = [s for s in oracle_states if s.ppo_r_rank_among_successful is not None]
    common_state_count = len(common_states)

    def _mean(values):
        return float(np.mean(values)) if values else None

    ppo_future_blocked_rates = [
        s.ppo_r_future_blocked / args.horizon
        for s in common_states
        if s.ppo_r_future_blocked is not None
    ]
    oracle_future_blocked_rates = [
        s.oracle_future_blocked / args.horizon
        for s in common_states
        if s.oracle_future_blocked is not None
    ]

    ppo_mean_block_rate = _mean(ppo_future_blocked_rates)
    oracle_mean_block_rate = _mean(oracle_future_blocked_rates)
    if ppo_mean_block_rate is not None and oracle_mean_block_rate is not None:
        oracle_headroom_pp = (ppo_mean_block_rate - oracle_mean_block_rate) * 100.0
    else:
        oracle_headroom_pp = None

    ppo_rank_values = [
        s.ppo_r_rank_among_successful
        for s in common_states
        if s.ppo_r_rank_among_successful is not None
    ]
    mean_ppo_rank = _mean(ppo_rank_values)

    # Per-seed summaries
    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        seed_states = [s for s in state_records if s.seed == seed]
        seed_multi = [s for s in seed_states if s.multi_action]
        seed_oracle = [s for s in seed_multi if s.oracle_r_action_idx is not None]
        seed_common = [s for s in seed_oracle if s.ppo_r_rank_among_successful is not None]
        ppo_rates = [s.ppo_r_future_blocked / args.horizon for s in seed_common if s.ppo_r_future_blocked is not None]
        ora_rates = [s.oracle_future_blocked / args.horizon for s in seed_common if s.oracle_future_blocked is not None]
        headroom = (_mean(ppo_rates) - _mean(ora_rates)) * 100.0 if ppo_rates and ora_rates else None
        per_seed[seed] = {
            "total_requests": len(seed_states),
            "r_mask_non_empty": sum(1 for s in seed_states if not s.r_mask_empty),
            "multi_action_count": len(seed_multi),
            "multi_action_rate": len(seed_multi) / max(len(seed_states), 1),
            "oracle_evaluable_count": len(seed_oracle),
            "common_comparison_count": len(seed_common),
            "ppo_mean_future_block_rate": _mean(ppo_rates),
            "oracle_mean_future_block_rate": _mean(ora_rates),
            "oracle_headroom_pp": headroom,
            "mask_empty_but_success": sum(s.mask_empty_but_success for s in seed_states),
        }

    report = {
        "config": {
            "agent_c_checkpoint": args.agent_c_checkpoint,
            "agent_r_checkpoint": args.agent_r_checkpoint,
            "topology": args.topology,
            "num_slots": args.num_slots,
            "num_servers": args.num_servers,
            "k_paths": args.k_paths,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "split_profile": args.split_profile,
            "num_splits": args.num_splits,
            "seeds": seeds,
            "episodes": args.episodes,
            "requests_per_episode": args.requests_per_episode,
            "horizon": args.horizon,
            "alpha": args.alpha,
            "util_threshold": args.util_threshold,
        },
        "summary": {
            "total_requests": total_requests,
            "r_mask_non_empty": r_mask_non_empty,
            "r_mask_empty": total_requests - r_mask_non_empty,
            "multi_action_count": multi_action_count,
            "multi_action_rate": multi_action_count / max(total_requests, 1),
            "multi_action_with_diff_future_blocking": multi_with_diff_block,
            "multi_action_with_diff_future_blocking_rate": multi_with_diff_block / max(multi_action_count, 1),
            "multi_action_with_diff_future_raw_empty": multi_with_diff_raw_empty,
            "multi_action_with_diff_future_raw_empty_rate": multi_with_diff_raw_empty / max(multi_action_count, 1),
            "oracle_evaluable_count": oracle_state_count,
            "common_comparison_count": common_state_count,
            "ppo_mean_future_block_rate": ppo_mean_block_rate,
            "oracle_mean_future_block_rate": oracle_mean_block_rate,
            "oracle_headroom_pp": oracle_headroom_pp,
            "mean_ppo_rank_among_successful": mean_ppo_rank,
            "mask_empty_but_success_count": mask_empty_but_success_count,
        },
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "state_records": [s.to_dict() for s in state_records],
        "action_records": [r.to_dict() for r in action_records],
        "elapsed_seconds": time.time() - t_start,
        "frozen_params_unchanged": {
            "agent_c": all(
                torch.equal(agent_c.policy_net.state_dict()[k], c_state_ref[k])
                for k in c_state_ref
            ),
            "agent_r": all(
                torch.equal(agent_r.policy_net.state_dict()[k], r_state_ref[k])
                for k in r_state_ref
            ),
        },
    }
    return report


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _build_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# R-Action H-Step Oracle Diagnostic",
        "",
        f"**Agent-C checkpoint:** `{report['config']['agent_c_checkpoint']}`",
        f"**PPO-R checkpoint:** `{report['config']['agent_r_checkpoint']}`",
        f"**Horizon H:** {report['config']['horizon']}",
        "",
        "## Setup",
        "",
        f"- Topology: {report['config']['topology']}",
        f"- Seeds: {report['config']['seeds']}",
        f"- Episodes/seed: {report['config']['episodes']}",
        f"- Requests/episode: {report['config']['requests_per_episode']}",
        f"- num_slots: {report['config']['num_slots']}",
        f"- num_servers: {report['config']['num_servers']}",
        f"- k_paths: {report['config']['k_paths']}",
        f"- max_blocks: {report['config']['max_blocks']}",
        "",
        "## Aggregate counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total requests | {s['total_requests']} |",
        f"| R-mask non-empty | {s['r_mask_non_empty']} |",
        f"| R-mask empty | {s['r_mask_empty']} |",
        f"| Multi-action (>=2 legal R) states | {s['multi_action_count']} ({s['multi_action_rate']:.2%}) |",
        f"| Mask-empty-but-success | {s['mask_empty_but_success_count']} |",
        "",
        "## Counterfactual impact variance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| States where R actions differ in future blocking | {s['multi_action_with_diff_future_blocking']} ({s['multi_action_with_diff_future_blocking_rate']:.2%}) |",
        f"| States where R actions differ in future raw-mask-empty | {s['multi_action_with_diff_future_raw_empty']} ({s['multi_action_with_diff_future_raw_empty_rate']:.2%}) |",
        "",
        "## Oracle comparison (per future request)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Oracle-evaluable states | {s['oracle_evaluable_count']} |",
        f"| Common comparison states (PPO action successful) | {s['common_comparison_count']} |",
        f"| PPO-R mean future block rate | {s['ppo_mean_future_block_rate']:.4f} |" if s['ppo_mean_future_block_rate'] is not None else "| PPO-R mean future block rate | N/A |",
        f"| Oracle-R-H mean future block rate | {s['oracle_mean_future_block_rate']:.4f} |" if s['oracle_mean_future_block_rate'] is not None else "| Oracle-R-H mean future block rate | N/A |",
        f"| Oracle headroom (pp) | {s['oracle_headroom_pp']:.3f} |" if s['oracle_headroom_pp'] is not None else "| Oracle headroom (pp) | N/A |",
        f"| Mean PPO-R rank among successful actions | {s['mean_ppo_rank_among_successful']:.2f} |" if s['mean_ppo_rank_among_successful'] is not None else "| Mean PPO-R rank among successful actions | N/A |",
        "",
        "## Per-seed results",
        "",
        "| Seed | Total | R non-empty | Multi-action | Oracle eval | Common | PPO block rate | Oracle block rate | Headroom (pp) | Mask-empty-success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, v in report["per_seed"].items():
        lines.append(
            f"| {seed} | {v['total_requests']} | {v['r_mask_non_empty']} | "
            f"{v['multi_action_count']} | {v['oracle_evaluable_count']} | {v.get('common_comparison_count', 'N/A')} | "
            f"{_fmt(v['ppo_mean_future_block_rate'])} | {_fmt(v['oracle_mean_future_block_rate'])} | "
            f"{_fmt(v['oracle_headroom_pp'])} | {v['mask_empty_but_success']} |"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
    ])
    proceed = True
    reasons = []
    if s["multi_action_rate"] < 0.10:
        proceed = False
        reasons.append(f"multi-action rate {s['multi_action_rate']:.2%} < 10%")
    if s["multi_action_with_diff_future_blocking_rate"] < 0.10:
        proceed = False
        reasons.append(f"future-blocking-difference rate {s['multi_action_with_diff_future_blocking_rate']:.2%} < 10%")
    if s["oracle_headroom_pp"] is None or s["oracle_headroom_pp"] < 1.0:
        proceed = False
        reasons.append(f"Oracle headroom {s['oracle_headroom_pp']} < 1 pp")
    if proceed:
        lines.append("**PROCEED_TO_POTENTIAL_DIAGNOSTIC**: multi-action states are common, R actions differ in future consequences, and Oracle-R-H provides ≥1 pp headroom.")
    else:
        lines.append("**STOP**: " + "; ".join(reasons))
    lines.append("")
    lines.append(f"Elapsed: {report['elapsed_seconds']:.1f}s")
    return "\n".join(lines)


def _fmt(x):
    if x is None:
        return "N/A"
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=3)
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
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--save_npz", type=str, default=None)
    args = parser.parse_args()

    report = _run_diagnostic(args)

    out_json = Path(args.output_json or f"sa_hmarl/experiments/r_action_horizon_oracle_h{args.horizon}.json")
    out_md = Path(args.output_md or f"sa_hmarl/experiments/r_action_horizon_oracle_h{args.horizon}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(_build_markdown(report))
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
    print(f"Oracle headroom (pp): {report['summary']['oracle_headroom_pp']}")

    if args.save_npz:
        try:
            import numpy as np
            np.savez_compressed(
                args.save_npz,
                state_records=np.array(report["state_records"], dtype=object),
                action_records=np.array(report["action_records"], dtype=object),
            )
            print(f"Saved NPZ {args.save_npz}")
        except Exception as e:
            print(f"NPZ save skipped: {e}")


if __name__ == "__main__":
    main()
