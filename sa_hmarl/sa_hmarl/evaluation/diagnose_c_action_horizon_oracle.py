"""C-action H-step Oracle diagnostic.

This diagnostic freezes Agent-C and an independent PPO-R. At each decision
state it enumerates every raw-mask-legal C action, lets the same PPO-R choose
the current R action, and rolls H future requests with the frozen C+R pair.
The oracle is counterfactual and single-decision: future C actions are not
chosen recursively by the oracle.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _rollout_future,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
    _spectrum_stats,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


@dataclass
class CActionRecord:
    seed: int
    episode: int
    request: int
    req_id: int
    c_action_idx: int
    split_id: int
    server_id: int
    r_action_idx: int
    current_success: bool
    current_reason: str
    current_delay_ms: Optional[float]
    current_fs: Optional[int]
    current_block_waste: Optional[float]
    k_c_valid_before: int
    k_r_total_before: int
    phi_spec_before: float
    frag_index_before: float
    frag_index_after: float
    lfb_ratio_before: float
    lfb_ratio_after: float
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
    demand_potential_after: Optional[float] = None


@dataclass
class CStateRecord:
    seed: int
    episode: int
    request: int
    req_id: int
    active_connections_before: int
    raw_c_count: int
    multi_action: bool
    full_horizon: bool
    mask_empty_but_success: bool
    ppo_c_action_idx: int
    ppo_c_current_success: bool
    ppo_c_future_blocked: Optional[int] = None
    ppo_c_future_raw_empty: Optional[int] = None
    ppo_c_rank_among_successful: Optional[int] = None
    oracle_c_action_idx: Optional[int] = None
    oracle_current_success: Optional[bool] = None
    oracle_future_blocked: Optional[int] = None
    oracle_future_raw_empty: Optional[int] = None


def _snapshot_before_c_decision(env, expected_req_id: int):
    """Copy an aligned state whose queue head is the current request."""
    if not env.event_queue:
        raise RuntimeError("Cannot snapshot an empty event queue")
    queued_req = env.event_queue[0][2]
    if int(queued_req.req_id) != int(expected_req_id):
        raise RuntimeError(
            "C counterfactual snapshot is not pre-decision: "
            f"expected request {expected_req_id}, queue head is {queued_req.req_id}"
        )
    return copy.deepcopy(env)


def _execute_c_with_frozen_r(env, req, c_action_idx: int, agent_r, num_servers: int):
    """Execute one C candidate using deterministic frozen PPO-R."""
    split_id, server_id = decode_agent_c_action(c_action_idx, num_servers)
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    r_action_idx, raw_r_mask = _select_r_action_from_obs(
        agent_r, obs_r, env.max_blocks
    )
    if int(raw_r_mask.sum()) == 0:
        r_action = (0, 0, 0)
    else:
        r_action = decode_agent_r_action(
            r_action_idx, len(obs_r["mod_names"]), env.max_blocks
        )
    _, _, _, info = env.step((split_id, server_id), r_action)
    return r_action_idx, info


def _oracle_sort_key(rec: CActionRecord):
    """Lexicographic long-horizon objective; lower is better."""
    return (
        rec.future_blocked_count,
        rec.future_raw_mask_empty_count,
        rec.future_no_suitable_block_count,
        rec.future_server_overload_count,
        -(rec.future_phi_spec_mean if rec.future_phi_spec_mean is not None else -np.inf),
        rec.current_block_waste if rec.current_block_waste is not None else np.inf,
        rec.current_delay_ms if rec.current_delay_ms is not None else np.inf,
    )


def _demand_aware_potential(
    env,
    request_history: List[Any],
    window: int,
    probe_limit: int,
    util_threshold: float,
    alpha: float,
) -> Optional[float]:
    """Estimate Phi(X, mu_t) using only current/past requests as probes."""
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
        k_q, n_q = _compute_spectrum_field(obs_probe, util_threshold)
        values.append(_phi_spec(k_q, n_q, alpha))
    return float(np.mean(values)) if values else None


def _make_env(args):
    return make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )


def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_proto = _make_env(args)
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    c_ref = {k: v.clone() for k, v in agent_c.policy_net.state_dict().items()}
    r_ref = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}

    episodes_by_seed: Dict[int, List[List[Any]]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            episodes.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
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
            ))
        episodes_by_seed[seed] = episodes

    action_records: List[CActionRecord] = []
    state_records: List[CStateRecord] = []
    started = time.time()

    for seed in seeds:
        for ep_idx, requests in enumerate(episodes_by_seed[seed]):
            env = _make_env(args)
            env.reset(requests)
            for step_idx, req in enumerate(requests):
                # Release first, then build masks: this is the deployment state.
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                legal_c = np.flatnonzero(raw_c_mask).tolist()
                raw_c_count = len(legal_c)
                full_horizon = step_idx + args.horizon < len(requests)
                k_c_before, k_r_before = _compute_spectrum_field(
                    obs_c, args.util_threshold
                )
                phi_before = _phi_spec(k_c_before, k_r_before, args.alpha)
                stats_before = _spectrum_stats(env)

                ppo_c_idx, _, _ = _select_c_action_from_obs(
                    agent_c, obs_c, env.net.num_slots
                )
                snapshot = _snapshot_before_c_decision(env, req.req_id)

                # Advance the reference trajectory with the deployed C+R pair.
                _, ppo_info = _execute_c_with_frozen_r(
                    env, req, ppo_c_idx, agent_r, args.num_servers
                )
                ppo_success = bool(ppo_info.get("success", False))
                state = CStateRecord(
                    seed=seed,
                    episode=ep_idx,
                    request=step_idx,
                    req_id=int(req.req_id),
                    active_connections_before=len(env.active_connections),
                    raw_c_count=raw_c_count,
                    multi_action=raw_c_count >= 2,
                    full_horizon=full_horizon,
                    mask_empty_but_success=raw_c_count == 0 and ppo_success,
                    ppo_c_action_idx=ppo_c_idx,
                    ppo_c_current_success=ppo_success,
                )

                # A complete H-step window is required for an unbiased rate.
                if raw_c_count < 2 or not full_horizon:
                    state_records.append(state)
                    continue

                step_records: List[CActionRecord] = []
                request_history = requests[:step_idx + 1]
                for c_idx in legal_c:
                    env_c = copy.deepcopy(snapshot)
                    r_idx, info = _execute_c_with_frozen_r(
                        env_c, req, int(c_idx), agent_r, args.num_servers
                    )
                    success = bool(info.get("success", False))
                    stats_after = _spectrum_stats(env_c)

                    next_req = requests[step_idx + 1]
                    probe_env = copy.deepcopy(env_c)
                    probe_env.advance_time(next_req.arrival_time)
                    obs_next = build_agent_c_observation(probe_env, next_req)
                    k_c_after, k_r_after = _compute_spectrum_field(
                        obs_next, args.util_threshold
                    )
                    phi_after = _phi_spec(k_c_after, k_r_after, args.alpha)

                    demand_potential_after = _demand_aware_potential(
                        env_c,
                        request_history,
                        args.potential_history_window,
                        args.potential_probe_limit,
                        args.util_threshold,
                        args.alpha,
                    )
                    future = _rollout_future(
                        env_c, requests, step_idx + 1, args.horizon,
                        agent_c, agent_r, args.num_servers,
                        args.util_threshold, args.alpha,
                    )
                    split_id, server_id = decode_agent_c_action(
                        int(c_idx), args.num_servers
                    )
                    step_records.append(CActionRecord(
                        seed=seed,
                        episode=ep_idx,
                        request=step_idx,
                        req_id=int(req.req_id),
                        c_action_idx=int(c_idx),
                        split_id=split_id,
                        server_id=server_id,
                        r_action_idx=r_idx,
                        current_success=success,
                        current_reason=info.get("reason", "unknown"),
                        current_delay_ms=info.get("delay_ms"),
                        current_fs=info.get("num_slots"),
                        current_block_waste=info.get("block_waste"),
                        k_c_valid_before=k_c_before,
                        k_r_total_before=k_r_before,
                        phi_spec_before=phi_before,
                        frag_index_before=stats_before["frag_index"],
                        frag_index_after=stats_after["frag_index"],
                        lfb_ratio_before=stats_before["lfb_ratio"],
                        lfb_ratio_after=stats_after["lfb_ratio"],
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
                        demand_potential_after=demand_potential_after,
                    ))

                action_records.extend(step_records)
                successful = [r for r in step_records if r.current_success]
                if successful:
                    ranked = sorted(successful, key=_oracle_sort_key)
                    oracle = ranked[0]
                    state.oracle_c_action_idx = oracle.c_action_idx
                    state.oracle_current_success = True
                    state.oracle_future_blocked = oracle.future_blocked_count
                    state.oracle_future_raw_empty = oracle.future_raw_mask_empty_count
                    ppo_rec = next(
                        (r for r in ranked if r.c_action_idx == ppo_c_idx), None
                    )
                    if ppo_rec is not None:
                        state.ppo_c_rank_among_successful = ranked.index(ppo_rec) + 1
                        state.ppo_c_future_blocked = ppo_rec.future_blocked_count
                        state.ppo_c_future_raw_empty = ppo_rec.future_raw_mask_empty_count
                state_records.append(state)

    def mean(values):
        return float(np.mean(values)) if values else None

    evaluable_states = [
        s for s in state_records if s.multi_action and s.full_horizon
    ]
    oracle_states = [s for s in evaluable_states if s.oracle_c_action_idx is not None]
    common_states = [
        s for s in oracle_states if s.ppo_c_rank_among_successful is not None
    ]

    def records_for(state):
        return [r for r in action_records if (
            r.seed == state.seed and r.episode == state.episode
            and r.request == state.request
        )]

    diff_block = sum(
        len({r.future_blocked_count for r in records_for(s)}) > 1
        for s in evaluable_states
    )
    diff_raw = sum(
        len({r.future_raw_mask_empty_count for r in records_for(s)}) > 1
        for s in evaluable_states
    )

    def summarize(states):
        ppo_rates = [s.ppo_c_future_blocked / args.horizon for s in states]
        oracle_rates = [s.oracle_future_blocked / args.horizon for s in states]
        ppo_rate = mean(ppo_rates)
        oracle_rate = mean(oracle_rates)
        return {
            "common_comparison_count": len(states),
            "ppo_mean_future_block_rate": ppo_rate,
            "oracle_mean_future_block_rate": oracle_rate,
            "oracle_headroom_pp": (
                (ppo_rate - oracle_rate) * 100.0
                if ppo_rate is not None and oracle_rate is not None else None
            ),
            "mean_ppo_rank_among_successful": mean([
                s.ppo_c_rank_among_successful for s in states
            ]),
        }

    summary = summarize(common_states)
    summary.update({
        "total_requests": len(state_records),
        "raw_c_non_empty": sum(s.raw_c_count > 0 for s in state_records),
        "raw_c_empty": sum(s.raw_c_count == 0 for s in state_records),
        "multi_action_count": sum(s.multi_action for s in state_records),
        "multi_action_rate": sum(s.multi_action for s in state_records) / max(len(state_records), 1),
        "full_horizon_multi_action_count": len(evaluable_states),
        "oracle_evaluable_count": len(oracle_states),
        "multi_action_with_diff_future_blocking": diff_block,
        "multi_action_with_diff_future_blocking_rate": diff_block / max(len(evaluable_states), 1),
        "multi_action_with_diff_future_raw_empty": diff_raw,
        "multi_action_with_diff_future_raw_empty_rate": diff_raw / max(len(evaluable_states), 1),
        "mask_empty_but_success_count": sum(s.mask_empty_but_success for s in state_records),
        "mean_active_connections_before": mean([
            s.active_connections_before for s in state_records
        ]),
        "p50_active_connections_before": float(np.percentile(
            [s.active_connections_before for s in state_records], 50
        )),
        "p95_active_connections_before": float(np.percentile(
            [s.active_connections_before for s in state_records], 95
        )),
        "max_active_connections_before": max(
            (s.active_connections_before for s in state_records), default=0
        ),
    })

    if args.potential_probe_limit > 0:
        potential_states = []
        potential_diff_states = []
        pair_correct = 0
        pair_total = 0
        for state in common_states:
            recs = [r for r in records_for(state) if (
                r.current_success and r.demand_potential_after is not None
            )]
            if not recs:
                continue
            selected = max(
                recs, key=lambda r: (r.demand_potential_after, -r.c_action_idx)
            )
            best_block = min(r.future_blocked_count for r in recs)
            potential_states.append((state, selected, best_block))
            if len({r.future_blocked_count for r in recs}) > 1:
                potential_diff_states.append((state, selected, best_block))
            for i, left in enumerate(recs):
                for right in recs[i + 1:]:
                    if left.future_blocked_count == right.future_blocked_count:
                        continue
                    pair_total += 1
                    better = left if left.future_blocked_count < right.future_blocked_count else right
                    worse = right if better is left else left
                    if better.demand_potential_after > worse.demand_potential_after:
                        pair_correct += 1

        def potential_metrics(rows):
            if not rows:
                return {
                    "count": 0,
                    "ppo_mean_future_block_rate": None,
                    "potential_mean_future_block_rate": None,
                    "oracle_mean_future_block_rate": None,
                    "potential_vs_ppo_improvement_pp": None,
                    "potential_oracle_gap_pp": None,
                    "potential_oracle_optimal_rate": None,
                }
            ppo_rate = mean([
                state.ppo_c_future_blocked / args.horizon
                for state, _, _ in rows
            ])
            potential_rate = mean([
                selected.future_blocked_count / args.horizon
                for _, selected, _ in rows
            ])
            oracle_rate = mean([best / args.horizon for _, _, best in rows])
            return {
                "count": len(rows),
                "ppo_mean_future_block_rate": ppo_rate,
                "potential_mean_future_block_rate": potential_rate,
                "oracle_mean_future_block_rate": oracle_rate,
                "potential_vs_ppo_improvement_pp": (
                    (ppo_rate - potential_rate) * 100.0
                ),
                "potential_oracle_gap_pp": (potential_rate - oracle_rate) * 100.0,
                "potential_oracle_optimal_rate": mean([
                    float(selected.future_blocked_count == best)
                    for _, selected, best in rows
                ]),
            }

        summary["demand_potential"] = {
            "history_window": args.potential_history_window,
            "probe_limit": args.potential_probe_limit,
            "all_common_states": potential_metrics(potential_states),
            "candidate_different_states": potential_metrics(potential_diff_states),
            "pairwise_blocking_order_accuracy": (
                pair_correct / pair_total if pair_total else None
            ),
            "pairwise_comparison_count": pair_total,
        }

    per_seed = {}
    for seed in seeds:
        seed_all = [s for s in state_records if s.seed == seed]
        seed_common = [s for s in common_states if s.seed == seed]
        seed_summary = summarize(seed_common)
        seed_summary.update({
            "total_requests": len(seed_all),
            "multi_action_count": sum(s.multi_action for s in seed_all),
            "raw_c_empty": sum(s.raw_c_count == 0 for s in seed_all),
            "mean_active_connections_before": mean([
                s.active_connections_before for s in seed_all
            ]),
        })
        per_seed[str(seed)] = seed_summary

    return {
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
            "potential_history_window": args.potential_history_window,
            "potential_probe_limit": args.potential_probe_limit,
            "oracle_semantics": "single-decision counterfactual; frozen C+R future rollout",
        },
        "summary": summary,
        "per_seed": per_seed,
        "state_records": [asdict(s) for s in state_records],
        "action_records": [asdict(r) for r in action_records],
        "elapsed_seconds": time.time() - started,
        "frozen_params_unchanged": {
            "agent_c": all(torch.equal(agent_c.policy_net.state_dict()[k], v) for k, v in c_ref.items()),
            "agent_r": all(torch.equal(agent_r.policy_net.state_dict()[k], v) for k, v in r_ref.items()),
        },
    }


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def _build_markdown(report: Dict[str, Any]) -> str:
    c, s = report["config"], report["summary"]
    proceed = (
        s["multi_action_rate"] >= 0.10
        and s["multi_action_with_diff_future_blocking_rate"] >= 0.10
        and s["oracle_headroom_pp"] is not None
        and s["oracle_headroom_pp"] >= 1.0
    )
    potential_proceed = False
    if "demand_potential" in s:
        p = s["demand_potential"]
        improvement = p["all_common_states"]["potential_vs_ppo_improvement_pp"]
        pair_accuracy = p["pairwise_blocking_order_accuracy"]
        potential_proceed = (
            improvement is not None and improvement >= 1.0
            and pair_accuracy is not None and pair_accuracy >= 0.60
            and s["oracle_headroom_pp"] is not None
            and s["oracle_headroom_pp"] >= 1.0
        )
    verdict = (
        "PROCEED_TO_POTENTIAL_CLOSED_LOOP"
        if potential_proceed else
        ("PROCEED_TO_C_SIDE_MEAN_FIELD" if proceed else "STOP_OR_REDESIGN")
    )
    lines = [
        "# C-Action H-Step Oracle Diagnostic", "",
        f"- Agent-C: `{c['agent_c_checkpoint']}`",
        f"- PPO-R: `{c['agent_r_checkpoint']}`",
        f"- H: {c['horizon']}",
        f"- Seeds: {c['seeds']}",
        f"- Episodes/seed: {c['episodes']}",
        f"- Requests/episode: {c['requests_per_episode']}", "",
        "## Aggregate", "",
        "| Metric | Value |", "|---|---:|",
        f"| Total requests | {s['total_requests']} |",
        f"| Raw C-mask empty | {s['raw_c_empty']} |",
        f"| Active connections, mean / P95 / max | {s['mean_active_connections_before']:.2f} / {s['p95_active_connections_before']:.1f} / {s['max_active_connections_before']} |",
        f"| Multi-C-action states | {s['multi_action_count']} ({s['multi_action_rate']:.2%}) |",
        f"| Full-H multi-action states | {s['full_horizon_multi_action_count']} |",
        f"| Different future blocking | {s['multi_action_with_diff_future_blocking']} ({s['multi_action_with_diff_future_blocking_rate']:.2%}) |",
        f"| Different future raw-mask-empty | {s['multi_action_with_diff_future_raw_empty']} ({s['multi_action_with_diff_future_raw_empty_rate']:.2%}) |",
        f"| Common comparison states | {s['common_comparison_count']} |",
        f"| PPO-C future block rate | {_fmt(s['ppo_mean_future_block_rate'])} |",
        f"| Oracle-C-H future block rate | {_fmt(s['oracle_mean_future_block_rate'])} |",
        f"| Oracle headroom (pp) | {_fmt(s['oracle_headroom_pp'])} |",
        f"| PPO-C mean Oracle rank | {_fmt(s['mean_ppo_rank_among_successful'])} |", "",
        "## Per Seed", "",
        "| Seed | Total | Multi-C | Common | PPO block | Oracle block | Headroom pp | Mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, row in report["per_seed"].items():
        lines.append(
            f"| {seed} | {row['total_requests']} | {row['multi_action_count']} | "
            f"{row['common_comparison_count']} | {_fmt(row['ppo_mean_future_block_rate'])} | "
            f"{_fmt(row['oracle_mean_future_block_rate'])} | {_fmt(row['oracle_headroom_pp'])} | "
            f"{_fmt(row['mean_ppo_rank_among_successful'])} |"
        )
    if "demand_potential" in s:
        p = s["demand_potential"]
        all_rows = p["all_common_states"]
        diff_rows = p["candidate_different_states"]
        lines.extend([
            "", "## Historical Demand-Aware Potential", "",
            f"History window: {p['history_window']}; probes/candidate: {p['probe_limit']}", "",
            "| Scope | States | PPO block | Potential block | Oracle block | Gain vs PPO pp | Gap to Oracle pp | Optimal rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| All common | {all_rows['count']} | {_fmt(all_rows['ppo_mean_future_block_rate'])} | {_fmt(all_rows['potential_mean_future_block_rate'])} | {_fmt(all_rows['oracle_mean_future_block_rate'])} | {_fmt(all_rows['potential_vs_ppo_improvement_pp'])} | {_fmt(all_rows['potential_oracle_gap_pp'])} | {_fmt(all_rows['potential_oracle_optimal_rate'])} |",
            f"| Candidate-different | {diff_rows['count']} | {_fmt(diff_rows['ppo_mean_future_block_rate'])} | {_fmt(diff_rows['potential_mean_future_block_rate'])} | {_fmt(diff_rows['oracle_mean_future_block_rate'])} | {_fmt(diff_rows['potential_vs_ppo_improvement_pp'])} | {_fmt(diff_rows['potential_oracle_gap_pp'])} | {_fmt(diff_rows['potential_oracle_optimal_rate'])} |",
            f"| Pairwise ordering accuracy | {p['pairwise_comparison_count']} pairs | {_fmt(p['pairwise_blocking_order_accuracy'])} |  |  |  |  |  |",
        ])
    lines.extend(["", "## Verdict", "", f"**{verdict}**", "",
                  "This is an empirical finite-horizon upper bound under a frozen future policy, not a global theoretical optimum."])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", default="42,123,456")
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
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--potential_history_window", type=int, default=12)
    parser.add_argument("--potential_probe_limit", type=int, default=0,
                        help="Past/current demand probes per candidate; 0 disables potential diagnostic.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    report = _run_diagnostic(args)
    out_json = Path(args.output_json or f"sa_hmarl/experiments/c_action_horizon_oracle_h{args.horizon}.json")
    out_md = Path(args.output_md or f"sa_hmarl/experiments/c_action_horizon_oracle_h{args.horizon}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(_build_markdown(report))
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
    print(f"Oracle headroom (pp): {report['summary']['oracle_headroom_pp']}")


if __name__ == "__main__":
    main()
