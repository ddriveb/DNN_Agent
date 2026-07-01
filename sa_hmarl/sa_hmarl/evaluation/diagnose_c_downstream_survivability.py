"""C-side downstream survivability diagnostic.

For each real request, evaluate every valid (split, server) candidate a_C by:
  1. Snapshotting the env just before the R decision.
  2. Applying a_C and letting the frozen v1 R-ranker pick an R action.
  3. Computing Phi_probe(s_plus): the fraction of probe requests that still have
     at least one legal C-R combination in the resulting post-decision state.

Outputs per-request candidate records and aggregate statistics, plus a lightweight
oracle_phi closed-loop run that always picks the candidate with highest Phi_probe.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    _select_rank_only_r_action,
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _make_probe_requests(
    env,
    current_req,
    probe_count: int,
    rng: np.random.RandomState,
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
) -> List[Any]:
    """Generate probe requests from the same distribution as the current request."""
    return generate_requests(
        env, rng, current_req.src_node, probe_count,
        arrival_interval, holding_min, holding_max,
        deadline_min, deadline_max, size_min_mb, size_max_mb,
        edge_cost_min, edge_cost_max, num_splits, split_profile,
    )


def _compute_probe_survivability(
    env_snapshot,
    probe_requests: List[Any],
) -> Dict[str, Any]:
    """Compute Phi_probe(s_plus) without allocating resources."""
    survivable = 0
    max_feasible_counts = []
    valid_c_counts = []
    for probe_req in probe_requests:
        obs_c = build_agent_c_observation(env_snapshot, probe_req)
        mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        valid_c_indices = np.flatnonzero(mask).tolist()
        valid_c_counts.append(len(valid_c_indices))
        if not valid_c_indices:
            max_feasible_counts.append(0)
            continue
        feasible_counts = [obs_c["candidate_features"][idx]["feasible_count"] for idx in valid_c_indices]
        max_fc = max(feasible_counts)
        max_feasible_counts.append(max_fc)
        if max_fc > 0:
            survivable += 1
    B = len(probe_requests)
    return {
        "phi": survivable / B if B > 0 else 0.0,
        "probe_count": B,
        "survivable_count": survivable,
        "avg_max_feasible_count": float(np.mean(max_feasible_counts)) if max_feasible_counts else 0.0,
        "avg_valid_c_count": float(np.mean(valid_c_counts)) if valid_c_counts else 0.0,
        "zero_legal_probe_rate": float(np.mean([m == 0 for m in max_feasible_counts])) if max_feasible_counts else 0.0,
    }


def _evaluate_candidate(
    env_snapshot,
    req,
    c_idx: int,
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    probe_requests: List[Any],
    device: str,
) -> Dict[str, Any]:
    """Apply one C candidate with v1 R-ranker and compute s_plus survivability."""
    split_id, server_id = decode_agent_c_action(c_idx, len(env_snapshot.mec.servers))
    env_c = copy.deepcopy(env_snapshot)

    obs_c = build_agent_c_observation(env_c, req)
    obs_r = build_agent_r_observation(env_c, req, split_id, server_id)
    r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    legal_r = int(r_mask.sum())

    if legal_r > 0:
        r_idx = _select_rank_only_r_action(
            env_c, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
            split_id, server_id, device,
        )
        r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env_c.max_blocks)
    else:
        r_idx = -1
        r_action = (0, 0, 0)

    _, _, _, info = env_c.step((split_id, server_id), r_action)
    current_success = bool(info.get("success", False))

    phi_info = _compute_probe_survivability(env_c, probe_requests)

    return {
        "c_idx": int(c_idx),
        "split_id": int(split_id),
        "server_id": int(server_id),
        "feasible_count": int(obs_c["candidate_features"][c_idx]["feasible_count"]),
        "legal_r_count": legal_r,
        "r_idx": int(r_idx),
        "current_success": current_success,
        "current_reason": info.get("reason", ""),
        "phi": phi_info["phi"],
        "probe_count": phi_info["probe_count"],
        "survivable_count": phi_info["survivable_count"],
        "avg_max_feasible_count": phi_info["avg_max_feasible_count"],
        "avg_valid_c_count": phi_info["avg_valid_c_count"],
        "zero_legal_probe_rate": phi_info["zero_legal_probe_rate"],
    }


def _run_ppo_c_trajectory(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    probe_count: int,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
) -> List[Dict[str, Any]]:
    """Run main trajectory with frozen PPO-C + v1 ranker and record per-request Phi stats."""
    records = []
    for req_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        raw_mask_empty = int(raw_c_mask.sum()) == 0

        selected_c_idx = agent_c.select_action(obs_c, deterministic=True)
        if selected_c_idx is None:
            selected_c_idx = 0
        selected_split, selected_server = decode_agent_c_action(int(selected_c_idx), args.num_servers)

        # Build valid candidate list.
        valid_c_indices = np.flatnonzero(raw_c_mask).tolist()
        if not valid_c_indices:
            env.step((selected_split, selected_server), (0, 0, 0))
            _record_outcome(metrics, {"success": False, "reason": "no_valid_c_action"}, True, obs_c,
                            selected_split, selected_server, 0.0, True)
            metrics.active_connections[-1] = len(env.active_connections)
            records.append({
                "seed": None,
                "episode": None,
                "req_id": int(req.req_id),
                "valid_c_count": 0,
                "selected_c_idx": int(selected_c_idx),
                "selected_phi": 0.0,
                "best_phi": 0.0,
                "selected_phi_rank": 0,
                "phi_range": 0.0,
                "phi_std": 0.0,
                "selected_current_success": False,
                "best_phi_current_success": False,
                "selected_valid_r_count": 0,
                "best_phi_valid_r_count": 0,
                "actual_result_success": False,
                "actual_block_reason": "no_valid_c_action",
                "raw_mask_empty_current": True,
                "candidates": [],
            })
            continue

        # Snapshot for candidate evaluation.
        snapshot = _snapshot_before_r_decision(env, req.req_id)

        # Generate probe requests (common random numbers for all candidates).
        probe_rng = np.random.RandomState(
            (args.base_probe_seed + (args._seed or 0) * 1000000 + (args._episode or 0) * 1000 + req_idx) % (2**32)
        )
        probe_requests = _make_probe_requests(
            env, req, probe_count, probe_rng,
            args.arrival_interval, args.holding_min, args.holding_max,
            args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
            args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
        )

        # Evaluate all valid C candidates.
        candidates = []
        phi_values = {}
        for c_idx in valid_c_indices:
            cand = _evaluate_candidate(
                snapshot, req, c_idx, agent_r, rank_model, rank_mean, rank_std,
                probe_requests, args.device,
            )
            candidates.append(cand)
            phi_values[c_idx] = cand["phi"]

        best_c_idx = max(valid_c_indices, key=lambda x: (phi_values[x], -x))
        best_phi = phi_values[best_c_idx]
        # Rank with tie handling: rank = 1 + #candidates strictly better than selected.
        selected_phi_rank = 1 + sum(1 for x in valid_c_indices if phi_values[x] > phi_values[selected_c_idx] + 1e-9)
        phi_list = list(phi_values.values())
        phi_range = max(phi_list) - min(phi_list)
        phi_std = float(np.std(phi_list, ddof=0)) if len(phi_list) > 1 else 0.0

        selected_cand = next(c for c in candidates if c["c_idx"] == selected_c_idx)
        best_cand = next(c for c in candidates if c["c_idx"] == best_c_idx)

        # Step the real env with PPO-C + v1 ranker.
        obs_r = build_agent_r_observation(env, req, selected_split, selected_server)
        r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if r_mask.any():
            r_idx = _select_rank_only_r_action(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                selected_split, selected_server, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        else:
            r_action = (0, 0, 0)
        _, _, _, info = env.step((selected_split, selected_server), r_action)
        _record_outcome(metrics, info, raw_mask_empty, obs_c, selected_split, selected_server, 0.0, True)
        metrics.active_connections[-1] = len(env.active_connections)

        records.append({
            "seed": args._seed,
            "episode": args._episode,
            "req_id": int(req.req_id),
            "valid_c_count": int(len(valid_c_indices)),
            "selected_c_idx": int(selected_c_idx),
            "selected_phi": float(selected_cand["phi"]),
            "best_phi": float(best_phi),
            "selected_phi_rank": selected_phi_rank,
            "phi_range": float(phi_range),
            "phi_std": float(phi_std),
            "selected_current_success": bool(selected_cand["current_success"]),
            "best_phi_current_success": bool(best_cand["current_success"]),
            "selected_valid_r_count": int(selected_cand["legal_r_count"]),
            "best_phi_valid_r_count": int(best_cand["legal_r_count"]),
            "actual_result_success": bool(info.get("success", False)),
            "actual_block_reason": info.get("reason", ""),
            "raw_mask_empty_current": bool(raw_mask_empty),
            "candidates": candidates,
        })
    return records


def _run_oracle_phi_trajectory(
    env,
    requests: List[Any],
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    probe_count: int,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
) -> List[Dict[str, Any]]:
    """Oracle that picks the C candidate with highest Phi_probe at each step."""
    for req_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        valid_c_indices = np.flatnonzero(raw_c_mask).tolist()

        if not valid_c_indices:
            # No valid C action: block.
            env.step((0, 0), (0, 0, 0))
            _record_outcome(metrics, {"success": False, "reason": "no_valid_c_action"}, True, obs_c,
                            0, 0, 0.0, True)
            metrics.active_connections[-1] = len(env.active_connections)
            continue

        snapshot = _snapshot_before_r_decision(env, req.req_id)
        probe_rng = np.random.RandomState(
            (args.base_probe_seed + (args._seed or 0) * 1000000 + (args._episode or 0) * 1000 + req_idx) % (2**32)
        )
        probe_requests = _make_probe_requests(
            env, req, probe_count, probe_rng,
            args.arrival_interval, args.holding_min, args.holding_max,
            args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
            args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
        )

        best_c_idx = None
        best_phi = -1.0
        for c_idx in valid_c_indices:
            cand = _evaluate_candidate(
                snapshot, req, c_idx, agent_r, rank_model, rank_mean, rank_std,
                probe_requests, args.device,
            )
            if cand["phi"] > best_phi or (cand["phi"] == best_phi and (best_c_idx is None or c_idx < best_c_idx)):
                best_phi = cand["phi"]
                best_c_idx = c_idx

        split_id, server_id = decode_agent_c_action(best_c_idx, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if r_mask.any():
            r_idx = _select_rank_only_r_action(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        else:
            r_action = (0, 0, 0)
        _, _, _, info = env.step((split_id, server_id), r_action)
        _record_outcome(metrics, info, int(raw_c_mask.sum()) == 0, obs_c, split_id, server_id, 0.0, True)
        metrics.active_connections[-1] = len(env.active_connections)
    return []


def _compute_correlations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _corr(a, b, method="pearson"):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return None
        if method == "pearson":
            r, p = stats.pearsonr(a, b)
        else:
            r, p = stats.spearmanr(a, b)
        return {"r": float(r), "p": float(p)}

    selected_phi = np.array([r["selected_phi"] for r in records])
    actual_success = np.array([1.0 if r["actual_result_success"] else 0.0 for r in records])
    selected_valid_r = np.array([r["selected_valid_r_count"] for r in records], dtype=float)
    phi_range = np.array([r["phi_range"] for r in records])
    selected_phi_rank = np.array([r["selected_phi_rank"] if r["selected_phi_rank"] is not None else 0 for r in records], dtype=float)
    blocked = np.array([0.0 if r["actual_result_success"] else 1.0 for r in records])

    return {
        "selected_phi_vs_actual_success": _corr(selected_phi, actual_success),
        "selected_phi_vs_selected_valid_r": _corr(selected_phi, selected_valid_r),
        "phi_range_vs_raw_mask_empty_next": None,  # optional, not computed in v1
        "selected_phi_rank_vs_blocked": _corr(selected_phi_rank, blocked),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)

    all_records = []
    per_seed_metrics = {"ppo_c_v1_ranker": {}, "oracle_phi": {}}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max, args.size_min_mb,
                args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile,
            ))

        for ep_idx, requests in enumerate(episodes):
            args._seed = seed
            args._episode = ep_idx

            # PPO-C + v1 ranker trajectory.
            env_ppo = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env_ppo.reset(requests)
            metrics_ppo = PerMethodMetrics()
            records = _run_ppo_c_trajectory(
                env_ppo, requests, agent_c, agent_r, rank_model, rank_mean, rank_std,
                args.probe_count, args, metrics_ppo,
            )
            for r in records:
                r["seed"] = int(seed)
                r["episode"] = int(ep_idx)
            all_records.extend(records)
            per_seed_metrics["ppo_c_v1_ranker"][str(seed)] = _aggregate_metrics(metrics_ppo)

            # Oracle-phi trajectory.
            env_oracle = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env_oracle.reset(requests)
            metrics_oracle = PerMethodMetrics()
            _run_oracle_phi_trajectory(
                env_oracle, requests, agent_r, rank_model, rank_mean, rank_std,
                args.probe_count, args, metrics_oracle,
            )
            per_seed_metrics["oracle_phi"][str(seed)] = _aggregate_metrics(metrics_oracle)

        print(f"[seed {seed}] ppo_c blocking={per_seed_metrics['ppo_c_v1_ranker'][str(seed)]['blocking_rate']:.2%} "
              f"oracle blocking={per_seed_metrics['oracle_phi'][str(seed)]['blocking_rate']:.2%}", flush=True)

    # Aggregate statistics.
    valid_records = [r for r in all_records if r["valid_c_count"] >= 2]
    total_requests = len(all_records)
    requests_with_valid_c_ge2 = len(valid_records)

    selected_phi_arr = np.array([r["selected_phi"] for r in valid_records])
    best_phi_arr = np.array([r["best_phi"] for r in valid_records])
    phi_gap_arr = best_phi_arr - selected_phi_arr
    phi_range_arr = np.array([r["phi_range"] for r in valid_records])

    selected_is_best = [r["selected_phi"] >= r["best_phi"] - 1e-9 for r in valid_records]
    selected_in_top2 = []
    for r in valid_records:
        rank = r["selected_phi_rank"]
        selected_in_top2.append(rank is not None and rank <= 2)

    correlations = _compute_correlations(valid_records if valid_records else all_records)

    aggregate = {
        "total_requests": total_requests,
        "requests_with_valid_c_ge2": requests_with_valid_c_ge2,
        "avg_valid_c_count": float(np.mean([r["valid_c_count"] for r in all_records])),
        "raw_mask_empty_rate": float(np.mean([r["raw_mask_empty_current"] for r in all_records])),
        "ppo_c_v1_ranker": {
            key: float(np.mean([m[key] for m in per_seed_metrics["ppo_c_v1_ranker"].values()]))
            for key in next(iter(per_seed_metrics["ppo_c_v1_ranker"].values()))
            if isinstance(next(iter(per_seed_metrics["ppo_c_v1_ranker"].values()))[key], (int, float))
        },
        "oracle_phi": {
            key: float(np.mean([m[key] for m in per_seed_metrics["oracle_phi"].values()]))
            for key in next(iter(per_seed_metrics["oracle_phi"].values()))
            if isinstance(next(iter(per_seed_metrics["oracle_phi"].values()))[key], (int, float))
        },
        "phi_stats": {
            "mean_phi_selected": float(np.mean(selected_phi_arr)) if len(selected_phi_arr) else 0.0,
            "mean_phi_best": float(np.mean(best_phi_arr)) if len(best_phi_arr) else 0.0,
            "mean_phi_gap": float(np.mean(phi_gap_arr)) if len(phi_gap_arr) else 0.0,
            "mean_phi_range": float(np.mean(phi_range_arr)) if len(phi_range_arr) else 0.0,
            "median_phi_range": float(np.median(phi_range_arr)) if len(phi_range_arr) else 0.0,
            "fraction_phi_range_gt_0": float(np.mean(phi_range_arr > 1e-9)) if len(phi_range_arr) else 0.0,
            "fraction_phi_range_gt_0.1": float(np.mean(phi_range_arr > 0.1)) if len(phi_range_arr) else 0.0,
            "fraction_phi_range_gt_0.2": float(np.mean(phi_range_arr > 0.2)) if len(phi_range_arr) else 0.0,
        },
        "ppo_c_selection_quality": {
            "selected_phi_rank_mean": float(np.mean([r["selected_phi_rank"] for r in valid_records if r["selected_phi_rank"] is not None])) if valid_records else 0.0,
            "selected_is_best_phi_rate": float(np.mean(selected_is_best)) if selected_is_best else 0.0,
            "selected_in_top2_phi_rate": float(np.mean(selected_in_top2)) if selected_in_top2 else 0.0,
        },
        "correlations": correlations,
    }

    report = {
        "config": vars(args),
        "aggregate": aggregate,
        "per_seed_metrics": per_seed_metrics,
        "records": all_records,
        "elapsed_seconds": time.time() - getattr(args, "_start_time", time.time()),
    }
    if getattr(args, "output_json", None):
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if getattr(args, "output_md", None):
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(out_md, report)
    return report


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    agg = report["aggregate"]
    phi = agg["phi_stats"]
    sel = agg["ppo_c_selection_quality"]
    ppo = agg["ppo_c_v1_ranker"]
    ora = agg["oracle_phi"]
    lines = [
        "# C-Side Downstream Survivability Diagnostic",
        "",
        "## 1. Basic statistics",
        "",
        f"- Total requests: {agg['total_requests']}",
        f"- Requests with >=2 valid C candidates: {agg['requests_with_valid_c_ge2']}",
        f"- Average valid C count: {agg['avg_valid_c_count']:.2f}",
        f"- Raw C mask empty rate: {agg['raw_mask_empty_rate']:.2%}",
        "",
        "## 2. Closed-loop blocking",
        "",
        "| Method | Blocking | Raw empty | NSB | Overload |",
        "|---|---:|---:|---:|---:|",
        f"| PPO-C + v1 ranker | {ppo['blocking_rate']:.2%} | {ppo['raw_mask_empty_rate']:.2%} | "
        f"{ppo['no_suitable_block_rate']:.2%} | {ppo['server_overload_rate']:.2%} |",
        f"| Oracle-Phi | {ora['blocking_rate']:.2%} | {ora['raw_mask_empty_rate']:.2%} | "
        f"{ora['no_suitable_block_rate']:.2%} | {ora['server_overload_rate']:.2%} |",
        "",
        "## 3. Phi difference across C candidates",
        "",
        f"- Mean Phi (selected): {phi['mean_phi_selected']:.4f}",
        f"- Mean Phi (best): {phi['mean_phi_best']:.4f}",
        f"- Mean Phi gap (best - selected): {phi['mean_phi_gap']:.4f}",
        f"- Mean Phi range: {phi['mean_phi_range']:.4f}",
        f"- Median Phi range: {phi['median_phi_range']:.4f}",
        f"- Fraction range > 0: {phi['fraction_phi_range_gt_0']:.2%}",
        f"- Fraction range > 0.1: {phi['fraction_phi_range_gt_0.1']:.2%}",
        f"- Fraction range > 0.2: {phi['fraction_phi_range_gt_0.2']:.2%}",
        "",
        "## 4. PPO-C selection quality",
        "",
        f"- Mean selected-Phi rank: {sel['selected_phi_rank_mean']:.2f}",
        f"- Selected is best-Phi rate: {sel['selected_is_best_phi_rate']:.2%}",
        f"- Selected in top-2 Phi rate: {sel['selected_in_top2_phi_rate']:.2%}",
        "",
        "## 5. Correlations",
        "",
    ]
    for name, corr in agg["correlations"].items():
        if corr is None:
            lines.append(f"- {name}: N/A")
        else:
            lines.append(f"- {name}: r={corr['r']:.3f}, p={corr['p']:.3g}")

    # Verdict per task specification.
    mean_gap = phi["mean_phi_gap"]
    frac_gt_01 = phi["fraction_phi_range_gt_0.1"]
    best_rate = sel["selected_is_best_phi_rate"]
    blocking_improvement = ppo["blocking_rate"] - ora["blocking_rate"]
    overload_ok = ora["server_overload_rate"] <= ppo["server_overload_rate"] + 0.005
    lines += ["", "## 6. Verdict"]
    pass_conditions = (
        mean_gap >= 0.05
        and frac_gt_01 >= 0.20
        and best_rate <= 0.60
        and blocking_improvement >= 0.01
        and overload_ok
    )
    marginal_conditions = (
        (0.003 <= blocking_improvement < 0.01)
        or (mean_gap >= 0.03 and blocking_improvement >= 0.003)
    )
    if pass_conditions:
        lines.append("**PASS**: C-side survivability has clear leverage. Proceed to C-ranker dataset/training.")
    elif marginal_conditions:
        lines.append("**MARGINAL**: Survivability signal exists but closed-loop gain is modest or a trade-off is present.")
    else:
        lines.append("**FAIL**: Phi_range is too small and/or PPO-C already picks best-Phi; C-side survivability is not the bottleneck.")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_k5m10_h5/ranking_model.pt")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--probe_count", type=int, default=16)
    parser.add_argument("--base_probe_seed", type=int, default=123456)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.15)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/c_downstream_survivability_diagnostic.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/c_downstream_survivability_diagnostic.md")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    args._seed = None
    args._episode = None
    report = evaluate(args)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")
