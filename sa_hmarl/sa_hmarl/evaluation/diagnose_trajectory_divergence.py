"""Trajectory-level divergence diagnostic: v1 R-ranker vs DeepRMSA.

For each request trace we run two closed-loop trajectories side-by-side:
  - v1: frozen PPO-C + v1 counterfactual R-ranker
  - DeepRMSA: frozen PPO-C + frozen DeepRMSA teacher for the R action

Both trajectories share the identical request sequence. We align per-request
decisions and record the first point where C or R actions differ, the resource
delta introduced by that difference, and how long afterwards each method hits
zero-legal / blocking.

The output answers:
  1. Where do the two policies first diverge? (C action or R action)
  2. How long after divergence does blocking appear?
  3. What resource metric(s) does DeepRMSA protect better?
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

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
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
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    _select_rank_only_r_action,
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


# ---------------------------------------------------------------------------
# Per-episode state container
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryState:
    """Lightweight mutable record used while stepping through one trajectory."""
    env: Any
    metrics: PerMethodMetrics = field(default_factory=PerMethodMetrics)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _spectrum_stats(env) -> Dict[str, float]:
    stats = env.net.get_global_spectrum_stats()
    return {
        "frag_index": float(stats.get("avg_frag_index", 0.0)),
        "lfb_ratio": float(stats.get("largest_free_block_ratio", 0.0)),
        "utilization": float(stats.get("spectrum_utilization", 0.0)),
        "avg_free_block_count": float(stats.get("avg_free_block_count", 0.0)),
        "num_links": int(stats.get("num_links", 0)),
    }


def _server_util_stats(env) -> Dict[str, float]:
    utils = [float(s.utilization) for s in env.mec.servers]
    return {
        "mean_server_util": float(np.mean(utils)) if utils else 0.0,
        "max_server_util": float(np.max(utils)) if utils else 0.0,
    }


def _action_features(obs_r: Dict[str, Any], action_idx: int) -> Optional[Dict[str, Any]]:
    """Return feature dict for a concrete R action index, if available."""
    if action_idx is None or action_idx < 0:
        return None
    feats = obs_r.get("action_features")
    if feats is None or action_idx >= len(feats):
        return None
    f = feats[action_idx]
    return {
        "path_idx": int(f[0]) if len(f) > 0 else None,
        "mod_idx": int(f[1]) if len(f) > 1 else None,
        "block_idx": int(f[2]) if len(f) > 2 else None,
        "path_length_km": float(f[3]) if len(f) > 3 else None,
        "required_fs": float(f[4]) if len(f) > 4 else None,
        "block_size": float(f[5]) if len(f) > 5 else None,
        "block_waste": float(f[6]) if len(f) > 6 else None,
        "se": float(f[7]) if len(f) > 7 else None,
    }


# ---------------------------------------------------------------------------
# Single-step rollout for one method
# ---------------------------------------------------------------------------
def _step_method(
    state: TrajectoryState,
    req: Any,
    agent_c: Any,
    agent_r: Any,
    rank_model: torch.nn.Module,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    deep_rmsa: Optional[DeepRMSAAgent],
    args: argparse.Namespace,
    method: str,
) -> Dict[str, Any]:
    """Advance one method by one request and return a per-request record."""
    env = state.env
    env.advance_time(req.arrival_time)

    obs_c = build_agent_c_observation(env, req)
    c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
    split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)

    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    legal_r = int(r_mask.sum())

    if method == "v1_ranker":
        if legal_r > 0:
            r_idx = _select_rank_only_r_action(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        else:
            r_idx = -1
            r_action = (0, 0, 0)
    elif method == "deep_rmsa":
        deep_idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
        if deep_idx is not None and legal_r > 0:
            r_idx = int(deep_idx)
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        else:
            r_idx = -1
            r_action = (0, 0, 0)
    else:
        raise ValueError(f"Unknown method: {method}")

    before_stats = _spectrum_stats(env)
    before_server = _server_util_stats(env)

    _, _, _, info = env.step((split_id, server_id), r_action)
    _record_outcome(
        state.metrics, info, int(raw_c_mask.sum()) == 0, obs_c,
        split_id, server_id, 0.0, True,
    )
    state.metrics.active_connections[-1] = len(env.active_connections)

    after_stats = _spectrum_stats(env)
    after_server = _server_util_stats(env)

    r_path, r_mod, r_block = r_action
    action_feat = _action_features(obs_r, r_idx) if r_idx >= 0 else None

    return {
        "method": method,
        "c_idx": int(c_idx),
        "split_id": int(split_id),
        "server_id": int(server_id),
        "r_idx": int(r_idx),
        "r_action_path": int(r_path),
        "r_action_mod": int(r_mod),
        "r_action_block": int(r_block),
        "legal_c_count": int(raw_c_mask.sum()),
        "legal_r_count": legal_r,
        "success": bool(info.get("success", False)),
        "reason": info.get("reason", ""),
        "delay_ms": float(info.get("delay_ms", 0.0)) if info.get("success", False) else None,
        "path_dist_km": float(info.get("path_dist_km", 0.0)) if info.get("success", False) else None,
        "num_hops": int(info.get("num_hops", 0)) if info.get("success", False) else None,
        "num_slots": float(info.get("num_slots", 0.0)) if info.get("success", False) else None,
        "block_waste": float(info.get("block_waste", 0.0)) if info.get("success", False) else None,
        "modulation": info.get("modulation", ""),
        "before_spectrum": before_stats,
        "after_spectrum": after_stats,
        "before_server_util": before_server,
        "after_server_util": after_server,
        "r_action_features": action_feat,
    }


# ---------------------------------------------------------------------------
# Align two trajectories on the same trace
# ---------------------------------------------------------------------------
def _run_aligned_pair(
    requests: List[Any],
    agent_c: Any,
    agent_r: Any,
    rank_model: torch.nn.Module,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    deep_rmsa: DeepRMSAAgent,
    args: argparse.Namespace,
    seed: int,
    episode: int,
) -> Dict[str, Any]:
    """Run v1 ranker and DeepRMSA on the same requests, align per request."""
    env_v1 = make_env(
        args.topology, args.num_slots, args.num_servers, seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    env_deep = make_env(
        args.topology, args.num_slots, args.num_servers, seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    env_v1.reset(requests)
    env_deep.reset(requests)

    state_v1 = TrajectoryState(env=env_v1)
    state_deep = TrajectoryState(env=env_deep)

    records = []
    first_divergence_index: Optional[int] = None
    first_divergence_type: Optional[str] = None

    for idx, req in enumerate(requests):
        rec_v1 = _step_method(
            state_v1, req, agent_c, agent_r, rank_model, rank_mean, rank_std,
            None, args, "v1_ranker",
        )
        rec_deep = _step_method(
            state_deep, req, agent_c, agent_r, rank_model, rank_mean, rank_std,
            deep_rmsa, args, "deep_rmsa",
        )

        c_same = rec_v1["c_idx"] == rec_deep["c_idx"]
        r_same = rec_v1["r_idx"] == rec_deep["r_idx"]
        actions_same = c_same and r_same

        if first_divergence_index is None and not actions_same:
            first_divergence_index = idx
            if not c_same and not r_same:
                first_divergence_type = "both"
            elif not c_same:
                first_divergence_type = "c"
            else:
                first_divergence_type = "r"

        post_divergence = first_divergence_index is not None
        first_here = post_divergence and idx == first_divergence_index

        records.append({
            "seed": seed,
            "episode": episode,
            "req_index": idx,
            "req_id": int(req.req_id),
            "arrival_time": float(req.arrival_time),
            "v1": rec_v1,
            "deep": rec_deep,
            "c_same": bool(c_same),
            "r_same": bool(r_same),
            "actions_same": bool(actions_same),
            "first_divergence_here": bool(first_here),
            "divergence_type": first_divergence_type if first_here else None,
            "post_divergence": bool(post_divergence),
            "divergence_distance": (idx - first_divergence_index) if post_divergence else None,
        })

    # Compute distances from first divergence to first block for each method.
    v1_first_block_after = None
    deep_first_block_after = None
    if first_divergence_index is not None:
        for rec in records[first_divergence_index:]:
            if v1_first_block_after is None and not rec["v1"]["success"]:
                v1_first_block_after = rec["divergence_distance"]
            if deep_first_block_after is None and not rec["deep"]["success"]:
                deep_first_block_after = rec["divergence_distance"]
            if v1_first_block_after is not None and deep_first_block_after is not None:
                break

    episode_summary = {
        "seed": seed,
        "episode": episode,
        "first_divergence_index": first_divergence_index,
        "first_divergence_req_id": (
            int(requests[first_divergence_index].req_id)
            if first_divergence_index is not None else None
        ),
        "first_divergence_type": first_divergence_type,
        "v1_first_block_after_divergence": v1_first_block_after,
        "deep_first_block_after_divergence": deep_first_block_after,
        "v1_blocks": int(state_v1.metrics.blocked),
        "deep_blocks": int(state_deep.metrics.blocked),
        "v1_blocking_rate": state_v1.metrics.blocked / max(state_v1.metrics.total, 1),
        "deep_blocking_rate": state_deep.metrics.blocked / max(state_deep.metrics.total, 1),
    }

    return {
        "records": records,
        "summary": episode_summary,
        "v1_metrics": _aggregate_metrics(state_v1.metrics),
        "deep_metrics": _aggregate_metrics(state_deep.metrics),
    }


# ---------------------------------------------------------------------------
# Aggregation and report generation
# ---------------------------------------------------------------------------
def _safe_mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _safe_median(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _safe_percentile(values: List[float], p: float) -> Optional[float]:
    return float(np.percentile(values, p)) if values else None


def _summarize(all_records: List[Dict[str, Any]], episode_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    diverged_episodes = [s for s in episode_summaries if s["first_divergence_index"] is not None]
    first_indices = [s["first_divergence_index"] for s in diverged_episodes]
    first_types = [s["first_divergence_type"] for s in diverged_episodes]

    v1_blocks_after = [s["v1_first_block_after_divergence"] for s in diverged_episodes
                       if s["v1_first_block_after_divergence"] is not None]
    deep_blocks_after = [s["deep_first_block_after_divergence"] for s in diverged_episodes
                         if s["deep_first_block_after_divergence"] is not None]

    # Resource deltas at the first-divergence request (v1 - deep).
    delta_keys = ["frag_index", "lfb_ratio", "utilization", "avg_free_block_count"]
    resource_deltas_all = {k: [] for k in delta_keys}
    resource_deltas_both_success = {k: [] for k in delta_keys}

    path_km_deltas = []
    fs_deltas = []
    waste_deltas = []
    delay_deltas = []

    for rec in all_records:
        if not rec["first_divergence_here"]:
            continue
        v1_after = rec["v1"]["after_spectrum"]
        deep_after = rec["deep"]["after_spectrum"]
        for k in delta_keys:
            d = v1_after[k] - deep_after[k]
            resource_deltas_all[k].append(d)
            if rec["v1"]["success"] and rec["deep"]["success"]:
                resource_deltas_both_success[k].append(d)

        if rec["v1"]["success"] and rec["deep"]["success"]:
            if rec["v1"]["path_dist_km"] is not None and rec["deep"]["path_dist_km"] is not None:
                path_km_deltas.append(rec["v1"]["path_dist_km"] - rec["deep"]["path_dist_km"])
            if rec["v1"]["num_slots"] is not None and rec["deep"]["num_slots"] is not None:
                fs_deltas.append(rec["v1"]["num_slots"] - rec["deep"]["num_slots"])
            if rec["v1"]["block_waste"] is not None and rec["deep"]["block_waste"] is not None:
                waste_deltas.append(rec["v1"]["block_waste"] - rec["deep"]["block_waste"])
            if rec["v1"]["delay_ms"] is not None and rec["deep"]["delay_ms"] is not None:
                delay_deltas.append(rec["v1"]["delay_ms"] - rec["deep"]["delay_ms"])

    total_requests = len(all_records)
    total_episodes = len(episode_summaries)

    v1_overall_blocks = sum(s["v1_blocks"] for s in episode_summaries)
    deep_overall_blocks = sum(s["deep_blocks"] for s in episode_summaries)

    # Blocking reason distribution among all blocked requests.
    v1_reasons = {}
    deep_reasons = {}
    for rec in all_records:
        if not rec["v1"]["success"]:
            r = rec["v1"]["reason"] or "unknown"
            v1_reasons[r] = v1_reasons.get(r, 0) + 1
        if not rec["deep"]["success"]:
            r = rec["deep"]["reason"] or "unknown"
            deep_reasons[r] = deep_reasons.get(r, 0) + 1

    summary = {
        "total_requests": total_requests,
        "total_episodes": total_episodes,
        "episodes_with_divergence": len(diverged_episodes),
        "divergence_rate": len(diverged_episodes) / max(total_episodes, 1),
        "v1_overall_blocking_rate": v1_overall_blocks / max(total_requests, 1),
        "deep_overall_blocking_rate": deep_overall_blocks / max(total_requests, 1),
        "first_divergence_index": {
            "mean": _safe_mean(first_indices),
            "median": _safe_median(first_indices),
            "p25": _safe_percentile(first_indices, 25),
            "p75": _safe_percentile(first_indices, 75),
            "min": min(first_indices) if first_indices else None,
            "max": max(first_indices) if first_indices else None,
        },
        "first_divergence_type_counts": {
            t: first_types.count(t) for t in set(first_types)
        },
        "blocks_after_divergence": {
            "v1_mean": _safe_mean(v1_blocks_after),
            "v1_median": _safe_median(v1_blocks_after),
            "deep_mean": _safe_mean(deep_blocks_after),
            "deep_median": _safe_median(deep_blocks_after),
        },
        "resource_delta_v1_minus_deep_at_divergence": {
            "all_first_divergences": {k: _safe_mean(v) for k, v in resource_deltas_all.items()},
            "both_success": {k: _safe_mean(v) for k, v in resource_deltas_both_success.items()},
        },
        "action_delta_v1_minus_deep_both_success": {
            "path_km_mean": _safe_mean(path_km_deltas),
            "fs_mean": _safe_mean(fs_deltas),
            "waste_mean": _safe_mean(waste_deltas),
            "delay_ms_mean": _safe_mean(delay_deltas),
        },
        "blocking_reason_counts": {
            "v1": v1_reasons,
            "deep": deep_reasons,
        },
    }

    # Verdict based on user-specified logic.
    r_first_frac = first_types.count("r") / max(len(first_types), 1)
    short_term_block_frac = (
        sum(1 for d in v1_blocks_after if d is not None and d <= 3)
        / max(len(v1_blocks_after), 1)
    )
    same_zero_legal_rate = _same_zero_legal_rate(all_records)

    if len(diverged_episodes) == 0:
        verdict = "NO_DIVERGENCE"
        verdict_text = "**NO DIVERGENCE**: v1 ranker and DeepRMSA behave identically on all traces. The 0.64 pp gap is statistical noise."
    elif r_first_frac >= 0.7 and short_term_block_frac >= 0.5:
        verdict = "R_SHORT_TERM"
        verdict_text = "**R-action short-term bottleneck**: Most first divergences are R-action choices and blocking tends to appear within 1-3 requests. Revisit R-ranker hard-case / trajectory-aware training."
    elif r_first_frac >= 0.7 and short_term_block_frac < 0.5:
        verdict = "R_LONG_TERM"
        verdict_text = "**R-action long-term effect**: First divergences are R-action choices, but blocking appears much later. The frozen C/R decomposition may be near its ceiling; consider joint online fine-tuning only if the gap justifies the cost."
    elif same_zero_legal_rate >= 0.5:
        verdict = "RESOURCE_FLOOR"
        verdict_text = "**Resource floor / statistical noise**: Most blocking occurs at identical zero-legal states after divergence. The 0.64 pp gap is likely not worth a large engineering effort."
    else:
        verdict = "C_DOMINATED"
        verdict_text = "**C-action dominated divergence**: First divergences are frequently C-action differences, suggesting the bottleneck is upstream of R-ranker."

    summary["verdict"] = verdict
    summary["verdict_text"] = verdict_text
    return summary


def _same_zero_legal_rate(all_records: List[Dict[str, Any]]) -> float:
    """Fraction of post-divergence blocked requests where both methods block at the same request with reason 'no_valid_c_action' or 'no_suitable_block'."""
    blocked_pairs = []
    for rec in all_records:
        if rec["post_divergence"] and not rec["v1"]["success"] and not rec["deep"]["success"]:
            blocked_pairs.append(rec)
    if not blocked_pairs:
        return 0.0
    same = sum(
        1 for r in blocked_pairs
        if r["v1"]["reason"] == r["deep"]["reason"]
        and r["v1"]["reason"] in ("no_valid_c_action", "no_suitable_block", "")
    )
    return same / len(blocked_pairs)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
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
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

    all_records: List[Dict[str, Any]] = []
    episode_summaries: List[Dict[str, Any]] = []
    per_seed_episodes: Dict[str, List[Dict[str, Any]]] = {}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        per_seed_episodes[str(seed)] = []
        for ep_idx in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            requests = generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max, args.size_min_mb,
                args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile,
            )
            result = _run_aligned_pair(
                requests, agent_c, agent_r, rank_model, rank_mean, rank_std,
                deep_rmsa, args, seed, ep_idx,
            )
            all_records.extend(result["records"])
            episode_summaries.append(result["summary"])
            per_seed_episodes[str(seed)].append({
                "episode": ep_idx,
                "summary": result["summary"],
                "v1_metrics": result["v1_metrics"],
                "deep_metrics": result["deep_metrics"],
            })

    summary = _summarize(all_records, episode_summaries)

    output = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "summary": summary,
        "episode_summaries": episode_summaries,
        "per_seed_episodes": per_seed_episodes,
        "records": all_records,
        "elapsed_seconds": time.time() - args._start_time,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")

    _write_markdown_report(summary, episode_summaries, args)
    return summary


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _write_markdown_report(summary: Dict[str, Any], episode_summaries: List[Dict[str, Any]], args: argparse.Namespace):
    lines = [
        "# Trajectory Divergence Diagnostic: v1 R-ranker vs DeepRMSA",
        "",
        "## 1. Setup",
        "",
        f"- Topology: `{args.topology}`",
        f"- Slots: {args.num_slots}, Servers: {args.num_servers}, k_paths: {args.k_paths}, max_blocks: {args.max_blocks}",
        f"- Block sort: `{args.block_sort_strategy}`, Split profile: `{args.split_profile}`",
        f"- Seeds: `{args.seeds}`, Episodes/seed: {args.episodes}, Requests/episode: {args.requests_per_episode}",
        f"- v1 ranker: `{args.ranking_checkpoint}`",
        f"- DeepRMSA: `{args.deep_rmsa_checkpoint}`",
        "",
        "## 2. Aggregate blocking",
        "",
        "| Method | Blocking |",
        "|---|---:|",
        f"| v1 R-ranker | {summary['v1_overall_blocking_rate']*100:.2f}% |",
        f"| DeepRMSA    | {summary['deep_overall_blocking_rate']*100:.2f}% |",
        "",
        "## 3. First divergence",
        "",
        f"- Episodes with divergence: {summary['episodes_with_divergence']} / {summary['total_episodes']} "
        f"({summary['divergence_rate']*100:.1f}%)",
        f"- Mean first-divergence request index: {summary['first_divergence_index']['mean']:.2f} "
        f"(median {summary['first_divergence_index']['median']:.0f}, "
        f"p25 {summary['first_divergence_index']['p25']:.0f}, "
        f"p75 {summary['first_divergence_index']['p75']:.0f})",
        f"- First-divergence type counts: {summary['first_divergence_type_counts']}",
        "",
        "## 4. Time from divergence to first block",
        "",
        "| Method | Mean distance (requests) | Median |",
        "|---|---:|---:|",
        f"| v1 ranker   | {summary['blocks_after_divergence']['v1_mean']:.2f} | "
        f"{summary['blocks_after_divergence']['v1_median']:.0f} |",
        f"| DeepRMSA    | {summary['blocks_after_divergence']['deep_mean']:.2f} | "
        f"{summary['blocks_after_divergence']['deep_median']:.0f} |",
        "",
        "## 5. Resource delta at first divergence (v1 − DeepRMSA)",
        "",
        "### All first-divergence requests",
        "",
        "| Metric | Mean delta |",
        "|---|---:|",
    ]
    for k, v in summary["resource_delta_v1_minus_deep_at_divergence"]["all_first_divergences"].items():
        lines.append(f"| {k} | {v:+.6f} |")

    lines += [
        "",
        "### Only when both methods succeed at the divergence request",
        "",
        "| Metric | Mean delta |",
        "|---|---:|",
    ]
    for k, v in summary["resource_delta_v1_minus_deep_at_divergence"]["both_success"].items():
        lines.append(f"| {k} | {v:+.6f} |")

    lines += [
        "",
        "### Action-level delta (both succeed)",
        "",
        "| Metric | Mean delta (v1 − DeepRMSA) |",
        "|---|---:|",
    ]
    for k, v in summary["action_delta_v1_minus_deep_both_success"].items():
        val = f"{v:+.4f}" if v is not None else "N/A"
        lines.append(f"| {k} | {val} |")

    lines += [
        "",
        "## 6. Blocking reasons",
        "",
        "| Reason | v1 | DeepRMSA |",
        "|---|---:|---:|",
    ]
    all_reasons = sorted(set(summary["blocking_reason_counts"]["v1"]) | set(summary["blocking_reason_counts"]["deep"]))
    for reason in all_reasons:
        v1_c = summary["blocking_reason_counts"]["v1"].get(reason, 0)
        deep_c = summary["blocking_reason_counts"]["deep"].get(reason, 0)
        lines.append(f"| {reason} | {v1_c} | {deep_c} |")

    lines += [
        "",
        "## 7. Verdict",
        "",
        summary["verdict_text"],
        "",
    ]

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_k5m10_h5/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--requests_per_episode", type=int, default=80)
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/trajectory_divergence_diagnostic.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/trajectory_divergence_diagnostic.md")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    summary = evaluate(args)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")
    print("Verdict:", summary["verdict_text"])
