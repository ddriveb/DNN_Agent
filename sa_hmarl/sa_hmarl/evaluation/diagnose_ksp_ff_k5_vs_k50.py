"""Diagnostic script for KSP-FF K=5 vs K=50 overload anomaly.

This script performs three tasks requested in the audit:

1. Code-fairness check: run legacy ``ppo_c+ksp_ff`` and
   ``ppo_c+ksp_ff_k50_hops`` together with the new strict-fairness modes
   (``ksp_ff_plain_k5_hops``, ``ksp_ff_plain_k50_hops``,
   ``ksp_ff_highest_mod_k5_hops``, ``ksp_ff_highest_mod_k50_hops``) and verify
   that K5/K50 variants with the same selector differ *only* in ``env.k``.

2. Per-request trace: for one seed, record every request's C/R decisions,
   masks, selected path/mod/block, and block reason, then compare K5 vs K50
   trajectories to locate the first R-action divergence and quantify C-action
   drift before/after divergence.

3. Same-snapshot single-step K-extension test: deep-copy the environment at
   each request arrival (after time advancement), fix the PPO-C action, and
   evaluate both K5 and K50 R observations with the *same* selector.  This
   removes closed-loop state evolution and isolates the immediate feasibility
   effect of widening the candidate path horizon.

Output:
    sa_hmarl/experiments/ksp_k5_vs_k50_overload_audit.json
    sa_hmarl/experiments/ksp_k5_vs_k50_overload_audit.md
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Re-use the long-horizon evaluator infrastructure.
from sa_hmarl.evaluation.eval_long_horizon_system_comparison import (
    _file_sha256,
    _git_info,
    _r_backend_env_config,
    _select_c_action,
    _select_r_action_idx,
    generate_long_horizon_requests,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    load_ranking_checkpoint,
)
from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.agents.post_decision_finalizer_policy import PostDecisionFinalizerPolicy
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.baselines.rmsa_baselines import (
    ksp_ff_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


# ---------------------------------------------------------------------------
# Trace / snapshot helpers
# ---------------------------------------------------------------------------

TraceRecord = Dict[str, Any]


def _extract_r_action_details(
    obs_r: Dict[str, Any], r_idx: Optional[int]
) -> Dict[str, Any]:
    """Decode a flat R action into human-readable fields."""
    if r_idx is None:
        return {
            "path_idx": None,
            "mod_idx": None,
            "block_idx": None,
            "path_length_km": None,
            "hop_count": None,
            "required_fs": None,
            "block_start": None,
            "block_size": None,
        }
    num_mods = len(obs_r["mod_names"])
    num_blocks = len(obs_r["agent_r_mask"]) // (len(obs_r["candidate_paths"]) * num_mods)
    path_idx, mod_idx, block_idx = decode_agent_r_action(r_idx, num_mods, num_blocks)
    path_feat = obs_r["path_features"][path_idx]
    req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    block_start, block_size = (
        blocks[block_idx] if block_idx < len(blocks) else (None, None)
    )
    return {
        "path_idx": int(path_idx),
        "mod_idx": int(mod_idx),
        "block_idx": int(block_idx),
        "path_length_km": float(path_feat["path_length_km"]),
        "hop_count": int(path_feat["hop_count"]),
        "required_fs": int(req_fs) if req_fs is not None else None,
        "block_start": int(block_start) if block_start is not None else None,
        "block_size": int(block_size) if block_size is not None else None,
    }


def _run_episode_trace(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_policy,
    postdec_finalizer_policy,
    deep_rmsa,
    method_name: str,
    c_mode: str,
    r_mode: str,
    args: argparse.Namespace,
    seed: int,
    record_warmup: bool = False,
) -> List[TraceRecord]:
    """Run one episode and return a per-request trace list."""
    rng = np.random.RandomState(seed + 100003)
    server_selected_count = np.zeros(args.num_servers, dtype=int)
    warmup_requests = getattr(args, "warmup_requests", 0)
    trace: List[TraceRecord] = []

    for step_idx, req in enumerate(requests):
        started = time.perf_counter()
        env.advance_time(req.arrival_time)

        # Pre-C env config is always the main experiment configuration.
        env.k = args.k_paths
        env.path_sort_strategy = args.path_sort_strategy
        env.block_sort_strategy = args.block_sort_strategy
        obs_c = build_agent_c_observation(env, req)
        c_idx, raw_c_mask = _select_c_action(
            c_mode, agent_c, env, req, obs_c, args, rng, server_selected_count
        )
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        server_selected_count[server_id] += 1

        # Post-C R-side env config.
        pre_r_k, pre_r_path_sort, pre_r_block_sort = _r_backend_env_config(r_mode, args)
        env.k = pre_r_k
        env.path_sort_strategy = pre_r_path_sort
        env.block_sort_strategy = pre_r_block_sort
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_idx = _select_r_action_idx(
            r_mode, env, req, obs_c, obs_r, agent_r, rank_policy,
            postdec_finalizer_policy, split_id, server_id, deep_rmsa, args,
        )
        r_action = (
            (0, 0, 0)
            if r_idx is None
            else decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
        )
        decision_ms = (time.perf_counter() - started) * 1000.0

        _, _, _, info = env.step((split_id, server_id), r_action)

        if (not record_warmup) and step_idx < warmup_requests:
            continue

        feasible_counts = obs_c["feasible_counts"]
        trace.append({
            "seed": seed,
            "step_idx": step_idx,
            "req_id": req.req_id,
            "src_node": req.src_node,
            "arrival_time": req.arrival_time,
            "pre_c_k": args.k_paths,
            "pre_c_path_sort_strategy": args.path_sort_strategy,
            "pre_c_block_sort_strategy": args.block_sort_strategy,
            "c_idx": int(c_idx),
            "split_id": int(split_id),
            "server_id": int(server_id),
            "raw_c_mask_sum": int(raw_c_mask.sum()),
            "feasible_counts": feasible_counts,
            "r_mode": r_mode,
            "post_c_k": pre_r_k,
            "post_c_path_sort_strategy": pre_r_path_sort,
            "post_c_block_sort_strategy": pre_r_block_sort,
            "r_idx": int(r_idx) if r_idx is not None else None,
            "r_action_decoded": _extract_r_action_details(obs_r, r_idx),
            "block_reason": None if info.get("success") else info.get("reason"),
            "info_success": bool(info.get("success")),
            "decision_ms": decision_ms,
        })
    return trace


# ---------------------------------------------------------------------------
# Same-snapshot K-extension test
# ---------------------------------------------------------------------------


def _snapshot_selector_outcome(
    env_snapshot,
    req,
    obs_c: Dict[str, Any],
    split_id: int,
    server_id: int,
    k: int,
    selector,
) -> Dict[str, Any]:
    """Evaluate a single selector on a deep-copied env snapshot with the given K.

    Does *not* mutate the snapshot beyond setting env.k/path_sort/block_sort.
    """
    env_snapshot.k = k
    env_snapshot.path_sort_strategy = "hops"
    env_snapshot.block_sort_strategy = "start_asc"
    obs_r = build_agent_r_observation(env_snapshot, req, split_id, server_id)
    r_idx = selector(obs_r)
    mask_sum = int(np.asarray(obs_r["agent_r_mask"], dtype=bool).sum())
    return {
        "k": k,
        "selector": selector.__name__,
        "r_idx": int(r_idx) if r_idx is not None else None,
        "raw_r_mask_sum": mask_sum,
        "action_details": _extract_r_action_details(obs_r, r_idx),
    }


def _run_snapshot_k_extension(
    env,
    requests: List[Any],
    agent_c,
    args: argparse.Namespace,
    seed: int,
    max_snapshots: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """At each request arrival, deep-copy env, fix C action, test K5/K50 selectors."""
    rng = np.random.RandomState(seed + 100003)
    server_selected_count = np.zeros(args.num_servers, dtype=int)
    results: List[Dict[str, Any]] = []

    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)

        # Use the same pre-C config as the main evaluator.
        env.k = args.k_paths
        env.path_sort_strategy = args.path_sort_strategy
        env.block_sort_strategy = args.block_sort_strategy
        obs_c = build_agent_c_observation(env, req)
        c_idx, _ = _select_c_action(
            "ppo_c", agent_c, env, req, obs_c, args, rng, server_selected_count
        )
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        server_selected_count[server_id] += 1

        # Deep-copy the env *after* time advancement but *before* any allocation.
        snapshot = copy.deepcopy(env)

        # Evaluate both K values with both selectors on the identical snapshot.
        plain_k5 = _snapshot_selector_outcome(
            snapshot, req, obs_c, split_id, server_id, 5, ksp_ff_action
        )
        plain_k50 = _snapshot_selector_outcome(
            snapshot, req, obs_c, split_id, server_id, 50, ksp_ff_action
        )
        hm_k5 = _snapshot_selector_outcome(
            snapshot, req, obs_c, split_id, server_id, 5, ksp_ff_highest_mod_action
        )
        hm_k50 = _snapshot_selector_outcome(
            snapshot, req, obs_c, split_id, server_id, 50, ksp_ff_highest_mod_action
        )

        results.append({
            "seed": seed,
            "step_idx": step_idx,
            "req_id": req.req_id,
            "src_node": req.src_node,
            "arrival_time": req.arrival_time,
            "c_idx": int(c_idx),
            "split_id": int(split_id),
            "server_id": int(server_id),
            "plain_k5": plain_k5,
            "plain_k50": plain_k50,
            "highest_mod_k5": hm_k5,
            "highest_mod_k50": hm_k50,
        })

        # Continue the actual episode with the configured K5 baseline so the
        # reference trajectory remains consistent across snapshots.
        env.k, env.path_sort_strategy, env.block_sort_strategy = _r_backend_env_config(
            "ksp_ff_plain_k5_hops", args
        )
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_idx = ksp_ff_action(obs_r)
        r_action = (
            (0, 0, 0)
            if r_idx is None
            else decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
        )
        env.step((split_id, server_id), r_action)

        if max_snapshots is not None and step_idx + 1 >= max_snapshots:
            break

    return results


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------


def _find_first_divergence(trace_a: List[TraceRecord], trace_b: List[TraceRecord]) -> Optional[Dict[str, Any]]:
    """Find the first step where the R action differs between two traces."""
    n = min(len(trace_a), len(trace_b))
    for i in range(n):
        rec_a, rec_b = trace_a[i], trace_b[i]
        if rec_a["step_idx"] != rec_b["step_idx"]:
            continue
        if rec_a["r_idx"] != rec_b["r_idx"]:
            return {
                "step_idx": rec_a["step_idx"],
                "req_id": rec_a["req_id"],
                "c_idx_a": rec_a["c_idx"],
                "c_idx_b": rec_b["c_idx"],
                "r_idx_a": rec_a["r_idx"],
                "r_idx_b": rec_b["r_idx"],
                "r_action_a": rec_a["r_action_decoded"],
                "r_action_b": rec_b["r_action_decoded"],
                "block_reason_a": rec_a.get("block_reason"),
                "block_reason_b": rec_b.get("block_reason"),
            }
    return None


def _c_action_agreement(trace_a: List[TraceRecord], trace_b: List[TraceRecord]) -> Dict[str, Any]:
    """Compute C-action agreement overall, before divergence, and after divergence."""
    div = _find_first_divergence(trace_a, trace_b)
    div_step = div["step_idx"] if div is not None else None

    total = 0
    agree_total = 0
    agree_before = 0
    n_before = 0
    agree_after = 0
    n_after = 0

    for rec_a, rec_b in zip(trace_a, trace_b):
        if rec_a["step_idx"] != rec_b["step_idx"]:
            continue
        total += 1
        agree = int(rec_a["c_idx"]) == int(rec_b["c_idx"])
        agree_total += agree
        if div_step is None or rec_a["step_idx"] < div_step:
            n_before += 1
            agree_before += agree
        else:
            n_after += 1
            agree_after += agree

    return {
        "first_divergence_step": div_step,
        "first_divergence_req_id": div["req_id"] if div else None,
        "overall_c_agreement": agree_total / total if total else None,
        "before_divergence_c_agreement": agree_before / n_before if n_before else None,
        "after_divergence_c_agreement": agree_after / n_after if n_after else None,
        "n_total": total,
        "n_before": n_before,
        "n_after": n_after,
    }


def _split_server_distributions(trace: List[TraceRecord]) -> Dict[str, Any]:
    """Aggregate split/server selection distributions from a trace."""
    n = len(trace)
    splits: Dict[int, int] = {}
    servers: Dict[int, int] = {}
    for rec in trace:
        splits[rec["split_id"]] = splits.get(rec["split_id"], 0) + 1
        servers[rec["server_id"]] = servers.get(rec["server_id"], 0) + 1
    return {
        "n_requests": n,
        "split_dist": {str(k): v / n for k, v in sorted(splits.items())},
        "server_dist": {str(k): v / n for k, v in sorted(servers.items())},
    }


def _selected_path_stats(trace: List[TraceRecord]) -> Dict[str, Any]:
    """Average selected path length/hops/required FS for admitted requests."""
    admitted = [r for r in trace if r["block_reason"] is None and r["r_idx"] is not None]
    if not admitted:
        return {}
    lens = [r["r_action_decoded"]["path_length_km"] for r in admitted if r["r_action_decoded"]["path_length_km"] is not None]
    hops = [r["r_action_decoded"]["hop_count"] for r in admitted if r["r_action_decoded"]["hop_count"] is not None]
    fs = [r["r_action_decoded"]["required_fs"] for r in admitted if r["r_action_decoded"]["required_fs"] is not None]
    return {
        "n_admitted": len(admitted),
        "mean_path_length_km": float(np.mean(lens)) if lens else None,
        "mean_hop_count": float(np.mean(hops)) if hops else None,
        "mean_required_fs": float(np.mean(fs)) if fs else None,
    }


# ---------------------------------------------------------------------------
# Snapshot analysis
# ---------------------------------------------------------------------------


def _analyze_snapshots(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize the same-snapshot K5 vs K50 selector behavior."""
    plain_same = 0
    plain_diff = 0
    plain_k50_admits_k5_blocks = 0
    plain_k50_more_feasible = 0
    plain_k50_less_feasible = 0
    plain_k50_equal_feasible = 0

    hm_same = 0
    hm_diff = 0

    plain_k5_path_lengths = []
    plain_k50_path_lengths = []
    hm_k5_path_lengths = []
    hm_k50_path_lengths = []

    plain_k5_hops = []
    plain_k50_hops = []
    hm_k5_hops = []
    hm_k50_hops = []

    plain_k5_fs = []
    plain_k50_fs = []
    hm_k5_fs = []
    hm_k50_fs = []

    for rec in snapshots:
        p5 = rec["plain_k5"]
        p50 = rec["plain_k50"]
        h5 = rec["highest_mod_k5"]
        h50 = rec["highest_mod_k50"]

        # Plain selector comparison.
        if p5["r_idx"] == p50["r_idx"]:
            plain_same += 1
        else:
            plain_diff += 1
        if p50["raw_r_mask_sum"] > p5["raw_r_mask_sum"]:
            plain_k50_more_feasible += 1
        elif p50["raw_r_mask_sum"] < p5["raw_r_mask_sum"]:
            plain_k50_less_feasible += 1
        else:
            plain_k50_equal_feasible += 1
        if p5["r_idx"] is not None and p50["r_idx"] is not None:
            # Does K50's chosen action lie in the K5 prefix (same flat index)?
            if p50["r_idx"] == p5["r_idx"]:
                plain_k50_admits_k5_blocks += 1
            for src, dst, key in (
                (p5, p50, "plain"),
                (h5, h50, "hm"),
            ):
                if key == "plain":
                    if src["action_details"]["path_length_km"] is not None:
                        plain_k5_path_lengths.append(src["action_details"]["path_length_km"])
                    if dst["action_details"]["path_length_km"] is not None:
                        plain_k50_path_lengths.append(dst["action_details"]["path_length_km"])
                    if src["action_details"]["hop_count"] is not None:
                        plain_k5_hops.append(src["action_details"]["hop_count"])
                    if dst["action_details"]["hop_count"] is not None:
                        plain_k50_hops.append(dst["action_details"]["hop_count"])
                    if src["action_details"]["required_fs"] is not None:
                        plain_k5_fs.append(src["action_details"]["required_fs"])
                    if dst["action_details"]["required_fs"] is not None:
                        plain_k50_fs.append(dst["action_details"]["required_fs"])
                else:
                    if src["action_details"]["path_length_km"] is not None:
                        hm_k5_path_lengths.append(src["action_details"]["path_length_km"])
                    if dst["action_details"]["path_length_km"] is not None:
                        hm_k50_path_lengths.append(dst["action_details"]["path_length_km"])
                    if src["action_details"]["hop_count"] is not None:
                        hm_k5_hops.append(src["action_details"]["hop_count"])
                    if dst["action_details"]["hop_count"] is not None:
                        hm_k50_hops.append(dst["action_details"]["hop_count"])
                    if src["action_details"]["required_fs"] is not None:
                        hm_k5_fs.append(src["action_details"]["required_fs"])
                    if dst["action_details"]["required_fs"] is not None:
                        hm_k50_fs.append(dst["action_details"]["required_fs"])

        # Highest-mod selector comparison.
        if h5["r_idx"] == h50["r_idx"]:
            hm_same += 1
        else:
            hm_diff += 1

    n = len(snapshots)
    return {
        "n_snapshots": n,
        "plain_selector": {
            "same_action_count": plain_same,
            "diff_action_count": plain_diff,
            "same_action_rate": plain_same / n if n else None,
            "k50_more_feasible_count": plain_k50_more_feasible,
            "k50_less_feasible_count": plain_k50_less_feasible,
            "k50_equal_feasible_count": plain_k50_equal_feasible,
            "k50_selects_k5_action_count": plain_k50_admits_k5_blocks,
            "mean_path_length_km_k5": float(np.mean(plain_k5_path_lengths)) if plain_k5_path_lengths else None,
            "mean_path_length_km_k50": float(np.mean(plain_k50_path_lengths)) if plain_k50_path_lengths else None,
            "mean_hop_count_k5": float(np.mean(plain_k5_hops)) if plain_k5_hops else None,
            "mean_hop_count_k50": float(np.mean(plain_k50_hops)) if plain_k50_hops else None,
            "mean_required_fs_k5": float(np.mean(plain_k5_fs)) if plain_k5_fs else None,
            "mean_required_fs_k50": float(np.mean(plain_k50_fs)) if plain_k50_fs else None,
        },
        "highest_mod_selector": {
            "same_action_count": hm_same,
            "diff_action_count": hm_diff,
            "same_action_rate": hm_same / n if n else None,
            "mean_path_length_km_k5": float(np.mean(hm_k5_path_lengths)) if hm_k5_path_lengths else None,
            "mean_path_length_km_k50": float(np.mean(hm_k50_path_lengths)) if hm_k50_path_lengths else None,
            "mean_hop_count_k5": float(np.mean(hm_k5_hops)) if hm_k5_hops else None,
            "mean_hop_count_k50": float(np.mean(hm_k50_hops)) if hm_k50_hops else None,
            "mean_required_fs_k5": float(np.mean(hm_k5_fs)) if hm_k5_fs else None,
            "mean_required_fs_k50": float(np.mean(hm_k50_fs)) if hm_k50_fs else None,
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------


def _load_dependencies(args: argparse.Namespace):
    """Load Agent-C, Agent-R, ranking policy, etc."""
    env_proto = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
        path_sort_strategy=args.path_sort_strategy,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, rank_ckpt = load_ranking_checkpoint(
        args.ranking_checkpoint, args.device
    )
    rank_policy = CounterfactualRRankerPolicy(
        rank_model,
        rank_mean,
        rank_std,
        device=args.device,
        feature_names=rank_ckpt.get("feature_names"),
        candidate_mode=rank_ckpt.get("candidate_mode", "all_legal"),
        max_candidates=rank_ckpt.get("max_candidates", 48),
        ppo_top_k=rank_ckpt.get("ppo_top_k", 8),
        num_random_candidates=rank_ckpt.get("num_random_candidates", 5),
        min_candidates=rank_ckpt.get("min_candidates", 15),
        candidate_seed=rank_ckpt.get("candidate_seed", 12345),
        ensure_ksp_action=args.ranker_ensure_ksp,
    )
    return env_proto, agent_c, agent_r, rank_policy, None


def _generate_requests_for_seed(env_proto, rng: np.random.RandomState, args: argparse.Namespace):
    """Generate one episode's requests."""
    return generate_long_horizon_requests(
        env_proto.net.NUM_NODES,
        rng,
        args.requests_per_episode,
        args.arrival_interval,
        args.holding_min,
        args.holding_max,
        args.deadline_min,
        args.deadline_max,
        args.size_min_mb,
        args.size_max_mb,
        args.edge_cost_min,
        args.edge_cost_max,
        args.num_splits,
        args.split_profile,
        poisson_arrivals=args.poisson_arrivals,
        exponential_holding=args.exponential_holding,
    )


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    env_proto, agent_c, agent_r, rank_policy, postdec = _load_dependencies(args)

    # --- Trace collection for one seed ---
    trace_seed = int(args.trace_seed)
    rng = np.random.RandomState(trace_seed)
    requests = _generate_requests_for_seed(env_proto, rng, args)

    trace_methods = {
        "ksp_ff_plain_k5_hops": "ksp_ff_plain_k5_hops",
        "ksp_ff_plain_k50_hops": "ksp_ff_plain_k50_hops",
        "ksp_ff_highest_mod_k5_hops": "ksp_ff_highest_mod_k5_hops",
        "ksp_ff_highest_mod_k50_hops": "ksp_ff_highest_mod_k50_hops",
    }
    traces: Dict[str, List[TraceRecord]] = {}
    for name, r_mode in trace_methods.items():
        env = make_env(
            args.topology,
            args.num_slots,
            args.num_servers,
            trace_seed,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
            k=args.k_paths,
            path_sort_strategy=args.path_sort_strategy,
        )
        env.reset(requests)
        traces[name] = _run_episode_trace(
            env, requests, agent_c, agent_r, rank_policy, postdec, None,
            name, "ppo_c", r_mode, args, trace_seed,
        )
        print(f"[trace] {name}: {len(traces[name])} records", flush=True)

    # --- Snapshot K-extension test on the same seed ---
    env = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        trace_seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
        path_sort_strategy=args.path_sort_strategy,
    )
    env.reset(requests)
    snapshots = _run_snapshot_k_extension(
        env, requests, agent_c, args, trace_seed,
        max_snapshots=args.max_snapshots,
    )
    print(f"[snapshot] collected {len(snapshots)} snapshots", flush=True)

    # --- Comparative analyses ---
    comparisons = {}
    pairs = [
        ("plain_k5_vs_k50", "ksp_ff_plain_k5_hops", "ksp_ff_plain_k50_hops"),
        ("highest_mod_k5_vs_k50", "ksp_ff_highest_mod_k5_hops", "ksp_ff_highest_mod_k50_hops"),
    ]
    for key, mode_a, mode_b in pairs:
        comparisons[key] = {
            "c_action_agreement": _c_action_agreement(traces[mode_a], traces[mode_b]),
            "split_server_a": _split_server_distributions(traces[mode_a]),
            "split_server_b": _split_server_distributions(traces[mode_b]),
            "selected_path_stats_a": _selected_path_stats(traces[mode_a]),
            "selected_path_stats_b": _selected_path_stats(traces[mode_b]),
        }

    snapshot_summary = _analyze_snapshots(snapshots)

    return {
        "config": vars(args),
        "trace_seed": trace_seed,
        "traces": {k: v for k, v in traces.items()},
        "snapshots": snapshots,
        "comparisons": comparisons,
        "snapshot_summary": snapshot_summary,
        "metadata": {
            "git": _git_info(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "checkpoint_sha256": {
                "agent_c": _file_sha256(args.agent_c_checkpoint),
                "agent_r": _file_sha256(args.agent_r_checkpoint),
                "ranking": _file_sha256(args.ranking_checkpoint),
            },
        },
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(value, fmt: str = ".4f") -> str:
    return f"{value:{fmt}}" if value is not None else "N/A"


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    cfg = report["config"]
    lines = [
        "# KSP-FF K=5 vs K=50 Overload Audit",
        "",
        "## Purpose",
        "",
        "Diagnose why `ppo_c+ksp_ff_k50_hops` shows higher blocking than "
        "`ppo_c+ksp_ff` under the COST239 long-horizon protocol.",
        "",
        "## Key Code-Audit Finding",
        "",
        "The legacy evaluators map:",
        "",
        "- `ksp_ff` / `ksp_ff_k5_hops` → **plain** First-Fit selector.",
        "- `ksp_ff_k50_hops` → **highest-modulation** First-Fit selector.",
        "",
        "Therefore the historical K5 vs K50 comparison was **not** a fair K-only "
        "comparison; it confounded path-horizon width with selector/modulation "
        "policy.  This audit introduces strict fairness modes where K5 and K50 "
        "share the identical selector and block ordering.",
        "",
        "## Configuration",
        "",
    ]
    for key in [
        "topology", "num_slots", "num_servers", "seeds", "requests_per_episode",
        "warmup_requests", "arrival_interval", "holding_min", "holding_max",
        "edge_cost_min", "edge_cost_max", "path_sort_strategy",
        "block_sort_strategy", "k_paths", "ksp_ff_k50_hops_k_paths",
        "agent_c_checkpoint",
    ]:
        lines.append(f"- `{key}`: `{cfg.get(key)}`")

    lines.extend(["", "## Per-Request Trace Comparison", ""])
    for comp_key, comp in report["comparisons"].items():
        lines.append(f"### {comp_key}")
        lines.append("")
        agree = comp["c_action_agreement"]
        lines.append("**C-action agreement**")
        lines.append(f"- First R-action divergence at step: {agree['first_divergence_step']}")
        overall = agree['overall_c_agreement']
        before = agree['before_divergence_c_agreement']
        after = agree['after_divergence_c_agreement']
        lines.append(f"- Overall C agreement: {_fmt(overall)}")
        lines.append(f"- Before divergence C agreement: {_fmt(before)} (n={agree['n_before']})")
        lines.append(f"- After divergence C agreement: {_fmt(after)} (n={agree['n_after']})")
        lines.append("")
        lines.append("**Split / server distributions**")
        for label, dist_key in (("K5", "split_server_a"), ("K50", "split_server_b")):
            d = comp[dist_key]
            lines.append(f"- {label}: split_dist={d['split_dist']}, server_dist={d['server_dist']}")
        lines.append("")
        lines.append("**Selected path statistics (admitted requests)**")
        for label, stats_key in (("K5", "selected_path_stats_a"), ("K50", "selected_path_stats_b")):
            s = comp[stats_key]
            lines.append(
                f"- {label}: n_admitted={s.get('n_admitted')}, "
                f"mean_path_length_km={_fmt(s.get('mean_path_length_km'), '.2f')}, "
                f"mean_hop_count={_fmt(s.get('mean_hop_count'), '.2f')}, "
                f"mean_required_fs={_fmt(s.get('mean_required_fs'), '.2f')}"
            )
        lines.append("")

    lines.extend(["## Same-Snapshot Single-Step K-Extension Test", ""])
    ss = report["snapshot_summary"]
    lines.append(f"- Snapshots evaluated: {ss['n_snapshots']}")
    lines.append("")
    lines.append("**Plain First-Fit selector**")
    p = ss["plain_selector"]
    lines.append(f"- Same action K5 vs K50: {p['same_action_count']} / {ss['n_snapshots']} ({_fmt(p['same_action_rate'])})")
    lines.append(
        f"- K50 more feasible: {p['k50_more_feasible_count']}, "
        f"equal: {p['k50_equal_feasible_count']}, less: {p['k50_less_feasible_count']}"
    )
    lines.append(f"- Mean path length km: K5={_fmt(p['mean_path_length_km_k5'], '.2f')}, K50={_fmt(p['mean_path_length_km_k50'], '.2f')}")
    lines.append(f"- Mean hop count: K5={_fmt(p['mean_hop_count_k5'], '.2f')}, K50={_fmt(p['mean_hop_count_k50'], '.2f')}")
    lines.append(f"- Mean required FS: K5={_fmt(p['mean_required_fs_k5'], '.2f')}, K50={_fmt(p['mean_required_fs_k50'], '.2f')}")
    lines.append("")
    lines.append("**Highest-modulation First-Fit selector**")
    h = ss["highest_mod_selector"]
    lines.append(f"- Same action K5 vs K50: {h['same_action_count']} / {ss['n_snapshots']} ({_fmt(h['same_action_rate'])})")
    lines.append(f"- Mean path length km: K5={_fmt(h['mean_path_length_km_k5'], '.2f')}, K50={_fmt(h['mean_path_length_km_k50'], '.2f')}")
    lines.append(f"- Mean hop count: K5={_fmt(h['mean_hop_count_k5'], '.2f')}, K50={_fmt(h['mean_hop_count_k50'], '.2f')}")
    lines.append(f"- Mean required FS: K5={_fmt(h['mean_required_fs_k5'], '.2f')}, K50={_fmt(h['mean_required_fs_k50'], '.2f')}")
    lines.append("")

    lines.extend([
        "## Interpretation Notes",
        "",
        "1. If the same-selector K50 blocking is still higher than K50 in the "
        "strict-fairness rerun, the cause is closed-loop resource evolution "
        "(e.g. K50 admits more load onto already-stressed servers, or selects "
        "longer paths that increase resource occupancy).",
        "",
        "2. If same-selector K50 blocking is lower or equal to K5, the original "
        "anomaly was primarily an implementation unfairness (different selectors).",
        "",
        "3. The snapshot test distinguishes immediate feasibility from closed-loop "
        "drift: if K50 is at least as feasible in the snapshot but worse in the "
        "full episode, the bottleneck is dynamic state evolution, not single-step "
        "path reachability.",
        "",
    ])

    meta = report.get("metadata", {})
    if meta:
        lines.extend(["## Reproducibility Metadata", ""])
        git = meta.get("git", {})
        lines.append(f"- Git commit: `{git.get('commit', 'N/A')}`")
        lines.append(f"- Git dirty: `{git.get('dirty', 'N/A')}`")
        lines.append(f"- Python: `{meta.get('python_version', 'N/A').split()[0]}`")
        lines.append(f"- Torch: `{meta.get('torch_version', 'N/A')}`")
        lines.append(f"- Numpy: `{meta.get('numpy_version', 'N/A')}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--trace_seed", type=int, default=3030)
    parser.add_argument("--max_snapshots", type=int, default=None,
                        help="Limit the number of snapshots for quick testing.")
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--ksp_ff_k50_hops_k_paths", type=int, default=50)
    parser.add_argument("--path_sort_strategy", default="hops", choices=["km", "hops"])
    parser.add_argument("--block_sort_strategy", default="start_asc")
    parser.add_argument("--requests_per_episode", type=int, default=10000)
    parser.add_argument("--warmup_requests", type=int, default=2000)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--poisson_arrivals", action="store_true")
    parser.add_argument("--exponential_holding", action="store_true")
    parser.add_argument("--ranker_ensure_ksp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/ksp_k5_vs_k50_overload_audit.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/ksp_k5_vs_k50_overload_audit.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_audit(args)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), report)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
