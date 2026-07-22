"""Main S100 system-level comparison for the final paper/PPT.

This script moves the main comparison from the very high-blocking S24 stress
setting to a more realistic S100 setting.  It evaluates both:

1. R-side backends under the same learned PPO-C policy.
2. C-side heuristic baselines paired with the final v1.2 planner-distilled
   R-ranker.

Default methods:
    ppo_c+v12, ppo_c+deep_rmsa, ppo_c+ppo_r, ppo_c+ksp_bf, ppo_c+ksp_ff,
    ppo_c+ksp_ff_k50_hops, greedy_c+v12, df_c+v12, rf_c+v12, wo_c+v12,
    iwd_c+v12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.agents.post_decision_finalizer_policy import PostDecisionFinalizerPolicy
from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.baselines.rmsa_baselines import (
    ksp_bf_action,
    ksp_ff_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
    _select_r_action,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.evaluation.server_diagnostics import ServerDiagnostics
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _r_backend_env_config(r_mode: str, args: argparse.Namespace) -> tuple[int, str, str]:
    """Return the R-side path breadth/order for a backend.

    PPO-C is kept on the main experiment path configuration.  Tuned R-only
    heuristic baselines can override the R observation and execution path set.

    The ``ppo_r_k50_hops`` and ``ppo_r_proposer_ksp_ff`` modes give PPO-R the
    same wider candidate horizon used by KSP-FF K50 and v1.2-k50, so we can
    isolate whether PPO-R is candidate-limited or structurally weak as a final
    backend.

    Strict fairness modes for KSP-FF audit (plain vs highest-mod × K5 vs K50)
    guarantee that the only difference between K5 and K50 variants is the
    ``env.k`` value; they share exactly the same action selector and block
    ordering.
    """
    if r_mode in ("ksp_ff_k50_hops", "ksp_ff_plain_k50_hops", "ksp_ff_highest_mod_k50_hops",
                  "v12_k50_hops", "v13_k50_hops", "ppo_r_k50_hops", "ppo_r_proposer_ksp_ff",
                  "ppo_r_proposer_postdec", "mixed32_postdec", "deep_rmsa_style_k50_hops"):
        return args.ksp_ff_k50_hops_k_paths, "hops", "start_asc"
    if r_mode in ("ksp_ff_plain_k5_hops", "ksp_ff_highest_mod_k5_hops", "v13_k5_hops"):
        return args.k_paths, "hops", "start_asc"
    return args.k_paths, args.path_sort_strategy, args.block_sort_strategy


def _parse_methods(spec: str) -> List[tuple[str, str, str]]:
    """Parse method specs like ``ppo_c+v12`` into (name, c_mode, r_mode)."""
    methods = []
    for item in spec.split(","):
        name = item.strip()
        if not name:
            continue
        if "+" not in name:
            raise ValueError(f"Method must look like c_mode+r_mode: {name}")
        c_mode, r_mode = name.split("+", 1)
        methods.append((name, c_mode.strip(), r_mode.strip()))
    return methods


def _select_c_action(
    c_mode: str,
    agent_c,
    env,
    req,
    obs_c: Dict[str, Any],
    args: argparse.Namespace,
    rng: np.random.RandomState,
    server_selected_count: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Select a flat C action id, returning raw mask for metric accounting."""
    raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    if c_mode == "ppo_c":
        c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        return int(c_idx if c_idx is not None else 0), np.asarray(raw_c_mask, dtype=bool)

    c_idx = select_offloading_action(
        c_mode.replace("_c", ""),
        env,
        req,
        obs_c,
        raw_mask,
        rng=rng,
        server_selected_count=server_selected_count,
    )
    return int(c_idx if c_idx is not None else 0), raw_mask


def _ppo_r_topk_actions(agent_r, obs_r: Dict[str, Any], top_k: int) -> List[int]:
    """Return the top-k legal flat action ids ranked by PPO-R logits."""
    import numpy as np
    import torch

    features, mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(legal) == 0:
        return []
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    top_k = min(top_k, len(legal))
    top_local = np.argsort(-logits[legal], kind="stable")[:top_k]
    return [int(legal[i]) for i in top_local]


def _ksp_ff_highest_mod_action_subset(
    obs_r: Dict[str, Any], allowed_set: set
) -> Optional[int]:
    """KSP-FF-highest-mod restricted to a subset of allowed flat actions."""
    import numpy as np

    mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    num_paths = len(obs_r["candidate_paths"])
    num_mods = len(obs_r["mod_names"])
    if num_paths == 0 or num_mods == 0:
        return None
    num_blocks = len(mask) // (num_paths * num_mods)

    for path_idx in range(num_paths):
        best_mod = None
        best_req_fs = float("inf")
        for mod_idx in range(num_mods):
            req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
            if req_fs is None or req_fs <= 0:
                continue
            start = path_idx * num_mods * num_blocks + mod_idx * num_blocks
            end = start + num_blocks
            if not any((a in allowed_set and mask[a]) for a in range(start, end)):
                continue
            if float(req_fs) < best_req_fs:
                best_req_fs = float(req_fs)
                best_mod = mod_idx

        if best_mod is None:
            continue

        start = path_idx * num_mods * num_blocks + best_mod * num_blocks
        blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][best_mod]
        valid_blocks = []
        for block_idx in range(num_blocks):
            action_idx = start + block_idx
            if not (action_idx in allowed_set and mask[action_idx]):
                continue
            block_start = blocks[block_idx][0] if block_idx < len(blocks) else block_idx
            valid_blocks.append((block_start, block_idx, action_idx))
        if valid_blocks:
            valid_blocks.sort(key=lambda item: (item[0], item[1]))
            return int(valid_blocks[0][2])
    return None


def _deep_rmsa_style_k50_action(obs_r: Dict[str, Any], mod_reg) -> Optional[int]:
    """Heuristic DeepRMSA-style selector over the K=50 hop-ordered path set.

    DeepRMSA's action space is (path, FS-block).  The modulation for the
    chosen path is always the highest-spectral-efficiency feasible format.
    We mimic this by, for each candidate path, evaluating only the legal
    blocks belonging to the best feasible modulation, and scoring them with
    a hand-crafted DeepRMSA-like objective: tight fit, short path, few slots.

    This lets us compare v1.3 against a DeepRMSA-style policy with the same
    path-support horizon (K=50, hop-ordered) without retraining the neural
    DeepRMSA checkpoint, whose input/output dimensions are tied to K=5.
    """
    import numpy as np

    mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return None

    num_paths = len(obs_r["candidate_paths"])
    num_mods = len(obs_r["mod_names"])
    if num_paths == 0 or num_mods == 0:
        return None
    num_blocks = len(mask) // (num_paths * num_mods)

    max_path_km = max(
        (float(f.get("path_length_km", 0.0)) for f in obs_r["path_features"]),
        default=1.0,
    )
    if max_path_km <= 0.0:
        max_path_km = 1.0
    num_slots = int(obs_r.get("num_slots", 320))

    # Precompute best (highest-SE feasible) modulation per path, matching DeepRMSA.
    best_mod_per_path: List[Optional[int]] = []
    for p_idx in range(num_paths):
        best_mod = None
        best_se = -1.0
        for m_idx in range(num_mods):
            if obs_r["feasible_mask_per_path_mod"][p_idx][m_idx]:
                se = mod_reg[m_idx].spectral_efficiency
                if se > best_se:
                    best_se = se
                    best_mod = m_idx
        best_mod_per_path.append(best_mod)

    best_action: Optional[int] = None
    best_score = float("-inf")
    for action_idx in legal:
        path_idx, mod_idx, block_idx = decode_agent_r_action(action_idx, num_mods, num_blocks)
        if path_idx >= num_paths:
            continue
        if best_mod_per_path[path_idx] != mod_idx:
            continue
        req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
        if req_fs is None or req_fs <= 0:
            continue
        blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        if block_idx >= len(blocks):
            continue
        block_start, block_size = blocks[block_idx]
        waste = (block_size - req_fs) / block_size if block_size > 0 else 1.0

        path_km = float(obs_r["path_features"][path_idx].get("path_length_km", 0.0))

        score = (
            -1.00 * waste
            - 0.30 * (path_km / max_path_km)
            - 0.20 * (req_fs / num_slots)
            - 0.05 * (block_start / num_slots)
        )
        if score > best_score:
            best_score = score
            best_action = int(action_idx)

    return best_action


def _select_r_action_idx(
    r_mode: str,
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    rank_policy: CounterfactualRRankerPolicy,
    postdec_finalizer_policy: Optional[PostDecisionFinalizerPolicy],
    split_id: int,
    server_id: int,
    deep_rmsa,
    args: argparse.Namespace,
) -> Optional[int]:
    if r_mode in ("v12", "v12_k50_hops", "v13_k5_hops", "v13_k50_hops"):
        return rank_policy.select_action(env, req, obs_c, obs_r, agent_r, split_id, server_id)
    if r_mode == "deep_rmsa":
        return deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
    if r_mode == "deep_rmsa_style_k50_hops":
        return _deep_rmsa_style_k50_action(obs_r, env.mod_reg)
    if r_mode == "ppo_r":
        return agent_r.select_action(obs_r, deterministic=True)
    if r_mode == "ppo_r_k50_hops":
        # Evaluate the existing PPO-R checkpoint on the wider K50/hops candidate set.
        return agent_r.select_action(obs_r, deterministic=True)
    if r_mode == "ppo_r_proposer_ksp_ff":
        # Use PPO-R only to propose a small candidate subset, then let KSP-FF
        # pick the final action.  This tests whether PPO-R is better as a
        # proposer than as a final backend.
        allowed = _ppo_r_topk_actions(agent_r, obs_r, args.ppo_r_proposer_top_k)
        if not allowed:
            return agent_r.select_action(obs_r, deterministic=True)
        selected = _ksp_ff_highest_mod_action_subset(obs_r, set(allowed))
        if selected is not None:
            return selected
        return agent_r.select_action(obs_r, deterministic=True)
    if r_mode == "ppo_r_proposer_postdec":
        # Use PPO-R as a proposer and a learned post-decision value ranker as
        # the finalizer.  This is the learned counterpart to
        # ``ppo_r_proposer_ksp_ff``.
        if postdec_finalizer_policy is None:
            raise RuntimeError(
                "ppo_r_proposer_postdec requires --post_decision_checkpoint"
            )
        return postdec_finalizer_policy.select_action(
            env, req, obs_c, obs_r, agent_r, split_id, server_id
        )
    if r_mode == "mixed32_postdec":
        # Mixed candidate retrieval: PPO-R top-K + heuristic anchors, scored by
        # a learned post-decision finalizer.
        if postdec_finalizer_policy is None:
            raise RuntimeError(
                "mixed32_postdec requires --post_decision_checkpoint"
            )
        return postdec_finalizer_policy.select_action(
            env, req, obs_c, obs_r, agent_r, split_id, server_id,
            candidate_mode="mixed32",
        )
    if r_mode == "ksp_bf":
        return ksp_bf_action(obs_r)
    # Legacy aliases (kept for backward compatibility).  Historically
    # ``ksp_ff`` mapped to plain First-Fit while ``ksp_ff_k50_hops`` mapped to
    # highest-modulation First-Fit, so they were NOT a fair K-only comparison.
    if r_mode in ("ksp_ff", "ksp_ff_k5_hops", "ksp_ff_plain_k5_hops", "ksp_ff_plain_k50_hops"):
        return ksp_ff_action(obs_r)
    if r_mode in ("ksp_ff_k50_hops", "ksp_ff_highest_mod_k5_hops", "ksp_ff_highest_mod_k50_hops"):
        return ksp_ff_highest_mod_action(obs_r)
    raise ValueError(f"Unknown R mode: {r_mode}")


def _run_episode(
    env,
    requests,
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
    metrics: PerMethodMetrics,
    server_diag: Optional[ServerDiagnostics] = None,
) -> None:
    rng = np.random.RandomState(seed + 100003)
    server_selected_count = np.zeros(args.num_servers, dtype=int)

    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)

        # Keep the C-side policy fixed on the main experiment configuration.
        env.k = args.k_paths
        env.path_sort_strategy = args.path_sort_strategy
        env.block_sort_strategy = args.block_sort_strategy
        obs_c = build_agent_c_observation(env, req)
        c_idx, raw_c_mask = _select_c_action(
            c_mode, agent_c, env, req, obs_c, args, rng, server_selected_count
        )
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        server_selected_count[server_id] += 1

        # R-side tuned baselines may use a wider/differently ordered path set.
        env.k, env.path_sort_strategy, env.block_sort_strategy = _r_backend_env_config(
            r_mode, args
        )
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
        if server_diag is not None:
            server_diag.record_step(env, split_id, server_id, info)
        same_as_ppo = True
        if r_idx is not None:
            ppo_idx = agent_r.select_action(obs_r, deterministic=True)
            same_as_ppo = int(r_idx) == int(ppo_idx if ppo_idx is not None else 0)

        profile = rank_policy.last_profile if r_mode in ("v12", "v12_k50_hops", "v13_k5_hops", "v13_k50_hops") else None
        _record_outcome(
            metrics,
            info,
            int(raw_c_mask.sum()) == 0,
            obs_c,
            split_id,
            server_id,
            decision_ms,
            same_as_ppo,
            profile=profile,
        )
        metrics.active_connections[-1] = len(env.active_connections)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = _parse_methods(args.methods)
    requested_r = {r for _, _, r in methods}

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
        candidate_mode=args.ranker_candidate_mode if args.ranker_candidate_mode is not None else rank_ckpt.get("candidate_mode", "all_legal"),
        max_candidates=args.ranker_max_candidates if args.ranker_max_candidates is not None else rank_ckpt.get("max_candidates", 48),
        ppo_top_k=args.ranker_ppo_top_k if args.ranker_ppo_top_k is not None else rank_ckpt.get("ppo_top_k", 8),
        num_random_candidates=args.ranker_num_random_candidates if args.ranker_num_random_candidates is not None else rank_ckpt.get("num_random_candidates", 5),
        min_candidates=args.ranker_min_candidates if args.ranker_min_candidates is not None else rank_ckpt.get("min_candidates", 15),
        candidate_seed=rank_ckpt.get("candidate_seed", 12345),
        ensure_ksp_action=args.ranker_ensure_ksp,
        enable_profile=args.ranker_enable_profile,
    )

    deep_rmsa = None
    if "deep_rmsa" in requested_r:
        deep_rmsa = _load_deep_rmsa(
            args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device
        )

    postdec_finalizer_policy = None
    if "ppo_r_proposer_postdec" in requested_r or "mixed32_postdec" in requested_r:
        if not args.post_decision_checkpoint:
            raise RuntimeError(
                "--post_decision_checkpoint is required for ppo_r_proposer_postdec"
            )
        postdec_finalizer_policy = PostDecisionFinalizerPolicy.from_checkpoint(
            args.post_decision_checkpoint,
            device=args.device,
            top_k=args.ppo_r_postdec_top_k,
            ensure_ksp_anchor=args.ppo_r_postdec_ensure_ksp,
        )

    episodes_by_seed = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(generate_requests(
                env_proto,
                rng,
                src,
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
            ))
        episodes_by_seed[seed] = episodes

    report_methods: Dict[str, Any] = {}
    for method_name, c_mode, r_mode in methods:
        spec: Dict[str, Any] = {
            "c_mode": c_mode,
            "r_mode": r_mode,
            "per_seed": {},
        }
        if args.collect_server_diagnostics:
            spec["server_diagnostics"] = {}
        for seed in seeds:
            metrics = PerMethodMetrics()
            server_diag = None
            if args.collect_server_diagnostics:
                server_diag = ServerDiagnostics(
                    num_servers=args.num_servers,
                    num_splits=args.num_splits,
                    high_util_threshold=args.server_diag_high_util_threshold,
                )
            for ep_idx, requests in enumerate(episodes_by_seed[seed]):
                env = make_env(
                    args.topology,
                    args.num_slots,
                    args.num_servers,
                    seed + ep_idx,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks,
                    block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                    path_sort_strategy=args.path_sort_strategy,
                )
                env.reset(requests)
                _run_episode(
                    env,
                    requests,
                    agent_c,
                    agent_r,
                    rank_policy,
                    postdec_finalizer_policy,
                    deep_rmsa,
                    method_name,
                    c_mode,
                    r_mode,
                    args,
                    seed + ep_idx,
                    metrics,
                    server_diag,
                )
            spec["per_seed"][str(seed)] = _aggregate_metrics(metrics)
            if server_diag is not None:
                spec["server_diagnostics"][str(seed)] = server_diag.aggregate()
            print(f"[{method_name}] completed seed {seed}", flush=True)

        sample = next(iter(spec["per_seed"].values()))
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key in sample
            if isinstance(sample[key], (int, float))
        }
        if "profile_stats" in sample and sample["profile_stats"]:
            profile_keys = sample["profile_stats"].keys()
            spec["aggregate"]["profile_stats"] = {
                key: float(np.mean([row["profile_stats"][key] for row in spec["per_seed"].values()]))
                for key in profile_keys
            }
        if args.collect_server_diagnostics and "server_diagnostics" in spec:
            spec["server_diagnostics_aggregate"] = _aggregate_server_diagnostics(
                spec["server_diagnostics"]
            )
        report_methods[method_name] = spec

    return {
        "config": vars(args),
        "methods": report_methods,
    }


def _aggregate_server_diagnostics(per_seed_diag: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate server diagnostics across seeds (simple mean for scalars)."""
    seeds = list(per_seed_diag.values())
    if not seeds:
        return {}
    n_servers = seeds[0]["num_servers"]
    n_splits = seeds[0]["num_splits"]
    thr = seeds[0]["high_util_threshold"]

    def mean_over_seeds(key):
        vals = [s[key] for s in seeds]
        if isinstance(vals[0], (int, float)):
            return float(np.mean(vals))
        if isinstance(vals[0], list) and isinstance(vals[0][0], (int, float)):
            return np.mean(vals, axis=0).tolist()
        if isinstance(vals[0], dict):
            keys = vals[0].keys()
            return {k: float(np.mean([v[k] for v in vals])) for k in keys}
        return vals[0]

    # For per-server nested stats, average across seeds.
    util_avg = []
    queue_avg = []
    for s in range(n_servers):
        util_avg.append({
            "mean": float(np.mean([seed["utilization"][s]["mean"] for seed in seeds])),
            "p95": float(np.mean([seed["utilization"][s]["p95"] for seed in seeds])),
            "max": float(np.mean([seed["utilization"][s]["max"] for seed in seeds])),
        })
        queue_avg.append({
            "mean": float(np.mean([seed["queue_delay_ms"][s]["mean"] for seed in seeds])),
            "p95": float(np.mean([seed["queue_delay_ms"][s]["p95"] for seed in seeds])),
            "max": float(np.mean([seed["queue_delay_ms"][s]["max"] for seed in seeds])),
        })

    first_high = {}
    for seed in seeds:
        for srv, t in seed["first_high_util_time"].items():
            srv_int = int(srv)
            first_high[srv_int] = min(first_high.get(srv_int, float("inf")), t)

    return {
        "num_servers": n_servers,
        "num_splits": n_splits,
        "high_util_threshold": thr,
        "total_samples": int(np.sum([s["total_samples"] for s in seeds])),
        "total_overloads": int(np.sum([s["total_overloads"] for s in seeds])),
        "selected_count": np.sum([s["selected_count"] for s in seeds], axis=0).tolist(),
        "selected_percentage": np.mean([s["selected_percentage"] for s in seeds], axis=0).tolist(),
        "utilization": util_avg,
        "queue_delay_ms": queue_avg,
        "high_util_fraction": np.mean([s["high_util_fraction"] for s in seeds], axis=0).tolist(),
        "first_high_util_time": first_high,
        "overload_by_server": np.sum([s["overload_by_server"] for s in seeds], axis=0).tolist(),
        "overload_by_server_percentage": np.mean([s["overload_by_server_percentage"] for s in seeds], axis=0).tolist(),
        "overload_by_split_server": np.sum([s["overload_by_split_server"] for s in seeds], axis=0).tolist(),
    }


def _write_server_diagnostics_section(lines: List[str], report: Dict[str, Any]) -> None:
    """Append server-level diagnostics tables to the markdown report."""
    lines.extend(["", "## Server-Level Diagnostics", ""])
    for name, spec in report["methods"].items():
        diag = spec.get("server_diagnostics_aggregate")
        if diag is None:
            continue
        lines.append(f"### {name}")
        lines.append("")
        n = diag["num_servers"]

        # Selection distribution
        lines.append("**Server selection distribution**")
        lines.append("| Server | Selected count | Selected % |")
        lines.append("|---:|---:|---:|")
        for s in range(n):
            lines.append(
                f"| {s} | {diag['selected_count'][s]} | {diag['selected_percentage'][s]:.2f}% |"
            )

        # Utilization
        lines.append("")
        lines.append("**Server utilization**")
        lines.append("| Server | Mean | P95 | Max | High-util fraction | First high-util time |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for s in range(n):
            u = diag["utilization"][s]
            fh = diag["first_high_util_time"].get(s)
            fh_str = f"{fh:.3f}" if fh is not None else "—"
            lines.append(
                f"| {s} | {u['mean']:.3f} | {u['p95']:.3f} | {u['max']:.3f} | "
                f"{diag['high_util_fraction'][s]:.2%} | {fh_str} |"
            )

        # Queue delay
        lines.append("")
        lines.append("**Server queue delay (ms)**")
        lines.append("| Server | Mean | P95 | Max |")
        lines.append("|---:|---:|---:|---:|")
        for s in range(n):
            q = diag["queue_delay_ms"][s]
            lines.append(f"| {s} | {q['mean']:.3f} | {q['p95']:.3f} | {q['max']:.3f} |")

        # Overload distribution
        lines.append("")
        lines.append("**Overload distribution**")
        total = diag["total_overloads"]
        lines.append(f"Total overloads: {total}")
        lines.append("| Server | Overloads | % of overloads |")
        lines.append("|---:|---:|---:|")
        for s in range(n):
            lines.append(
                f"| {s} | {diag['overload_by_server'][s]} | {diag['overload_by_server_percentage'][s]:.2f}% |"
            )

        # Split-server matrix
        lines.append("")
        lines.append("**Overload by split × server**")
        header = "| Split | " + " | ".join([f"S{s}" for s in range(n)]) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (n + 1)) + "|")
        for split in range(diag["num_splits"]):
            row = diag["overload_by_split_server"][split]
            lines.append(f"| {split} | " + " | ".join([str(v) for v in row]) + " |")
        lines.append("")


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Main S100 System-Level Comparison",
        "",
        "This is the recommended main comparison table for paper/PPT use.",
        "",
        "## Configuration",
        "",
    ]
    cfg = report["config"]
    for key in [
        "topology", "num_slots", "split_profile", "arrival_interval",
        "holding_min", "holding_max", "size_min_mb", "size_max_mb",
        "seeds", "episodes", "requests_per_episode", "k_paths",
        "path_sort_strategy", "ksp_ff_k50_hops_k_paths",
    ]:
        lines.append(f"- `{key}`: `{cfg[key]}`")

    lines.extend([
        "",
        "## Results",
        "",
        "| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, spec in report["methods"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {name} | {spec['c_mode']} | {spec['r_mode']} | "
            f"{agg['blocking_rate']:.2%} | {agg['raw_mask_empty_rate']:.2%} | "
            f"{agg['no_suitable_block_rate']:.2%} | {agg['server_overload_rate']:.2%} | "
            f"{agg.get('deadline_failure_rate', 0.0):.2%} | "
            f"{agg.get('other_failure_rate', 0.0):.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{agg['mean_decision_time_ms']:.3f}/{agg['p95_decision_time_ms']:.3f} ms |"
        )

    if "ppo_c+v12" in report["methods"]:
        base = report["methods"]["ppo_c+v12"]["aggregate"]
        lines.extend(["", "## Delta vs PPO-C + v1.2", ""])
        lines.append("| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, spec in report["methods"].items():
            if name == "ppo_c+v12":
                continue
            agg = spec["aggregate"]
            lines.append(
                f"| {name} | "
                f"{(agg['blocking_rate'] - base['blocking_rate']) * 100:+.2f} pp | "
                f"{(agg['no_suitable_block_rate'] - base['no_suitable_block_rate']) * 100:+.2f} pp | "
                f"{(agg['server_overload_rate'] - base['server_overload_rate']) * 100:+.2f} pp | "
                f"{agg['mean_delay_ms'] - base['mean_delay_ms']:+.3f} ms |"
            )

    if any("server_diagnostics_aggregate" in spec for spec in report["methods"].values()):
        _write_server_diagnostics_section(lines, report)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_s100_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument(
        "--methods",
        default=(
            "ppo_c+v12,ppo_c+deep_rmsa,ppo_c+ppo_r,ppo_c+ksp_bf,ppo_c+ksp_ff,"
            "ppo_c+ksp_ff_k50_hops,greedy_c+v12,df_c+v12,rf_c+v12,"
            "wo_c+v12,iwd_c+v12"
        ),
    )
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=100)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--path_sort_strategy", default="km", choices=["km", "hops"])
    parser.add_argument("--ksp_ff_k50_hops_k_paths", type=int, default=50)
    parser.add_argument("--ppo_r_proposer_top_k", type=int, default=8,
                        help="Top-K actions proposed by PPO-R for the ppo_r_proposer_ksp_ff backend.")
    parser.add_argument("--post_decision_checkpoint",
                        default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt",
                        help="Checkpoint for the post-decision value finalizer used by ppo_r_proposer_postdec.")
    parser.add_argument("--ppo_r_postdec_top_k", type=int, default=8,
                        help="Top-K actions proposed by PPO-R for the ppo_r_proposer_postdec backend.")
    parser.add_argument("--ppo_r_postdec_ensure_ksp", action="store_true",
                        help="Ensure the KSP-FF action is included in the post-decision candidate set.")
    parser.add_argument("--ranker_candidate_mode", default=None, choices=["all_legal", "v1", "legalctx48", "highest_se_path_block", "top2_se_path_block"],
                        help="Override the candidate_mode stored in the ranking checkpoint.")
    parser.add_argument("--ranker_max_candidates", type=int, default=None)
    parser.add_argument("--ranker_ppo_top_k", type=int, default=None)
    parser.add_argument("--ranker_num_random_candidates", type=int, default=None)
    parser.add_argument("--ranker_min_candidates", type=int, default=None)
    parser.add_argument("--ranker_ensure_ksp", action="store_true",
                        help="Ensure the KSP-FF K=50 hops action is in the ranker candidate set.")
    parser.add_argument("--ranker_enable_profile", action="store_true",
                        help="Enable per-request profiling of the v1.3 ranker pipeline.")
    parser.add_argument("--collect_server_diagnostics", action="store_true",
                        help="Collect per-server utilization/queue/overload diagnostics.")
    parser.add_argument("--server_diag_high_util_threshold", type=float, default=0.90,
                        help="Utilization threshold for 'near limit' diagnostics.")
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/main_s100_system_comparison.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/main_s100_system_comparison.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), report)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
