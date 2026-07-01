"""Diagnostic: why TD R-post underperforms on K5/M10.

Runs three sets of analyses on the same request traces used for the formal
system comparison:

1. Separate trajectories for td_post / ppo_r / deep_rmsa on identical traces,
   recording per-request legal R action count, selected_valid_r_actions,
   selected action features, and outcome (blocked / reason).
2. A probe pass over the PPO-R trajectory: at every request, after the C
   decision, query what td_post and DeepRMSA would do in the same state,
   compute the full TD post-value distribution over legal R actions, and
   record ranks.
3. Aggregate into a JSON + markdown diagnostic report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
    _select_r_action,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import (
    _load_deep_rmsa,
    _select_post_r_action,
    load_post_checkpoint,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES, _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


R_FEATURE_ORDER = {
    "path_length_km": 0,
    "hop_count": 1,
    "lfb": 2,
    "free_ratio": 3,
    "frag_index": 4,
    "spectral_efficiency": 5,
    "reach_km": 6,
    "required_fs": 7,
    "block_size": 8,
    "block_waste": 9,
    "path_mod_feasible": 10,
}


def _selected_action_features(obs_r: Dict[str, Any], r_features: np.ndarray, action_idx: int):
    num_mods = len(obs_r["mod_names"])
    max_blocks = len(obs_r["agent_r_mask"]) // (len(obs_r["candidate_paths"]) * num_mods)
    path_idx, mod_idx, block_idx = decode_agent_r_action(action_idx, num_mods, max_blocks)
    feat = r_features[action_idx]
    return {
        "path_idx": int(path_idx),
        "mod_idx": int(mod_idx),
        "block_idx": int(block_idx),
        "path_length_km": float(feat[R_FEATURE_ORDER["path_length_km"]]),
        "hop_count": float(feat[R_FEATURE_ORDER["hop_count"]]),
        "spectral_efficiency": float(feat[R_FEATURE_ORDER["spectral_efficiency"]]),
        "required_fs": float(feat[R_FEATURE_ORDER["required_fs"]]),
        "block_size": float(feat[R_FEATURE_ORDER["block_size"]]),
        "block_waste": float(feat[R_FEATURE_ORDER["block_waste"]]),
        "path_mod_feasible": int(feat[R_FEATURE_ORDER["path_mod_feasible"]]),
    }


def _value_distribution(
    env, req, obs_c, obs_r, agent_r, model, mean, std, split_id, server_id, device
):
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    selected_valid = int(obs_c["feasible_counts"][split_id][server_id])
    if not legal:
        return {
            "legal_count": 0,
            "selected_valid": int(selected_valid),
            "values": {},
            "ranks": {},
            "best_action": None,
            "worst_action": None,
            "value_mean": None,
            "value_std": None,
            "value_range": 0.0,
        }
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        values = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    order = np.argsort(-values, kind="stable")
    ranks = {int(legal[i]): int(rank) + 1 for rank, i in enumerate(order)}
    best_action = int(legal[order[0]])
    worst_action = int(legal[order[-1]])
    return {
        "legal_count": int(len(legal)),
        "selected_valid": int(selected_valid),
        "values": {int(a): float(v) for a, v in zip(legal, values)},
        "ranks": ranks,
        "best_action": best_action,
        "worst_action": worst_action,
        "value_mean": float(np.mean(values)),
        "value_std": float(np.std(values)),
        "value_range": float(values.max() - values.min()),
    }


def _run_separate_trajectory(
    method: str,
    episodes_by_seed: Dict[int, List[List[Any]]],
    agent_c,
    agent_r,
    deep_rmsa,
    post_pack,
    args,
):
    """Run each method on its own trajectory and record per-request features/outcomes."""
    records = []
    per_seed_metrics = {}
    model, mean, std = post_pack
    for seed, episodes in episodes_by_seed.items():
        metrics = PerMethodMetrics()
        for requests in episodes:
            env = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            server_selected_count = np.zeros(args.num_servers, dtype=int)
            for req in requests:
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                c_idx = agent_c.select_action(obs_c, deterministic=True)
                if c_idx is None:
                    c_idx = 0
                split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
                server_selected_count[server_id] += 1
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                r_features, _ = agent_r.build_action_features(obs_r)
                legal_count = int(np.asarray(obs_r["agent_r_mask"], dtype=bool).sum())
                selected_valid = int(obs_c["feasible_counts"][split_id][server_id])

                if method == "td_post":
                    r_idx = _select_post_r_action(
                        env, req, obs_c, obs_r, agent_r, model, mean, std,
                        split_id, server_id, "td_post_only", 1.0, args.device,
                    )
                    r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
                elif method == "deep_rmsa":
                    r_idx = deep_rmsa.select_action(obs_r)
                    r_action = (0, 0, 0) if r_idx is None else decode_agent_r_action(
                        r_idx, len(obs_r["mod_names"]), env.max_blocks
                    )
                    if r_idx is None:
                        r_idx = -1
                elif method == "ppo_r":
                    r_idx = agent_r.select_action(obs_r, deterministic=True)
                    if r_idx is None:
                        r_idx = -1
                        r_action = (0, 0, 0)
                    else:
                        r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
                else:
                    raise ValueError(method)

                _, _, _, info = env.step((split_id, server_id), r_action)
                _record_outcome(
                    metrics, info,
                    int(np.asarray(obs_c["agent_c_mask"], dtype=bool).sum()) == 0,
                    obs_c, split_id, server_id, 0.0, True,
                )
                metrics.active_connections[-1] = len(env.active_connections)

                feat = _selected_action_features(obs_r, r_features, r_idx) if r_idx >= 0 else None
                records.append({
                    "method": method,
                    "seed": int(seed),
                    "req_id": int(req.req_id),
                    "legal_count": legal_count,
                    "selected_valid": selected_valid,
                    "selected_r_idx": int(r_idx),
                    "success": bool(info.get("success", False)),
                    "blocked": not bool(info.get("success", False)),
                    "reason": info.get("reason", ""),
                    "action_features": feat,
                })
        per_seed_metrics[str(seed)] = _aggregate_metrics(metrics)
        print(f"[{method}] completed seed {seed}", flush=True)
    aggregate = {
        key: float(np.mean([row[key] for row in per_seed_metrics.values()]))
        for key in next(iter(per_seed_metrics.values()))
        if isinstance(next(iter(per_seed_metrics.values()))[key], (int, float))
    }
    return records, {"per_seed": per_seed_metrics, "aggregate": aggregate}


def _run_probe_on_ppo_r_trajectory(
    episodes_by_seed: Dict[int, List[List[Any]]],
    agent_c,
    agent_r,
    deep_rmsa,
    post_pack,
    args,
):
    """Run PPO-R trajectory and probe what td_post / DeepRMSA would do at each state."""
    model, mean, std = post_pack
    probe_records = []
    per_seed_metrics = {}
    for seed, episodes in episodes_by_seed.items():
        metrics = PerMethodMetrics()
        for requests in episodes:
            env = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            for req in requests:
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                c_idx = agent_c.select_action(obs_c, deterministic=True)
                if c_idx is None:
                    c_idx = 0
                split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
                obs_r = build_agent_r_observation(env, req, split_id, server_id)

                ppo_idx = agent_r.select_action(obs_r, deterministic=True)
                if ppo_idx is None:
                    ppo_idx = -1
                td_idx = _select_post_r_action(
                    env, req, obs_c, obs_r, agent_r, model, mean, std,
                    split_id, server_id, "td_post_only", 1.0, args.device,
                )
                deep_idx = deep_rmsa.select_action(obs_r)

                val_dist = _value_distribution(
                    env, req, obs_c, obs_r, agent_r, model, mean, std,
                    split_id, server_id, args.device,
                )

                # Step with PPO-R action.
                r_action = (0, 0, 0) if ppo_idx < 0 else decode_agent_r_action(
                    ppo_idx, len(obs_r["mod_names"]), env.max_blocks
                )
                _, _, _, info = env.step((split_id, server_id), r_action)
                _record_outcome(
                    metrics, info,
                    int(np.asarray(obs_c["agent_c_mask"], dtype=bool).sum()) == 0,
                    obs_c, split_id, server_id, 0.0, True,
                )
                metrics.active_connections[-1] = len(env.active_connections)

                probe_records.append({
                    "seed": int(seed),
                    "req_id": int(req.req_id),
                    "legal_count": val_dist["legal_count"],
                    "selected_valid": val_dist["selected_valid"],
                    "ppo_r_idx": int(ppo_idx),
                    "td_post_idx": int(td_idx),
                    "deep_rmsa_idx": int(deep_idx) if deep_idx is not None else -1,
                    "ppo_r_rank_in_td_value": val_dist["ranks"].get(int(ppo_idx)),
                    "td_post_rank_in_td_value": val_dist["ranks"].get(int(td_idx)),
                    "deep_rmsa_rank_in_td_value": val_dist["ranks"].get(int(deep_idx) if deep_idx is not None else -1),
                    "post_value_of_ppo_r": val_dist["values"].get(int(ppo_idx)),
                    "post_value_of_td_post": val_dist["values"].get(int(td_idx)),
                    "post_value_of_deep_rmsa": val_dist["values"].get(int(deep_idx) if deep_idx is not None else -1),
                    "post_value_best": val_dist["values"].get(val_dist["best_action"]) if val_dist["best_action"] is not None else None,
                    "post_value_worst": val_dist["values"].get(val_dist["worst_action"]) if val_dist["worst_action"] is not None else None,
                    "post_value_mean": val_dist["value_mean"],
                    "post_value_std": val_dist["value_std"],
                    "post_value_range": val_dist["value_range"],
                    "ppo_r_success": bool(info.get("success", False)),
                })
        per_seed_metrics[str(seed)] = _aggregate_metrics(metrics)
        print(f"[probe on ppo_r] completed seed {seed}", flush=True)
    aggregate = {
        key: float(np.mean([row[key] for row in per_seed_metrics.values()]))
        for key in next(iter(per_seed_metrics.values()))
        if isinstance(next(iter(per_seed_metrics.values()))[key], (int, float))
    }
    return probe_records, {"per_seed": per_seed_metrics, "aggregate": aggregate}


def _bin_blocking(records: List[Dict[str, Any]], key: str, bins: List[Tuple[int, int]], methods: List[str]):
    rows = []
    for method in methods:
        for lo, hi in bins:
            subset = [
                r for r in records
                if r["method"] == method and lo <= r[key] <= hi
            ]
            if not subset:
                continue
            rows.append({
                "method": method,
                "bin": f"{lo}" if lo == hi else f"{lo}-{hi}",
                "count": len(subset),
                "blocked": int(sum(r["blocked"] for r in subset)),
                "blocking_rate": float(np.mean([r["blocked"] for r in subset])),
                "mean_legal_count": float(np.mean([r["legal_count"] for r in subset])) if key != "legal_count" else None,
                "mean_selected_valid": float(np.mean([r["selected_valid"] for r in subset])) if key != "selected_valid" else None,
            })
    return rows


def _action_feature_summary(records: List[Dict[str, Any]], methods: List[str]):
    summary = []
    for method in methods:
        feats = [r["action_features"] for r in records if r["method"] == method and r["action_features"] is not None]
        if not feats:
            continue
        for name in ["path_idx", "mod_idx", "block_idx", "path_length_km", "hop_count",
                     "spectral_efficiency", "required_fs", "block_size", "block_waste"]:
            vals = [f[name] for f in feats]
            summary.append({
                "method": method,
                "feature": name,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
            })
    return summary


def _action_agreement(records: List[Dict[str, Any]], method_a: str, method_b: str):
    pairs = []
    for r in records:
        if r["method"] != method_a:
            continue
        # Find matching request in method_b under same seed.
        match = next((x for x in records if x["method"] == method_b and x["seed"] == r["seed"] and x["req_id"] == r["req_id"]), None)
        if match is None:
            continue
        if r["selected_r_idx"] >= 0 and match["selected_r_idx"] >= 0:
            pairs.append(int(r["selected_r_idx"] == match["selected_r_idx"]))
    return float(np.mean(pairs)) if pairs else None, int(len(pairs))


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)
    post_model, post_mean, post_std, _ = load_post_checkpoint(args.post_checkpoint, args.device)
    post_pack = (post_model, post_mean, post_std)

    episodes_by_seed: Dict[int, List[List[Any]]] = {}
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
        episodes_by_seed[seed] = episodes

    methods = ["td_post", "ppo_r", "deep_rmsa"]
    all_records = []
    separate_metrics = {}
    for method in methods:
        records, metrics = _run_separate_trajectory(
            method, episodes_by_seed, agent_c, agent_r, deep_rmsa, post_pack, args
        )
        all_records.extend(records)
        separate_metrics[method] = metrics

    probe_records, probe_metrics = _run_probe_on_ppo_r_trajectory(
        episodes_by_seed, agent_c, agent_r, deep_rmsa, post_pack, args
    )

    legal_bins = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 50), (51, 9999)]
    selected_valid_bins = [(0, 0), (1, 5), (6, 15), (16, 30), (31, 50), (51, 9999)]
    blocking_by_legal = _bin_blocking(all_records, "legal_count", legal_bins, methods)
    blocking_by_selected_valid = _bin_blocking(all_records, "selected_valid", selected_valid_bins, methods)

    action_features = _action_feature_summary(all_records, methods)

    agreement_td_deep, n_td_deep = _action_agreement(all_records, "td_post", "deep_rmsa")
    agreement_td_ppo, n_td_ppo = _action_agreement(all_records, "td_post", "ppo_r")
    agreement_deep_ppo, n_deep_ppo = _action_agreement(all_records, "deep_rmsa", "ppo_r")

    # Probe summaries.
    ppo_ranks = [r["ppo_r_rank_in_td_value"] for r in probe_records if r["ppo_r_rank_in_td_value"] is not None]
    td_ranks = [r["td_post_rank_in_td_value"] for r in probe_records if r["td_post_rank_in_td_value"] is not None]
    deep_ranks = [r["deep_rmsa_rank_in_td_value"] for r in probe_records if r["deep_rmsa_rank_in_td_value"] is not None]
    td_values = [r["post_value_of_td_post"] for r in probe_records if r["post_value_of_td_post"] is not None]
    ppo_values = [r["post_value_of_ppo_r"] for r in probe_records if r["post_value_of_ppo_r"] is not None]
    deep_values = [r["post_value_of_deep_rmsa"] for r in probe_records if r["post_value_of_deep_rmsa"] is not None]
    best_values = [r["post_value_best"] for r in probe_records if r["post_value_best"] is not None]

    rank_distribution_ppo = {}
    if ppo_ranks:
        for rank in range(1, max(ppo_ranks) + 1):
            rank_distribution_ppo[str(rank)] = int(np.sum(np.asarray(ppo_ranks) == rank))
    rank_distribution_deep = {}
    if deep_ranks:
        for rank in range(1, max(deep_ranks) + 1):
            rank_distribution_deep[str(rank)] = int(np.sum(np.asarray(deep_ranks) == rank))

    report = {
        "config": vars(args),
        "separate_metrics": separate_metrics,
        "probe_metrics": probe_metrics,
        "agreement": {
            "td_post_vs_deep_rmsa": {"rate": agreement_td_deep, "pairs": n_td_deep},
            "td_post_vs_ppo_r": {"rate": agreement_td_ppo, "pairs": n_td_ppo},
            "deep_rmsa_vs_ppo_r": {"rate": agreement_deep_ppo, "pairs": n_deep_ppo},
        },
        "blocking_by_legal_count": blocking_by_legal,
        "blocking_by_selected_valid": blocking_by_selected_valid,
        "action_feature_summary": action_features,
        "probe_summary": {
            "ppo_r_rank_mean": float(np.mean(ppo_ranks)) if ppo_ranks else None,
            "ppo_r_rank_median": float(np.median(ppo_ranks)) if ppo_ranks else None,
            "td_post_rank_mean": float(np.mean(td_ranks)) if td_ranks else None,
            "td_post_rank_median": float(np.median(td_ranks)) if td_ranks else None,
            "deep_rmsa_rank_mean": float(np.mean(deep_ranks)) if deep_ranks else None,
            "deep_rmsa_rank_median": float(np.median(deep_ranks)) if deep_ranks else None,
            "td_post_selected_value_mean": float(np.mean(td_values)) if td_values else None,
            "td_post_selected_value_std": float(np.std(td_values)) if td_values else None,
            "ppo_r_selected_value_mean": float(np.mean(ppo_values)) if ppo_values else None,
            "ppo_r_selected_value_std": float(np.std(ppo_values)) if ppo_values else None,
            "deep_rmsa_selected_value_mean": float(np.mean(deep_values)) if deep_values else None,
            "deep_rmsa_selected_value_std": float(np.std(deep_values)) if deep_values else None,
            "best_value_mean": float(np.mean(best_values)) if best_values else None,
            "best_value_std": float(np.std(best_values)) if best_values else None,
            "ppo_r_rank_distribution": rank_distribution_ppo,
            "deep_rmsa_rank_distribution": rank_distribution_deep,
        },
        "elapsed_seconds": time.time() - getattr(args, "_start_time", time.time()),
    }
    return report


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# TD R-Post Diagnostic Report", ""]

    lines += ["## 1. Separate-trajectory aggregate metrics", ""]
    lines += ["| Method | Blocking | NSB | Overload | Delay mean/P95 |", "|---|---:|---:|---:|---:|"]
    for method, spec in report["separate_metrics"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {method} | {agg['blocking_rate']:.2%} | {agg['no_suitable_block_rate']:.2%} | "
            f"{agg['server_overload_rate']:.2%} | {agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms |"
        )
    lines.append("")

    lines += ["## 2. R-action agreement (same state)", ""]
    lines += ["| Pair | Agreement | Pairs |", "|---|---:|---:|"]
    for pair, spec in report["agreement"].items():
        rate = spec["rate"]
        rate_str = f"{rate:.2%}" if rate is not None else "N/A"
        lines.append(f"| {pair} | {rate_str} | {spec['pairs']} |")
    lines.append("")

    lines += ["## 3. PPO-R / DeepRMSA action rank under TD value (on PPO-R states)", ""]
    ps = report["probe_summary"]
    lines += [
        f"- DeepRMSA mean rank: {ps['deep_rmsa_rank_mean']:.2f}",
        f"- DeepRMSA median rank: {ps['deep_rmsa_rank_median']:.2f}",
        f"- PPO-R mean rank: {ps['ppo_r_rank_mean']:.2f}",
        f"- PPO-R median rank: {ps['ppo_r_rank_median']:.2f}",
        "",
        "DeepRMSA rank distribution:",
        "",
        "| Rank | Count |",
        "|---|---:|",
    ]
    for rank, count in sorted(ps["deep_rmsa_rank_distribution"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {rank} | {count} |")
    lines.append("")

    lines += ["## 4. Selected action value distribution", ""]
    lines += [
        f"- TD-post selected value mean ± std: {ps['td_post_selected_value_mean']:.4f} ± {ps['td_post_selected_value_std']:.4f}",
        f"- PPO-R selected value mean ± std: {ps['ppo_r_selected_value_mean']:.4f} ± {ps['ppo_r_selected_value_std']:.4f}",
        f"- DeepRMSA selected value mean ± std: {ps['deep_rmsa_selected_value_mean']:.4f} ± {ps['deep_rmsa_selected_value_std']:.4f}",
        f"- Best legal value mean ± std: {ps['best_value_mean']:.4f} ± {ps['best_value_std']:.4f}",
        "",
    ]

    lines += ["## 5. Blocking by legal R action count", ""]
    lines += ["| Method | Bin | Count | Blocked | Blocking rate |", "|---|---|---:|---:|---:|"]
    for row in report["blocking_by_legal_count"]:
        lines.append(
            f"| {row['method']} | {row['bin']} | {row['count']} | {row['blocked']} | {row['blocking_rate']:.2%} |"
        )
    lines.append("")

    lines += ["## 6. Blocking by selected_valid_r_actions", ""]
    lines += ["| Method | Bin | Count | Blocked | Blocking rate |", "|---|---|---:|---:|---:|"]
    for row in report["blocking_by_selected_valid"]:
        lines.append(
            f"| {row['method']} | {row['bin']} | {row['count']} | {row['blocked']} | {row['blocking_rate']:.2%} |"
        )
    lines.append("")

    lines += ["## 7. Selected action feature summary", ""]
    lines += ["| Method | Feature | Mean | Median | Std |", "|---|---|---:|---:|---:|"]
    for row in report["action_feature_summary"]:
        lines.append(
            f"| {row['method']} | {row['feature']} | {row['mean']:.4f} | {row['median']:.4f} | {row['std']:.4f} |"
        )
    lines.append("")

    lines += ["## 8. Interpretation and minimal next-step recommendations", ""]
    lines += [
        "1. If TD-post selected value mean is close to PPO-R/DeepRMSA selected value mean,",
        "   the value network is not confidently discriminating better actions.",
        "2. If PPO-R and DeepRMSA actions rank > 1 under TD value on PPO-R states, TD-post",
        "   often disagrees with strong baselines even when they succeed.",
        "3. Compare feature means across methods: systematic differences in modulation,",
        "   path length, or block waste point to inductive bias or value misalignment.",
        "4. Next steps:",
        "   - Verify Bellman targets are not dominated by immediate -1 blocked rewards.",
        "   - Increase replay diversity (e.g., collect from multiple R policies / epsilon exploration).",
        "   - Tune gamma / target-update frequency; consider Double-DQN style target reduction.",
        "   - Add ranking or margin loss so the best unblocked action is valued above others.",
        "   - Distill from DeepRMSA as a teacher if TD bootstrapping remains unstable.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--post_checkpoint", default="sa_hmarl/checkpoints/r_td_post_decision_k5m10/td_post_decision.pt")
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_td_post_decision_diagnostic.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_td_post_decision_diagnostic.md")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    report = evaluate(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(out_md, report)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
