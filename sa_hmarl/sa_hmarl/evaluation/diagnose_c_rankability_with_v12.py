"""C-side rankability diagnostic with frozen v1.2 R-ranker.

For every decision state, this script enumerates raw-mask-valid C candidates
``(split, server)``.  For each candidate it lets the frozen v1.2 R-ranker choose
the current RMSA action, then rolls out a short H-step future with the frozen
PPO-C + v1.2 R-ranker policy.

The goal is not to train a new C policy yet.  The goal is to answer whether C
has enough long-horizon headroom to justify a C-side downstream-RMSA-aware
ranker.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
    _select_c_action_from_obs,
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _parse_seeds(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _safe_corr(x: Iterable[float], y: Iterable[float]) -> Dict[str, Optional[float]]:
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 3 or float(np.std(xs)) == 0.0 or float(np.std(ys)) == 0.0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(stats.pearsonr(xs, ys).statistic),
        "spearman": float(stats.spearmanr(xs, ys).statistic),
    }


def _make_env(args: argparse.Namespace, seed: int):
    return make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )


def _load_agents(args: argparse.Namespace):
    env_proto = _make_env(args, 42)
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(
        args.ranking_checkpoint, args.device
    )
    return env_proto, agent_c, agent_r, rank_model, rank_mean, rank_std


def _r_ranker_scores(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    split_id: int,
    server_id: int,
    device: str,
) -> Dict[str, Any]:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return {
            "legal_r_count": 0,
            "best_r_idx": -1,
            "best_r_score": None,
            "top3_r_score_mean": None,
            "score_range": None,
        }
    if len(legal) == 1:
        return {
            "legal_r_count": 1,
            "best_r_idx": int(legal[0]),
            "best_r_score": 0.0,
            "top3_r_score_mean": 0.0,
            "score_range": 0.0,
        }
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - rank_mean) / rank_std
    with torch.no_grad():
        scores = (
            rank_model(torch.as_tensor(normalized, dtype=torch.float32, device=device))
            .cpu()
            .numpy()
            .reshape(-1)
        )
    order = np.argsort(-scores)
    top = scores[order[: min(3, len(order))]]
    return {
        "legal_r_count": int(len(legal)),
        "best_r_idx": int(legal[int(order[0])]),
        "best_r_score": float(scores[int(order[0])]),
        "top3_r_score_mean": float(np.mean(top)),
        "score_range": float(np.max(scores) - np.min(scores)),
    }


def _execute_current_with_v12(
    env,
    req,
    c_idx: int,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
    obs_c = build_agent_c_observation(env, req)
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    r_summary = _r_ranker_scores(
        env,
        req,
        obs_c,
        obs_r,
        agent_r,
        rank_model,
        rank_mean,
        rank_std,
        split_id,
        server_id,
        args.device,
    )
    if r_summary["legal_r_count"] <= 0:
        r_action = (0, 0, 0)
    else:
        r_action = decode_agent_r_action(
            int(r_summary["best_r_idx"]), len(obs_r["mod_names"]), env.max_blocks
        )
    _, _, _, info = env.step((split_id, server_id), r_action)
    return {
        "split_id": int(split_id),
        "server_id": int(server_id),
        "info": info,
        **r_summary,
    }


def _select_c_with_ppo(agent_c, obs_c: Dict[str, Any], num_slots: int) -> int:
    c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, num_slots)
    return int(c_idx)


def _rollout_future_v12(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    blocked = 0
    raw_empty = 0
    no_suitable_block = 0
    server_overload = 0
    deadline_infeasible = 0
    delay_sum = 0.0
    fs_sum = 0.0
    steps = 0

    for t in range(start_idx, min(start_idx + horizon, len(requests))):
        req = requests[t]
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if int(raw_c_mask.sum()) == 0:
            raw_empty += 1
            blocked += 1
            env.step((0, 0), (0, 0, 0))
            steps += 1
            continue

        c_idx = _select_c_with_ppo(agent_c, obs_c, env.net.num_slots)
        outcome = _execute_current_with_v12(
            env, req, c_idx, agent_r, rank_model, rank_mean, rank_std, args
        )
        info = outcome["info"]
        steps += 1
        if not info.get("success", False):
            blocked += 1
            reason = str(info.get("reason", ""))
            if reason == "no_suitable_block":
                no_suitable_block += 1
            elif reason == "server_overload":
                server_overload += 1
            elif reason == "deadline_infeasible":
                deadline_infeasible += 1
        else:
            delay_sum += float(info.get("delay_ms", 0.0))
            fs_sum += float(info.get("num_slots", 0.0))

    denom = max(steps, 1)
    return {
        "steps": int(steps),
        "future_blocked": int(blocked),
        "future_raw_empty": int(raw_empty),
        "future_nsb": int(no_suitable_block),
        "future_overload": int(server_overload),
        "future_deadline": int(deadline_infeasible),
        "future_delay_mean": float(delay_sum / denom),
        "future_fs_mean": float(fs_sum / denom),
    }


def _candidate_return(current: Dict[str, Any], future: Dict[str, Any], args: argparse.Namespace) -> float:
    info = current["info"]
    current_block = 0.0 if info.get("success", False) else 1.0
    current_nsb = 1.0 if str(info.get("reason", "")) == "no_suitable_block" else 0.0
    current_overload = 1.0 if str(info.get("reason", "")) == "server_overload" else 0.0
    current_delay = float(info.get("delay_ms", 0.0)) if info.get("success", False) else 0.0
    current_fs = float(info.get("num_slots", 0.0)) if info.get("success", False) else 0.0
    return float(
        -args.current_block_coef * current_block
        -args.future_block_coef * future["future_blocked"]
        -args.future_nsb_coef * future["future_nsb"]
        -args.future_overload_coef * future["future_overload"]
        -args.current_nsb_coef * current_nsb
        -args.current_overload_coef * current_overload
        -args.delay_coef * (current_delay + future["future_delay_mean"])
        -args.fs_coef * (current_fs + future["future_fs_mean"])
    )


def _generate_episodes(args: argparse.Namespace, env_proto) -> Dict[int, List[List[Any]]]:
    episodes_by_seed: Dict[int, List[List[Any]]] = {}
    for seed in _parse_seeds(args.seeds):
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(
                generate_requests(
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
                )
            )
        episodes_by_seed[seed] = episodes
    return episodes_by_seed


def run(args: argparse.Namespace) -> Dict[str, Any]:
    env_proto, agent_c, agent_r, rank_model, rank_mean, rank_std = _load_agents(args)
    episodes_by_seed = _generate_episodes(args, env_proto)

    state_records: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []

    for seed, episodes in episodes_by_seed.items():
        for ep_idx, requests in enumerate(episodes):
            env = _make_env(args, seed)
            env.reset(requests)
            for req_idx, req in enumerate(requests):
                if args.max_states > 0 and len(state_records) >= args.max_states:
                    break
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                legal_c = np.flatnonzero(raw_c_mask).tolist()
                ppo_c_idx = _select_c_with_ppo(agent_c, obs_c, env.net.num_slots)

                if not legal_c:
                    env.step((0, 0), (0, 0, 0))
                    state_records.append({
                        "seed": seed,
                        "episode": ep_idx,
                        "request": req_idx,
                        "valid_c_count": 0,
                        "ppo_c_idx": int(ppo_c_idx),
                        "skipped_reason": "no_valid_c",
                    })
                    continue

                snapshot = copy.deepcopy(env)
                cand_rows: List[Dict[str, Any]] = []
                for c_idx in legal_c:
                    branch = copy.deepcopy(snapshot)
                    current = _execute_current_with_v12(
                        branch, req, c_idx, agent_r, rank_model, rank_mean, rank_std, args
                    )
                    future = _rollout_future_v12(
                        branch,
                        requests,
                        req_idx + 1,
                        args.horizon,
                        agent_c,
                        agent_r,
                        rank_model,
                        rank_mean,
                        rank_std,
                        args,
                    )
                    score = _candidate_return(current, future, args)
                    candidate = {
                        "seed": seed,
                        "episode": ep_idx,
                        "request": req_idx,
                        "req_id": int(req.req_id),
                        "c_idx": int(c_idx),
                        "split_id": int(current["split_id"]),
                        "server_id": int(current["server_id"]),
                        "selected_by_ppo": int(c_idx) == int(ppo_c_idx),
                        "current_success": bool(current["info"].get("success", False)),
                        "current_reason": str(current["info"].get("reason", "")),
                        "current_delay_ms": float(current["info"].get("delay_ms", 0.0))
                        if current["info"].get("success", False) else 0.0,
                        "current_fs": float(current["info"].get("num_slots", 0.0))
                        if current["info"].get("success", False) else 0.0,
                        "legal_r_count": int(current["legal_r_count"]),
                        "best_r_score": current["best_r_score"],
                        "top3_r_score_mean": current["top3_r_score_mean"],
                        "r_score_range": current["score_range"],
                        **future,
                        "return": score,
                    }
                    cand_rows.append(candidate)
                    candidate_records.append(candidate)

                best = max(cand_rows, key=lambda row: row["return"])
                ppo = next((row for row in cand_rows if row["c_idx"] == int(ppo_c_idx)), cand_rows[0])
                ordered = sorted(cand_rows, key=lambda row: row["return"], reverse=True)
                ppo_rank = next(i + 1 for i, row in enumerate(ordered) if row["c_idx"] == ppo["c_idx"])
                returns = np.asarray([row["return"] for row in cand_rows], dtype=float)
                legal_r_counts = np.asarray([row["legal_r_count"] for row in cand_rows], dtype=float)
                r_scores = np.asarray([
                    row["best_r_score"] if row["best_r_score"] is not None else np.nan
                    for row in cand_rows
                ], dtype=float)
                top3_scores = np.asarray([
                    row["top3_r_score_mean"] if row["top3_r_score_mean"] is not None else np.nan
                    for row in cand_rows
                ], dtype=float)
                state_records.append({
                    "seed": seed,
                    "episode": ep_idx,
                    "request": req_idx,
                    "req_id": int(req.req_id),
                    "valid_c_count": int(len(legal_c)),
                    "multi_c": len(legal_c) >= 2,
                    "ppo_c_idx": int(ppo_c_idx),
                    "oracle_c_idx": int(best["c_idx"]),
                    "ppo_is_oracle": int(ppo["c_idx"]) == int(best["c_idx"]),
                    "ppo_rank": int(ppo_rank),
                    "ppo_return": float(ppo["return"]),
                    "oracle_return": float(best["return"]),
                    "return_gap": float(best["return"] - ppo["return"]),
                    "return_range": float(np.max(returns) - np.min(returns)),
                    "return_std": float(np.std(returns)),
                    "ppo_future_blocked": int(ppo["future_blocked"]),
                    "oracle_future_blocked": int(best["future_blocked"]),
                    "ppo_future_nsb": int(ppo["future_nsb"]),
                    "oracle_future_nsb": int(best["future_nsb"]),
                    "ppo_legal_r_count": int(ppo["legal_r_count"]),
                    "oracle_legal_r_count": int(best["legal_r_count"]),
                    "legal_r_count_range": float(np.max(legal_r_counts) - np.min(legal_r_counts)),
                    "r_score_range": float(np.nanmax(r_scores) - np.nanmin(r_scores))
                    if np.any(np.isfinite(r_scores)) else 0.0,
                    "top3_r_score_range": float(np.nanmax(top3_scores) - np.nanmin(top3_scores))
                    if np.any(np.isfinite(top3_scores)) else 0.0,
                })

                # Advance the real trajectory with PPO-C + v1.2, so future states are on-policy.
                _execute_current_with_v12(
                    env, req, ppo_c_idx, agent_r, rank_model, rank_mean, rank_std, args
                )
            if args.max_states > 0 and len(state_records) >= args.max_states:
                break
        if args.max_states > 0 and len(state_records) >= args.max_states:
            break

    summary = _summarize(state_records, candidate_records, args)
    return {
        "config": vars(args),
        "summary": summary,
        "states": state_records,
        "candidates": candidate_records,
    }


def _mean(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _summarize(
    states: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    valid_states = [s for s in states if s.get("valid_c_count", 0) > 0]
    multi = [s for s in valid_states if s.get("multi_c")]
    total_decisions = max(len(valid_states), 1)
    horizon_requests = max(len(valid_states) * args.horizon, 1)
    ppo_future_blocks = sum(int(s["ppo_future_blocked"]) for s in valid_states)
    oracle_future_blocks = sum(int(s["oracle_future_blocked"]) for s in valid_states)
    ppo_future_nsb = sum(int(s["ppo_future_nsb"]) for s in valid_states)
    oracle_future_nsb = sum(int(s["oracle_future_nsb"]) for s in valid_states)
    return_gaps = [float(s["return_gap"]) for s in valid_states]
    return_ranges = [float(s["return_range"]) for s in valid_states]
    legal_ranges = [float(s["legal_r_count_range"]) for s in valid_states]

    cand_returns = [float(c["return"]) for c in candidates]
    cand_legal = [float(c["legal_r_count"]) for c in candidates]
    cand_best_r = [
        float(c["best_r_score"]) if c["best_r_score"] is not None else np.nan
        for c in candidates
    ]
    cand_top3 = [
        float(c["top3_r_score_mean"]) if c["top3_r_score_mean"] is not None else np.nan
        for c in candidates
    ]
    cand_future_block = [float(c["future_blocked"]) for c in candidates]
    cand_future_nsb = [float(c["future_nsb"]) for c in candidates]

    oracle_headroom_pp = 100.0 * (ppo_future_blocks - oracle_future_blocks) / horizon_requests
    oracle_nsb_headroom_pp = 100.0 * (ppo_future_nsb - oracle_future_nsb) / horizon_requests
    selected_best_rate = float(np.mean([bool(s["ppo_is_oracle"]) for s in valid_states])) if valid_states else 0.0
    verdict = "FAIL"
    if oracle_headroom_pp >= args.pass_headroom_pp and selected_best_rate <= args.pass_selected_best_rate:
        verdict = "PROCEED_TO_C_RANKER"
    elif oracle_headroom_pp >= 0.25:
        verdict = "MARGINAL"

    return {
        "verdict": verdict,
        "states_total": int(len(states)),
        "valid_c_states": int(len(valid_states)),
        "multi_c_states": int(len(multi)),
        "candidate_total": int(len(candidates)),
        "selected_best_rate": selected_best_rate,
        "mean_ppo_rank": _mean([s["ppo_rank"] for s in valid_states]),
        "mean_return_gap": _mean(return_gaps),
        "mean_return_range": _mean(return_ranges),
        "return_range_gt_0_rate": float(np.mean([r > 0 for r in return_ranges])) if return_ranges else 0.0,
        "return_range_gt_0p1_rate": float(np.mean([r > 0.1 for r in return_ranges])) if return_ranges else 0.0,
        "mean_legal_r_count_range": _mean(legal_ranges),
        "ppo_future_block_rate": 100.0 * ppo_future_blocks / horizon_requests,
        "oracle_future_block_rate": 100.0 * oracle_future_blocks / horizon_requests,
        "oracle_headroom_pp": oracle_headroom_pp,
        "ppo_future_nsb_rate": 100.0 * ppo_future_nsb / horizon_requests,
        "oracle_future_nsb_rate": 100.0 * oracle_future_nsb / horizon_requests,
        "oracle_nsb_headroom_pp": oracle_nsb_headroom_pp,
        "correlations": {
            "legal_r_count_vs_return": _safe_corr(cand_legal, cand_returns),
            "best_r_score_vs_return": _safe_corr(cand_best_r, cand_returns),
            "top3_r_score_vs_return": _safe_corr(cand_top3, cand_returns),
            "legal_r_count_vs_future_blocked": _safe_corr(cand_legal, cand_future_block),
            "best_r_score_vs_future_blocked": _safe_corr(cand_best_r, cand_future_block),
            "top3_r_score_vs_future_blocked": _safe_corr(cand_top3, cand_future_block),
            "legal_r_count_vs_future_nsb": _safe_corr(cand_legal, cand_future_nsb),
            "best_r_score_vs_future_nsb": _safe_corr(cand_best_r, cand_future_nsb),
            "top3_r_score_vs_future_nsb": _safe_corr(cand_top3, cand_future_nsb),
        },
    }


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    s = report["summary"]
    lines = [
        "# C-side Rankability Diagnostic with v1.2 R-ranker",
        "",
        f"**Verdict:** `{s['verdict']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| states_total | {s['states_total']} |",
        f"| valid_c_states | {s['valid_c_states']} |",
        f"| multi_c_states | {s['multi_c_states']} |",
        f"| candidate_total | {s['candidate_total']} |",
        f"| selected_best_rate | {s['selected_best_rate']:.2%} |",
        f"| mean_ppo_rank | {s['mean_ppo_rank']:.2f} |",
        f"| mean_return_gap | {s['mean_return_gap']:.4f} |",
        f"| mean_return_range | {s['mean_return_range']:.4f} |",
        f"| return_range_gt_0_rate | {s['return_range_gt_0_rate']:.2%} |",
        f"| return_range_gt_0p1_rate | {s['return_range_gt_0p1_rate']:.2%} |",
        f"| mean_legal_r_count_range | {s['mean_legal_r_count_range']:.2f} |",
        f"| ppo_future_block_rate | {s['ppo_future_block_rate']:.2f}% |",
        f"| oracle_future_block_rate | {s['oracle_future_block_rate']:.2f}% |",
        f"| oracle_headroom_pp | {s['oracle_headroom_pp']:.2f} pp |",
        f"| ppo_future_nsb_rate | {s['ppo_future_nsb_rate']:.2f}% |",
        f"| oracle_future_nsb_rate | {s['oracle_future_nsb_rate']:.2f}% |",
        f"| oracle_nsb_headroom_pp | {s['oracle_nsb_headroom_pp']:.2f} pp |",
        "",
        "## Candidate Feature Correlations",
        "",
        "| Pair | Pearson | Spearman |",
        "|---|---:|---:|",
    ]
    for key, val in s["correlations"].items():
        pearson = "NA" if val["pearson"] is None else f"{val['pearson']:.4f}"
        spearman = "NA" if val["spearman"] is None else f"{val['spearman']:.4f}"
        lines.append(f"| {key} | {pearson} | {spearman} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max_states", type=int, default=0)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.09)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=14.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--current_block_coef", type=float, default=3.0)
    parser.add_argument("--current_nsb_coef", type=float, default=2.0)
    parser.add_argument("--current_overload_coef", type=float, default=2.0)
    parser.add_argument("--future_block_coef", type=float, default=4.0)
    parser.add_argument("--future_nsb_coef", type=float, default=3.0)
    parser.add_argument("--future_overload_coef", type=float, default=2.0)
    parser.add_argument("--delay_coef", type=float, default=0.03)
    parser.add_argument("--fs_coef", type=float, default=0.05)
    parser.add_argument("--pass_headroom_pp", type=float, default=0.5)
    parser.add_argument("--pass_selected_best_rate", type=float, default=0.9)
    parser.add_argument("--output_json", default="sa_hmarl/experiments/c_rankability_with_v12_diagnostic.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/c_rankability_with_v12_diagnostic.md")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = run(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_md(out_md, result)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
