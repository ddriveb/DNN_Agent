"""Diagnose whether R-side spectrum block position still has oracle headroom.

The v1.2 R-ranker already learns path length and required-FS preferences.  This
diagnostic isolates a narrower question: for the same selected C action and the
same (path, modulation, required_fs), do different legal spectrum blocks lead
to meaningfully different H-step outcomes?

If the answer is yes, a fragmentation-aware slot-position v1.4 may be worth
training.  If the answer is no, the current action abstraction has little
remaining R-side position headroom.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _rollout_future,
    _select_c_action_from_obs,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    _select_rank_only_r_action,
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    xr = _rankdata(np.asarray(x, dtype=float))
    yr = _rankdata(np.asarray(y, dtype=float))
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _path_availability_after(avail: np.ndarray, start: int, req_fs: int) -> np.ndarray:
    after = np.asarray(avail, dtype=bool).copy()
    after[start : start + req_fs] = False
    return after


def _count_blocks(avail: np.ndarray) -> int:
    count = 0
    in_block = False
    for item in avail:
        if item and not in_block:
            count += 1
            in_block = True
        elif not item:
            in_block = False
    return count


def _max_consecutive(avail: np.ndarray) -> int:
    best = cur = 0
    for item in avail:
        if item:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _position_features(env, obs_r: Dict[str, Any], action_idx: int) -> Optional[Dict[str, float]]:
    num_mods = len(obs_r["mod_names"])
    path_idx, mod_idx, block_idx = decode_agent_r_action(action_idx, num_mods, env.max_blocks)
    if path_idx >= len(obs_r["candidate_paths"]):
        return None
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    if block_idx >= len(blocks):
        return None
    req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    if req_fs is None or req_fs <= 0:
        return None
    start, block_size = blocks[block_idx]
    path = obs_r["candidate_paths"][path_idx]
    avail = env.net.get_available_slots(path)
    after = _path_availability_after(avail, int(start), int(req_fs))

    # Because the current environment allocates at block.start_slot, left_free
    # relative to the selected contiguous free block is normally zero.  We keep
    # both left/right features to make this action-space property visible.
    left_free = 0
    right_free = max(int(block_size) - int(req_fs), 0)
    lfb_before = _max_consecutive(avail)
    lfb_after = _max_consecutive(after)
    blocks_before = _count_blocks(avail)
    blocks_after = _count_blocks(after)
    total_free_before = int(np.sum(avail))
    total_free_after = int(np.sum(after))

    return {
        "path_idx": float(path_idx),
        "mod_idx": float(mod_idx),
        "block_idx": float(block_idx),
        "start_slot": float(start),
        "required_fs": float(req_fs),
        "block_size": float(block_size),
        "left_free": float(left_free),
        "right_free": float(right_free),
        "is_left_edge": float(start == 0),
        "is_right_edge_after": float(int(start) + int(req_fs) >= env.net.num_slots),
        "start_norm": float(start) / max(env.net.num_slots - 1, 1),
        "right_free_norm": float(right_free) / max(env.net.num_slots, 1),
        "lfb_before": float(lfb_before),
        "lfb_after": float(lfb_after),
        "lfb_drop": float(lfb_before - lfb_after),
        "free_blocks_before": float(blocks_before),
        "free_blocks_after": float(blocks_after),
        "free_block_delta": float(blocks_after - blocks_before),
        "total_free_before": float(total_free_before),
        "total_free_after": float(total_free_after),
        "frag_proxy_after": 1.0 - float(lfb_after) / max(float(total_free_after), 1.0),
        "block_consumed_ratio": float(req_fs) / max(float(block_size), 1.0),
    }


def _execute_fixed_r(env, req, split_id: int, server_id: int, action_idx: int, obs_r: Dict[str, Any]):
    action = decode_agent_r_action(int(action_idx), len(obs_r["mod_names"]), env.max_blocks)
    _, _, _, info = env.step((split_id, server_id), action)
    return info


def _return_score(info: Dict[str, Any], future: Dict[str, Any], actual_horizon: int) -> float:
    denom = max(int(actual_horizon), 1)
    current_block = 0.0 if bool(info.get("success", False)) else 1.0
    return float(
        -3.0 * current_block
        -4.0 * float(future.get("blocked", 0)) / denom
        -3.0 * float(future.get("no_suitable_block", 0)) / denom
        -1.0 * float(future.get("server_overload", 0)) / denom
    )


def _group_legal_by_path_mod_fs(env, obs_r: Dict[str, Any], legal: List[int]) -> Dict[Tuple[int, int, int], List[int]]:
    groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    num_mods = len(obs_r["mod_names"])
    for action_idx in legal:
        path_idx, mod_idx, block_idx = decode_agent_r_action(action_idx, num_mods, env.max_blocks)
        if path_idx >= len(obs_r["candidate_paths"]):
            continue
        req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
        blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        if req_fs is None or req_fs <= 0 or block_idx >= len(blocks):
            continue
        groups[(path_idx, mod_idx, int(req_fs))].append(int(action_idx))
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _select_deep_action(deep_rmsa, obs_r: Dict[str, Any]) -> Optional[int]:
    if deep_rmsa is None:
        return None
    action = deep_rmsa.select_action(obs_r)
    return None if action is None else int(action)


def run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
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
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

    records: List[Dict[str, Any]] = []
    group_records: List[Dict[str, Any]] = []
    state_count = 0
    multi_action_states = 0
    position_group_states = 0
    evaluated_groups = 0

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for episode in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            requests = generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max, args.size_min_mb,
                args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile,
            )
            env = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            for request_index, req in enumerate(requests):
                state_count += 1
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
                split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                r_features, r_mask = agent_r.build_action_features(obs_r)
                legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()

                v12_idx = None
                deep_idx = None
                if legal:
                    v12_idx = _select_rank_only_r_action(
                        env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                        split_id, server_id, args.device,
                    )
                    deep_idx = _select_deep_action(deep_rmsa, obs_r)

                if len(legal) >= 2:
                    multi_action_states += 1
                position_groups = _group_legal_by_path_mod_fs(env, obs_r, legal)
                if position_groups:
                    position_group_states += 1

                if position_groups:
                    snapshot = _snapshot_before_r_decision(env, req.req_id)
                    actual_horizon = max(min(args.horizon, len(requests) - request_index - 1), 0)
                    for group_key, actions in position_groups.items():
                        rows = []
                        for action_idx in actions:
                            pos = _position_features(env, obs_r, action_idx)
                            if pos is None:
                                continue
                            branch = copy.deepcopy(snapshot)
                            info = _execute_fixed_r(branch, req, split_id, server_id, action_idx, obs_r)
                            future = _rollout_future(
                                copy.deepcopy(branch),
                                requests,
                                request_index + 1,
                                actual_horizon,
                                agent_c,
                                agent_r,
                                len(env.mec.servers),
                                args.util_threshold,
                                args.alpha,
                            )
                            ret = _return_score(info, future, actual_horizon)
                            row = {
                                "seed": seed,
                                "episode": episode,
                                "request_index": request_index,
                                "req_id": int(req.req_id),
                                "group_key": list(group_key),
                                "action_idx": int(action_idx),
                                "return": ret,
                                "future_blocked": int(future.get("blocked", 0)),
                                "future_nsb": int(future.get("no_suitable_block", 0)),
                                "current_success": bool(info.get("success", False)),
                                "selected_by_v12": int(v12_idx == action_idx) if v12_idx is not None else 0,
                                "selected_by_deep": int(deep_idx == action_idx) if deep_idx is not None else 0,
                                **pos,
                            }
                            rows.append(row)
                        if len(rows) < 2:
                            continue
                        evaluated_groups += 1
                        returns = np.asarray([r["return"] for r in rows], dtype=float)
                        best_i = int(np.argmax(returns))
                        worst_i = int(np.argmin(returns))
                        ret_range = float(returns[best_i] - returns[worst_i])
                        group_record = {
                            "seed": seed,
                            "episode": episode,
                            "request_index": request_index,
                            "req_id": int(req.req_id),
                            "group_key": list(group_key),
                            "num_actions": len(rows),
                            "return_range": ret_range,
                            "future_block_range": int(max(r["future_blocked"] for r in rows) - min(r["future_blocked"] for r in rows)),
                            "future_nsb_range": int(max(r["future_nsb"] for r in rows) - min(r["future_nsb"] for r in rows)),
                            "v12_in_group": int(any(r["selected_by_v12"] for r in rows)),
                            "deep_in_group": int(any(r["selected_by_deep"] for r in rows)),
                            "v12_best": int(any(r["selected_by_v12"] and r is rows[best_i] for r in rows)),
                            "deep_best": int(any(r["selected_by_deep"] and r is rows[best_i] for r in rows)),
                        }
                        group_records.append(group_record)
                        records.extend(rows)

                # Advance real trajectory with v1.2 if possible, else PPO-R fallback.
                if v12_idx is None:
                    v12_idx = agent_r.select_action(obs_r, deterministic=True)
                    if v12_idx is None:
                        v12_idx = 0
                r_action = decode_agent_r_action(int(v12_idx), len(obs_r["mod_names"]), env.max_blocks)
                env.step((split_id, server_id), r_action)

    feature_names = [
        "start_norm", "right_free_norm", "lfb_after", "lfb_drop",
        "free_block_delta", "frag_proxy_after", "block_consumed_ratio",
        "block_idx", "block_size",
    ]
    correlations = {}
    for name in feature_names:
        correlations[name] = _spearman(
            [float(r[name]) for r in records],
            [float(r["return"]) for r in records],
        )

    nonzero_groups = [g for g in group_records if g["return_range"] > 1e-9]
    block_diff_groups = [g for g in group_records if g["future_block_range"] > 0]
    nsb_diff_groups = [g for g in group_records if g["future_nsb_range"] > 0]
    v12_groups = [g for g in group_records if g["v12_in_group"]]
    deep_groups = [g for g in group_records if g["deep_in_group"]]

    summary = {
        "total_states": state_count,
        "multi_action_states": multi_action_states,
        "position_group_states": position_group_states,
        "evaluated_position_groups": evaluated_groups,
        "evaluated_actions": len(records),
        "position_group_state_rate": position_group_states / max(state_count, 1),
        "nonzero_return_group_rate": len(nonzero_groups) / max(evaluated_groups, 1),
        "future_block_diff_group_rate": len(block_diff_groups) / max(evaluated_groups, 1),
        "future_nsb_diff_group_rate": len(nsb_diff_groups) / max(evaluated_groups, 1),
        "mean_return_range": float(np.mean([g["return_range"] for g in group_records])) if group_records else 0.0,
        "p95_return_range": float(np.percentile([g["return_range"] for g in group_records], 95)) if group_records else 0.0,
        "v12_best_rate_when_in_group": (
            float(np.mean([g["v12_best"] for g in v12_groups])) if v12_groups else 0.0
        ),
        "deep_best_rate_when_in_group": (
            float(np.mean([g["deep_best"] for g in deep_groups])) if deep_groups else 0.0
        ),
        "v12_in_group_count": len(v12_groups),
        "deep_in_group_count": len(deep_groups),
        "feature_spearman_with_return": correlations,
    }

    # A conservative proceed rule: enough groups must have real future blocking
    # differences, not just tiny shaped-return differences.
    verdict = "PROCEED_TO_V14_POSITION_RANKER"
    reasons = []
    if summary["future_block_diff_group_rate"] < args.min_block_diff_rate:
        verdict = "STOP_POSITION_FEATURES"
        reasons.append(
            f"future_block_diff_group_rate {summary['future_block_diff_group_rate']:.2%} < "
            f"{args.min_block_diff_rate:.2%}"
        )
    if summary["evaluated_position_groups"] < args.min_groups:
        verdict = "STOP_POSITION_FEATURES"
        reasons.append(
            f"evaluated_position_groups {summary['evaluated_position_groups']} < {args.min_groups}"
        )
    summary["verdict"] = verdict
    summary["verdict_reasons"] = reasons

    return {
        "config": vars(args),
        "summary": summary,
        "group_records_sample": group_records[: args.max_records_in_json],
        "action_records_sample": records[: args.max_records_in_json],
    }


def _write_markdown(path: Path, result: Dict[str, Any]) -> None:
    s = result["summary"]
    lines = [
        "# R Slot-Position Fragmentation Oracle Diagnostic",
        "",
        f"- Verdict: **{s['verdict']}**",
        f"- Total states: {s['total_states']}",
        f"- Multi-action states: {s['multi_action_states']} ({s['multi_action_states']/max(s['total_states'],1):.2%})",
        f"- States with same path/mod/fs but multiple blocks: {s['position_group_states']} ({s['position_group_state_rate']:.2%})",
        f"- Evaluated position groups: {s['evaluated_position_groups']}",
        f"- Evaluated actions: {s['evaluated_actions']}",
        "",
        "## Group-Level Headroom",
        "",
        f"- Nonzero return range group rate: {s['nonzero_return_group_rate']:.2%}",
        f"- Future-block difference group rate: {s['future_block_diff_group_rate']:.2%}",
        f"- Future-NSB difference group rate: {s['future_nsb_diff_group_rate']:.2%}",
        f"- Mean return range: {s['mean_return_range']:.4f}",
        f"- P95 return range: {s['p95_return_range']:.4f}",
        f"- v1.2 best rate when selected action is in a position group: {s['v12_best_rate_when_in_group']:.2%} (n={s['v12_in_group_count']})",
        f"- DeepRMSA best rate when selected action is in a position group: {s['deep_best_rate_when_in_group']:.2%} (n={s['deep_in_group_count']})",
        "",
        "## Feature Spearman Correlation With H-Step Return",
        "",
        "| Feature | Spearman |",
        "|---|---:|",
    ]
    for key, value in sorted(
        s["feature_spearman_with_return"].items(),
        key=lambda kv: 0.0 if kv[1] is None else -abs(kv[1]),
    ):
        lines.append(f"| {key} | {'n/a' if value is None else f'{value:.3f}'} |")
    if s["verdict_reasons"]:
        lines.extend(["", "## Stop Reasons", ""])
        lines.extend([f"- {reason}" for reason in s["verdict_reasons"]])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_s80_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=80)
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
    parser.add_argument("--size_max_mb", type=float, default=40.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min_block_diff_rate", type=float, default=0.02)
    parser.add_argument("--min_groups", type=int, default=100)
    parser.add_argument("--max_records_in_json", type=int, default=200)
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_slot_position_oracle_s80.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_slot_position_oracle_s80.md")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = run_diagnostic(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(out_md, result)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
