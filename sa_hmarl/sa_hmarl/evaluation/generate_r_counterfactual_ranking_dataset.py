"""Generate planner-distillation data for the v1.2 R-side controller.

For each sampled state we:
  1. Freeze PPO-C and use it to pick split/server.
  2. Build a diverse candidate set from legal R actions (PPO top-K + heuristics +
     random), without using DeepRMSA.
  3. Snapshot the env before the R decision.
  4. For every candidate action, deepcopy the snapshot, apply the same C decision
     and the candidate R action, then roll out H future requests with frozen
     PPO-C + PPO-R using the same future trace.
  5. Compute an H-step return and store the candidate features/returns as one
     ranked group.

The H-step branch evaluation is the finite-horizon planner.  The downstream
ranker is trained to imitate/distill this planner's within-state action ranking,
so online inference can avoid rollout and use a single neural forward pass over
the current legal RMSA candidates.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _rollout_future as _rollout_future_base,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES, _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


SPLIT_SEEDS = {
    "train": (1001, 1002, 1003),
    "val": (2001,),
    "test": (3001,),
}

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


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _group_id(split_name: str, seed: int, episode: int, request_index: int) -> int:
    split_code = {"train": 1, "val": 2, "test": 3}[split_name]
    return split_code * 10**15 + seed * 10**7 + episode * 1000 + request_index


def _get_r_logits(agent_r, obs_r: Dict[str, Any]) -> np.ndarray:
    features, mask = agent_r.build_action_features(obs_r)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    return logits


def _rollout_future(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c,
    agent_r,
    num_servers: int,
    util_threshold: float,
    alpha: float,
) -> Dict[str, Any]:
    """Roll out H future requests and collect blocking/NSB/delay/FS/phi stats."""
    blocked = 0
    raw_empty = 0
    no_suitable_block = 0
    server_overload = 0
    phi_specs: List[float] = []
    delays: List[float] = []
    fs_values: List[float] = []

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
            delays.append(0.0)
            fs_values.append(0.0)
            continue

        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split, server = decode_agent_c_action(c_idx, num_servers)
        obs_r = build_agent_r_observation(env, req, split, server)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if int(raw_r_mask.sum()) == 0:
            raw_empty += 1
            blocked += 1
            env.step((split, server), (0, 0, 0))
            delays.append(0.0)
            fs_values.append(0.0)
            continue

        r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split, server), r_action)

        if not info.get("success", False):
            blocked += 1
            reason = info.get("reason", "")
            if reason == "no_suitable_block":
                no_suitable_block += 1
            elif reason == "server_overload":
                server_overload += 1
            delays.append(0.0)
            fs_values.append(0.0)
        else:
            delays.append(float(info.get("delay_ms", 0.0)))
            fs_values.append(float(info.get("num_slots", 0)))

    denom = max(len(delays), 1)
    return {
        "blocked": blocked,
        "raw_mask_empty": raw_empty,
        "no_suitable_block": no_suitable_block,
        "server_overload": server_overload,
        "phi_spec_mean": float(np.mean(phi_specs)) if phi_specs else None,
        "phi_spec_min": float(np.min(phi_specs)) if phi_specs else None,
        "phi_spec_end": phi_specs[-1] if phi_specs else None,
        "delay_mean": float(np.sum(delays) / denom),
        "avg_fs": float(np.sum(fs_values) / denom),
    }


def _compute_phi_after_obs_c_after(
    branch_env,
    next_req,
    args: argparse.Namespace,
) -> float:
    """Compute post-decision spectrum viability from the next-request C-observation.

    Returns a scalar in roughly [0, alpha+beta] that is larger when the post-decision
    state leaves many valid C actions and many feasible R actions for the next request.
    """
    if next_req is None:
        return 0.0
    try:
        obs_c = build_agent_c_observation(branch_env, next_req)
    except Exception:
        # If building the next observation fails for any reason, treat as no viability.
        return 0.0

    raw_c_mask = np.asarray(obs_c.get("agent_c_mask", []), dtype=bool)
    k_c_valid = int(raw_c_mask.sum())

    feasible_counts = obs_c.get("feasible_counts", [])
    k_r_total = 0
    if isinstance(feasible_counts, (list, tuple, np.ndarray)):
        k_r_total = int(np.sum([int(x) for row in feasible_counts for x in (row if hasattr(row, "__iter__") else [row])]))

    num_splits = len(next_req.splits)
    num_servers = len(branch_env.mec.servers)
    num_c_actions = max(num_splits * num_servers, 1)

    num_mods = len(getattr(branch_env, "mod_reg", None) and branch_env.mod_reg.names or [])
    if num_mods == 0:
        # Fallback: try to infer from the env modulation profile.
        try:
            num_mods = len(ModulationRegistry.from_profile(args.modulation_profile).names)
        except Exception:
            num_mods = 1
    max_r_per_c = max(branch_env.k * num_mods * branch_env.max_blocks, 1)
    max_r_total = max(num_c_actions * max_r_per_c, 1)

    k_c_norm = float(np.log(1 + k_c_valid) / np.log(1 + num_c_actions))
    k_r_norm = float(np.log(1 + k_r_total) / np.log(1 + max_r_total))

    return float(args.viability_alpha_kc * k_c_norm + args.viability_beta_kr * k_r_norm)


def _compute_return(
    info: Dict[str, Any],
    future: Dict[str, Any],
    args: argparse.Namespace,
    path_km_norm: float = 0.0,
    required_fs_norm: float = 0.0,
) -> float:
    current_blocked = 0.0 if info.get("success", False) else 1.0
    return float(
        -args.return_current_block_coef * current_blocked
        - args.return_future_block_coef * float(future.get("blocked", 0))
        - args.return_future_nsb_coef * float(future.get("no_suitable_block", 0))
        - args.return_delay_coef * float(future.get("delay_mean", 0.0))
        - args.return_fs_coef * float(future.get("avg_fs", 0.0))
        - args.return_future_server_overload_coef * float(future.get("server_overload", 0))
        - args.path_penalty_coef * path_km_norm
        - args.fs_penalty_coef * required_fs_norm
    )


def _select_candidate_actions_v1(
    agent_r,
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    legal: List[int],
    max_blocks: int,
    args: argparse.Namespace,
) -> List[int]:
    """Original v1 diverse candidate subset of legal R actions (no DeepRMSA)."""
    if not legal:
        return []

    candidates: List[int] = []

    # 1. PPO-R top-K.
    logits = _get_r_logits(agent_r, obs_r)
    top_k = min(args.ppo_top_k, len(legal))
    ppo_top = np.argsort(-logits[legal], kind="stable")[:top_k]
    candidates.extend([int(legal[i]) for i in ppo_top])

    # Decode legal actions for heuristic selection.
    num_mods = len(obs_r["mod_names"])
    # Heuristic buckets keyed by path_idx.
    by_path: Dict[int, List[int]] = {}
    for a in legal:
        path_idx, mod_idx, block_idx = decode_agent_r_action(a, num_mods, max_blocks)
        by_path.setdefault(path_idx, []).append(a)

    # 2. Diversity heuristics per path.
    for path_idx, actions in by_path.items():
        feats = {a: r_features[a] for a in actions}

        # first-fit: smallest block_idx (earliest in candidate list).
        first_fit = min(actions, key=lambda a: decode_agent_r_action(a, num_mods, max_blocks)[2])
        candidates.append(first_fit)

        # best-fit: block_size closest to required_fs.
        best_fit = min(
            actions,
            key=lambda a: abs(feats[a][R_FEATURE_ORDER["block_size"]] - feats[a][R_FEATURE_ORDER["required_fs"]]),
        )
        candidates.append(best_fit)

        # largest-block.
        largest = max(actions, key=lambda a: feats[a][R_FEATURE_ORDER["block_size"]])
        candidates.append(largest)

        # minimum waste.
        min_waste = min(actions, key=lambda a: feats[a][R_FEATURE_ORDER["block_waste"]])
        candidates.append(min_waste)

    # 3. Shortest-path candidate.
    shortest = min(legal, key=lambda a: r_features[a][R_FEATURE_ORDER["path_length_km"]])
    candidates.append(shortest)

    # 4. Random fillers.
    rng = np.random.RandomState(args.candidate_seed)
    remaining = [a for a in legal if a not in candidates]
    if remaining:
        n_random = min(args.num_random_candidates, len(remaining))
        random_sel = rng.choice(remaining, size=n_random, replace=False).tolist()
        candidates.extend([int(a) for a in random_sel])

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[int] = []
    for a in candidates:
        if a not in seen and a in legal:
            seen.add(a)
            deduped.append(a)

    # If too few, fall back to all legal actions.
    if len(deduped) < args.min_candidates and len(legal) <= args.max_candidates:
        deduped = list(dict.fromkeys(legal).keys())

    return deduped[: args.max_candidates]


def _select_candidate_actions_highest_se(
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    legal: List[int],
    max_blocks: int,
    args: argparse.Namespace,
) -> List[int]:
    """Highest-SE (or top-2 SE) path-block candidate set.

    For each path, keep only the feasible modulation(s) with the highest
    spectral efficiency, then keep all legal blocks for that modulation.
    This encodes the physical prior that higher SE uses fewer slots.
    """
    if not legal:
        return []

    num_mods = len(obs_r["mod_names"])
    num_paths = len(obs_r["candidate_paths"])

    # Bucket legal actions by (path, mod).
    by_pm: Dict[Tuple[int, int], List[int]] = {}
    for a in legal:
        path_idx, mod_idx, _ = decode_agent_r_action(a, num_mods, max_blocks)
        by_pm.setdefault((path_idx, mod_idx), []).append(a)

    n_keep = 2 if args.candidate_mode == "top2_se_path_block" else 1
    candidates: List[int] = []
    for path_idx in range(num_paths):
        mods = []
        for (p, m), actions in by_pm.items():
            if p != path_idx:
                continue
            # All actions for the same path-mod share the same SE.
            se = float(r_features[actions[0]][R_FEATURE_ORDER["spectral_efficiency"]])
            mods.append((se, m, actions))
        if not mods:
            continue
        mods.sort(key=lambda x: (-x[0], x[1]))
        for _, _, actions in mods[:n_keep]:
            # Preserve block order.
            ordered = sorted(actions, key=lambda a: decode_agent_r_action(a, num_mods, max_blocks)[2])
            candidates.extend(ordered)

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[int] = []
    for a in candidates:
        if a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped


def _select_candidate_actions(
    agent_r,
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    legal: List[int],
    max_blocks: int,
    args: argparse.Namespace,
) -> List[int]:
    """Dispatch to the requested candidate selection strategy."""
    if args.candidate_mode in ("highest_se_path_block", "top2_se_path_block"):
        return _select_candidate_actions_highest_se(obs_r, r_features, legal, max_blocks, args)
    return _select_candidate_actions_v1(agent_r, obs_r, r_features, legal, max_blocks, args)


def _execute_fixed_r(env, req, split_id: int, server_id: int, r_action_idx: int, obs_r: Dict[str, Any]):
    r_action = decode_agent_r_action(int(r_action_idx), len(obs_r["mod_names"]), env.max_blocks)
    _, _, _, info = env.step((split_id, server_id), r_action)
    return info


def _generate_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    split_name: str,
    seed: int,
    episode_index: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Return (groups, sampled_states, skipped_groups) for one episode."""
    groups: List[Dict[str, Any]] = []
    env.reset(requests)
    sampled_states = 0
    skipped_groups = 0

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()

        # Always select PPO-R action for the deployed trajectory.
        ppo_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        if len(legal) >= 2:
            sampled_states += 1
            snapshot = _snapshot_before_r_decision(env, req.req_id)
            candidate_actions = _select_candidate_actions(
                agent_r, obs_r, r_features, legal, env.max_blocks, args,
            )

            if len(candidate_actions) < 2:
                skipped_groups += 1
            else:
                features = []
                returns = []
                action_ids = []
                current_successes = []
                future_blocked_counts = []
                future_nsb_counts = []
                phi_after_values = []

                # Pre-compute legal-action statistics for resource-penalty normalization.
                legal_path_km = np.array([float(r_features[int(a)][R_FEATURE_ORDER["path_length_km"]]) for a in legal])
                legal_required_fs = np.array([float(r_features[int(a)][R_FEATURE_ORDER["required_fs"]]) for a in legal])
                path_mean = legal_path_km.mean()
                path_std = legal_path_km.std() + 1e-6
                fs_mean = legal_required_fs.mean()
                fs_std = legal_required_fs.std() + 1e-6

                next_req = requests[request_index + 1] if request_index + 1 < len(requests) else None
                actual_horizon = max(min(args.horizon, len(requests) - request_index - 1), 0)

                for r_action_idx in candidate_actions:
                    branch = copy.deepcopy(snapshot)
                    info = _execute_fixed_r(branch, req, split_id, server_id, int(r_action_idx), obs_r)
                    phi_after = _compute_phi_after_obs_c_after(branch, next_req, args)
                    future = _rollout_future(
                        copy.deepcopy(branch), requests, request_index + 1, actual_horizon,
                        agent_c, agent_r, len(env.mec.servers), args.util_threshold, args.alpha,
                    )
                    path_km_norm = (float(r_features[int(r_action_idx)][R_FEATURE_ORDER["path_length_km"]]) - path_mean) / path_std
                    required_fs_norm = (float(r_features[int(r_action_idx)][R_FEATURE_ORDER["required_fs"]]) - fs_mean) / fs_std
                    ret_v12 = _compute_return(info, future, args, path_km_norm, required_fs_norm)
                    ret = ret_v12 + args.viability_phi_coef * phi_after
                    features.append(_r_feature_vector(
                        env, req, obs_c, obs_r, r_features, int(r_action_idx), split_id, server_id
                    ))
                    returns.append(ret)
                    action_ids.append(int(r_action_idx))
                    current_successes.append(bool(info.get("success", False)))
                    future_blocked_counts.append(int(future.get("blocked", 0)))
                    future_nsb_counts.append(int(future.get("no_suitable_block", 0)))
                    phi_after_values.append(phi_after)

                features = np.stack(features).astype(np.float32)
                returns = np.asarray(returns, dtype=np.float32)
                action_ids = np.asarray(action_ids, dtype=np.int64)
                mask = np.ones(len(action_ids), dtype=bool)

                ppo_action_index = int(np.where(action_ids == int(ppo_r_idx))[0][0]) if int(ppo_r_idx) in action_ids else -1

                groups.append({
                    "group_id": _group_id(split_name, seed, episode_index, request_index),
                    "seed": seed,
                    "episode": episode_index,
                    "request_index": request_index,
                    "split": split_name,
                    "legal_count": len(legal),
                    "features": features,
                    "returns": returns,
                    "action_ids": action_ids,
                    "mask": mask,
                    "ppo_action_index": ppo_action_index,
                    "ppo_action_id": int(ppo_r_idx),
                    "current_successes": np.asarray(current_successes, dtype=bool),
                    "future_blocked_counts": np.asarray(future_blocked_counts, dtype=np.int64),
                    "future_nsb_counts": np.asarray(future_nsb_counts, dtype=np.int64),
                    "phi_after": np.asarray(phi_after_values, dtype=np.float32),
                })

        # Advance the real trajectory with PPO-R.
        ppo_action = decode_agent_r_action(int(ppo_r_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), ppo_action)

    return groups, sampled_states, skipped_groups


def _concat_groups(group_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not group_list:
        return {
            "features": np.empty((0, 0, len(FEATURE_NAMES)), dtype=np.float32),
            "returns": np.empty((0, 0), dtype=np.float32),
            "action_ids": np.empty((0, 0), dtype=np.int64),
            "mask": np.empty((0, 0), dtype=bool),
            "ppo_action_index": np.empty((0,), dtype=np.int64),
            "group_id": np.empty((0,), dtype=np.int64),
            "seed": np.empty((0,), dtype=np.int64),
            "episode": np.empty((0,), dtype=np.int64),
            "request_index": np.empty((0,), dtype=np.int64),
            "legal_count": np.empty((0,), dtype=np.int64),
            "current_successes": np.empty((0, 0), dtype=bool),
            "future_blocked_counts": np.empty((0, 0), dtype=np.int64),
            "future_nsb_counts": np.empty((0, 0), dtype=np.int64),
            "phi_after": np.empty((0, 0), dtype=np.float32),
        }
    max_c = max(int(g["features"].shape[0]) for g in group_list)
    n = len(group_list)
    d = group_list[0]["features"].shape[1]

    features = np.zeros((n, max_c, d), dtype=np.float32)
    returns = np.full((n, max_c), -1e9, dtype=np.float32)
    action_ids = np.full((n, max_c), -1, dtype=np.int64)
    mask = np.zeros((n, max_c), dtype=bool)
    current_successes = np.zeros((n, max_c), dtype=bool)
    future_blocked_counts = np.zeros((n, max_c), dtype=np.int64)
    future_nsb_counts = np.zeros((n, max_c), dtype=np.int64)
    phi_after = np.zeros((n, max_c), dtype=np.float32)

    for i, g in enumerate(group_list):
        c = g["features"].shape[0]
        features[i, :c] = g["features"]
        returns[i, :c] = g["returns"]
        action_ids[i, :c] = g["action_ids"]
        mask[i, :c] = g["mask"]
        current_successes[i, :c] = g["current_successes"]
        future_blocked_counts[i, :c] = g["future_blocked_counts"]
        future_nsb_counts[i, :c] = g["future_nsb_counts"]
        phi_after[i, :c] = g["phi_after"]

    return {
        "features": features,
        "returns": returns,
        "action_ids": action_ids,
        "mask": mask,
        "ppo_action_index": np.asarray([g["ppo_action_index"] for g in group_list], dtype=np.int64),
        "ppo_action_id": np.asarray([g["ppo_action_id"] for g in group_list], dtype=np.int64),
        "group_id": np.asarray([g["group_id"] for g in group_list], dtype=np.int64),
        "seed": np.asarray([g["seed"] for g in group_list], dtype=np.int64),
        "episode": np.asarray([g["episode"] for g in group_list], dtype=np.int64),
        "request_index": np.asarray([g["request_index"] for g in group_list], dtype=np.int64),
        "legal_count": np.asarray([g["legal_count"] for g in group_list], dtype=np.int64),
        "current_successes": current_successes,
        "future_blocked_counts": future_blocked_counts,
        "future_nsb_counts": future_nsb_counts,
        "phi_after": phi_after,
    }


def _dataset_diagnostics(data: Dict[str, np.ndarray], horizon: int = 1) -> Dict[str, Any]:
    returns = data["returns"]
    mask = data["mask"]
    n = returns.shape[0]
    if n == 0:
        return {
            "groups": 0,
            "avg_candidates": 0.0,
            "nonzero_return_range_rate": 0.0,
            "ppo_top1_rate": 0.0,
            "oracle_headroom_pp": 0.0,
            "future_blocked_variance": 0.0,
        }

    ranges = []
    ppo_top1 = []
    oracle_headrooms = []
    future_blocked_vars = []
    phi_after_all = []
    phi_after_valid_pairs = []

    for i in range(n):
        valid = mask[i]
        ret = returns[i, valid]
        if ret.size == 0:
            continue
        ranges.append(float(ret.max() - ret.min()))
        future_blocked = data["future_blocked_counts"][i, valid]
        future_blocked_vars.append(float(future_blocked.var(ddof=0)) if future_blocked.size > 0 else 0.0)

        if "phi_after" in data:
            phi = data["phi_after"][i, valid]
            phi_after_all.extend(phi.tolist())
            if phi.size >= 2 and ret.size >= 2:
                rho, _ = stats.spearmanr(phi, ret)
                if not np.isnan(rho):
                    phi_after_valid_pairs.append(float(rho))

        ppo_idx = int(data["ppo_action_index"][i])
        if ppo_idx >= 0:
            best_idx = int(ret.argmax())
            ppo_top1.append(int(ppo_idx == best_idx))
            # Oracle headroom in percentage points of future blocking rate.
            ppo_blocked = float(future_blocked[ppo_idx])
            best_blocked = float(future_blocked.min())
            oracle_headrooms.append(float(ppo_blocked - best_blocked) / max(horizon, 1))

    nonzero_range_rate = float(np.mean([r > 1e-6 for r in ranges])) if ranges else 0.0
    phi_arr = np.asarray(phi_after_all)
    result = {
        "groups": n,
        "avg_candidates": float(mask.sum(axis=1).mean()),
        "nonzero_return_range_rate": nonzero_range_rate,
        "ppo_top1_rate": float(np.mean(ppo_top1)) if ppo_top1 else 0.0,
        "oracle_headroom_pp": float(np.mean(oracle_headrooms)) if oracle_headrooms else 0.0,
        "future_blocked_variance": float(np.mean(future_blocked_vars)) if future_blocked_vars else 0.0,
    }
    if phi_arr.size > 0:
        result["phi_after_mean"] = float(phi_arr.mean())
        result["phi_after_std"] = float(phi_arr.std())
        result["phi_after_range_nonzero_rate"] = float(np.mean(phi_arr > 1e-6))
        result["phi_after_spearman_with_return"] = float(np.mean(phi_after_valid_pairs)) if phi_after_valid_pairs else 0.0
    return result


def _candidate_feature_stats(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Return descriptive stats about the candidate set."""
    mask = data["mask"]
    n = mask.shape[0]
    if n == 0:
        return {}
    counts = mask.sum(axis=1)
    valid_features = data["features"][mask]
    stats: Dict[str, float] = {
        "p50_candidates": float(np.median(counts)),
        "p90_candidates": float(np.percentile(counts, 90)),
    }
    if valid_features.size > 0:
        stats.update({
            "avg_spectral_efficiency": float(valid_features[:, R_FEATURE_ORDER["spectral_efficiency"]].mean()),
            "avg_block_size": float(valid_features[:, R_FEATURE_ORDER["block_size"]].mean()),
            "avg_block_waste": float(valid_features[:, R_FEATURE_ORDER["block_waste"]].mean()),
            "avg_path_idx_norm": float(valid_features[:, -3].mean()),
            "avg_mod_idx_norm": float(valid_features[:, -2].mean()),
            "avg_block_idx_norm": float(valid_features[:, -1].mean()),
        })
    return stats


def generate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    r_before = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}

    split_episodes = {
        split: getattr(args, f"{split}_episodes", None) or args.episodes
        for split in splits
    }

    all_groups: Dict[str, List[Dict[str, Any]]] = {split: [] for split in splits}
    sampled_states = {split: 0 for split in splits}
    skipped_groups = {split: 0 for split in splits}

    for split in splits:
        num_episodes = split_episodes[split]
        for seed in SPLIT_SEEDS[split]:
            proto = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            rng = np.random.RandomState(seed)
            for episode in range(num_episodes):
                src = int(rng.randint(0, proto.net.NUM_NODES))
                requests = generate_requests(
                    proto, rng, src, args.requests_per_episode,
                    args.arrival_interval, args.holding_min, args.holding_max,
                    args.deadline_min, args.deadline_max, args.size_min_mb,
                    args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                    args.num_splits, args.split_profile,
                )
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                t0 = time.time()
                groups, sampled, skipped = _generate_episode(
                    env, requests, agent_c, agent_r, split, seed, episode, args
                )
                all_groups[split].extend(groups)
                sampled_states[split] += sampled
                skipped_groups[split] += skipped
                print(
                    f"[split={split}][seed={seed}][episode={episode + 1}/{num_episodes}] "
                    f"groups={len(groups)} sampled={sampled} skipped={skipped} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    diagnostics = {}
    split_data = {}
    for split in splits:
        data = _concat_groups(all_groups[split])
        split_data[split] = data
        diag = _dataset_diagnostics(data, args.horizon)
        diag.update(_candidate_feature_stats(data))
        sampled = sampled_states.get(split, 0)
        skipped = skipped_groups.get(split, 0)
        diag["sampled_states"] = sampled
        diag["skipped_groups"] = skipped
        diag["skipped_ratio"] = float(skipped / max(sampled, 1))
        diagnostics[split] = diag
        _atomic_save_npz(out / f"{split}.npz", **data)

    # Normalization statistics from train features (valid candidates only).
    train_data = split_data["train"]
    train_features = train_data["features"]
    train_mask = train_data["mask"]
    valid_features = train_features[train_mask].astype(np.float64)
    feature_mean = valid_features.mean(axis=0).tolist()
    feature_std = np.maximum(valid_features.std(axis=0), 1e-6).tolist()

    r_unchanged = all(
        torch.equal(value, r_before[key])
        for key, value in agent_r.policy_net.state_dict().items()
    )

    # Quality gates.
    quality_ok = all(
        diag["nonzero_return_range_rate"] >= args.min_nonzero_range_rate
        and diag["oracle_headroom_pp"] >= args.min_oracle_headroom_pp
        for diag in diagnostics.values()
    )

    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "return_coefs": {
            "current_block": args.return_current_block_coef,
            "future_block": args.return_future_block_coef,
            "future_nsb": args.return_future_nsb_coef,
            "delay": args.return_delay_coef,
            "fs": args.return_fs_coef,
            "path_penalty": args.path_penalty_coef,
            "fs_penalty": args.fs_penalty_coef,
        },
        "horizon": args.horizon,
        "gamma": args.gamma,
        "splits": {split: {"seeds": SPLIT_SEEDS[split], "episodes": split_episodes[split]} for split in splits},
        "diagnostics": diagnostics,
        "train_feature_mean": feature_mean,
        "train_feature_std": feature_std,
        "agent_r_unchanged": r_unchanged,
        "elapsed_seconds": time.time() - started,
        "candidate_mode": args.candidate_mode,
        "viability": {
            "viability_phi_coef": args.viability_phi_coef,
            "viability_alpha_kc": args.viability_alpha_kc,
            "viability_beta_kr": args.viability_beta_kr,
            "viability_mode": args.viability_mode,
        },
        "verdict": "PROCEED_TO_RANKING_TRAINING" if quality_ok and r_unchanged else "STOP_AND_AUDIT_DATASET",
    }
    _atomic_write_json(out / "metadata.json", metadata)
    _write_report(out / "generation_report.md", metadata)
    return metadata


def _write_report(path: Path, metadata: Dict[str, Any]) -> None:
    return_line = (
        f"- Return: `-{metadata['return_coefs']['current_block']}*cur_blocked - "
        f"{metadata['return_coefs']['future_block']}*future_blocked - "
        f"{metadata['return_coefs']['future_nsb']}*future_nsb - "
        f"{metadata['return_coefs']['delay']}*delay_mean - "
        f"{metadata['return_coefs']['fs']}*avg_fs"
    )
    if metadata['return_coefs'].get('path_penalty') or metadata['return_coefs'].get('fs_penalty'):
        return_line += (
            f" - {metadata['return_coefs']['path_penalty']}*norm(path_km)"
            f" - {metadata['return_coefs']['fs_penalty']}*norm(required_fs)`"
        )
    else:
        return_line += "`"

    viability = metadata.get("viability", {})
    viability_line = ""
    if viability.get("viability_phi_coef", 0.0) != 0.0:
        viability_line = (
            f"- Viability Phi_after: coef={viability['viability_phi_coef']}, "
            f"alpha_kc={viability['viability_alpha_kc']}, "
            f"beta_kr={viability['viability_beta_kr']}, "
            f"mode={viability['viability_mode']}"
        )

    lines = [
        "# Counterfactual R-Side Ranking Dataset Generation", "",
        f"- Candidate mode: `{metadata.get('candidate_mode', 'v1')}`",
        f"- Horizon H: {metadata['horizon']}",
        return_line,
    ]
    if viability_line:
        lines.append(viability_line)
    lines += ["",
        "| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, diag in metadata["diagnostics"].items():
        lines.append(
            f"| {split} | {diag['groups']} | {diag['skipped_groups']} ({diag['skipped_ratio']:.1%}) | "
            f"{diag['avg_candidates']:.2f} | {diag.get('p50_candidates', 0):.0f} | {diag.get('p90_candidates', 0):.0f} | "
            f"{diag['nonzero_return_range_rate']:.2%} | {diag['ppo_top1_rate']:.2%} | "
            f"{diag['oracle_headroom_pp']:.4f} | {diag['future_blocked_variance']:.4f} |"
        )
    lines += ["", "### Candidate feature profile", "",
        "| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, diag in metadata["diagnostics"].items():
        lines.append(
            f"| {split} | {diag.get('avg_spectral_efficiency', 0):.3f} | "
            f"{diag.get('avg_block_size', 0):.2f} | {diag.get('avg_block_waste', 0):.3f} | "
            f"{diag.get('avg_path_idx_norm', 0):.3f} | {diag.get('avg_mod_idx_norm', 0):.3f} | "
            f"{diag.get('avg_block_idx_norm', 0):.3f} |"
        )
    # Phi_after diagnostics if viability was used.
    if any("phi_after_mean" in diag for diag in metadata["diagnostics"].values()):
        lines += ["", "### Phi-after viability profile", "",
            "| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |",
            "|---|---:|---:|---:|---:|",
        ]
        for split, diag in metadata["diagnostics"].items():
            lines.append(
                f"| {split} | {diag.get('phi_after_mean', 0):.4f} | "
                f"{diag.get('phi_after_std', 0):.4f} | "
                f"{diag.get('phi_after_range_nonzero_rate', 0):.2%} | "
                f"{diag.get('phi_after_spearman_with_return', 0):.3f} |"
            )

    lines += ["", f"**Verdict: {metadata['verdict']}**"]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--output_dir", default="sa_hmarl/datasets/r_counterfactual_ranking_k5m10_h5")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--train_episodes", type=int, default=None)
    parser.add_argument("--val_episodes", type=int, default=None)
    parser.add_argument("--test_episodes", type=int, default=None)
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
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--ppo_top_k", type=int, default=8)
    parser.add_argument("--num_random_candidates", type=int, default=5)
    parser.add_argument("--min_candidates", type=int, default=15)
    parser.add_argument("--max_candidates", type=int, default=30)
    parser.add_argument("--candidate_seed", type=int, default=12345)
    parser.add_argument("--return_current_block_coef", type=float, default=3.0)
    parser.add_argument("--return_future_block_coef", type=float, default=4.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=3.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.03)
    parser.add_argument("--return_fs_coef", type=float, default=0.05)
    parser.add_argument("--return_future_server_overload_coef", type=float, default=0.0,
                        help="Penalty weight on future server_overload count in the H-step return.")
    parser.add_argument("--path_penalty_coef", type=float, default=0.0,
                        help="Penalty coefficient for normalized path_length_km in the return.")
    parser.add_argument("--fs_penalty_coef", type=float, default=0.0,
                        help="Penalty coefficient for normalized required_fs in the return.")
    parser.add_argument("--viability_phi_coef", type=float, default=0.0,
                        help="Weight on the post-decision spectrum-viability potential Phi_after.")
    parser.add_argument("--viability_alpha_kc", type=float, default=1.0,
                        help="Weight on normalized log of valid C actions in Phi_after.")
    parser.add_argument("--viability_beta_kr", type=float, default=0.3,
                        help="Weight on normalized log of total feasible R actions in Phi_after.")
    parser.add_argument("--viability_mode", default="obs_c_after", choices=["obs_c_after", "future_type_probe"],
                        help="How to compute Phi_after. obs_c_after uses the next-request C-observation.")
    parser.add_argument("--candidate_mode", default="v1", choices=["v1", "highest_se_path_block", "top2_se_path_block"],
                        help="Candidate selection strategy.")
    parser.add_argument("--min_nonzero_range_rate", type=float, default=0.20)
    parser.add_argument("--min_oracle_headroom_pp", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    result = generate_dataset(build_parser().parse_args())
    print(result["verdict"])
    for split, diag in result["diagnostics"].items():
        print(f"  {split}: groups={diag['groups']} avg_cand={diag['avg_candidates']:.2f} "
              f"range={diag['nonzero_return_range_rate']:.2%} headroom={diag['oracle_headroom_pp']:.4f}")
