"""Generate a counterfactual R-side ranking dataset.

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

The dataset trains a listwise ranking model that learns relative preference over
actions in the same decision state.
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


def _compute_return(
    info: Dict[str, Any],
    future: Dict[str, Any],
    args: argparse.Namespace,
) -> float:
    current_blocked = 0.0 if info.get("success", False) else 1.0
    return float(
        -args.return_current_block_coef * current_blocked
        - args.return_future_block_coef * float(future.get("blocked", 0))
        - args.return_future_nsb_coef * float(future.get("no_suitable_block", 0))
        - args.return_delay_coef * float(future.get("delay_mean", 0.0))
        - args.return_fs_coef * float(future.get("avg_fs", 0.0))
    )


def _select_candidate_actions(
    agent_r,
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    legal: List[int],
    max_blocks: int,
    args: argparse.Namespace,
) -> List[int]:
    """Build a diverse candidate set across the full R action space.

    Candidates are *not* restricted to currently legal actions: every path-mod
    pair is represented (first-fit block if feasible, otherwise block 0).  This
    lets the ranking model learn relative preferences over path/modulation
    choices even when many concrete actions are infeasible in a given state.
    """
    if not legal:
        return []

    candidates: List[int] = []
    num_mods = len(obs_r["mod_names"])
    num_paths = len(obs_r["candidate_paths"])
    total_actions = len(r_features)

    # 1. PPO-R top-K (legal only).
    logits = _get_r_logits(agent_r, obs_r)
    top_k = min(args.ppo_top_k, len(legal))
    ppo_top = np.argsort(-logits[legal], kind="stable")[:top_k]
    candidates.extend([int(legal[i]) for i in ppo_top])

    # Bucket legal actions by (path, mod) for representative selection.
    legal_by_pm: Dict[Tuple[int, int], List[int]] = {}
    for a in legal:
        path_idx, mod_idx, _ = decode_agent_r_action(a, num_mods, max_blocks)
        legal_by_pm.setdefault((path_idx, mod_idx), []).append(a)

    # 2. One representative per path-mod pair (covers the whole action space).
    for path_idx in range(num_paths):
        for mod_idx in range(num_mods):
            actions = legal_by_pm.get((path_idx, mod_idx), [])
            if actions:
                # first-fit among feasible blocks for this path-mod.
                rep = min(actions, key=lambda a: decode_agent_r_action(a, num_mods, max_blocks)[2])
            else:
                # Infeasible path-mod pair: represent with block 0.
                rep = path_idx * (num_mods * max_blocks) + mod_idx * max_blocks
            candidates.append(rep)

    # 3. Block-level diversity heuristics per path (legal only).
    by_path: Dict[int, List[int]] = {}
    for a in legal:
        path_idx, mod_idx, block_idx = decode_agent_r_action(a, num_mods, max_blocks)
        by_path.setdefault(path_idx, []).append(a)
    for path_idx, actions in by_path.items():
        feats = {a: r_features[a] for a in actions}

        # first-fit: smallest block_idx.
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

    # 4. Global diversity candidates (legal only).
    shortest = min(legal, key=lambda a: r_features[a][R_FEATURE_ORDER["path_length_km"]])
    candidates.append(shortest)

    highest_mod = max(legal, key=lambda a: r_features[a][R_FEATURE_ORDER["spectral_efficiency"]])
    candidates.append(highest_mod)

    global_low_frag = max(legal, key=lambda a: r_features[a][R_FEATURE_ORDER["free_ratio"]])
    candidates.append(global_low_frag)

    # 5. Random fillers from remaining legal actions to increase block diversity.
    rng = np.random.RandomState(args.candidate_seed)
    seen = set(candidates)
    remaining = [a for a in legal if a not in seen]
    if remaining:
        n_random = min(args.num_random_candidates, len(remaining))
        random_sel = rng.choice(remaining, size=n_random, replace=False).tolist()
        candidates.extend([int(a) for a in random_sel])

    # Deduplicate while preserving order.  Allow both legal and infeasible reps.
    seen = set()
    deduped: List[int] = []
    for a in candidates:
        if 0 <= a < total_actions and a not in seen:
            seen.add(a)
            deduped.append(a)

    # Fallback: if still too few, append every legal action.
    if len(deduped) < args.min_candidates:
        for a in legal:
            if a not in seen:
                seen.add(a)
                deduped.append(a)

    return deduped[: args.max_candidates]


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
) -> List[Dict[str, Any]]:
    """Return a list of group dictionaries, one per sampled state."""
    groups: List[Dict[str, Any]] = []
    env.reset(requests)

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
            snapshot = _snapshot_before_r_decision(env, req.req_id)
            candidate_actions = _select_candidate_actions(
                agent_r, obs_r, r_features, legal, env.max_blocks, args,
            )

            features = []
            returns = []
            action_ids = []
            current_successes = []
            future_blocked_counts = []
            future_nsb_counts = []

            for r_action_idx in candidate_actions:
                branch = copy.deepcopy(snapshot)
                info = _execute_fixed_r(branch, req, split_id, server_id, int(r_action_idx), obs_r)
                actual_horizon = max(min(args.horizon, len(requests) - request_index - 1), 0)
                future = _rollout_future(
                    copy.deepcopy(branch), requests, request_index + 1, actual_horizon,
                    agent_c, agent_r, len(env.mec.servers), args.util_threshold, args.alpha,
                )
                ret = _compute_return(info, future, args)
                features.append(_r_feature_vector(
                    env, req, obs_c, obs_r, r_features, int(r_action_idx), split_id, server_id
                ))
                returns.append(ret)
                action_ids.append(int(r_action_idx))
                current_successes.append(bool(info.get("success", False)))
                future_blocked_counts.append(int(future.get("blocked", 0)))
                future_nsb_counts.append(int(future.get("no_suitable_block", 0)))

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
            })

        # Advance the real trajectory with PPO-R.
        ppo_action = decode_agent_r_action(int(ppo_r_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), ppo_action)

    return groups


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

    for i, g in enumerate(group_list):
        c = g["features"].shape[0]
        features[i, :c] = g["features"]
        returns[i, :c] = g["returns"]
        action_ids[i, :c] = g["action_ids"]
        mask[i, :c] = g["mask"]
        current_successes[i, :c] = g["current_successes"]
        future_blocked_counts[i, :c] = g["future_blocked_counts"]
        future_nsb_counts[i, :c] = g["future_nsb_counts"]

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

    for i in range(n):
        valid = mask[i]
        ret = returns[i, valid]
        if ret.size == 0:
            continue
        ranges.append(float(ret.max() - ret.min()))
        future_blocked = data["future_blocked_counts"][i, valid]
        future_blocked_vars.append(float(future_blocked.var(ddof=0)) if future_blocked.size > 0 else 0.0)

        ppo_idx = int(data["ppo_action_index"][i])
        if ppo_idx >= 0:
            best_idx = int(ret.argmax())
            ppo_top1.append(int(ppo_idx == best_idx))
            # Oracle headroom in percentage points of future blocking rate.
            ppo_blocked = float(future_blocked[ppo_idx])
            best_blocked = float(future_blocked.min())
            oracle_headrooms.append(float(ppo_blocked - best_blocked) / max(horizon, 1))

    nonzero_range_rate = float(np.mean([r > 1e-6 for r in ranges])) if ranges else 0.0
    return {
        "groups": n,
        "avg_candidates": float(mask.sum(axis=1).mean()),
        "nonzero_return_range_rate": nonzero_range_rate,
        "ppo_top1_rate": float(np.mean(ppo_top1)) if ppo_top1 else 0.0,
        "oracle_headroom_pp": float(np.mean(oracle_headrooms)) if oracle_headrooms else 0.0,
        "future_blocked_variance": float(np.mean(future_blocked_vars)) if future_blocked_vars else 0.0,
    }


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
                groups = _generate_episode(
                    env, requests, agent_c, agent_r, split, seed, episode, args
                )
                all_groups[split].extend(groups)
                print(
                    f"[split={split}][seed={seed}][episode={episode + 1}/{num_episodes}] "
                    f"groups={len(groups)} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    diagnostics = {}
    split_data = {}
    for split in splits:
        data = _concat_groups(all_groups[split])
        split_data[split] = data
        diagnostics[split] = _dataset_diagnostics(data, args.horizon)
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
        },
        "horizon": args.horizon,
        "gamma": args.gamma,
        "splits": {split: {"seeds": SPLIT_SEEDS[split], "episodes": split_episodes[split]} for split in splits},
        "diagnostics": diagnostics,
        "train_feature_mean": feature_mean,
        "train_feature_std": feature_std,
        "agent_r_unchanged": r_unchanged,
        "elapsed_seconds": time.time() - started,
        "verdict": "PROCEED_TO_RANKING_TRAINING" if quality_ok and r_unchanged else "STOP_AND_AUDIT_DATASET",
    }
    _atomic_write_json(out / "metadata.json", metadata)
    _write_report(out / "generation_report.md", metadata)
    return metadata


def _write_report(path: Path, metadata: Dict[str, Any]) -> None:
    lines = [
        "# Counterfactual R-Side Ranking Dataset Generation", "",
        f"- Horizon H: {metadata['horizon']}",
        f"- Return: `-{metadata['return_coefs']['current_block']}*cur_blocked - "
        f"{metadata['return_coefs']['future_block']}*future_blocked - "
        f"{metadata['return_coefs']['future_nsb']}*future_nsb - "
        f"{metadata['return_coefs']['delay']}*delay_mean - "
        f"{metadata['return_coefs']['fs']}*avg_fs`", "",
        "| Split | Groups | Avg candidates | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, diag in metadata["diagnostics"].items():
        lines.append(
            f"| {split} | {diag['groups']} | {diag['avg_candidates']:.2f} | "
            f"{diag['nonzero_return_range_rate']:.2%} | {diag['ppo_top1_rate']:.2%} | "
            f"{diag['oracle_headroom_pp']:.4f} | {diag['future_blocked_variance']:.4f} |"
        )
    lines += ["", f"**Verdict: {metadata['verdict']}**"]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--output_dir", default="sa_hmarl/datasets/r_counterfactual_ranking_k5m10_h5_v2")
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
    parser.add_argument("--ppo_top_k", type=int, default=12)
    parser.add_argument("--num_random_candidates", type=int, default=15)
    parser.add_argument("--min_candidates", type=int, default=12)
    parser.add_argument("--max_candidates", type=int, default=40)
    parser.add_argument("--candidate_seed", type=int, default=12345)
    parser.add_argument("--return_current_block_coef", type=float, default=3.0)
    parser.add_argument("--return_future_block_coef", type=float, default=6.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=4.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.01)
    parser.add_argument("--return_fs_coef", type=float, default=0.02)
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
