"""Collect TD post-decision replay transitions for R-side value learning.

This script runs normal closed-loop trajectories (frozen Agent-C + behavior
R policy) and stores, for each request:

- z_t: post-decision feature vector for the executed R action
- r_t: shaped step reward
- done_t: whether the request is the last one in the episode
- next_z_candidates: all legal R-action post-decision features at the next
  decision state (after time advances to the next request)
- next_mask_empty: whether the next decision state has no legal R actions
- diagnostic fields: success, blocked, reason, delay_ms, etc.

No H-step counterfactual rollouts are performed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import (
    _select_post_r_action,
    load_post_checkpoint,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES, _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


SPLIT_SEEDS = {
    "train": (1001, 1002, 1003),
    "val": (2001,),
    "test": (3001,),
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


def _select_c_action(agent_c, obs_c: Dict[str, Any]) -> int:
    idx = agent_c.select_action(obs_c, deterministic=True)
    return int(idx) if idx is not None else 0


def _select_behavior_r_action(
    backend: str,
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    post_pack: Optional[Tuple],
    split_id: int,
    server_id: int,
    device: str,
) -> Tuple[int, Tuple[int, int, int]]:
    legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()
    if not legal:
        return 0, (0, 0, 0)

    if backend == "ppo_r":
        idx = agent_r.select_action(obs_r, deterministic=True)
        if idx is None:
            return int(legal[0]), (0, 0, 0)
        r_action = decode_agent_r_action(int(idx), len(obs_r["mod_names"]), env.max_blocks)
        return int(idx), r_action

    if backend == "r_post":
        if post_pack is None:
            raise ValueError("r_post behavior requires --behavior_r_post_checkpoint")
        model, mean, std, _ = post_pack
        idx = _select_post_r_action(
            env, req, obs_c, obs_r, agent_r, model, mean, std,
            split_id, server_id, "post_only", 1.0, device,
        )
        r_action = decode_agent_r_action(int(idx), len(obs_r["mod_names"]), env.max_blocks)
        return int(idx), r_action

    raise ValueError(f"Unknown behavior R backend: {backend}")


def _reward_from_info(info: Dict[str, Any], req, args: argparse.Namespace) -> float:
    success = bool(info.get("success", False))
    if not success:
        return -args.reward_block_coef
    delay_ms = float(info.get("delay_ms", 0.0))
    delay_norm = delay_ms / max(float(req.deadline_ms), 1.0)
    return -args.reward_delay_coef * delay_norm


def _compute_z(
    env, req, obs_c: Dict[str, Any], obs_r: Dict[str, Any],
    r_features: np.ndarray, action_idx: int, split_id: int, server_id: int,
) -> np.ndarray:
    return _r_feature_vector(
        env, req, obs_c, obs_r, r_features, int(action_idx), split_id, server_id
    )


def _compute_next_candidates(
    env,
    requests: List[Any],
    request_index: int,
    agent_c,
    agent_r,
    args,
) -> Tuple[np.ndarray, int, bool]:
    """Return concatenated next candidate features, count, and empty flag."""
    if request_index + 1 >= len(requests):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), 0, True

    next_req = requests[request_index + 1]
    env.advance_time(next_req.arrival_time)
    next_obs_c = build_agent_c_observation(env, next_req)

    raw_c_mask = np.asarray(next_obs_c["agent_c_mask"], dtype=bool)
    if int(raw_c_mask.sum()) == 0:
        # No legal C action at next state; step a dummy to keep env consistent.
        env.step((0, 0), (0, 0, 0))
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), 0, True

    next_c_action = _select_c_action(agent_c, next_obs_c)
    next_split, next_server = decode_agent_c_action(next_c_action, args.num_servers)
    next_obs_r = build_agent_r_observation(env, next_req, next_split, next_server)
    next_r_features, next_r_mask = agent_r.build_action_features(next_obs_r)
    next_legal = np.flatnonzero(np.asarray(next_r_mask, dtype=bool)).tolist()

    if not next_legal:
        env.step((next_split, next_server), (0, 0, 0))
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), 0, True

    candidates = np.stack([
        _compute_z(
            env, next_req, next_obs_c, next_obs_r, next_r_features,
            int(a), next_split, next_server,
        )
        for a in next_legal
    ]).astype(np.float32)
    return candidates, len(next_legal), False


def _collect_episode(
    env, requests: List[Any], agent_c, agent_r, post_pack: Optional[Tuple],
    split_name: str, seed: int, episode_index: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rows = {
        "z_t": [],
        "reward": [],
        "done": [],
        "next_candidate_lengths": [],
        "next_mask_empty": [],
        "success": [],
        "blocked": [],
        "reason": [],
        "delay_ms": [],
        "no_suitable_block": [],
        "server_overload": [],
        "raw_mask_empty": [],
        "seed": [],
        "episode": [],
        "request_index": [],
    }
    next_candidates_list: List[np.ndarray] = []

    env.reset(requests)
    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_action = _select_c_action(agent_c, obs_c)
        split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()

        # Behavior selection.
        r_idx, r_action = _select_behavior_r_action(
            args.behavior_r_backend, env, req, obs_c, obs_r,
            agent_r, post_pack, split_id, server_id, args.device,
        )

        # z_t for the executed action.
        z_t = _compute_z(env, req, obs_c, obs_r, r_features, r_idx, split_id, server_id)

        # Step the environment with the behavior action.
        _, _, _, info = env.step((split_id, server_id), r_action)
        reward = _reward_from_info(info, req, args)
        done = request_index == len(requests) - 1

        # Compute next-state candidate post-decision features.
        next_z, next_len, next_empty = _compute_next_candidates(
            env, requests, request_index, agent_c, agent_r, args
        )

        rows["z_t"].append(z_t)
        rows["reward"].append(reward)
        rows["done"].append(done)
        rows["next_candidate_lengths"].append(next_len)
        rows["next_mask_empty"].append(next_empty)
        rows["success"].append(bool(info.get("success", False)))
        rows["blocked"].append(not bool(info.get("success", False)))
        rows["reason"].append(info.get("reason", "") or "")
        rows["delay_ms"].append(float(info.get("delay_ms", 0.0)))
        rows["no_suitable_block"].append(
            bool(info.get("reason", "") == "no_suitable_block")
        )
        rows["server_overload"].append(
            bool(info.get("reason", "") == "server_overload")
        )
        rows["raw_mask_empty"].append(len(legal) == 0)
        rows["seed"].append(seed)
        rows["episode"].append(episode_index)
        rows["request_index"].append(request_index)
        if next_len > 0:
            next_candidates_list.append(next_z)

    z_t_array = np.stack(rows["z_t"]).astype(np.float32)
    next_candidates = (
        np.concatenate(next_candidates_list, axis=0).astype(np.float32)
        if next_candidates_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    return {
        "z_t": z_t_array,
        "reward": np.asarray(rows["reward"], dtype=np.float32),
        "done": np.asarray(rows["done"], dtype=bool),
        "next_candidate_lengths": np.asarray(rows["next_candidate_lengths"], dtype=np.int64),
        "next_mask_empty": np.asarray(rows["next_mask_empty"], dtype=bool),
        "next_candidates": next_candidates,
        "success": np.asarray(rows["success"], dtype=bool),
        "blocked": np.asarray(rows["blocked"], dtype=bool),
        "reason": np.asarray(rows["reason"]),
        "delay_ms": np.asarray(rows["delay_ms"], dtype=np.float32),
        "no_suitable_block": np.asarray(rows["no_suitable_block"], dtype=bool),
        "server_overload": np.asarray(rows["server_overload"], dtype=bool),
        "raw_mask_empty": np.asarray(rows["raw_mask_empty"], dtype=bool),
        "seed": np.asarray(rows["seed"], dtype=np.int64),
        "episode": np.asarray(rows["episode"], dtype=np.int64),
        "request_index": np.asarray(rows["request_index"], dtype=np.int64),
    }


def _dataset_diagnostics(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    lengths = data["next_candidate_lengths"]
    rewards = data["reward"]
    return {
        "transitions": int(len(data["z_t"])),
        "next_candidate_total": int(lengths.sum()),
        "next_candidate_mean": float(lengths.mean()),
        "next_candidate_std": float(lengths.std()),
        "next_candidate_p50": float(np.median(lengths)),
        "next_candidate_p95": float(np.percentile(lengths, 95)),
        "next_candidate_max": int(lengths.max()),
        "next_mask_empty_rate": float(data["next_mask_empty"].mean()),
        "done_rate": float(data["done"].mean()),
        "success_rate": float(data["success"].mean()),
        "blocked_rate": float(data["blocked"].mean()),
        "no_suitable_block_rate": float(data["no_suitable_block"].mean()),
        "server_overload_rate": float(data["server_overload"].mean()),
        "raw_mask_empty_rate": float(data["raw_mask_empty"].mean()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "feature_finite_rate": float(np.isfinite(data["z_t"]).mean()),
    }


def collect_replay(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    split_episodes = {
        split: getattr(args, f"{split}_episodes", None) or args.episodes
        for split in splits
    }
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    post_pack = None
    if args.behavior_r_backend == "r_post":
        post_pack = load_post_checkpoint(args.behavior_r_post_checkpoint, args.device)

    diagnostics = {}
    for split in splits:
        split_data = None
        for seed in SPLIT_SEEDS[split]:
            proto = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            rng = np.random.RandomState(seed)
            for episode in range(split_episodes[split]):
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
                arrays = _collect_episode(
                    env, requests, agent_c, agent_r, post_pack,
                    split, seed, episode, args,
                )
                if split_data is None:
                    split_data = {key: [value] for key, value in arrays.items()}
                else:
                    for key in arrays:
                        split_data[key].append(arrays[key])
                print(
                    f"[split={split}][seed={seed}][episode={episode + 1}/{split_episodes[split]}] "
                    f"transitions={len(arrays['z_t'])} next_candidates={len(arrays['next_candidates'])} "
                    f"blocked={int(arrays['blocked'].sum())}",
                    flush=True,
                )
        # Concatenate across seeds/episodes for arrays that are not ragged.
        concat_data = {}
        for key, parts in split_data.items():
            if key == "next_candidates":
                concat_data[key] = np.concatenate(parts, axis=0).astype(np.float32)
            elif key == "reason":
                concat_data[key] = np.concatenate(parts)
            else:
                concat_data[key] = np.concatenate(parts)
        _atomic_save_npz(out / f"{split}.npz", **concat_data)
        diagnostics[split] = _dataset_diagnostics(concat_data)

    # Compute normalization statistics from train z_t.
    train_x = np.load(out / "train.npz", allow_pickle=False)["z_t"].astype(np.float64)
    feature_mean = train_x.mean(axis=0).tolist()
    feature_std = np.maximum(train_x.std(axis=0), 1e-6).tolist()

    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "behavior_r_backend": args.behavior_r_backend,
        "reward_block_coef": args.reward_block_coef,
        "reward_delay_coef": args.reward_delay_coef,
        "gamma": args.gamma,
        "splits": {split: {"seeds": SPLIT_SEEDS[split], "episodes": split_episodes[split]} for split in splits},
        "diagnostics": diagnostics,
        "train_feature_mean": feature_mean,
        "train_feature_std": feature_std,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_write_json(out / "replay_report.json", metadata)
    _write_report(out / "replay_report.md", metadata)
    return metadata


def _write_report(path: Path, metadata: Dict[str, Any]) -> None:
    lines = [
        "# TD R-Side Post-Decision Replay Collection", "",
        f"- Behavior R backend: `{metadata['behavior_r_backend']}`",
        f"- Reward: `-{metadata['reward_block_coef']} * I[blocked] - {metadata['reward_delay_coef']} * delay_norm`",
        f"- Gamma: `{metadata['gamma']}`",
        f"- Feature dimension: `{metadata['feature_dim']}`", "",
        "| Split | Transitions | Success | Blocked | NSB | Overload | Next-empty | Next-cand mean/P95/max | Reward mean/std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, diag in metadata["diagnostics"].items():
        lines.append(
            f"| {split} | {diag['transitions']} | "
            f"{diag['success_rate']:.2%} | {diag['blocked_rate']:.2%} | "
            f"{diag['no_suitable_block_rate']:.2%} | {diag['server_overload_rate']:.2%} | "
            f"{diag['next_mask_empty_rate']:.2%} | "
            f"{diag['next_candidate_mean']:.1f}/{diag['next_candidate_p95']:.0f}/{diag['next_candidate_max']} | "
            f"{diag['reward_mean']:.4f}/{diag['reward_std']:.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="sa_hmarl/datasets/r_td_post_decision_k5m10")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--train_episodes", type=int, default=50)
    parser.add_argument("--val_episodes", type=int, default=10)
    parser.add_argument("--test_episodes", type=int, default=10)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--behavior_r_backend", default="ppo_r", choices=["ppo_r", "r_post"])
    parser.add_argument("--behavior_r_post_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h5_k5m10_blocking_focused/c_post_decision_joint.pt")
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
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
    parser.add_argument("--reward_block_coef", type=float, default=1.0)
    parser.add_argument("--reward_delay_coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    result = collect_replay(build_parser().parse_args())
    print("Collection complete.")
    for split, diag in result["diagnostics"].items():
        print(f"  {split}: transitions={diag['transitions']} blocked={diag['blocked_rate']:.2%} "
              f"next_empty={diag['next_mask_empty_rate']:.2%}")
