"""Paired feature-only dataset generator for v1.35.

For every canonical decision group we generate:
  - the same group_id, state hash, fixed C action, candidate action_ids/mask
  - the same H-step labels and label components
  - two feature matrices: features_v1 (25-d) and features_poststate_v1 (41-d)

Candidate selection is forced to `ppo_r_topk_only` with max_candidates=30 to match
the historical v1.3 setting and isolate the feature effect.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

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
    _select_r_action_from_obs,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import (
    _compute_phi_after_obs_c_after,
    _compute_return,
    _group_id,
    _rollout_future as _rollout_future_base,
    _select_candidate_actions,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    build_poststate_v1_feature,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


# v1.35 fixed protocol defaults.
DEFAULT_K_PATHS_C = 5
DEFAULT_K_PATHS_R = 50
DEFAULT_PATH_SORT = "hops"
DEFAULT_BLOCK_SORT = "start_asc"
DEFAULT_MAX_CANDIDATES = 30
DEFAULT_PPO_TOP_K = 30
DEFAULT_HORIZON = 5


def _state_hash(env, req, split_id: int, server_id: int) -> str:
    """Stable hash of the current decision state."""
    h = hashlib.sha256()
    h.update(f"t={env.time:.9f}|req={req.req_id}|split={split_id}|server={server_id}".encode())
    for key in sorted(env.net.link_states.keys()):
        h.update(str(key).encode())
        h.update(np.asarray(env.net.link_states[key], dtype=np.int64).tobytes())
    utils = np.asarray([s.utilization for s in env.mec.servers], dtype=np.float32)
    h.update(utils.tobytes())
    return h.hexdigest()[:16]


def _get_r_logits(agent_r, obs_r: Dict[str, Any]) -> np.ndarray:
    features, mask = agent_r.build_action_features(obs_r)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    return logits


def _candidate_args_from_cfg(cfg: Dict[str, Any]):
    """Build a minimal argparse-like namespace for candidate selection."""
    from types import SimpleNamespace
    return SimpleNamespace(
        candidate_mode=cfg["candidate_mode"],
        ppo_top_k=cfg["ppo_top_k"],
        max_candidates=cfg["max_candidates"],
        num_random_candidates=cfg["num_random_candidates"],
        min_candidates=cfg["min_candidates"],
        candidate_seed=cfg["candidate_seed"],
        ensure_ksp_action=cfg.get("ensure_ksp_action", False),
    )


def _execute_fixed_r(env, req, split_id: int, server_id: int, r_action_idx: int, obs_r: Dict[str, Any]):
    r_action = decode_agent_r_action(int(r_action_idx), len(obs_r["mod_names"]), env.max_blocks)
    _, _, _, info = env.step((split_id, server_id), r_action)
    return info


def _rollout_future(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c,
    agent_r,
    num_servers: int,
    gamma: float = 1.0,
) -> Dict[str, Any]:
    """Simplified future rollout using only PPO-C + PPO-R."""
    return _rollout_future_base(
        env, requests, start_idx, horizon, agent_c, agent_r,
        num_servers=num_servers,
        util_threshold=0.95,
        alpha=0.3,
        future_rollout_policy="ppo_r",
        rng=np.random.RandomState(0),
        gamma=gamma,
    )


def _generate_episode_groups(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    split_name: str,
    seed: int,
    episode_index: int,
    horizon: int,
    gamma: float,
    max_candidates: int,
    cand_args,
    return_coefs: Dict[str, float],
    path_penalty_coef: float,
    fs_penalty_coef: float,
    viability_phi_coef: float,
) -> List[Dict[str, Any]]:
    """Generate canonical paired-feature groups for one episode."""
    groups: List[Dict[str, Any]] = []
    env.reset(requests)
    num_servers = len(env.mec.servers)

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)

        # C-side observation with K=5.
        env.k = DEFAULT_K_PATHS_C
        env.path_sort_strategy = DEFAULT_PATH_SORT
        env.block_sort_strategy = DEFAULT_BLOCK_SORT
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), num_servers)

        # R-side observation with K=50.
        env.k = DEFAULT_K_PATHS_R
        env.path_sort_strategy = DEFAULT_PATH_SORT
        env.block_sort_strategy = DEFAULT_BLOCK_SORT
        obs_r = build_agent_r_observation(env, req, split_id, server_id)

        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        ppo_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        if len(legal) >= 2:
            snapshot = _snapshot_before_r_decision(env, req.req_id)
            candidate_actions = _select_candidate_actions(
                env, agent_r, obs_r, r_features, legal, env.max_blocks, cand_args,
            )
            if len(candidate_actions) > max_candidates:
                candidate_actions = candidate_actions[:max_candidates]
            if len(candidate_actions) >= 2:
                # Pre-compute normalization stats over legal actions.
                legal_path_km = np.array([float(r_features[int(a)][0]) for a in legal])
                legal_required_fs = np.array([float(r_features[int(a)][7]) for a in legal])
                path_mean = legal_path_km.mean()
                path_std = legal_path_km.std() + 1e-6
                fs_mean = legal_required_fs.mean()
                fs_std = legal_required_fs.std() + 1e-6

                next_req = requests[request_index + 1] if request_index + 1 < len(requests) else None
                actual_horizon = max(min(horizon, len(requests) - request_index - 1), 0)

                base_future_rng = np.random.RandomState(seed + episode_index * 100000 + request_index + 12345)
                base_rng_state = base_future_rng.get_state()

                features_v1 = []
                features_post = []
                returns = []
                action_ids = []
                current_successes = []
                future_blocked_counts = []
                future_nsb_counts = []
                future_overload_counts = []
                future_alloc_failed_counts = []
                future_other_counts = []
                delay_sums = []
                fs_sums = []
                phi_after_values = []

                for r_action_idx in candidate_actions:
                    branch = copy.deepcopy(snapshot)
                    info = _execute_fixed_r(branch, req, split_id, server_id, int(r_action_idx), obs_r)
                    phi_after = (
                        _compute_phi_after_obs_c_after(branch, next_req, _dummy_args(return_coefs, viability_phi_coef, path_penalty_coef, fs_penalty_coef))
                        if viability_phi_coef != 0.0 else 0.0
                    )
                    candidate_rng = np.random.RandomState(0)
                    candidate_rng.set_state(base_rng_state)
                    future = _rollout_future(
                        copy.deepcopy(branch), requests, request_index + 1, actual_horizon,
                        agent_c, agent_r, num_servers, gamma=gamma,
                    )
                    path_km_norm = (float(r_features[int(r_action_idx)][0]) - path_mean) / path_std
                    required_fs_norm = (float(r_features[int(r_action_idx)][7]) - fs_mean) / fs_std

                    # Build v1 return exactly as in original generator.
                    ret = _compute_return(
                        info, future,
                        _dummy_args(return_coefs, viability_phi_coef, path_penalty_coef, fs_penalty_coef),
                        path_km_norm, required_fs_norm,
                    )
                    if viability_phi_coef != 0.0:
                        ret += viability_phi_coef * phi_after

                    fv1 = _r_feature_vector(
                        env, req, obs_c, obs_r, r_features, int(r_action_idx), split_id, server_id,
                        feature_names=FEATURE_NAMES,
                    )
                    fpost = build_poststate_v1_feature(
                        env, req, obs_c, obs_r, r_features, int(r_action_idx), split_id, server_id,
                        feature_names=POSTSTATE_V1_FEATURE_NAMES,
                    )

                    features_v1.append(fv1)
                    features_post.append(fpost)
                    returns.append(ret)
                    action_ids.append(int(r_action_idx))
                    current_successes.append(bool(info.get("success", False)))
                    future_blocked_counts.append(int(future.get("blocked", 0)))
                    future_nsb_counts.append(int(future.get("no_suitable_block", 0)))
                    future_overload_counts.append(int(future.get("server_overload", 0)))
                    future_alloc_failed_counts.append(int(future.get("allocation_failed", 0)))
                    future_other_counts.append(int(future.get("other", 0)))
                    delay_sums.append(float(future.get("delay_sum", 0.0)))
                    fs_sums.append(float(future.get("fs_sum", 0.0)))
                    phi_after_values.append(phi_after)

                features_v1 = np.stack(features_v1).astype(np.float32)
                features_post = np.stack(features_post).astype(np.float32)
                returns = np.asarray(returns, dtype=np.float32)
                action_ids = np.asarray(action_ids, dtype=np.int64)
                mask = np.ones(len(action_ids), dtype=bool)
                ppo_action_index = int(np.where(action_ids == int(ppo_r_idx))[0][0]) if int(ppo_r_idx) in action_ids else -1

                groups.append({
                    "group_id": _group_id(split_name, seed, episode_index, request_index),
                    "seed": np.int64(seed),
                    "episode": np.int64(episode_index),
                    "request_index": np.int64(request_index),
                    "split": split_name,
                    "state_hash": _state_hash(env, req, split_id, server_id),
                    "fixed_c_action": np.int64(c_idx),
                    "split_id": np.int64(split_id),
                    "server_id": np.int64(server_id),
                    "legal_count": np.int64(len(legal)),
                    "features_v1": features_v1,
                    "features_poststate_v1": features_post,
                    "returns": returns,
                    "action_ids": action_ids,
                    "mask": mask,
                    "ppo_action_index": np.int64(ppo_action_index),
                    "ppo_action_id": np.int64(ppo_r_idx),
                    "planner_action_id": np.int64(action_ids[int(np.argmax(returns))]),
                    "current_successes": np.asarray(current_successes, dtype=bool),
                    "future_blocked_counts": np.asarray(future_blocked_counts, dtype=np.int64),
                    "future_nsb_counts": np.asarray(future_nsb_counts, dtype=np.int64),
                    "future_server_overload_counts": np.asarray(future_overload_counts, dtype=np.int64),
                    "future_allocation_failed_counts": np.asarray(future_alloc_failed_counts, dtype=np.int64),
                    "future_other_counts": np.asarray(future_other_counts, dtype=np.int64),
                    "delay_sums": np.asarray(delay_sums, dtype=np.float32),
                    "fs_sums": np.asarray(fs_sums, dtype=np.float32),
                    "phi_after": np.asarray(phi_after_values, dtype=np.float32),
                })

        # Advance real trajectory with PPO-R.
        deployed_action = decode_agent_r_action(int(ppo_r_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), deployed_action)

    return groups


def _dummy_args(return_coefs: Dict[str, float], viability_phi_coef: float, path_penalty_coef: float = 0.0, fs_penalty_coef: float = 0.0):
    """Minimal argparse-like namespace for _compute_return / phi_after."""
    class Args:
        pass
    a = Args()
    for k, v in return_coefs.items():
        setattr(a, k, v)
    a.return_mode = "v1"
    a.path_penalty_coef = path_penalty_coef
    a.fs_penalty_coef = fs_penalty_coef
    a.viability_phi_coef = viability_phi_coef
    a.viability_alpha_kc = 1.0
    a.viability_beta_kr = 1.0
    a.modulation_profile = "default"
    return a


def _concat_groups(group_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Pad and stack groups into split-level arrays."""
    if not group_list:
        return {}
    max_c = max(g["features_v1"].shape[0] for g in group_list)
    n = len(group_list)

    def _pad(name: str, dtype: np.dtype, shape_tail: Tuple[int, ...] = ()) -> np.ndarray:
        out = np.zeros((n, max_c) + shape_tail, dtype=dtype)
        for i, g in enumerate(group_list):
            arr = g[name]
            out[i, :len(arr)] = arr
        return out

    arrays = {
        "group_id": np.asarray([g["group_id"] for g in group_list], dtype=np.int64),
        "seed": np.asarray([g["seed"] for g in group_list], dtype=np.int64),
        "episode": np.asarray([g["episode"] for g in group_list], dtype=np.int64),
        "request_index": np.asarray([g["request_index"] for g in group_list], dtype=np.int64),
        "split": np.asarray([g["split"] for g in group_list], dtype=object),
        "state_hash": np.asarray([g["state_hash"] for g in group_list], dtype=object),
        "fixed_c_action": np.asarray([g["fixed_c_action"] for g in group_list], dtype=np.int64),
        "split_id": np.asarray([g["split_id"] for g in group_list], dtype=np.int64),
        "server_id": np.asarray([g["server_id"] for g in group_list], dtype=np.int64),
        "legal_count": np.asarray([g["legal_count"] for g in group_list], dtype=np.int64),
        "features_v1": _pad("features_v1", np.float32, (len(FEATURE_NAMES),)),
        "features_poststate_v1": _pad("features_poststate_v1", np.float32, (len(POSTSTATE_V1_FEATURE_NAMES),)),
        "returns": _pad("returns", np.float32),
        "action_ids": _pad("action_ids", np.int64),
        "mask": _pad("mask", bool),
        "ppo_action_index": np.asarray([g["ppo_action_index"] for g in group_list], dtype=np.int64),
        "ppo_action_id": np.asarray([g["ppo_action_id"] for g in group_list], dtype=np.int64),
        "planner_action_id": np.asarray([g["planner_action_id"] for g in group_list], dtype=np.int64),
        "current_successes": _pad("current_successes", bool),
        "future_blocked_counts": _pad("future_blocked_counts", np.int64),
        "future_nsb_counts": _pad("future_nsb_counts", np.int64),
        "future_server_overload_counts": _pad("future_server_overload_counts", np.int64),
        "future_allocation_failed_counts": _pad("future_allocation_failed_counts", np.int64),
        "future_other_counts": _pad("future_other_counts", np.int64),
        "delay_sums": _pad("delay_sums", np.float32),
        "fs_sums": _pad("fs_sums", np.float32),
        "phi_after": _pad("phi_after", np.float32),
    }
    return arrays


def _shard_worker(shard: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    """Worker function for the parallel launcher."""
    import resource
    cfg = shard["config"]
    mod_reg = ModulationRegistry.from_profile(cfg["modulation_profile"])
    agent_c = _load_ppo_c(cfg["agent_c_checkpoint"], cfg["device"])
    agent_r = _load_ppo_r(cfg["agent_r_checkpoint"], mod_reg, cfg["device"])

    env = make_env(
        cfg["topology"], cfg["num_slots"], cfg["num_servers"], shard["seed"],
        modulation_profile=cfg["modulation_profile"],
        max_blocks=cfg["max_blocks"],
        block_sort_strategy=DEFAULT_BLOCK_SORT,
        path_sort_strategy=DEFAULT_PATH_SORT,
        k=DEFAULT_K_PATHS_R,
    )
    rng = np.random.RandomState(shard["seed"])
    src = int(rng.randint(0, env.net.NUM_NODES))
    total_requests = cfg["requests_per_episode"] + cfg.get("warmup_requests", 0)
    requests = generate_requests(
        env, rng, src, total_requests,
        cfg["arrival_interval"], cfg["holding_min"], cfg["holding_max"],
        cfg["deadline_min"], cfg["deadline_max"], cfg["size_min_mb"], cfg["size_max_mb"],
        cfg["edge_cost_min"], cfg["edge_cost_max"], cfg["num_splits"], cfg["split_profile"],
    )

    groups = _generate_episode_groups(
        env, requests, agent_c, agent_r,
        split_name=shard["split"],
        seed=shard["seed"],
        episode_index=shard["episode"],
        horizon=cfg["horizon"],
        gamma=cfg["gamma"],
        max_candidates=cfg["max_candidates"],
        cand_args=_candidate_args_from_cfg(cfg),
        return_coefs=cfg["return_coefs"],
        path_penalty_coef=cfg["path_penalty_coef"],
        fs_penalty_coef=cfg["fs_penalty_coef"],
        viability_phi_coef=cfg["viability_phi_coef"],
    )
    arrays = _concat_groups(groups)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "arrays": arrays,
        "groups": len(groups),
        "peak_rss_mb": peak_rss,
    }


def _make_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "agent_c_checkpoint": args.agent_c_checkpoint,
        "agent_r_checkpoint": args.agent_r_checkpoint,
        "topology": args.topology,
        "num_slots": args.num_slots,
        "num_servers": args.num_servers,
        "modulation_profile": args.modulation_profile,
        "max_blocks": args.max_blocks,
        "split_profile": args.split_profile,
        "num_splits": args.num_splits,
        "arrival_interval": args.arrival_interval,
        "holding_min": args.holding_min,
        "holding_max": args.holding_max,
        "deadline_min": args.deadline_min,
        "deadline_max": args.deadline_max,
        "size_min_mb": args.size_min_mb,
        "size_max_mb": args.size_max_mb,
        "edge_cost_min": args.edge_cost_min,
        "edge_cost_max": args.edge_cost_max,
        "horizon": args.horizon,
        "gamma": args.gamma,
        "max_candidates": args.max_candidates,
        "ppo_top_k": args.ppo_top_k,
        "candidate_mode": args.candidate_mode,
        "num_random_candidates": args.num_random_candidates,
        "min_candidates": args.min_candidates,
        "candidate_seed": args.candidate_seed,
        "ensure_ksp_action": args.ensure_ksp_action,
        "return_coefs": {
            "return_current_block_coef": args.return_current_block_coef,
            "return_future_block_coef": args.return_future_block_coef,
            "return_future_nsb_coef": args.return_future_nsb_coef,
            "return_future_server_overload_coef": args.return_future_server_overload_coef,
            "return_delay_coef": args.return_delay_coef,
            "return_fs_coef": args.return_fs_coef,
        },
        "path_penalty_coef": args.path_penalty_coef,
        "fs_penalty_coef": args.fs_penalty_coef,
        "viability_phi_coef": args.viability_phi_coef,
        "device": args.device,
        "requests_per_episode": args.requests_per_episode,
        "warmup_requests": args.warmup_requests,
        "schema": "paired_feature_v1",
        "version": "1.0",
    }


def _make_shards(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg = _make_config(args)
    shards = []
    for split_name, seeds in args.splits.items():
        n_eps = getattr(args, f"{split_name}_episodes", args.episodes)
        for seed in seeds:
            for episode in range(n_eps):
                shards.append({
                    "shard_id": f"{split_name}_s{seed}_e{episode}",
                    "split": split_name,
                    "seed": seed,
                    "episode": episode,
                    "config": cfg,
                })
    return shards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="sa_hmarl/datasets/v135_paired_feature_ablation")
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max_candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--ppo_top_k", type=int, default=DEFAULT_PPO_TOP_K)
    parser.add_argument("--candidate_mode", default="v1",
                        choices=["v1", "all_legal", "ppo_r_topk_only", "legalctx48", "highest_se_path_block", "top2_se_path_block", "poststate_anchor48"])
    parser.add_argument("--num_random_candidates", type=int, default=5)
    parser.add_argument("--min_candidates", type=int, default=15)
    parser.add_argument("--candidate_seed", type=int, default=12345)
    parser.add_argument("--ensure_ksp_action", action="store_true")
    parser.add_argument("--return_current_block_coef", type=float, default=1.0)
    parser.add_argument("--return_future_block_coef", type=float, default=1.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=1.0)
    parser.add_argument("--return_future_server_overload_coef", type=float, default=0.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.0)
    parser.add_argument("--return_fs_coef", type=float, default=0.0)
    parser.add_argument("--path_penalty_coef", type=float, default=0.0)
    parser.add_argument("--fs_penalty_coef", type=float, default=0.0)
    parser.add_argument("--viability_phi_coef", type=float, default=0.0)
    parser.add_argument("--requests_per_episode", type=int, default=1000)
    parser.add_argument("--warmup_requests", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--train_episodes", type=int, default=None)
    parser.add_argument("--val_episodes", type=int, default=None)
    parser.add_argument("--test_episodes", type=int, default=None)
    parser.add_argument("--train_seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    parser.add_argument("--val_seeds", type=int, nargs="+", default=[2001])
    parser.add_argument("--test_seeds", type=int, nargs="+", default=[3001])
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.splits = {
        "train": args.train_seeds,
        "val": args.val_seeds,
        "test": args.test_seeds,
    }
    for split in ("train", "val", "test"):
        if getattr(args, f"{split}_episodes") is None:
            setattr(args, f"{split}_episodes", args.episodes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(args)
    shards = _make_shards(args)

    from v135_parallel_launcher import ShardedParallelLauncher, _atomic_write_json, _config_hash

    launcher = ShardedParallelLauncher(
        output_dir=output_dir,
        config=config,
        worker_fn=_shard_worker,
        max_workers=args.max_workers,
        shard_subdir="shards",
    )
    results = launcher.run(shards)

    # Merge shards deterministically.
    print("[dataset] Merging shards...", flush=True)
    split_groups: Dict[str, List[Dict[str, np.ndarray]]] = {"train": [], "val": [], "test": []}
    meta_map = {}
    for r in results:
        if r.status != "ok":
            continue
        meta_path = output_dir / "shards" / f"{r.shard_id}.metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_map[r.shard_id] = meta
        split = r.shard_id.split("_s")[0]
        npz = np.load(str(output_dir / "shards" / f"{r.shard_id}.npz"), allow_pickle=True)
        split_groups[split].append(npz)

    for split, npz_list in split_groups.items():
        if not npz_list:
            continue
        # Sort by group_id arrays to get deterministic order.
        order = np.argsort(npz_list[0]["group_id"], kind="stable")
        merged = {}
        for key in npz_list[0].files:
            parts = [npz[key] for npz in npz_list]
            merged[key] = np.concatenate(parts, axis=0)
        # Deterministic sort by group_id.
        order = np.argsort(merged["group_id"], kind="stable")
        for key in merged:
            merged[key] = merged[key][order]
        np.savez_compressed(output_dir / f"{split}.npz", **merged)

    metadata = {
        "config": config,
        "config_hash": _config_hash(config),
        "feature_names_v1": list(FEATURE_NAMES),
        "feature_dim_v1": len(FEATURE_NAMES),
        "feature_names_poststate_v1": list(POSTSTATE_V1_FEATURE_NAMES),
        "feature_dim_poststate_v1": len(POSTSTATE_V1_FEATURE_NAMES),
        "splits": {s: {"seeds": args.splits[s], "episodes": getattr(args, f"{s}_episodes")} for s in args.splits},
        "shards": meta_map,
    }
    _atomic_write_json(output_dir / "metadata.json", metadata)
    print(f"[dataset] Saved merged dataset to {output_dir}")


if __name__ == "__main__":
    main()
