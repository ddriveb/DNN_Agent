"""Generate supervised R-side post-decision value datasets.

For each request, Agent-C is frozen and selects a split/server.  We then
either enumerate every raw-mask-legal Agent-R action (the default) or only the
top-K actions proposed by PPO-R, execute each candidate in a copied environment,
roll out H future requests with the frozen C+R pair, and assign a label that
rewards current success, low future blocking, and healthy spectrum structure.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_c_action_horizon_oracle import (
    _execute_c_with_frozen_r,
    _snapshot_before_c_decision,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _rollout_future,
    _select_c_action_from_obs,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


SPLIT_SEEDS = {
    "train": (42, 123, 456),
    "val": (789, 101112),
    "test": (2024, 2025),
}


def _ppo_r_topk_actions(agent_r, obs_r: Dict[str, Any], top_k: int) -> List[int]:
    """Return the top-k legal flat action ids ranked by PPO-R logits."""
    features, mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(mask, dtype=bool))
    if legal.size == 0:
        return []
    with torch.no_grad():
        x = (
            torch.as_tensor(features, dtype=torch.float32, device=agent_r.device)
            .unsqueeze(0)
        )
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    top_k = min(top_k, legal.size)
    top_local = np.argsort(-logits[legal], kind="stable")[:top_k]
    return [int(legal[i]) for i in top_local]

R_BASE_FEATURE_NAMES = [
    "path_length_km", "hop_count", "lfb", "free_ratio", "frag_index",
    "spectral_efficiency", "reach_km", "required_fs", "block_size",
    "block_waste", "path_mod_feasible",
]
R_CONTEXT_FEATURE_NAMES = [
    "split_norm", "server_norm", "deadline_norm", "holding_norm",
    "intermediate_size_norm", "server_utilization", "selected_valid_r_ratio",
]
R_FIELD_FEATURE_NAMES = [
    "k_c_valid_ratio", "k_r_total_ratio", "phi_spec_norm",
    "raw_r_valid_ratio",
]
R_ACTION_FEATURE_NAMES = ["path_idx_norm", "mod_idx_norm", "block_idx_norm"]
FEATURE_NAMES = (
    R_BASE_FEATURE_NAMES + R_CONTEXT_FEATURE_NAMES
    + R_FIELD_FEATURE_NAMES + R_ACTION_FEATURE_NAMES
)
STRUCTURAL_SCALAR_FEATURE_NAMES = (
    "block_start_norm",
    "block_end_norm",
    "block_center_norm",
)


def structured_feature_names(env) -> List[str]:
    """Return v1.2 feature names plus topology-specific structural fields."""
    edge_names = [
        f"edge_{min(int(u), int(v))}_{max(int(u), int(v))}"
        for u, v in sorted(env.net.G.edges())
    ]
    return list(FEATURE_NAMES) + list(STRUCTURAL_SCALAR_FEATURE_NAMES) + edge_names


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


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _r_feature_vector(
    env, req, obs_c: Dict[str, Any], obs_r: Dict[str, Any],
    r_features: np.ndarray, r_action_idx: int, split_id: int, server_id: int,
    feature_names: Iterable[str] = None,
) -> np.ndarray:
    num_paths = max(len(obs_r["candidate_paths"]), 1)
    num_mods = max(len(obs_r["mod_names"]), 1)
    max_blocks = max(int(env.max_blocks), 1)
    path_idx = r_action_idx // (num_mods * max_blocks)
    rem = r_action_idx % (num_mods * max_blocks)
    mod_idx = rem // max_blocks
    block_idx = rem % max_blocks

    raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    k_c, k_r = _compute_spectrum_field(obs_c, 0.95)
    num_c = max(len(obs_c["candidate_features"]), 1)
    max_r_total = max(num_c * num_paths * num_mods * max_blocks, 1)
    selected_valid = int(obs_c["feasible_counts"][split_id][server_id])

    c_ctx = obs_r.get("c_context", {})
    context = np.asarray([
        float(split_id) / max(len(req.splits) - 1, 1),
        float(server_id) / max(len(env.mec.servers) - 1, 1),
        float(req.deadline_ms) / 100.0,
        float(req.holding_time) / 10.0,
        float(c_ctx.get("intermediate_size_mb", 0.0)) / 100.0,
        float(obs_c["server_utilizations"][server_id]),
        selected_valid / max(num_paths * num_mods * max_blocks, 1),
    ], dtype=np.float32)
    field = np.asarray([
        k_c / num_c,
        k_r / max_r_total,
        _phi_spec(k_c, k_r, 0.3) / 8.0,
        int(raw_r_mask.sum()) / max(len(raw_r_mask), 1),
    ], dtype=np.float32)
    action = np.asarray([
        path_idx / max(num_paths - 1, 1),
        mod_idx / max(num_mods - 1, 1),
        block_idx / max(max_blocks - 1, 1),
    ], dtype=np.float32)
    base_vector = np.concatenate([
        np.asarray(r_features[r_action_idx], dtype=np.float32),
        context, field, action,
    ]).astype(np.float32)

    if feature_names is None:
        feature_names = FEATURE_NAMES
    feature_names = list(feature_names)

    if tuple(feature_names) == tuple(FEATURE_NAMES):
        vector = base_vector
    else:
        feature_values = {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, base_vector.tolist())
        }

        blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        if block_idx < len(blocks):
            block_start = float(blocks[block_idx][0])
            block_size = float(blocks[block_idx][1])
            block_end = block_start + block_size
        else:
            block_start = block_size = block_end = 0.0
        num_slots = max(float(obs_r.get("num_slots", env.net.num_slots)), 1.0)
        feature_values.update({
            "block_start_norm": block_start / num_slots,
            "block_end_norm": block_end / num_slots,
            "block_center_norm": (block_start + 0.5 * block_size) / num_slots,
        })

        path = obs_r["candidate_paths"][path_idx] if path_idx < len(obs_r["candidate_paths"]) else []
        path_edges = {
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in zip(path[:-1], path[1:])
        }
        for u, v in sorted(env.net.G.edges()):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            feature_values[f"edge_{key[0]}_{key[1]}"] = 1.0 if key in path_edges else 0.0

        vector = np.asarray([feature_values.get(name, 0.0) for name in feature_names], dtype=np.float32)

    if vector.shape != (len(feature_names),) or not np.all(np.isfinite(vector)):
        raise ValueError("Invalid R post-decision feature vector")
    return vector


def _execute_fixed_r(env, req, split_id: int, server_id: int, r_action_idx: int, obs_r: Dict[str, Any]):
    r_action = decode_agent_r_action(
        int(r_action_idx), len(obs_r["mod_names"]), env.max_blocks
    )
    _, _, _, info = env.step((split_id, server_id), r_action)
    return info


def _label(
    phi_after: float, info: Dict[str, Any], req, future: Dict[str, Any],
    actual_horizon: int, args: argparse.Namespace,
) -> float:
    denom = max(int(actual_horizon), 1)
    current_block = 0.0 if bool(info.get("success", False)) else 1.0
    delay_norm = 0.0
    if bool(info.get("success", False)):
        delay_norm = float(info.get("delay_ms", req.deadline_ms)) / max(float(req.deadline_ms), 1.0)
    return float(
        args.label_phi_coef * phi_after
        - args.label_current_block_coef * current_block
        - args.label_future_block_coef * float(future.get("blocked", 0)) / denom
        - args.label_future_nsb_coef * float(future.get("no_suitable_block", 0)) / denom
        - args.label_delay_coef * delay_norm
    )


def _generate_episode(
    env, requests: List[Any], agent_c, agent_r, split_name: str, seed: int,
    episode_index: int, args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    rows = {key: [] for key in [
        "features", "labels", "group_ids", "r_actions", "seeds",
        "episode_ids", "request_indices", "current_success",
        "future_blocked_count", "selected_by_ppo", "label_rank_in_group",
    ]}
    groups = {key: [] for key in [
        "all_group_ids", "group_r_mask_empty", "group_seeds",
        "group_episode_ids", "group_request_indices",
    ]}
    env.reset(requests)

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_action_idx, raw_c_mask, _ = _select_c_action_from_obs(
            agent_c, obs_c, env.net.num_slots
        )
        split_id, server_id = decode_agent_c_action(c_action_idx, len(env.mec.servers))
        gid = _group_id(split_name, seed, episode_index, request_index)
        groups["all_group_ids"].append(gid)
        groups["group_seeds"].append(seed)
        groups["group_episode_ids"].append(episode_index)
        groups["group_request_indices"].append(request_index)

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        groups["group_r_mask_empty"].append(len(legal) == 0)

        if args.candidate_source == "ppo_r_topk":
            candidates = _ppo_r_topk_actions(agent_r, obs_r, args.ppo_r_top_k)
        else:
            candidates = legal

        if candidates:
            snapshot = _snapshot_before_c_decision(env, req.req_id)
            ppo_r_action = agent_r.select_action(obs_r, deterministic=True)
            labels = []
            records = []
            for r_action_idx in candidates:
                branch = copy.deepcopy(snapshot)
                info = _execute_fixed_r(branch, req, split_id, server_id, int(r_action_idx), obs_r)
                actual_horizon = max(min(args.label_horizon, len(requests) - request_index - 1), 0)
                future = _rollout_future(
                    copy.deepcopy(branch), requests, request_index + 1, actual_horizon,
                    agent_c, agent_r, len(env.mec.servers), 0.95, args.alpha,
                )
                next_req = requests[min(request_index + 1, len(requests) - 1)]
                probe = copy.deepcopy(branch)
                probe.advance_time(next_req.arrival_time)
                obs_next = build_agent_c_observation(probe, next_req)
                k_c_after, k_r_after = _compute_spectrum_field(obs_next, 0.95)
                phi_after = _phi_spec(k_c_after, k_r_after, args.alpha)
                labels.append(_label(phi_after, info, req, future, actual_horizon, args))
                records.append((r_action_idx, info, future))
            label_array = np.asarray(labels, dtype=np.float32)
            ranks = _rank_desc(label_array)
            for local_idx, (r_action_idx, info, future) in enumerate(records):
                rows["features"].append(_r_feature_vector(
                    env, req, obs_c, obs_r, r_features, int(r_action_idx), split_id, server_id
                ))
                rows["labels"].append(float(label_array[local_idx]))
                rows["group_ids"].append(gid)
                rows["r_actions"].append(int(r_action_idx))
                rows["seeds"].append(seed)
                rows["episode_ids"].append(episode_index)
                rows["request_indices"].append(request_index)
                rows["current_success"].append(bool(info.get("success", False)))
                rows["future_blocked_count"].append(int(future.get("blocked", 0)))
                rows["selected_by_ppo"].append(int(r_action_idx) == int(ppo_r_action))
                rows["label_rank_in_group"].append(int(ranks[local_idx]))

        # Advance deployed trajectory with frozen C+R.
        _execute_c_with_frozen_r(env, req, int(c_action_idx), agent_r, len(env.mec.servers))

    feature_array = (
        np.stack(rows["features"]).astype(np.float32)
        if rows["features"] else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    return {
        "features": feature_array,
        "labels": np.asarray(rows["labels"], dtype=np.float32),
        "group_ids": np.asarray(rows["group_ids"], dtype=np.int64),
        "r_actions": np.asarray(rows["r_actions"], dtype=np.int64),
        "seeds": np.asarray(rows["seeds"], dtype=np.int64),
        "episode_ids": np.asarray(rows["episode_ids"], dtype=np.int64),
        "request_indices": np.asarray(rows["request_indices"], dtype=np.int64),
        "current_success": np.asarray(rows["current_success"], dtype=bool),
        "future_blocked_count": np.asarray(rows["future_blocked_count"], dtype=np.int64),
        "selected_by_ppo": np.asarray(rows["selected_by_ppo"], dtype=bool),
        "label_rank_in_group": np.asarray(rows["label_rank_in_group"], dtype=np.int64),
        "all_group_ids": np.asarray(groups["all_group_ids"], dtype=np.int64),
        "group_r_mask_empty": np.asarray(groups["group_r_mask_empty"], dtype=bool),
        "group_seeds": np.asarray(groups["group_seeds"], dtype=np.int64),
        "group_episode_ids": np.asarray(groups["group_episode_ids"], dtype=np.int64),
        "group_request_indices": np.asarray(groups["group_request_indices"], dtype=np.int64),
    }


def _concat_shards(paths: Iterable[Path]) -> Dict[str, np.ndarray]:
    loaded = [dict(np.load(path, allow_pickle=False)) for path in sorted(paths)]
    if not loaded:
        raise ValueError("No dataset shards found")
    return {
        key: np.concatenate([shard[key] for shard in loaded], axis=0)
        for key in loaded[0]
    }


def _dataset_diagnostics(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    labels = data["labels"]
    groups = np.unique(data["group_ids"])
    ranges, ppo_matches = [], []
    nonzero = 0
    multi = 0
    for gid in groups:
        idx = np.flatnonzero(data["group_ids"] == gid)
        if len(idx) > 1:
            multi += 1
            value_range = float(labels[idx].max() - labels[idx].min())
            ranges.append(value_range)
            nonzero += value_range > 1e-8
        chosen_ppo = idx[data["selected_by_ppo"][idx]]
        best = idx[data["label_rank_in_group"][idx] == 1]
        if len(chosen_ppo) and len(best):
            ppo_matches.append(int(chosen_ppo[0] == best[0]))
    return {
        "groups": int(len(data["all_group_ids"])),
        "candidate_groups": int(len(groups)),
        "candidates": int(len(labels)),
        "r_mask_empty_rate": float(np.mean(data["group_r_mask_empty"])),
        "label_mean": float(np.mean(labels)) if len(labels) else 0.0,
        "label_std": float(np.std(labels)) if len(labels) else 0.0,
        "multi_action_groups": multi,
        "nonzero_label_range_rate": float(nonzero / max(multi, 1)),
        "label_range_mean": float(np.mean(ranges)) if ranges else 0.0,
        "ppo_top1_agreement": float(np.mean(ppo_matches)) if ppo_matches else 0.0,
        "feature_finite_rate": float(np.mean(np.isfinite(data["features"]))) if data["features"].size else 1.0,
    }


def _write_report(path: Path, metadata: Dict[str, Any]) -> None:
    lines = [
        "# R-Side Post-Decision Dataset Generation", "",
        f"- Feature dimension: {metadata['feature_dim']}",
        f"- Label: `{metadata['label_name']}`", "",
        "| Split | Groups | Candidates | R-empty | Label std | Nonzero-range | PPO top-1 agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, diag in metadata["diagnostics"].items():
        lines.append(
            f"| {split} | {diag['groups']} | {diag['candidates']} | "
            f"{diag['r_mask_empty_rate']:.2%} | {diag['label_std']:.4f} | "
            f"{diag['nonzero_label_range_rate']:.2%} | {diag['ppo_top1_agreement']:.2%} |"
        )
    lines += ["", f"**Verdict: {metadata['verdict']}**"]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    out = Path(args.output_dir)
    shard_dir = out / "shards"
    out.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    r_before = {key: value.clone() for key, value in agent_r.policy_net.state_dict().items()}

    split_episodes = {
        split: getattr(args, f"{split}_episodes", None) or args.episodes
        for split in splits
    }
    for split in splits:
        num_episodes = split_episodes[split]
        for seed in SPLIT_SEEDS[split]:
            proto = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                args.slot_bw_hz, args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            rng = np.random.RandomState(seed)
            for episode in range(num_episodes):
                shard = shard_dir / f"{split}_s{seed}_e{episode:03d}.npz"
                if args.resume and shard.exists():
                    continue
                src = int(rng.randint(0, proto.net.NUM_NODES))
                requests = generate_requests(
                    proto, rng, src, args.requests_per_episode,
                    args.arrival_interval, args.holding_min, args.holding_max,
                    args.deadline_min, args.deadline_max, args.size_min_mb,
                    args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                    args.num_splits, args.split_profile, args.traffic_mode,
                    args.regime_stay_prob,
                )
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    args.slot_bw_hz, args.guard_band_fs,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                t0 = time.time()
                arrays = _generate_episode(env, requests, agent_c, agent_r, split, seed, episode, args)
                _atomic_save_npz(shard, **arrays)
                _atomic_write_json(shard.with_suffix(".timing.json"), {
                    "split": split, "seed": seed, "episode": episode,
                    "groups": int(len(arrays["all_group_ids"])),
                    "candidates": int(len(arrays["labels"])),
                    "generation_seconds": time.time() - t0,
                })
                print(
                    f"[split={split}][seed={seed}][episode={episode + 1}/{num_episodes}] "
                    f"groups={len(arrays['all_group_ids'])} candidates={len(arrays['labels'])} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    diagnostics = {}
    split_data = {}
    for split in splits:
        paths = [
            shard_dir / f"{split}_s{seed}_e{episode:03d}.npz"
            for seed in SPLIT_SEEDS[split]
            for episode in range(split_episodes[split])
        ]
        data = _concat_shards(paths)
        split_data[split] = data
        diagnostics[split] = _dataset_diagnostics(data)
        _atomic_save_npz(out / f"{split}.npz", **data)

    quality_ok = all(
        diag["feature_finite_rate"] == 1.0
        and diag["label_std"] > 1e-8
        and diag["nonzero_label_range_rate"] >= 0.10
        for diag in diagnostics.values()
    )
    r_unchanged = all(
        torch.equal(value, r_before[key])
        for key, value in agent_r.policy_net.state_dict().items()
    )
    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "label_name": (
            f"r_h{args.label_horizon}_phi{args.label_phi_coef:g}"
            f"_curblk{args.label_current_block_coef:g}"
            f"_futblk{args.label_future_block_coef:g}"
            f"_futnsb{args.label_future_nsb_coef:g}"
            f"_delay{args.label_delay_coef:g}"
        ),
        "diagnostics": diagnostics,
        "label_horizon": args.label_horizon,
        "candidate_source": args.candidate_source,
        "ppo_r_top_k": args.ppo_r_top_k,
        "agent_r_unchanged": r_unchanged,
        "elapsed_seconds": time.time() - started,
        "verdict": (
            "PROCEED_TO_POST_DECISION_TRAINING"
            if quality_ok and r_unchanged else "STOP_AND_AUDIT_DATASET"
        ),
    }
    train_x = split_data["train"]["features"].astype(np.float64)
    metadata["train_feature_mean"] = train_x.mean(axis=0).tolist()
    metadata["train_feature_std"] = np.maximum(train_x.std(axis=0), 1e-6).tolist()
    _atomic_write_json(out / "metadata.json", metadata)
    _write_report(out / "generation_report.md", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--output_dir", default="sa_hmarl/datasets/r_post_decision")
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
    parser.add_argument("--traffic_mode", default="iid")
    parser.add_argument("--regime_stay_prob", type=float, default=0.9)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--label_horizon", type=int, default=3)
    parser.add_argument("--label_phi_coef", type=float, default=0.2)
    parser.add_argument("--label_current_block_coef", type=float, default=2.0)
    parser.add_argument("--label_future_block_coef", type=float, default=2.0)
    parser.add_argument("--label_future_nsb_coef", type=float, default=0.0)
    parser.add_argument("--label_delay_coef", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--candidate_source",
        default="all_legal",
        choices=["all_legal", "ppo_r_topk"],
        help="Which candidate set to use for post-decision labels.",
    )
    parser.add_argument(
        "--ppo_r_top_k",
        type=int,
        default=8,
        help="Number of PPO-R top-K proposals when candidate_source=ppo_r_topk.",
    )
    return parser


if __name__ == "__main__":
    result = generate_dataset(build_parser().parse_args())
    print(result["verdict"])
