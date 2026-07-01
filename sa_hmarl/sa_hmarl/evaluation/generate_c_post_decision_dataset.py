"""Generate supervised C-side post-decision spectrum-potential datasets.

The online features are computed before taking the C action.  The target is
the explicit spectrum-only potential after executing that candidate with the
frozen deterministic PPO-R.  Episode shards make long generation jobs atomic
and resumable.
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

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.evaluation.diagnose_c_action_horizon_oracle import (
    _execute_c_with_frozen_r,
    _snapshot_before_c_decision,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _rollout_future,
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    _demand_aware_potential_spectrum_only,
    _select_ppo_c_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


SPLIT_SEEDS = {
    "train": (42, 123, 456),
    "val": (789, 101112),
    "test": (2024, 2025),
}

BASE_FEATURE_NAMES = [
    "intermediate_size_mb", "local_compute_ms", "edge_compute_ms",
    "server_utilization", "best_fs_estimate", "safe_fs_estimate",
    "feasible_count", "lfb_max", "lfb_mean", "lfb_min", "lfb_p25",
    "frag_mean", "frag_std", "free_mean", "num_feasible_paths",
    "path_diversity", "min_delay_est",
]
R_FEAS_FEATURE_NAMES = [
    "r_valid_action_ratio", "r_valid_action_log_ratio", "r_valid_path_mod_ratio",
    "r_valid_path_ratio", "r_valid_mod_ratio", "r_no_valid_indicator",
    "r_min_required_fs_norm", "r_safe_required_fs_norm",
    "r_best_block_fit_pressure", "r_avg_block_waste",
]
CONTEXT_FEATURE_NAMES = [
    "request_deadline_norm", "request_holding_norm", "request_source_norm",
    "candidate_split_norm", "candidate_server_norm",
]
FIELD_FEATURE_NAMES = [
    "k_c_valid_ratio", "k_r_total_ratio",
    "mf_small_block_ratio", "mf_medium_block_ratio", "mf_large_block_ratio",
    "mf_mean_lfb_norm", "mf_mean_fragmentation", "mf_mean_free_slot_ratio",
    "mf_server_idle_ratio", "mf_server_normal_ratio", "mf_server_busy_ratio",
    "mf_server_overload_risk_ratio", "mf_mean_available_compute_ratio",
]
IMPACT_FEATURE_NAMES = [
    "candidate_feasible_count_percentile", "candidate_best_required_fs_norm",
    "candidate_server_margin_after_norm", "candidate_path_distance_norm",
    "candidate_safe_spectrum_consumption_norm",
]
FEATURE_NAMES = (
    BASE_FEATURE_NAMES + R_FEAS_FEATURE_NAMES + CONTEXT_FEATURE_NAMES
    + FIELD_FEATURE_NAMES + IMPACT_FEATURE_NAMES
)


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


def _should_generate_shard(path: Path, resume: bool) -> bool:
    return not (resume and path.exists())


def _candidate_feature_vector(
    env, req, obs_c: Dict[str, Any], policy_features: np.ndarray,
    candidate_idx: int, split_id: int, server_id: int,
) -> np.ndarray:
    """Build label-free features available at the C decision instant."""
    if policy_features.shape[1] != 27:
        raise ValueError(
            f"Expected r_feasibility Agent-C features with dim 27, got {policy_features.shape[1]}"
        )
    feat = obs_c["candidate_features"][candidate_idx]
    num_actions = max(len(obs_c["candidate_features"]), 1)
    num_slots = max(int(obs_c["num_slots"]), 1)
    counts = np.asarray(obs_c["feasible_counts"], dtype=float).reshape(-1)
    count = float(feat.get("feasible_count", 0.0))
    percentile = float(np.mean(counts <= count)) if len(counts) else 0.0
    max_r_actions = max(
        num_actions * int(obs_c.get("_k_paths", 1))
        * int(feat.get("_mod_formats", 1)) * int(env.max_blocks), 1
    )
    k_c = int(np.asarray(obs_c["agent_c_mask"], dtype=bool).sum())
    k_r = int(counts.sum())

    num_splits = max(len(req.splits), 1)
    num_servers = max(len(env.mec.servers), 1)
    context = np.asarray([
        float(req.deadline_ms) / 100.0,
        float(req.holding_time) / 10.0,
        float(req.src_node) / max(env.net.NUM_NODES - 1, 1),
        float(split_id) / max(num_splits - 1, 1),
        float(server_id) / max(num_servers - 1, 1),
    ], dtype=np.float32)
    field = np.concatenate([
        np.asarray([k_c / num_actions, k_r / max_r_actions], dtype=np.float32),
        np.asarray(obs_c["mu_res_spec"], dtype=np.float32),
        np.asarray(obs_c["mu_res_compute"], dtype=np.float32),
    ])

    server = env.mec.servers[server_id]
    capacity = max(float(server.compute_capacity), 1e-6)
    edge_cost = float(feat.get("edge_compute_cost", 0.0))
    server_margin = (float(server.available_compute) - edge_cost) / capacity
    best_fs = feat.get("best_fs_estimate")
    safe_fs = feat.get("safe_fs_estimate")
    impact = np.asarray([
        percentile,
        float(best_fs if best_fs is not None else num_slots) / num_slots,
        float(np.clip(server_margin, -1.0, 1.0)),
        float(feat.get("server_mean_path_km", 0.0)) / 10000.0,
        float(safe_fs if safe_fs is not None else num_slots) / num_slots,
    ], dtype=np.float32)
    result = np.concatenate([
        np.asarray(policy_features[candidate_idx], dtype=np.float32), context, field, impact
    ]).astype(np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("Invalid post-decision input feature vector")
    return result


def _features_for_post_decision(agent_c, obs_c: Dict[str, Any]) -> np.ndarray:
    """Build the 27-dim r-feasibility prefix expected by post-decision features.

    The rollout policy may be an older delay-aware PPO-C checkpoint trained with
    default 17-dim features.  The supervised post-decision model still uses the
    richer label-free r-feasibility description, so construct that stream
    independently when necessary.
    """
    features, _ = agent_c.build_action_features(obs_c)
    if features.shape[1] == 27:
        return features
    if features.shape[1] != 17:
        raise ValueError(
            "Post-decision dataset expects default 17-dim or r_feasibility "
            f"27-dim C policy features, got {features.shape[1]}"
        )

    class _RFeasFeatureBuilder:
        feature_mode = "r_feasibility"
        ablation = False
        zero_spectrum = False

    r_features, _ = AgentC.build_action_features(_RFeasFeatureBuilder(), obs_c)
    return r_features


def _delay_aware_label(phi_after: float, info: Dict[str, Any], req, delay_coef: float) -> float:
    """Trade post-decision spectrum potential against current successful delay."""
    if delay_coef <= 0:
        return float(phi_after)
    if not info_success(info):
        return float(phi_after) - float(delay_coef)
    delay_norm = float(info.get("delay_ms", req.deadline_ms)) / max(float(req.deadline_ms), 1.0)
    return float(phi_after) - float(delay_coef) * delay_norm


def _horizon_delay_aware_label(
    phi_after: float,
    info: Dict[str, Any],
    req,
    future: Dict[str, Any],
    actual_horizon: int,
    *,
    phi_coef: float,
    current_block_coef: float,
    future_block_coef: float,
    delay_coef: float,
) -> float:
    """Label a post-decision candidate by H-step admission and delay effects.

    The label is intentionally simple and bounded by normalized counts.  It
    keeps the spectrum potential as a weak tie-breaker while making future
    blocking the primary supervised signal.
    """
    denom = max(int(actual_horizon), 1)
    current_block = 0.0 if info_success(info) else 1.0
    future_block_rate = float(future.get("blocked", 0)) / denom

    delay_norm = 0.0
    if info_success(info):
        delay_norm = float(info.get("delay_ms", req.deadline_ms)) / max(float(req.deadline_ms), 1.0)

    return float(
        phi_coef * float(phi_after)
        - current_block_coef * current_block
        - future_block_coef * future_block_rate
        - delay_coef * delay_norm
    )


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _generate_episode(
    env, requests: List[Any], agent_c, agent_r, split_name: str, seed: int,
    episode_index: int, history_window: int, probe_limit: int, alpha: float,
    label_delay_coef: float, label_horizon: int = 0,
    label_phi_coef: float = 1.0, label_current_block_coef: float = 1.0,
    label_future_block_coef: float = 1.0,
) -> Dict[str, np.ndarray]:
    rows: Dict[str, List[Any]] = {key: [] for key in [
        "features", "labels", "group_ids", "candidate_actions", "seeds",
        "episode_ids", "request_indices", "current_success", "explicit_phi_before",
        "explicit_phi_after", "delta_phi", "valid_r_actions", "failure_reason",
        "selected_by_ppo", "selected_by_explicit_potential", "label_rank_in_group",
    ]}
    groups = {key: [] for key in [
        "all_group_ids", "group_raw_mask_empty", "group_seeds", "group_episode_ids",
        "group_request_indices",
    ]}
    env.reset(requests)

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        legal = np.flatnonzero(raw_mask).tolist()
        gid = _group_id(split_name, seed, episode_index, request_index)
        groups["all_group_ids"].append(gid)
        groups["group_raw_mask_empty"].append(len(legal) == 0)
        groups["group_seeds"].append(seed)
        groups["group_episode_ids"].append(episode_index)
        groups["group_request_indices"].append(request_index)

        history = requests[:request_index + 1]
        ppo_action, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
        if legal:
            policy_features = _features_for_post_decision(agent_c, obs_c)
            snapshot = _snapshot_before_c_decision(env, req.req_id)
            phi_before = _demand_aware_potential_spectrum_only(
                snapshot, history, history_window, probe_limit, alpha
            )
            group_records = []
            for candidate_idx in legal:
                split_id, server_id = divmod(int(candidate_idx), len(env.mec.servers))
                branch = copy.deepcopy(snapshot)
                _, info = _execute_c_with_frozen_r(
                    branch, req, int(candidate_idx), agent_r, len(env.mec.servers)
                )
                phi_after = _demand_aware_potential_spectrum_only(
                    branch, history, history_window, probe_limit, alpha
                )
                if phi_before is None or phi_after is None:
                    raise ValueError("Spectrum potential label is unavailable")
                future = None
                actual_horizon = 0
                if label_horizon > 0:
                    actual_horizon = max(
                        min(int(label_horizon), len(requests) - request_index - 1), 0
                    )
                    future = _rollout_future(
                        copy.deepcopy(branch),
                        requests,
                        request_index + 1,
                        actual_horizon,
                        agent_c,
                        agent_r,
                        len(env.mec.servers),
                        0.95,
                        alpha,
                    )
                group_records.append((
                    candidate_idx, split_id, server_id, info, float(phi_after),
                    future, actual_horizon,
                ))

            if label_horizon > 0:
                labels = np.asarray([
                    _horizon_delay_aware_label(
                        record[4], record[3], req, record[5], record[6],
                        phi_coef=label_phi_coef,
                        current_block_coef=label_current_block_coef,
                        future_block_coef=label_future_block_coef,
                        delay_coef=label_delay_coef,
                    )
                    for record in group_records
                ], dtype=np.float32)
            else:
                labels = np.asarray([
                    _delay_aware_label(record[4], record[3], req, label_delay_coef)
                    for record in group_records
                ], dtype=np.float32)
            ranks = _rank_desc(labels)
            successful = [i for i, record in enumerate(group_records) if info_success(record[3])]
            best_success = max(successful, key=lambda i: (labels[i], -group_records[i][0])) if successful else -1
            for local_idx, (
                candidate_idx, split_id, server_id, info, phi_after, future, actual_horizon
            ) in enumerate(group_records):
                rows["features"].append(_candidate_feature_vector(
                    env, req, obs_c, policy_features, candidate_idx, split_id, server_id
                ))
                rows["labels"].append(float(labels[local_idx]))
                rows["group_ids"].append(gid)
                rows["candidate_actions"].append((split_id, server_id))
                rows["seeds"].append(seed)
                rows["episode_ids"].append(episode_index)
                rows["request_indices"].append(request_index)
                rows["current_success"].append(info_success(info))
                rows["explicit_phi_before"].append(phi_before)
                rows["explicit_phi_after"].append(phi_after)
                rows["delta_phi"].append(float(labels[local_idx]) - phi_before)
                rows["valid_r_actions"].append(
                    int(obs_c["feasible_counts"][split_id][server_id])
                )
                rows["failure_reason"].append(str(info.get("reason", "unknown")))
                rows["selected_by_ppo"].append(int(candidate_idx) == int(ppo_action))
                rows["selected_by_explicit_potential"].append(local_idx == best_success)
                rows["label_rank_in_group"].append(int(ranks[local_idx]))

        # Advance the single rollout trajectory with the frozen deployed policies.
        _execute_c_with_frozen_r(env, req, int(ppo_action), agent_r, len(env.mec.servers))

    feature_array = (
        np.stack(rows["features"]).astype(np.float32)
        if rows["features"] else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    )
    return {
        "features": feature_array,
        "labels": np.asarray(rows["labels"], dtype=np.float32),
        "group_ids": np.asarray(rows["group_ids"], dtype=np.int64),
        "candidate_actions": np.asarray(rows["candidate_actions"], dtype=np.int64).reshape(-1, 2),
        "seeds": np.asarray(rows["seeds"], dtype=np.int64),
        "episode_ids": np.asarray(rows["episode_ids"], dtype=np.int64),
        "request_indices": np.asarray(rows["request_indices"], dtype=np.int64),
        "current_success": np.asarray(rows["current_success"], dtype=bool),
        "explicit_phi_before": np.asarray(rows["explicit_phi_before"], dtype=np.float32),
        "explicit_phi_after": np.asarray(rows["explicit_phi_after"], dtype=np.float32),
        "delta_phi": np.asarray(rows["delta_phi"], dtype=np.float32),
        "valid_r_actions": np.asarray(rows["valid_r_actions"], dtype=np.int64),
        "failure_reason": np.asarray(rows["failure_reason"], dtype="<U32"),
        "selected_by_ppo": np.asarray(rows["selected_by_ppo"], dtype=bool),
        "selected_by_explicit_potential": np.asarray(rows["selected_by_explicit_potential"], dtype=bool),
        "label_rank_in_group": np.asarray(rows["label_rank_in_group"], dtype=np.int64),
        "all_group_ids": np.asarray(groups["all_group_ids"], dtype=np.int64),
        "group_raw_mask_empty": np.asarray(groups["group_raw_mask_empty"], dtype=bool),
        "group_seeds": np.asarray(groups["group_seeds"], dtype=np.int64),
        "group_episode_ids": np.asarray(groups["group_episode_ids"], dtype=np.int64),
        "group_request_indices": np.asarray(groups["group_request_indices"], dtype=np.int64),
    }


def info_success(info: Dict[str, Any]) -> bool:
    return bool(info.get("success", False))


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
    candidate_groups = np.unique(data["group_ids"])
    ranges, gaps = [], []
    ppo_matches = []
    nonzero_multi = 0
    multi = 0
    for gid in candidate_groups:
        idx = np.flatnonzero(data["group_ids"] == gid)
        if len(idx) > 1:
            multi += 1
            vals = labels[idx]
            value_range = float(vals.max() - vals.min())
            ranges.append(value_range)
            nonzero_multi += value_range > 1e-8
            sorted_values = np.sort(vals)[::-1]
            gaps.append(float(sorted_values[0] - sorted_values[1]))
        chosen_ppo = idx[data["selected_by_ppo"][idx]]
        chosen_pot = idx[data["selected_by_explicit_potential"][idx]]
        if len(chosen_ppo) and len(chosen_pot):
            ppo_matches.append(int(chosen_ppo[0] == chosen_pot[0]))
    return {
        "groups": int(len(data["all_group_ids"])),
        "candidate_groups": int(len(candidate_groups)),
        "candidates": int(len(labels)),
        "raw_mask_empty_rate": float(np.mean(data["group_raw_mask_empty"])),
        "current_success_rate": float(np.mean(data["current_success"])) if len(labels) else 0.0,
        "candidates_per_nonempty_group_mean": float(len(labels) / max(len(candidate_groups), 1)),
        "label_mean": float(np.mean(labels)) if len(labels) else 0.0,
        "label_std": float(np.std(labels)) if len(labels) else 0.0,
        "label_p5_p50_p95": np.percentile(labels, [5, 50, 95]).tolist() if len(labels) else [0, 0, 0],
        "multi_candidate_groups": multi,
        "nonzero_label_range_rate": float(nonzero_multi / max(multi, 1)),
        "label_range_mean": float(np.mean(ranges)) if ranges else 0.0,
        "top1_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
        "ppo_explicit_top1_agreement": float(np.mean(ppo_matches)) if ppo_matches else 0.0,
        "feature_finite_rate": float(np.mean(np.isfinite(data["features"]))) if data["features"].size else 1.0,
        "nan_or_inf_count": int(np.size(data["features"]) - np.isfinite(data["features"]).sum()),
        "split_counts": {
            str(split_id): int(np.sum(data["candidate_actions"][:, 0] == split_id))
            for split_id in np.unique(data["candidate_actions"][:, 0])
        } if len(labels) else {},
        "server_counts": {
            str(server_id): int(np.sum(data["candidate_actions"][:, 1] == server_id))
            for server_id in np.unique(data["candidate_actions"][:, 1])
        } if len(labels) else {},
    }


def _write_report(path: Path, metadata: Dict[str, Any]) -> None:
    lines = [
        "# C-Side Post-Decision Dataset Generation", "",
        f"- Feature dimension: {metadata['feature_dim']}",
        f"- Label: `{metadata['label_name']}`", "",
        "## Split Quality", "",
        "| Split | Groups | Candidates | Empty | Label std | Nonzero-range multi-groups | Top-1 agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name, diag in metadata["diagnostics"].items():
        lines.append(
            f"| {split_name} | {diag['groups']} | {diag['candidates']} | "
            f"{diag['raw_mask_empty_rate']:.2%} | {diag['label_std']:.4f} | "
            f"{diag['nonzero_label_range_rate']:.2%} | "
            f"{diag['ppo_explicit_top1_agreement']:.2%} |"
        )
    lines += ["", f"**Verdict: {metadata['verdict']}**", "",
              "## Generation Timing", ""]
    for split_name, timing in metadata.get("generation_timing", {}).items():
        lines.append(
            f"- {split_name}: {timing['episode_worker_seconds']:.1f}s worker time; "
            f"{timing['mean_episode_seconds']:.2f}s/episode; "
            f"{timing['mean_candidate_seconds'] * 1000:.2f}ms/candidate"
        )
    lines.append(f"- Total worker time: {metadata['elapsed_seconds']:.1f}s")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [name.strip() for name in args.splits.split(",") if name.strip()]
    if any(name not in SPLIT_SEEDS for name in splits):
        raise ValueError(f"Unknown split in {splits}")

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    r_before = {key: value.clone() for key, value in agent_r.policy_net.state_dict().items()}

    for split_name in splits:
        for seed in SPLIT_SEEDS[split_name]:
            proto = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                args.slot_bw_hz, args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            rng = np.random.RandomState(seed)
            for episode_index in range(args.episodes):
                shard = shard_dir / f"{split_name}_s{seed}_e{episode_index:03d}.npz"
                src = int(rng.randint(0, proto.net.NUM_NODES))
                requests = generate_requests(
                    proto, rng, src, args.requests_per_episode,
                    args.arrival_interval, args.holding_min, args.holding_max,
                    args.deadline_min, args.deadline_max, args.size_min_mb,
                    args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                    args.num_splits, args.split_profile, args.traffic_mode,
                    args.regime_stay_prob,
                )
                if not _should_generate_shard(shard, args.resume):
                    continue
                episode_started = time.time()
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    args.slot_bw_hz, args.guard_band_fs,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                arrays = _generate_episode(
                    env, requests, agent_c, agent_r, split_name, seed, episode_index,
                    args.potential_history_window, args.potential_probe_limit, args.alpha,
                    args.label_delay_coef, args.label_horizon, args.label_phi_coef,
                    args.label_current_block_coef, args.label_future_block_coef,
                )
                _atomic_save_npz(shard, **arrays)
                generation_seconds = time.time() - episode_started
                _atomic_write_json(
                    shard.with_suffix(".timing.json"),
                    {
                        "split": split_name,
                        "seed": seed,
                        "episode": episode_index,
                        "groups": int(len(arrays["all_group_ids"])),
                        "candidates": int(len(arrays["labels"])),
                        "generation_seconds": generation_seconds,
                    },
                )
                print(
                    f"[split={split_name}][seed={seed}][episode={episode_index + 1}/{args.episodes}] "
                    f"groups={len(arrays['all_group_ids'])} candidates={len(arrays['labels'])} "
                    f"elapsed={generation_seconds:.1f}s", flush=True,
                )

    diagnostics = {}
    split_data = {}
    generation_timing = {}
    for split_name in splits:
        paths = [
            shard_dir / f"{split_name}_s{seed}_e{episode_index:03d}.npz"
            for seed in SPLIT_SEEDS[split_name]
            for episode_index in range(args.episodes)
        ]
        if not all(path.exists() for path in paths):
            raise RuntimeError(f"Incomplete shards for {split_name}")
        data = _concat_shards(paths)
        split_data[split_name] = data
        diagnostics[split_name] = _dataset_diagnostics(data)
        _atomic_save_npz(output_dir / f"{split_name}.npz", **data)
        timing_records = []
        for shard_path in paths:
            timing_path = shard_path.with_suffix(".timing.json")
            if timing_path.exists():
                timing_records.append(json.loads(timing_path.read_text(encoding="utf-8")))
        total_seconds = float(sum(item["generation_seconds"] for item in timing_records))
        total_candidates = int(sum(item["candidates"] for item in timing_records))
        generation_timing[split_name] = {
            "episode_worker_seconds": total_seconds,
            "mean_episode_seconds": total_seconds / max(len(timing_records), 1),
            "mean_candidate_seconds": total_seconds / max(total_candidates, 1),
            "timed_episodes": len(timing_records),
        }

    seed_sets = [set(SPLIT_SEEDS[name]) for name in splits]
    no_seed_overlap = all(
        seed_sets[i].isdisjoint(seed_sets[j])
        for i in range(len(seed_sets)) for j in range(i + 1, len(seed_sets))
    )
    group_sets = [set(split_data[name]["all_group_ids"].tolist()) for name in splits]
    no_group_overlap = all(
        group_sets[i].isdisjoint(group_sets[j])
        for i in range(len(group_sets)) for j in range(i + 1, len(group_sets))
    )
    quality_ok = all(
        diag["feature_finite_rate"] == 1.0
        and diag["label_std"] > 1e-8
        and diag["nonzero_label_range_rate"] >= 0.20
        for diag in diagnostics.values()
    ) and no_seed_overlap and no_group_overlap
    r_unchanged = all(
        torch.equal(value, r_before[key])
        for key, value in agent_r.policy_net.state_dict().items()
    )
    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_dim": len(FEATURE_NAMES),
        "label_name": (
            f"h{args.label_horizon}_phi{args.label_phi_coef:g}"
            f"_curblk{args.label_current_block_coef:g}"
            f"_futblk{args.label_future_block_coef:g}"
            f"_delay{args.label_delay_coef:g}"
            if args.label_horizon > 0 else (
                "explicit_spectrum_only_phi_after"
                if args.label_delay_coef <= 0
                else f"explicit_spectrum_phi_after_minus_{args.label_delay_coef:g}_delay_norm"
            )
        ),
        "diagnostic_only_fields": [
            "explicit_phi_before", "explicit_phi_after", "delta_phi",
            "label_rank_in_group", "failure_reason",
        ],
        "split_seeds": {name: list(SPLIT_SEEDS[name]) for name in splits},
        "episodes_per_seed": args.episodes,
        "requests_per_episode": args.requests_per_episode,
        "label_delay_coef": args.label_delay_coef,
        "label_horizon": args.label_horizon,
        "label_phi_coef": args.label_phi_coef,
        "label_current_block_coef": args.label_current_block_coef,
        "label_future_block_coef": args.label_future_block_coef,
        "no_seed_overlap": no_seed_overlap,
        "no_group_overlap": no_group_overlap,
        "agent_r_unchanged": r_unchanged,
        "diagnostics": diagnostics,
        "generation_timing": generation_timing,
        "assembly_elapsed_seconds": time.time() - started,
        "elapsed_seconds": float(sum(
            item["episode_worker_seconds"] for item in generation_timing.values()
        )),
        "verdict": (
            "PROCEED_TO_POST_DECISION_TRAINING"
            if quality_ok and r_unchanged else "STOP_AND_AUDIT_DATASET"
        ),
    }
    if "train" in split_data and len(split_data["train"]["features"]):
        train_x = split_data["train"]["features"].astype(np.float64)
        metadata["train_feature_mean"] = train_x.mean(axis=0).tolist()
        metadata["train_feature_std"] = np.maximum(train_x.std(axis=0), 1e-6).tolist()
    _atomic_write_json(output_dir / "metadata.json", metadata)
    _write_report(output_dir / "generation_report.md", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--output_dir", default="sa_hmarl/datasets/c_post_decision")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
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
    parser.add_argument("--potential_history_window", type=int, default=12)
    parser.add_argument("--potential_probe_limit", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument(
        "--label_delay_coef", type=float, default=0.0,
        help="One-step mode: subtract coef * delay/deadline. H-step mode: delay penalty weight.",
    )
    parser.add_argument(
        "--label_horizon", type=int, default=0,
        help="If >0, use H-step labels with future blocking and delay terms.",
    )
    parser.add_argument(
        "--label_phi_coef", type=float, default=1.0,
        help="H-step label weight for post-decision spectrum potential.",
    )
    parser.add_argument(
        "--label_current_block_coef", type=float, default=1.0,
        help="H-step label penalty for current candidate failure.",
    )
    parser.add_argument(
        "--label_future_block_coef", type=float, default=1.0,
        help="H-step label penalty for future blocked count divided by horizon.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    result = generate_dataset(build_parser().parse_args())
    print(result["verdict"])
