"""High-load horizon probe for v1.35.

For each seed we:
  1. Run a 2000-request warmup to reach a high-load steady state.
  2. Continue through the episode and sample decision states, aiming to cover
     server-utilization buckets.
  3. For each sampled state, enumerate the same PPO-R top-K candidates and
     compute H-step returns for H=5, 12, 20 using the same future request trace
     and common-random-number continuation.
  4. Save per-seed shards and merge into a diagnostic report.
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

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action, ksp_ff_highest_mod_action
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
    _compute_return,
    _rollout_future as _rollout_future_base,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


C_K = 5
R_K = 50
PATH_SORT = "hops"
BLOCK_SORT = "start_asc"
HORIZONS = [5, 12, 20]
UTIL_BUCKETS = [
    (0.0, 0.5),
    (0.5, 0.7),
    (0.7, 0.85),
    (0.85, 0.95),
]


def _state_hash(env, req, split_id: int, server_id: int) -> str:
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


def _select_ppo_topk(agent_r, obs_r, legal: List[int], top_k: int) -> List[int]:
    if not legal:
        return []
    logits = _get_r_logits(agent_r, obs_r)
    top_k = min(top_k, len(legal))
    top_local = np.argsort(-logits[legal], kind="stable")[:top_k]
    return [int(legal[i]) for i in top_local]


def _execute_fixed_r(env, req, split_id: int, server_id: int, r_action_idx: int, obs_r: Dict[str, Any]):
    r_action = decode_agent_r_action(int(r_action_idx), len(obs_r["mod_names"]), env.max_blocks)
    _, _, _, info = env.step((split_id, server_id), r_action)
    return info


def _rollout_future(env, requests, start_idx, horizon, agent_c, agent_r, num_servers):
    return _rollout_future_base(
        env, requests, start_idx, horizon, agent_c, agent_r,
        num_servers=num_servers, util_threshold=0.95, alpha=0.3,
        future_rollout_policy="ppo_r", rng=np.random.RandomState(0), gamma=1.0,
    )


def _util_bucket(util: float) -> int:
    for i, (lo, hi) in enumerate(UTIL_BUCKETS):
        if lo <= util < hi or (i == len(UTIL_BUCKETS) - 1 and lo <= util <= hi):
            return i
    return len(UTIL_BUCKETS) - 1


def _dummy_args(return_coefs: Dict[str, float]):
    class Args:
        pass
    a = Args()
    for k, v in return_coefs.items():
        setattr(a, k, v)
    a.return_mode = "v1"
    a.path_penalty_coef = 0.0
    a.fs_penalty_coef = 0.0
    a.viability_phi_coef = 0.0
    return a


def _sample_seed(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    seed: int,
    horizon_list: List[int],
    max_candidates: int,
    ppo_top_k: int,
    warmup_requests: int,
    target_per_bucket: int,
    return_coefs: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Sample states from one seed after warmup and compute multi-horizon returns."""
    env.reset(requests)
    num_servers = len(env.mec.servers)
    bucket_counts = [0] * len(UTIL_BUCKETS)
    states = []

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        env.k = C_K
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), num_servers)

        env.k = R_K
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        ppo_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        is_warmup = request_index < warmup_requests
        if not is_warmup and len(legal) >= 2:
            util = float(obs_c["server_utilizations"][server_id])
            bucket = _util_bucket(util)
            if bucket_counts[bucket] < target_per_bucket:
                snapshot = _snapshot_before_r_decision(env, req.req_id)
                candidates = _select_ppo_topk(agent_r, obs_r, legal, ppo_top_k)[:max_candidates]
                if len(candidates) >= 2:
                    bucket_counts[bucket] += 1
                    state = _evaluate_state(
                        snapshot, req, requests, request_index,
                        split_id, server_id, c_idx, seed, obs_c, obs_r, r_features, legal,
                        candidates, ppo_r_idx, horizon_list,
                        agent_c, agent_r, num_servers, return_coefs, env,
                    )
                    states.append(state)
                    if all(c >= target_per_bucket for c in bucket_counts):
                        break

        # Advance real trajectory with PPO-R.
        deployed_action = decode_agent_r_action(int(ppo_r_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), deployed_action)

    return states


def _evaluate_state(
    snapshot,
    req,
    requests,
    request_index,
    split_id,
    server_id,
    c_idx,
    seed,
    obs_c,
    obs_r,
    r_features,
    legal,
    candidates,
    ppo_r_idx,
    horizon_list,
    agent_c,
    agent_r,
    num_servers,
    return_coefs,
    env,
):
    util = float(obs_c["server_utilizations"][server_id])
    state_hash = _state_hash(env, req, split_id, server_id)
    max_h = max(horizon_list)
    if request_index + 1 + max_h > len(requests):
        # Not enough future requests; skip heavy computation by returning empty.
        return None

    base_future_rng = np.random.RandomState(12345 + request_index)
    base_rng_state = base_future_rng.get_state()

    # Normalize stats over legal actions (not used in return here but kept for consistency).
    legal_path_km = np.array([float(r_features[int(a)][0]) for a in legal])
    legal_required_fs = np.array([float(r_features[int(a)][7]) for a in legal])
    path_mean = legal_path_km.mean()
    path_std = legal_path_km.std() + 1e-6
    fs_mean = legal_required_fs.mean()
    fs_std = legal_required_fs.std() + 1e-6

    results_by_h = {h: {"returns": [], "blocked": [], "optical": [], "overload": [],
                        "alloc_failed": [], "other": [], "admitted": [],
                        "delay_sum": [], "fs_sum": []} for h in horizon_list}

    ksp_plain = ksp_ff_action(obs_r)
    ksp_highest = ksp_ff_highest_mod_action(obs_r)

    for r_action_idx in candidates:
        branch = copy.deepcopy(snapshot)
        info = _execute_fixed_r(branch, req, split_id, server_id, int(r_action_idx), obs_r)
        current_success = bool(info.get("success", False))
        path_km_norm = (float(r_features[int(r_action_idx)][0]) - path_mean) / path_std
        required_fs_norm = (float(r_features[int(r_action_idx)][7]) - fs_mean) / fs_std

        for h in horizon_list:
            candidate_rng = np.random.RandomState(0)
            candidate_rng.set_state(base_rng_state)
            future_branch = copy.deepcopy(branch)
            future = _rollout_future(future_branch, requests, request_index + 1, h,
                                     agent_c, agent_r, num_servers)
            ret = _compute_return(info, future, _dummy_args(return_coefs),
                                  path_km_norm, required_fs_norm)
            blocked = int(future.get("blocked", 0))
            optical = int(future.get("no_suitable_block", 0))
            overload = int(future.get("server_overload", 0))
            alloc_failed = int(future.get("allocation_failed", 0))
            other = int(future.get("other", 0))
            admitted = h - blocked  # approximate; rollout horizon may stop early
            results_by_h[h]["returns"].append(float(ret))
            results_by_h[h]["blocked"].append(blocked)
            results_by_h[h]["optical"].append(optical)
            results_by_h[h]["overload"].append(overload)
            results_by_h[h]["alloc_failed"].append(alloc_failed)
            results_by_h[h]["other"].append(other)
            results_by_h[h]["admitted"].append(admitted)
            results_by_h[h]["delay_sum"].append(float(future.get("delay_sum", 0.0)))
            results_by_h[h]["fs_sum"].append(float(future.get("fs_sum", 0.0)))

    out = {
        "seed": seed,
        "request_index": request_index,
        "state_hash": state_hash,
        "fixed_c_action": int(c_idx),
        "split_id": split_id,
        "server_id": server_id,
        "server_utilization": util,
        "util_bucket": _util_bucket(util),
        "candidate_action_ids": np.asarray(candidates, dtype=np.int64),
        "ppo_action_id": int(ppo_r_idx),
        "ksp_plain_action_id": int(ksp_plain) if ksp_plain is not None and int(ksp_plain) in candidates else -1,
        "ksp_highest_action_id": int(ksp_highest) if ksp_highest is not None and int(ksp_highest) in candidates else -1,
    }
    for h in horizon_list:
        prefix = f"H{h}_"
        rets = np.asarray(results_by_h[h]["returns"], dtype=np.float32)
        out[prefix + "returns"] = rets
        out[prefix + "blocked"] = np.asarray(results_by_h[h]["blocked"], dtype=np.int64)
        out[prefix + "optical"] = np.asarray(results_by_h[h]["optical"], dtype=np.int64)
        out[prefix + "overload"] = np.asarray(results_by_h[h]["overload"], dtype=np.int64)
        out[prefix + "alloc_failed"] = np.asarray(results_by_h[h]["alloc_failed"], dtype=np.int64)
        out[prefix + "other"] = np.asarray(results_by_h[h]["other"], dtype=np.int64)
        out[prefix + "admitted"] = np.asarray(results_by_h[h]["admitted"], dtype=np.int64)
        out[prefix + "delay_sum"] = np.asarray(results_by_h[h]["delay_sum"], dtype=np.float32)
        out[prefix + "fs_sum"] = np.asarray(results_by_h[h]["fs_sum"], dtype=np.float32)
    return out


def _shard_worker(shard: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    import resource
    cfg = shard["config"]
    mod_reg = ModulationRegistry.from_profile(cfg["modulation_profile"])
    agent_c = _load_ppo_c(cfg["agent_c_checkpoint"], cfg["device"])
    agent_r = _load_ppo_r(cfg["agent_r_checkpoint"], mod_reg, cfg["device"])
    env = make_env(
        cfg["topology"], cfg["num_slots"], cfg["num_servers"], shard["seed"],
        modulation_profile=cfg["modulation_profile"],
        max_blocks=cfg["max_blocks"], block_sort_strategy=BLOCK_SORT,
        path_sort_strategy=PATH_SORT, k=R_K,
    )
    rng = np.random.RandomState(shard["seed"])
    src = int(rng.randint(0, env.net.NUM_NODES))
    total_requests = cfg["warmup_requests"] + cfg["max_requests_after_warmup"] + max(HORIZONS) + 1
    requests = generate_requests(
        env, rng, src, total_requests,
        cfg["arrival_interval"], cfg["holding_min"], cfg["holding_max"],
        cfg["deadline_min"], cfg["deadline_max"], cfg["size_min_mb"], cfg["size_max_mb"],
        cfg["edge_cost_min"], cfg["edge_cost_max"], cfg["num_splits"], cfg["split_profile"],
    )
    states = _sample_seed(
        env, requests, agent_c, agent_r, shard["seed"],
        horizon_list=cfg["horizon_list"],
        max_candidates=cfg["max_candidates"],
        ppo_top_k=cfg["ppo_top_k"],
        warmup_requests=cfg["warmup_requests"],
        target_per_bucket=cfg["target_per_bucket"],
        return_coefs=cfg["return_coefs"],
    )
    states = [s for s in states if s is not None]
    if not states:
        arrays = {}
    else:
        max_c = max(int(s["candidate_action_ids"].shape[0]) for s in states)
        arrays = {}
        candidate_keys = [k for k in states[0].keys() if isinstance(states[0][k], np.ndarray) and states[0][k].shape[0] == states[0]["candidate_action_ids"].shape[0]]
        scalar_keys = [k for k in states[0].keys() if isinstance(states[0][k], np.ndarray) and states[0][k].shape == ()]
        other_array_keys = [k for k in states[0].keys() if isinstance(states[0][k], np.ndarray) and k not in candidate_keys and k not in scalar_keys]
        for k in candidate_keys:
            out = []
            for s in states:
                arr = s[k]
                if arr.shape[0] < max_c:
                    pad = [(0, max_c - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
                    arr = np.pad(arr, pad, mode="constant")
                out.append(arr)
            arrays[k] = np.stack(out)
        arrays["mask"] = np.ones((len(states), max_c), dtype=bool)
        for i, s in enumerate(states):
            arrays["mask"][i, s["candidate_action_ids"].shape[0]:] = False
        for k in other_array_keys:
            arrays[k] = np.stack([s[k] for s in states])
        for k, v in states[0].items():
            if not isinstance(v, np.ndarray):
                arrays[k] = np.asarray([s[k] for s in states])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"arrays": arrays, "groups": len(states), "peak_rss_mb": peak_rss}


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
        "warmup_requests": args.warmup_requests,
        "max_requests_after_warmup": args.max_requests_after_warmup,
        "horizon_list": HORIZONS,
        "max_candidates": args.max_candidates,
        "ppo_top_k": args.ppo_top_k,
        "target_per_bucket": args.target_per_bucket,
        "return_coefs": {
            "return_current_block_coef": args.return_current_block_coef,
            "return_future_block_coef": args.return_future_block_coef,
            "return_future_nsb_coef": args.return_future_nsb_coef,
            "return_future_server_overload_coef": args.return_future_server_overload_coef,
            "return_delay_coef": args.return_delay_coef,
            "return_fs_coef": args.return_fs_coef,
        },
        "device": args.device,
        "schema": "horizon_probe_v1",
        "version": "1.0",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v135_parallel_paired/horizon_probe")
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
    parser.add_argument("--warmup_requests", type=int, default=2000)
    parser.add_argument("--max_requests_after_warmup", type=int, default=3000)
    parser.add_argument("--max_candidates", type=int, default=30)
    parser.add_argument("--ppo_top_k", type=int, default=30)
    parser.add_argument("--target_per_bucket", type=int, default=10)
    parser.add_argument("--return_current_block_coef", type=float, default=1.0)
    parser.add_argument("--return_future_block_coef", type=float, default=1.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=1.0)
    parser.add_argument("--return_future_server_overload_coef", type=float, default=0.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.0)
    parser.add_argument("--return_fs_coef", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3030, 4040, 5050])
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _make_config(args)
    shards = [{"shard_id": f"seed{s}", "seed": s, "config": config} for s in args.seeds]

    from v135_parallel_launcher import ShardedParallelLauncher, _atomic_write_json, _config_hash
    launcher = ShardedParallelLauncher(output_dir, config, _shard_worker, max_workers=args.max_workers, shard_subdir="shards")
    results = launcher.run(shards)

    print("[horizon_probe] Merging shards...", flush=True)
    all_states = []
    meta_map = {}
    for r in results:
        if r.status not in ("ok", "skipped"):
            continue
        meta_path = output_dir / "shards" / f"{r.shard_id}.metadata.json"
        meta_map[r.shard_id] = json.loads(meta_path.read_text(encoding="utf-8"))
        npz_path = output_dir / "shards" / f"{r.shard_id}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(str(npz_path), allow_pickle=True)
        if len(npz.get("state_hash", [])) == 0:
            continue
        n = len(npz["state_hash"])
        for i in range(n):
            state = {k: npz[k][i] for k in npz.files}
            all_states.append(state)

    if not all_states:
        print("[horizon_probe] No states sampled.", flush=True)
        return

    # Sort deterministically.
    all_states.sort(key=lambda s: (int(s["seed"]), int(s["request_index"])))
    _write_report(all_states, output_dir, config)
    print(f"[horizon_probe] Saved report to {output_dir}")


def _write_report(states: List[Dict[str, Any]], output_dir: Path, config: Dict[str, Any]):
    import scipy.stats as stats
    from v135_parallel_launcher import _config_hash, _atomic_write_json
    n = len(states)
    per_bucket = {i: [] for i in range(len(UTIL_BUCKETS))}
    for s in states:
        per_bucket[int(s["util_bucket"])].append(s)

    report = {
        "config": config,
        "config_hash": _config_hash(config),
        "num_states": n,
        "bucket_counts": {i: len(per_bucket[i]) for i in per_bucket},
        "horizons": {},
    }
    md_lines = [
        "# High-Load Horizon Probe Report",
        "",
        f"- Seeds: {config.get('seeds', 'see shards')}",
        f"- Warmup requests: {config['warmup_requests']}",
        f"- Sampled states: {n}",
        f"- Bucket counts: {report['bucket_counts']}",
        "",
    ]

    for h in HORIZONS:
        prefix = f"H{h}_"
        overload_pos = []
        overload_var = []
        optical_pos = []
        optical_var = []
        ret_ranges = []
        top1_top2_gaps = []
        ppo_regret = []
        ksp_plain_regret = []
        ksp_highest_regret = []

        for s in states:
            rets = s[prefix + "returns"]
            overload = s[prefix + "overload"]
            optical = s[prefix + "optical"]
            overload_pos.append(float(np.any(overload > 0)))
            overload_var.append(float(np.var(overload)))
            optical_pos.append(float(np.any(optical > 0)))
            optical_var.append(float(np.var(optical)))
            ret_ranges.append(float(np.max(rets) - np.min(rets)))
            sorted_rets = np.sort(rets)[::-1]
            top1_top2_gaps.append(float(sorted_rets[0] - sorted_rets[1]) if len(sorted_rets) > 1 else 0.0)
            best = float(np.max(rets))
            ppo_idx = int(np.where(s["candidate_action_ids"] == int(s["ppo_action_id"]))[0][0]) if int(s["ppo_action_id"]) in s["candidate_action_ids"] else -1
            if ppo_idx >= 0:
                ppo_regret.append(best - float(rets[ppo_idx]))
            kp = int(s["ksp_plain_action_id"])
            if kp >= 0:
                kp_idx = int(np.where(s["candidate_action_ids"] == kp)[0][0])
                ksp_plain_regret.append(best - float(rets[kp_idx]))
            kh = int(s["ksp_highest_action_id"])
            if kh >= 0:
                kh_idx = int(np.where(s["candidate_action_ids"] == kh)[0][0])
                ksp_highest_regret.append(best - float(rets[kh_idx]))

        # Spearman between H and H5 returns.
        spear_vs_h5 = None
        if h != 5:
            rhos = []
            for s in states:
                r1 = s["H5_returns"]
                r2 = s[prefix + "returns"]
                if len(r1) > 1 and len(r2) > 1 and np.std(r1) > 0 and np.std(r2) > 0:
                    rho, _ = stats.spearmanr(r1, r2)
                    rhos.append(float(rho))
            spear_vs_h5 = float(np.mean(rhos)) if rhos else None

        report["horizons"][h] = {
            "overload_positive_rate_mean": float(np.mean(overload_pos)),
            "overload_positive_rate_std": float(np.std(overload_pos)),
            "overload_variance_mean": float(np.mean(overload_var)),
            "overload_variance_std": float(np.std(overload_var)),
            "optical_positive_rate_mean": float(np.mean(optical_pos)),
            "optical_variance_mean": float(np.mean(optical_var)),
            "return_range_mean": float(np.mean(ret_ranges)),
            "return_range_std": float(np.std(ret_ranges)),
            "top1_top2_gap_mean": float(np.mean(top1_top2_gaps)),
            "ppo_regret_mean": float(np.mean(ppo_regret)) if ppo_regret else None,
            "ksp_plain_regret_mean": float(np.mean(ksp_plain_regret)) if ksp_plain_regret else None,
            "ksp_highest_regret_mean": float(np.mean(ksp_highest_regret)) if ksp_highest_regret else None,
            "spearman_vs_H5_mean": spear_vs_h5,
        }

        md_lines.extend([
            f"## H={h}",
            "",
            f"- future overload positive rate: {report['horizons'][h]['overload_positive_rate_mean']:.2%} ± {report['horizons'][h]['overload_positive_rate_std']:.2%}",
            f"- future overload variance: {report['horizons'][h]['overload_variance_mean']:.4f} ± {report['horizons'][h]['overload_variance_std']:.4f}",
            f"- future optical positive rate: {report['horizons'][h]['optical_positive_rate_mean']:.2%}",
            f"- return range: {report['horizons'][h]['return_range_mean']:.4f} ± {report['horizons'][h]['return_range_std']:.4f}",
            f"- top1-top2 gap: {report['horizons'][h]['top1_top2_gap_mean']:.4f}",
            f"- PPO regret: {report['horizons'][h]['ppo_regret_mean']}",
            f"- KSP plain regret: {report['horizons'][h]['ksp_plain_regret_mean']}",
            f"- KSP highest regret: {report['horizons'][h]['ksp_highest_regret_mean']}",
        ])
        if spear_vs_h5 is not None:
            md_lines.append(f"- Spearman vs H5: {spear_vs_h5:.4f}")
        md_lines.append("")

    # Top-1 overlap between horizons.
    overlap_h5_h12 = []
    overlap_h5_h20 = []
    overlap_h12_h20 = []
    top5_overlap_h5_h12 = []
    for s in states:
        a5 = np.argsort(-s["H5_returns"], kind="stable")
        a12 = np.argsort(-s["H12_returns"], kind="stable")
        a20 = np.argsort(-s["H20_returns"], kind="stable")
        overlap_h5_h12.append(float(a5[0] == a12[0]))
        overlap_h5_h20.append(float(a5[0] == a20[0]))
        overlap_h12_h20.append(float(a12[0] == a20[0]))
        top5_overlap_h5_h12.append(float(len(set(a5[:5]) & set(a12[:5]))) / 5.0)
    report["top1_overlap"] = {
        "H5_H12": float(np.mean(overlap_h5_h12)),
        "H5_H20": float(np.mean(overlap_h5_h20)),
        "H12_H20": float(np.mean(overlap_h12_h20)),
        "top5_H5_H12": float(np.mean(top5_overlap_h5_h12)),
    }
    md_lines.extend([
        "## Horizon action overlap",
        "",
        f"- top-1 H5/H12: {report['top1_overlap']['H5_H12']:.2%}",
        f"- top-1 H5/H20: {report['top1_overlap']['H5_H20']:.2%}",
        f"- top-1 H12/H20: {report['top1_overlap']['H12_H20']:.2%}",
        f"- top-5 H5/H12: {report['top1_overlap']['top5_H5_H12']:.2%}",
        "",
    ])

    # Per-bucket overload variance.
    bucket_report = {}
    md_lines.append("## Per-bucket overload variance (H=20)")
    md_lines.append("")
    md_lines.append("| Bucket | Count | overload pos rate | overload variance | return range |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for i, (lo, hi) in enumerate(UTIL_BUCKETS):
        bucket_states = per_bucket[i]
        if not bucket_states:
            continue
        overload_pos_b = [float(np.any(s["H20_overload"] > 0)) for s in bucket_states]
        overload_var_b = [float(np.var(s["H20_overload"])) for s in bucket_states]
        ret_range_b = [float(np.max(s["H20_returns"]) - np.min(s["H20_returns"])) for s in bucket_states]
        bucket_report[i] = {
            "range": [lo, hi],
            "count": len(bucket_states),
            "overload_positive_rate": float(np.mean(overload_pos_b)),
            "overload_variance": float(np.mean(overload_var_b)),
            "return_range": float(np.mean(ret_range_b)),
        }
        md_lines.append(
            f"| [{lo},{hi}) | {len(bucket_states)} | {bucket_report[i]['overload_positive_rate']:.2%} | "
            f"{bucket_report[i]['overload_variance']:.4f} | {bucket_report[i]['return_range']:.4f} |"
        )
    report["per_bucket"] = bucket_report

    md_lines.append("")
    if report["horizons"][20]["overload_variance_mean"] < 1e-6:
        md_lines.append("**Conclusion**: H=20 candidate-wise overload variance is near zero. Future overload is essentially determined by the fixed C-side/server choice. Do not launch overload-aware R-side full training.")
    else:
        md_lines.append("**Conclusion**: H=20 shows non-zero candidate-wise overload variance. A longer-horizon overload-aware label may be useful, but a costed plan should be generated before committing to a full dataset.")

    _atomic_write_json(output_dir / "HIGH_LOAD_HORIZON_PROBE.json", report)
    (output_dir / "HIGH_LOAD_HORIZON_PROBE.md").write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
