"""R Action-Space Expansion Sweep.

Sweeps k_paths, max_blocks, and block_sort_strategy to understand
whether expanding the R action space (more paths, more blocks, different
sorting) can reduce blocking by lowering no_suitable_block failures.

Four C+R methods per config:
  1. Agent-C + BC-PPO-R          (baseline)
  2. Agent-C + BC-PPO-R + BF     (brute-force fallback on PPO-R failure)
  3. DF-C   + BC-PPO-R
  4. IWD-C  + BC-PPO-R

Usage::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.sweep_r_action_space
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWEEP_AXES = {
    "k_paths": [3, 5, 7],
    "max_blocks": [5, 10, 15],
    "block_sort_strategy": ["size_desc", "mixed", "waste_asc"],
}

METHOD_ORDER = ["agent", "agent_bf", "df", "iwd"]

METHOD_LABELS = {
    "agent": "Agent-C + PPO-R",
    "agent_bf": "Agent-C + PPO-R + BF",
    "df": "DF-C + PPO-R",
    "iwd": "IWD-C + PPO-R",
}


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"),
    )
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=ckpt.get("agent_r_feature_mode", "default"),
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Agent-C selection + mask stats
# ---------------------------------------------------------------------------

def _select_agent_c_with_stats(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[Optional[int], int, int]:
    """Select Agent-C action. Returns (action_idx, n_valid_c, n_valid_r_max).

    n_valid_r_max is the max valid R actions across all valid C candidates
    (computed on the original mask for diagnostic purposes).
    """
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    n_valid_c = int(mask.sum())

    # Best-case R action count: max feasible (path,mod,block) across valid C
    n_valid_r_max = 0
    for idx in np.where(mask)[0]:
        split_id, server_id = decode_agent_c_action(int(idx), len(obs_c["server_utilizations"]))
        fc = obs_c["candidate_features"][int(idx)]
        n_feas = fc.get("n_feas_path_mod", fc.get("feasible_count", 0))
        n_valid_r_max = max(n_valid_r_max, int(n_feas))

    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action, n_valid_c, n_valid_r_max


# ---------------------------------------------------------------------------
# Brute-force R-action fallback
# ---------------------------------------------------------------------------

def _fallback_bf_select_r(
    env, req, split_id: int, server_id: int, obs_r: Dict[str, Any],
) -> Tuple[Optional[int], Tuple[int, int, int], bool]:
    """Brute-force search over all valid R actions on deep-copied env.

    Returns:
        (flat_action_idx, (path_idx, mod_idx, block_idx), found_any)
    """
    mask: np.ndarray = obs_r["agent_r_mask"]
    valid = np.where(mask)[0]
    num_mods = len(obs_r["mod_names"])
    max_blocks = env.max_blocks

    if len(valid) == 0:
        return None, (0, 0, 0), False

    best_action_idx: Optional[int] = None
    best_tuple: Tuple[int, int, int] = (0, 0, 0)
    best_waste: float = float("inf")
    best_fs: float = float("inf")
    best_path: float = float("inf")
    found_any = False

    # Sort valid actions by a heuristic to find good candidates early:
    # prefer lower path_idx (shorter path), lower mod_idx (more robust mod),
    # lower block_idx (first available block).
    valid_sorted = sorted(valid, key=lambda a: int(a))

    # Hard limit: at most 100 deep-copy trials per BF invocation
    max_trials = min(100, len(valid_sorted))

    for action_idx in valid_sorted[:max_trials]:
        action_idx = int(action_idx)
        path_idx, mod_idx, block_idx = decode_agent_r_action(
            action_idx, num_mods, max_blocks,
        )

        # Try on deep copy
        env_try = copy.deepcopy(env)
        _, _, _, info = env_try.step(
            (split_id, server_id), (path_idx, mod_idx, block_idx),
        )

        if info.get("success", False):
            found_any = True
            waste = float(info.get("block_waste", float("inf")))
            fs_val = float(info.get("num_slots", float("inf")))
            path_len = float(info.get("path_dist_km", float("inf")))

            # Lexicographic: min waste, then min fs, then min path
            if (waste < best_waste
                    or (waste == best_waste and fs_val < best_fs)
                    or (waste == best_waste and fs_val == best_fs and path_len < best_path)):
                best_waste = waste
                best_fs = fs_val
                best_path = path_len
                best_action_idx = action_idx
                best_tuple = (path_idx, mod_idx, block_idx)

            # Early exit: waste=0 with min FS is near-optimal
            if waste <= 0.01 and fs_val <= best_fs:
                break  # good enough, stop searching

    if found_any:
        return best_action_idx, best_tuple, True
    return None, (0, 0, 0), False


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def _eval_method_on_episode(
    method: str,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    requests: List,
    env_proto,
    args,
    ep_idx: int,
) -> Dict[str, Any]:
    """Evaluate one method on one episode."""
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities,
        k=args.k_paths,
    )
    env.reset(requests)

    server_selected_count = np.zeros(num_servers, dtype=int)
    seed_val = int(getattr(args, "seeds", "42").split(",")[0])
    rng = np.random.RandomState(seed_val + ep_idx)

    total = blocked = success = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    mod_counter = Counter()

    # Mask / action-space stats (from Agent-C trajectory)
    mask_valid_c_counts: List[int] = []
    mask_zero_c_count = 0
    mask_valid_r_counts: List[int] = []
    nsb_count = 0
    bf_used_count = 0
    bf_success_count = 0
    bf_total_candidates: List[int] = []
    bf_time_total = 0.0

    for req in requests:
        obs_c = build_agent_c_observation(env, req)

        # --- C action selection ---
        if method in ("agent", "agent_bf"):
            action_idx_c, n_valid_c, n_valid_r_max = _select_agent_c_with_stats(
                agent_c, obs_c, env.net.num_slots,
            )
            mask_valid_c_counts.append(n_valid_c)
            if n_valid_c == 0:
                mask_zero_c_count += 1
        else:
            raw_mask = obs_c["agent_c_mask"]
            action_idx_c = select_offloading_action(
                method, env, req, obs_c, raw_mask,
                rng=rng, server_selected_count=server_selected_count,
            )
            n_valid_c = int(raw_mask.sum())
            mask_valid_c_counts.append(n_valid_c)
            if n_valid_c == 0:
                mask_zero_c_count += 1

        action_c = (
            (0, 0) if action_idx_c is None
            else decode_agent_c_action(action_idx_c, num_servers)
        )
        split_id, server_id = action_c
        split_counter[f"split{split_id}"] += 1
        server_counter[f"s{server_id}"] += 1
        server_selected_count[server_id] += 1

        # --- R action selection ---
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask = obs_r["agent_r_mask"]
        n_valid_r = int(r_mask.sum())
        mask_valid_r_counts.append(n_valid_r)

        bf_used = False
        action_r_tuple: Tuple[int, int, int] = (0, 0, 0)

        if method == "agent_bf":
            # Try PPO-R first
            action_r_idx = agent_r.select_action(obs_r, deterministic=True)
            if action_r_idx is not None and r_mask[action_r_idx]:
                action_r_tuple = decode_agent_r_action(
                    action_r_idx, len(obs_r["mod_names"]), env.max_blocks,
                )
            else:
                # BF fallback
                bf_used = True
                t0 = time.time()
                bf_idx, bf_tuple, bf_found = _fallback_bf_select_r(
                    env, req, split_id, server_id, obs_r,
                )
                bf_time_total += time.time() - t0
                bf_total_candidates.append(n_valid_r)
                if bf_found:
                    action_r_idx = bf_idx
                    action_r_tuple = bf_tuple
                    bf_success_count += 1
                else:
                    action_r_idx = None
            bf_used_count += int(bf_used)
        else:
            action_r_idx = agent_r.select_action(obs_r, deterministic=True)
            if action_r_idx is not None:
                action_r_tuple = decode_agent_r_action(
                    action_r_idx, len(obs_r["mod_names"]), env.max_blocks,
                )

        # --- Step ---
        _, _, _, info = env.step(action_c, action_r_tuple)
        total += 1

        server_obj = env.mec.servers[server_id]
        reward = compute_agent_c_reward(
            info, req.deadline_ms, args.waste_coef, server_obj.utilization,
        )
        total_reward += reward

        if info.get("success", False):
            success += 1
            delay = float(info.get("delay_ms", 0.0))
            total_delay += delay
            total_fs += float(info.get("num_slots", 0))
            total_waste += float(info.get("block_waste", 0.0))
            total_path += float(info.get("path_dist_km", 0.0))
            mod_counter[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reason_counter[reason] += 1
            if reason == "no_suitable_block":
                nsb_count += 1

    n = max(total, 1)
    s = max(success, 1)

    result = {
        "total": total, "blocked": blocked, "success": success,
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "reason_counter": dict(reason_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "mod_counter": dict(mod_counter),
    }

    # C action-space stats
    result["mask_valid_c_counts"] = mask_valid_c_counts
    result["mask_zero_c_count"] = mask_zero_c_count
    result["no_valid_c_rate_ep"] = mask_zero_c_count / n

    # R action-space stats
    result["mask_valid_r_counts"] = mask_valid_r_counts
    result["avg_valid_r_actions"] = float(np.mean(mask_valid_r_counts)) if mask_valid_r_counts else 0.0

    # no_suitable_block ratio
    nsb_total = reason_counter.get("no_suitable_block", 0)
    result["no_suitable_block_ratio"] = nsb_total / n

    # BF stats
    result["bf_used_count"] = bf_used_count
    result["bf_success_count"] = bf_success_count
    result["bf_total_candidates_avg"] = (
        float(np.mean(bf_total_candidates)) if bf_total_candidates else 0.0
    )
    result["bf_time_s"] = bf_time_total

    return result


# ---------------------------------------------------------------------------
# Evaluate one config (all methods)
# ---------------------------------------------------------------------------

def evaluate_config(
    config_key: str,
    config_args,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    env_proto,
    all_episodes: Dict[int, List[List]],
    base_args,
) -> Dict[str, Any]:
    """Evaluate all 4 methods for one config."""
    seeds = [int(s) for s in base_args.seeds.split(",")]
    config_result: Dict[str, Any] = {"config": config_key}

    for method in METHOD_ORDER:
        ep_results = []
        for seed in seeds:
            for ep_idx, requests in enumerate(all_episodes[seed]):
                res = _eval_method_on_episode(
                    method, agent_c, agent_r, requests, env_proto, config_args, ep_idx,
                )
                ep_results.append(res)

        scalar_keys = [
            "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
            "avg_fs", "avg_waste", "avg_path_len_km",
        ]
        agg: Dict[str, Any] = {}
        for key in scalar_keys:
            vals = [float(r[key]) for r in ep_results]
            agg[key] = float(np.mean(vals))
        for key in ["reason_counter", "split_counter", "server_counter", "mod_counter"]:
            c = Counter()
            for r in ep_results:
                c.update(r.get(key, {}))
            agg[key] = dict(c)

        # no_suitable_block ratio
        reason = agg.get("reason_counter", {})
        total_reasons = sum(reason.values())
        nsb = reason.get("no_suitable_block", 0)
        agg["no_suitable_block_ratio"] = nsb / max(total_reasons, 1)

        # server_overload ratio
        so = reason.get("server_overload", 0) + reason.get("server_saturated", 0)
        agg["server_overload_ratio"] = so / max(total_reasons, 1)

        # Mask stats (from agent/agent_bf trajectory)
        if method in ("agent", "agent_bf"):
            all_c = []; zero_c = 0; req_c = 0
            all_r = []
            for r in ep_results:
                if "mask_valid_c_counts" in r:
                    all_c.extend(r["mask_valid_c_counts"])
                    zero_c += r.get("mask_zero_c_count", 0)
                    req_c += r["total"]
                if "mask_valid_r_counts" in r:
                    all_r.extend(r["mask_valid_r_counts"])
            agg["no_valid_c_rate"] = zero_c / max(req_c, 1)
            agg["avg_valid_c_actions"] = float(np.mean(all_c)) if all_c else 0.0
            agg["avg_valid_r_actions"] = float(np.mean(all_r)) if all_r else 0.0

        # BF stats (agent_bf only)
        if method == "agent_bf":
            bf_used = sum(r.get("bf_used_count", 0) for r in ep_results)
            bf_success = sum(r.get("bf_success_count", 0) for r in ep_results)
            bf_time = sum(r.get("bf_time_s", 0.0) for r in ep_results)
            agg["bf_used_total"] = bf_used
            agg["bf_success_total"] = bf_success
            agg["bf_time_s"] = bf_time
            agg["bf_trigger_rate"] = bf_used / max(sum(r["total"] for r in ep_results), 1)

        # Runtime
        agg["runtime_s"] = sum(
            r.get("bf_time_s", 0.0) for r in ep_results
        ) + 0.0  # base runtime negligible compared to BF

        config_result[method] = agg

    return config_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_label(params: Dict[str, Any]) -> str:
    return f"K{params['k_paths']}_B{params['max_blocks']}_{params['block_sort_strategy']}"


def _fmt_pct(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    items = sorted(c.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v / total:.1%}" for k, v in items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.20)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/r_action_space_sweep.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/r_action_space_sweep.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Load models once
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    # Build sweep grid
    axes = list(SWEEP_AXES.keys())
    values = list(SWEEP_AXES.values())
    grid = list(product(*values))
    total_configs = len(grid)
    print(f"\nR Action-Space Expansion Sweep")
    print(f"Axes: {axes}")
    print(f"Total configs: {total_configs}  Seeds: {seeds}  Eps/seed: {args.episodes}")
    print(f"Methods: {METHOD_ORDER}")
    print(f"Requests/ep: {args.requests_per_episode}  Slots: {args.num_slots}")
    print(f"{'=' * 70}")

    # Pre-generate episodes (independent of R params — same traffic for all configs)
    env_proto_base = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=10, block_sort_strategy="mixed",
        k=3,
    )
    all_episodes: Dict[int, List[List]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto_base.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto_base, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps

    # ------------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for idx, combo in enumerate(grid):
        params = dict(zip(axes, combo))
        label = _config_label(params)

        config_args = argparse.Namespace(**{
            **vars(args),
            "k_paths": params["k_paths"],
            "max_blocks": params["max_blocks"],
            "block_sort_strategy": params["block_sort_strategy"],
        })

        # Build env prototype for this config
        env_proto = make_env(
            topology=args.topology, num_slots=args.num_slots,
            num_servers=args.num_servers, seed=42,
            slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=params["max_blocks"],
            block_sort_strategy=params["block_sort_strategy"],
            k=params["k_paths"],
        )

        config_result = evaluate_config(
            label, config_args, agent_c, agent_r, env_proto, all_episodes, args,
        )
        all_results.append(config_result)

        # Progress
        agent_data = config_result["agent"]
        agent_bf_data = config_result["agent_bf"]
        elapsed = time.time() - t_start
        eta = elapsed / (idx + 1) * (total_configs - idx - 1)
        print(
            f"[{idx + 1:2d}/{total_configs}] {label}  "
            f"blk={agent_data['blocking_rate']:.3f}→{agent_bf_data['blocking_rate']:.3f}(BF)  "
            f"nsb={agent_data['no_suitable_block_ratio']:.2f}→{agent_bf_data['no_suitable_block_ratio']:.2f}  "
            f"noC={agent_data['no_valid_c_rate']:.2f}  "
            f"avgRacts={agent_data['avg_valid_r_actions']:.0f}  "
            f"elap={elapsed:.0f}s  ETA={eta:.0f}s"
        )

    total_elapsed = time.time() - t_start
    print(f"\nDone. Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Identify best config
    # ------------------------------------------------------------------
    best_config = None
    best_delta = -999.0
    for cr in all_results:
        agent_blk = cr["agent"]["blocking_rate"]
        agent_bf_blk = cr["agent_bf"]["blocking_rate"]
        delta = agent_blk - agent_bf_blk
        if delta > best_delta:
            best_delta = delta
            best_config = cr

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_output = {
        "args": vars(args),
        "sweep_axes": SWEEP_AXES,
        "total_configs": total_configs,
        "total_time_s": total_elapsed,
        "best_delta_blocking": best_delta,
        "best_config": best_config["config"] if best_config else None,
        "all_results": _clean(all_results),
    }
    out_json.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown output
    # ------------------------------------------------------------------
    md: List[str] = []
    md.append("# R Action-Space Expansion Sweep\n\n")
    md.append(f"**Topology:** {args.topology}  **Slots:** {args.num_slots}  "
              f"**Servers:** {args.num_servers}  **Req/ep:** {args.requests_per_episode}\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}\n\n")
    md.append(f"**Total configs:** {total_configs}  "
              f"**Total time:** {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)\n\n")

    md.append("## Sweep Axes\n\n")
    md.append("| Axis | Values |\n|---|---|\n")
    for axis, vals in SWEEP_AXES.items():
        md.append(f"| {axis} | {vals} |\n")

    # Results table
    md.append("\n## Results\n\n")
    hdr = (
        "| Config | AgentBlk | BFBlk | DFBlk | IWDBlk | "
        "Δ(BF) | NSB(Agent) | NSB(BF) | noC% | avgRacts | "
        "AgentDelay | BFDelay | AgentFS | BFFS | BFused% |\n"
    )
    md.append(hdr)
    md.append("|" + "---|" * 15 + "\n")
    for cr in all_results:
        a = cr["agent"]; bf = cr["agent_bf"]; df = cr["df"]; iwd = cr["iwd"]
        delta = a["blocking_rate"] - bf["blocking_rate"]
        md.append(
            f"| {cr['config']} "
            f"| {a['blocking_rate']:.4f} | {bf['blocking_rate']:.4f} "
            f"| {df['blocking_rate']:.4f} | {iwd['blocking_rate']:.4f} "
            f"| {delta:+.4f} "
            f"| {a['no_suitable_block_ratio']:.3f} | {bf['no_suitable_block_ratio']:.3f} "
            f"| {a['no_valid_c_rate']:.3f} | {a['avg_valid_r_actions']:.0f} "
            f"| {a['avg_delay_ms']:.1f} | {bf['avg_delay_ms']:.1f} "
            f"| {a['avg_fs']:.2f} | {bf['avg_fs']:.2f} "
            f"| {bf.get('bf_trigger_rate', 0):.1%} |\n"
        )

    # Best config detail
    if best_config:
        md.append(f"\n## Best Config: {best_config['config']}  "
                  f"(Δblocking = {best_delta:+.4f})\n\n")
        for method in METHOD_ORDER:
            m = best_config[method]
            md.append(f"### {METHOD_LABELS[method]}\n\n")
            md.append(f"- blocking: {m['blocking_rate']:.4f}\n")
            md.append(f"- no_suitable_block: {m['no_suitable_block_ratio']:.3f}\n")
            md.append(f"- server_overload: {m['server_overload_ratio']:.3f}\n")
            md.append(f"- delay: {m['avg_delay_ms']:.1f} ms\n")
            md.append(f"- avg_fs: {m['avg_fs']:.2f}\n")
            md.append(f"- path: {m['avg_path_len_km']:.1f} km\n")
            md.append(f"- modulations: {_fmt_pct(m.get('mod_counter', {}))}\n")
            if method == "agent_bf":
                md.append(f"- BF trigger rate: {m.get('bf_trigger_rate', 0):.1%}\n")
                md.append(f"- BF success rate: {m.get('bf_success_total', 0)} / {m.get('bf_used_total', 0)}\n")
                md.append(f"- BF time: {m.get('bf_time_s', 0):.1f}s\n")
            md.append("\n")

    # Failure reason summary
    md.append("## Failure Reason Summary\n\n")
    md.append("| Config | Method | no_suitable_block | server_overload | other |\n")
    md.append("|---|---|---|---|---|\n")
    for cr in all_results:
        for method in METHOD_ORDER:
            m = cr[method]
            reason = m.get("reason_counter", {})
            total_r = sum(reason.values())
            nsb = reason.get("no_suitable_block", 0)
            so = reason.get("server_overload", 0) + reason.get("server_saturated", 0)
            other = total_r - nsb - so
            md.append(
                f"| {cr['config']} | {METHOD_LABELS[method]} "
                f"| {nsb / max(total_r, 1):.3f} "
                f"| {so / max(total_r, 1):.3f} "
                f"| {other / max(total_r, 1):.3f} |\n"
            )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total configs: {total_configs}")
    print(f"Best Δblocking (BF vs Agent): {best_delta:+.4f}")
    if best_config:
        print(f"  Config: {best_config['config']}")
        print(f"  Agent blk={best_config['agent']['blocking_rate']:.4f}  "
              f"BF blk={best_config['agent_bf']['blocking_rate']:.4f}")
    print(f"\nCriterion: Δblocking ≥ 0.03 (3pp) → {'✓ expand R space' if best_delta >= 0.03 else '✗ R expansion insufficient'}")

    # Per-axis breakdown
    print(f"\nPer-axis Δblocking (BF vs Agent):")
    for axis in SWEEP_AXES:
        by_val: Dict[Any, List[float]] = {}
        for cr in all_results:
            # Parse value from config label
            parts = cr["config"].split("_")
            if axis == "k_paths":
                val = parts[0].replace("K", "")
            elif axis == "max_blocks":
                val = parts[1].replace("B", "")
            elif axis == "block_sort_strategy":
                val = parts[2]
            else:
                continue
            delta = cr["agent"]["blocking_rate"] - cr["agent_bf"]["blocking_rate"]
            by_val.setdefault(val, []).append(delta)
        print(f"  {axis}:")
        for val in SWEEP_AXES[axis]:
            str_val = str(val)
            deltas = by_val.get(str_val, [])
            avg_d = np.mean(deltas) if deltas else 0.0
            print(f"    {str_val}: Δ={avg_d:+.4f}")


if __name__ == "__main__":
    main()
