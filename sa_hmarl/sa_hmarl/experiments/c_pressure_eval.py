"""Comprehensive C-side evaluation: Original / PressureAware / TopK-Rerank / Oracle / Baselines.

Evaluates on identical test episodes for fair comparison:
  - Original-Agent-C      (existing checkpoint)
  - PressureAware-Agent-C (new checkpoint)
  - TopK-Rerank-Original  (inference-time rerank on Original)
  - True-Oracle-Step      (deepcopy env, execute R-selected action for each C candidate)
  - Oracle-R-Query        (cheap R-query heuristic, not an upper bound)
  - Oracle-Pressure       (pressure-based selection)
  - IWD-C, DF-C, Greedy-C, WO-C, RF-C

Fixed R backend: ppo_r_deeprmsa_bc_snap24_best.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_joint_alternating import _build_eval_set
from sa_hmarl.training.utils import (
    compute_agent_c_reward,
    compute_agent_c_reward_delay_aware,
    generate_requests,
    make_env,
)
from sa_hmarl.utils.checkpoint import load_checkpoint


SEEDS = [42, 123, 456, 789, 2024]


def _load_frozen_ppo_r(ckpt_path: str, mod_reg, device: str):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=ckpt.get("agent_r_feature_mode", "default"),
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    for p in agent.policy_net.parameters():
        p.requires_grad = False
    agent.policy_net.eval()
    return agent


def _load_agent_c(ckpt_path: str, device: str, feature_mode: Optional[str] = None):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    input_dim = ckpt.get("input_dim", 17)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = feature_mode or ckpt_args.get(
        "agent_c_feature_mode",
        ckpt.get("agent_c_feature_mode", "default"),
    )
    agent = PPOAgentC(
        input_dim=input_dim,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    for p in agent.policy_net.parameters():
        p.requires_grad = False
    agent.policy_net.eval()
    return agent


def _compute_objective(blocking_rate: float, avg_delay_ms: float) -> float:
    return blocking_rate + 0.5 * (avg_delay_ms / 100.0)


def _finite(value, default: float = 1e9) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(val):
        return default
    return val


def _candidate_pressure_score(obs_c: Dict, action_idx: int, num_slots: int) -> float:
    """Long-term pressure proxy for one split/server candidate.

    Lower is better.  This deliberately penalizes tight RMSA feasibility
    margins before delay so that the oracle is not just a one-step delay
    greedy policy.
    """
    feat = obs_c["candidate_features"][int(action_idx)]
    spec = feat.get("spectrum_summary", [])
    if isinstance(spec, np.ndarray):
        spec = spec.tolist()

    feasible = _finite(feat.get("feasible_count", 0.0), 0.0)
    best_fs = _finite(feat.get("best_fs_estimate", num_slots), float(num_slots))
    safe_fs = _finite(feat.get("safe_fs_estimate", best_fs), best_fs)

    lfb_max = max(_finite(spec[0] if len(spec) > 0 else num_slots, float(num_slots)), 1.0)
    lfb_p25 = max(_finite(spec[3] if len(spec) > 3 else lfb_max, lfb_max), 1.0)
    frag_mean = np.clip(_finite(spec[4] if len(spec) > 4 else 0.0, 0.0), 0.0, 1.0)
    free_mean = np.clip(_finite(spec[6] if len(spec) > 6 else 0.0, 0.0), 0.0, 1.0)

    best_pressure = min(best_fs / lfb_max, 10.0)
    safe_pressure = min(safe_fs / lfb_p25, 10.0)
    scarcity = 1.0 / (1.0 + max(feasible, 0.0))
    server_util = np.clip(_finite(feat.get("server_utilization", 0.0), 0.0), 0.0, 2.0)

    return (
        1.2 * scarcity
        + 0.9 * best_pressure
        + 0.9 * safe_pressure
        + 0.5 * frag_mean
        + 0.3 * (1.0 - free_mean)
        + 0.4 * server_util
    )


def _true_oracle_step_score(info: Dict, obs_c: Dict, action_idx: int, req, args) -> Tuple:
    """Score a candidate after a real env_try.step execution.

    Tuple ordering is lexicographic.  We prioritize current feasibility, then
    spectrum pressure, then delay/waste.  This is still a one-step diagnostic,
    not a full future-aware planner, but it is a real environment trial rather
    than a cheap R-query heuristic.
    """
    pressure = _candidate_pressure_score(obs_c, int(action_idx), args.num_slots)
    if info.get("success", False):
        delay = _finite(info.get("delay_ms", 1e9))
        delay_ratio = delay / max(_finite(getattr(req, "deadline_ms", args.deadline_max), args.deadline_max), 1.0)
        waste = _finite(info.get("block_waste", 1.0), 1.0)
        fs = _finite(info.get("num_slots", 1e9), 1e9)
        path = _finite(info.get("path_dist_km", 1e9), 1e9)
        deadline_violation = 1 if delay > getattr(req, "deadline_ms", args.deadline_max) else 0
        return (0, deadline_violation, pressure, delay_ratio, waste, fs, path)

    reason = info.get("reason", "unknown")
    reason_rank = {
        "server_overload": 1,
        "server_saturated": 1,
        "no_suitable_block": 2,
        "fs_too_large": 3,
        "deadline_infeasible": 4,
    }.get(reason, 9)
    return (1, reason_rank, pressure, 1e9, 1e9, 1e9, 1e9)


def select_true_oracle_step(env, req, obs_c: Dict, mask: np.ndarray, frozen_r, args) -> Optional[int]:
    """True one-step Oracle-C diagnostic.

    For each valid (split, server), deep-copy the current environment, let the
    frozen R actor choose its deterministic RMSA action, execute env_try.step,
    and score the real resulting info.
    """
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None

    best_idx = None
    best_score = None
    num_servers = len(env.mec.servers)

    for idx in valid:
        idx = int(idx)
        split_id, server_id = decode_agent_c_action(idx, num_servers)
        env_try = copy.deepcopy(env)
        try:
            obs_r = build_agent_r_observation(env_try, req, split_id, server_id)
            action_idx_r = frozen_r.select_action(obs_r, deterministic=True)
            action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env_try.max_blocks
            )
            _, _, _, info_try = env_try.step(action_c=(split_id, server_id), action_r=action_r)
        except Exception as exc:
            info_try = {"success": False, "reason": f"oracle_exception:{type(exc).__name__}"}

        score = _true_oracle_step_score(info_try, obs_c, idx, req, args)
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def evaluate_method(
    method_name: str,
    agent_c,
    frozen_r,
    episodes_list: List[List],
    env_prototype,
    args,
    top_k: int = 0,
    pressure_alpha: float = 2.0,
) -> Dict:
    """Evaluate one C method on pre-generated episodes."""
    total = 0
    blocked = 0
    total_reward = 0.0
    successes = 0
    total_delay = 0.0
    deadline_met = 0
    total_fs = 0.0
    total_path = 0.0
    reasons = Counter()
    mods = Counter()
    splits = Counter()
    servers = Counter()
    server_overloads = 0
    no_suitable_blocks = 0

    for requests in episodes_list:
        env = make_env(
            topology=args.topology,
            num_servers=args.num_servers,
            seed=args.seed,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )
        env.reset(requests)

        server_selected_count = np.zeros(args.num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            mask = obs_c["agent_c_mask"]

            if method_name == "original":
                action_idx_c = agent_c.select_action(obs_c, deterministic=True)
            elif method_name == "pressureaware":
                action_idx_c = agent_c.select_action(obs_c, deterministic=True)
            elif method_name == "topk_rerank":
                action_idx_c = agent_c.select_action_topk_rerank(
                    obs_c, top_k=top_k, pressure_alpha=pressure_alpha, deterministic=True
                )
            elif method_name == "true_oracle_step":
                action_idx_c = select_true_oracle_step(env, req, obs_c, mask, frozen_r, args)
            elif method_name == "oracle_r_query":
                action_idx_c = select_offloading_action(
                    method_name, env, req, obs_c, mask,
                    agent_r=frozen_r,
                )
            elif method_name == "oracle_pressure":
                action_idx_c = select_offloading_action(
                    method_name, env, req, obs_c, mask,
                )
            else:
                # Baselines
                action_idx_c = select_offloading_action(
                    method_name, env, req, obs_c, mask,
                    server_selected_count=server_selected_count,
                )

            if action_idx_c is not None:
                split_id, server_id = decode_agent_c_action(
                    action_idx_c, len(env.mec.servers)
                )
                splits[f"split{split_id}"] += 1
                servers[f"s{server_id}"] += 1
                server_selected_count[server_id] += 1
            else:
                split_id, server_id = 0, 0

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = frozen_r.select_action(obs_r, deterministic=True)
            action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )
            _, _, _, info = env.step(action_c=(split_id, server_id), action_r=action_r)
            server = env.mec.servers[server_id]

            reward_c = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )

            total += 1
            total_reward += reward_c
            if info.get("success", False):
                successes += 1
                delay = info.get("delay_ms", 0.0)
                total_delay += delay
                if delay <= req.deadline_ms:
                    deadline_met += 1
                total_fs += info.get("num_slots", 0)
                total_path += info.get("path_dist_km", 0.0)
                mods[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reasons[info.get("reason", "unknown")] += 1
                if info.get("reason") == "server_overload":
                    server_overloads += 1
                elif info.get("reason") == "no_suitable_block":
                    no_suitable_blocks += 1

    s = max(successes, 1)
    blocking_rate = blocked / max(total, 1)
    avg_delay_ms = total_delay / s
    objective = _compute_objective(blocking_rate, avg_delay_ms)
    total_cnt = sum(splits.values())
    srv_cnt = sum(servers.values())

    return {
        "blocking_rate": blocking_rate,
        "success_rate": successes / max(total, 1),
        "avg_reward": total_reward / max(total, 1),
        "avg_delay_ms": avg_delay_ms,
        "deadline_sat_rate": deadline_met / max(total, 1),
        "avg_fs": total_fs / s,
        "avg_path_len_km": total_path / s,
        "objective": objective,
        "reasons": dict(reasons),
        "mods": dict(mods),
        "splits": dict(splits),
        "servers": dict(servers),
        "split_distribution": {k: v / total_cnt for k, v in splits.items()} if total_cnt else {},
        "server_distribution": {k: v / srv_cnt for k, v in servers.items()} if srv_cnt else {},
        "server_overload_ratio": server_overloads / max(total, 1),
        "no_suitable_block_ratio": no_suitable_blocks / max(total, 1),
    }


def run_eval(args):
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    frozen_r = _load_frozen_ppo_r(args.frozen_r_checkpoint, mod_reg, args.device)
    print(f"Loaded frozen PPO-R from {args.frozen_r_checkpoint}")

    # Load agents
    agents = {}
    if args.original_c_ckpt and Path(args.original_c_ckpt).exists():
        agents["original"] = _load_agent_c(args.original_c_ckpt, args.device)
        print(f"Loaded Original Agent-C from {args.original_c_ckpt}")
    if args.pressure_c_ckpt and Path(args.pressure_c_ckpt).exists():
        agents["pressureaware"] = _load_agent_c(args.pressure_c_ckpt, args.device)
        print(f"Loaded PressureAware Agent-C from {args.pressure_c_ckpt}")

    # Build test episodes
    env_proto = make_env(
        topology=args.topology,
        num_servers=args.num_servers,
            seed=args.seed,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )
    all_episodes = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes_per_seed):
            ep = generate_requests(
                env_proto,
                rng,
                src_node=rng.randint(0, env_proto.net.NUM_NODES) if args.src_node < 0 else args.src_node,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min,
                holding_max=args.holding_max,
                deadline_min=args.deadline_min,
                deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb,
                size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min,
                edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits,
                split_profile=args.split_profile,
            )
            episodes.append(ep)
        all_episodes[seed] = episodes

    methods = []
    if "original" in agents:
        methods.extend(["original", "topk_rerank"])
    if "pressureaware" in agents:
        methods.append("pressureaware")
    methods.extend(["true_oracle_step", "oracle_r_query", "oracle_pressure"])
    methods.extend(["iwd", "df", "greedy", "wo", "rf"])

    results = {m: [] for m in methods}

    total_episodes = 0
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ({args.episodes_per_seed} episodes) ---")
        episodes = all_episodes[seed]
        total_episodes += len(episodes)

        for method in methods:
            print(f"  Evaluating {method}...", end="", flush=True)
            kwargs = {}
            if method == "original":
                kwargs["agent_c"] = agents["original"]
            elif method == "pressureaware":
                kwargs["agent_c"] = agents["pressureaware"]
            elif method == "topk_rerank":
                kwargs["agent_c"] = agents["original"]
                kwargs["top_k"] = args.top_k
                kwargs["pressure_alpha"] = args.pressure_alpha

            res = evaluate_method(
                method,
                kwargs.get("agent_c"),
                frozen_r,
                episodes,
                env_proto,
                args,
                top_k=kwargs.get("top_k", 0),
                pressure_alpha=kwargs.get("pressure_alpha", 2.0),
            )
            results[method].append(res)
            print(f" blk={res['blocking_rate']:.4f} delay={res['avg_delay_ms']:.2f}ms obj={res['objective']:.4f}")

    # Aggregate
    print("\n" + "=" * 80)
    print(f"AGGREGATE RESULTS ({len(SEEDS)} seeds x {args.episodes_per_seed} episodes = {total_episodes} episodes)")
    print("=" * 80)
    agg = {}
    for method in methods:
        ress = results[method]
        agg[method] = {
            "blocking_rate": float(np.mean([r["blocking_rate"] for r in ress])),
            "blocking_rate_std": float(np.std([r["blocking_rate"] for r in ress])),
            "avg_delay_ms": float(np.mean([r["avg_delay_ms"] for r in ress])),
            "avg_delay_ms_std": float(np.std([r["avg_delay_ms"] for r in ress])),
            "objective": float(np.mean([r["objective"] for r in ress])),
            "objective_std": float(np.std([r["objective"] for r in ress])),
            "avg_reward": float(np.mean([r["avg_reward"] for r in ress])),
            "deadline_sat_rate": float(np.mean([r["deadline_sat_rate"] for r in ress])),
            "server_overload_ratio": float(np.mean([r["server_overload_ratio"] for r in ress])),
            "no_suitable_block_ratio": float(np.mean([r["no_suitable_block_ratio"] for r in ress])),
        }
        # Aggregate split/server distributions
        all_splits = defaultdict(list)
        all_servers = defaultdict(list)
        for r in ress:
            for k, v in r.get("split_distribution", {}).items():
                all_splits[k].append(v)
            for k, v in r.get("server_distribution", {}).items():
                all_servers[k].append(v)
        agg[method]["split_distribution"] = {k: float(np.mean(v)) for k, v in all_splits.items()}
        agg[method]["server_distribution"] = {k: float(np.mean(v)) for k, v in all_servers.items()}

    # Print table
    print(f"\n{'Method':<20} {'Blocking':>10} {'Delay(ms)':>10} {'Objective':>10} {'Overload%':>10} {'NoBlock%':>10}")
    print("-" * 80)
    for method in methods:
        a = agg[method]
        print(f"{method:<20} {a['blocking_rate']:>10.4f} {a['avg_delay_ms']:>10.2f} "
              f"{a['objective']:>10.4f} {a['server_overload_ratio']*100:>9.1f}% {a['no_suitable_block_ratio']*100:>9.1f}%")
    print(f"\nTotal episodes evaluated: {total_episodes} ({len(SEEDS)} seeds x {args.episodes_per_seed} episodes)")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "c_pressure_eval.json"
    with open(out_json, "w") as f:
        json.dump({"aggregate": agg, "per_seed": results, "args": vars(args)}, f, indent=2)

    out_md = out_dir / "c_pressure_eval.md"
    with open(out_md, "w") as f:
        f.write("# C-Side Pressure Diagnostic Evaluation\n\n")
        f.write(f"Topology: {args.topology}, Servers: {args.num_servers}, Slots: {args.num_slots}\n\n")
        f.write(f"| Method | Blocking | Delay(ms) | Objective | Overload% | NoBlock% |\n")
        f.write(f"|--------|----------|-----------|-----------|-----------|----------|\n")
        for method in methods:
            a = agg[method]
            f.write(f"| {method} | {a['blocking_rate']:.4f} | {a['avg_delay_ms']:.2f} | "
                    f"{a['objective']:.4f} | {a['server_overload_ratio']*100:.1f}% | {a['no_suitable_block_ratio']*100:.1f}% |\n")
        f.write("\n## Split Distributions\n\n")
        for method in methods:
            f.write(f"**{method}**: {agg[method]['split_distribution']}\n\n")

    print(f"\nSaved results to {out_json} and {out_md}")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--max_blocks", type=int, default=5)
    parser.add_argument("--block_sort_strategy", type=str, default="size_desc",
                        choices=["size_desc", "waste_asc", "start_asc", "center_asc", "mixed"])
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--src_node", type=int, default=-1,
                        help="Use -1 for random source per episode; otherwise fixed source node.")
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
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--frozen_r_checkpoint", type=str, default=None)
    parser.add_argument("--original_c_ckpt", type=str, default=None)
    parser.add_argument("--pressure_c_ckpt", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--pressure_alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes_per_seed", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="sa_hmarl/experiments/c_pressure_eval_hard")
    args = parser.parse_args()

    if args.frozen_r_checkpoint is None:
        root = Path(__file__).resolve().parents[2]
        args.frozen_r_checkpoint = str(root / "checkpoints" / "ppo_r_deeprmsa_bc_snap24_best.pt")
    if args.original_c_ckpt is None:
        root = Path(__file__).resolve().parents[2]
        args.original_c_ckpt = str(root / "checkpoints" / "agent_c_meanpath_cgap_msval_best.pt")

    run_eval(args)


if __name__ == "__main__":
    main()
