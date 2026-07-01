"""Unified evaluation of methods on identical test episodes.

Methods:
  1. Stage2 Frozen-R+PPO-C  (agent_c_frozen_r_best.pt + v2_cshape R)
  2. Stage3 v3 MAPPO KL=0.05 (plan_stage3_mappo_finetune_v3_*_best.pt)
  3. Stage3 v4 MAPPO KL=0.01 (plan_stage3_mappo_finetune_v4_*_best.pt)
  4. Compute-Greedy + PPO-R   (greedy C + v2_cshape R)
  5. Separate C+R             (separate_agent_c_best.pt + separate_agent_r.pt)
  6. Compute-Greedy + KSP-BF  (greedy C + KSP first-fit)
  7. Compute-Greedy + DeepRMSA-A2C/A3C-style (optional)

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_unified \
        --topologies metro24_bottleneck --seeds 42,123,456,789,2024
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _load_dqn_c(ckpt_path: str, device: str = "cpu") -> AgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = AgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.q_net.load_state_dict(ckpt["model_state"])
    if "target_state" in ckpt:
        agent.target_net.load_state_dict(ckpt["target_state"])
    else:
        agent.target_net.load_state_dict(ckpt["model_state"])
    agent.q_net.eval()
    agent.target_net.eval()
    return agent


def _load_dqn_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> AgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = AgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.q_net.load_state_dict(ckpt["model_state"])
    if "target_state" in ckpt:
        agent.target_net.load_state_dict(ckpt["target_state"])
    else:
        agent.target_net.load_state_dict(ckpt["model_state"])
    agent.q_net.eval()
    agent.target_net.eval()
    return agent


def _load_deep_rmsa(ckpt_path: str, env, mod_reg: ModulationRegistry,
                    device: str = "cpu") -> DeepRMSAAgent:
    """Load a topology-specific DeepRMSA-style A2C checkpoint."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_nodes = ckpt.get("num_nodes")
    if ckpt_nodes is not None and ckpt_nodes != env.net.NUM_NODES:
        raise ValueError(
            "DeepRMSA checkpoint topology mismatch: "
            f"ckpt has num_nodes={ckpt_nodes}, "
            f"but env has NUM_NODES={env.net.NUM_NODES}. "
            "Please pass a checkpoint trained for this topology."
        )

    ckpt_slots = ckpt.get("num_slots")
    if ckpt_slots is not None and ckpt_slots != env.net.num_slots:
        raise ValueError(
            "DeepRMSA checkpoint slot mismatch: "
            f"ckpt has num_slots={ckpt_slots}, "
            f"but env has num_slots={env.net.num_slots}."
        )

    agent = DeepRMSAAgent(
        num_nodes=env.net.NUM_NODES,
        num_slots=env.net.num_slots,
        k_path=ckpt.get("k_path", env.k),
        m_blocks=ckpt.get("m_blocks", env.max_blocks),
        mod_registry=mod_reg,
        gamma=ckpt.get("gamma", 0.95),
        device=device,
    )
    agent.load_state_dict(ckpt)
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def _select_c_ppo(agent, obs_c) -> Optional[int]:
    return agent.select_action(obs_c, deterministic=True)


def _select_c_dqn(agent, obs_c) -> Optional[int]:
    return agent.select_action(obs_c, epsilon=0.0)


def _select_c_greedy(obs_c, valid) -> Optional[int]:
    """Compute-greedy: pick the server with minimum edge_compute_ms."""
    if len(valid) == 0:
        return None
    best = int(valid[0])
    best_val = obs_c["candidate_features"][best]["edge_compute_ms"]
    if best_val == float("inf"):
        best_val = 1e9
    for idx in valid[1:]:
        val = obs_c["candidate_features"][int(idx)]["edge_compute_ms"]
        if val == float("inf"):
            val = 1e9
        if val < best_val:
            best = int(idx)
            best_val = val
    return best


def _select_r_ppo(agent, obs_r) -> Optional[int]:
    return agent.select_action(obs_r, deterministic=True)


def _select_r_dqn(agent, obs_r) -> Optional[int]:
    return agent.select_action(obs_r, epsilon=0.0)


def _select_r_deep(agent, obs_r) -> Optional[int]:
    return agent.select_action(obs_r)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def _fmt_pct(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in counter.most_common())


def _fmt_cnt(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in counter.most_common())


def evaluate_method(env, agent_c, agent_r,
                    c_type: str, r_type: str,
                    episodes_data: List[List],
                    waste_coef: float = 0.8) -> Dict[str, Any]:
    """Run one method over all evaluation episodes."""
    total = 0
    blocked = 0
    success = 0
    total_reward = 0.0
    total_delay = 0.0
    total_fs = 0.0
    total_waste = 0.0
    total_path = 0.0
    deadline_met = 0

    mod_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    server_success_counter = Counter()
    server_overload_counter = Counter()
    reason_counter = Counter()

    for requests in episodes_data:
        env.reset(requests)
        for req in requests:
            # --- Agent-C ---
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]
            valid_c = np.where(mask_c)[0]

            if c_type == "ppo":
                action_idx_c = _select_c_ppo(agent_c, obs_c)
            elif c_type == "dqn":
                action_idx_c = _select_c_dqn(agent_c, obs_c)
            elif c_type == "greedy":
                action_idx_c = _select_c_greedy(obs_c, valid_c)
            else:
                raise ValueError(f"Unknown c_type: {c_type}")

            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, len(env.mec.servers)
            )
            split_id, server_id = action_c

            # Track split/server choice (even for blocked requests)
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1

            # --- Agent-R ---
            obs_r = build_agent_r_observation(env, req, split_id, server_id)

            if r_type == "ppo":
                action_idx_r = _select_r_ppo(agent_r, obs_r)
            elif r_type == "dqn":
                action_idx_r = _select_r_dqn(agent_r, obs_r)
            elif r_type == "deep_rmsa":
                action_idx_r = _select_r_deep(agent_r, obs_r)
            elif r_type == "ksp_bf":
                action_idx_r = ksp_bf_action(obs_r)
            else:
                raise ValueError(f"Unknown r_type: {r_type}")

            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                )

            # --- Step ---
            _, _, _, info = env.step(action_c, action_r)
            total += 1

            # Reward
            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, waste_coef, server.utilization
            )
            total_reward += reward

            if info.get("success", False):
                success += 1
                server_success_counter[f"s{server_id}"] += 1
                delay = info.get("delay_ms", 0.0)
                total_delay += delay
                if delay <= req.deadline_ms:
                    deadline_met += 1
                total_fs += info.get("num_slots", 0)
                total_waste += info.get("block_waste", 0.0)
                total_path += info.get("path_dist_km", 0.0)
                mod_counter[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1
                if reason in ("server_overload", "server_saturated"):
                    server_overload_counter[f"s{server_id}"] += 1

    n = max(total, 1)
    s = max(success, 1)
    return {
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "deadline_sat_rate": deadline_met / n,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "server_overload_ratio": reason_counter.get("server_overload", 0) / max(blocked, 1),
        "no_suitable_block_ratio": reason_counter.get("no_suitable_block", 0) / max(blocked, 1),
        "mod_counter": dict(mod_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "server_success_counter": dict(server_success_counter),
        "server_overload_counter": dict(server_overload_counter),
        "reason_counter": dict(reason_counter),
    }


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def is_dominated(a: Dict, b: Dict) -> bool:
    """Return True if method `a` is strictly dominated by method `b`.

    Domination means b is better-or-equal on all four dimensions and strictly
    better on at least one.
    """
    better = 0
    equal = 0
    for key, direction in [("blocking_rate", -1), ("avg_reward", +1),
                           ("avg_delay_ms", -1),  ("avg_fs", -1)]:
        va = a[key]
        vb = b[key]
        if direction > 0:
            if vb > va:
                better += 1
            elif abs(vb - va) < 1e-9:
                equal += 1
            else:
                return False
        else:
            if vb < va:
                better += 1
            elif abs(vb - va) < 1e-9:
                equal += 1
            else:
                return False
    return better >= 1 and better + equal == 4


def compute_pareto_front(methods: Dict[str, Dict]):
    """Return the set of non-dominated method names."""
    dominated = set()
    names = list(methods.keys())
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                continue
            if is_dominated(methods[na], methods[nb]):
                dominated.add(na)
                break
    return [n for n in names if n not in dominated]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation of SA-HMARL methods")
    parser.add_argument("--topologies", type=str, default="metro24_bottleneck")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=60)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None,
                        help="If set, write results to this JSON file")
    parser.add_argument("--stage2_c_checkpoint", type=str, default=None)
    parser.add_argument("--stage2_r_checkpoint", type=str, default=None)
    parser.add_argument("--stage3_v3_c_checkpoint", type=str, default=None)
    parser.add_argument("--stage3_v3_r_checkpoint", type=str, default=None)
    parser.add_argument("--stage3_v4_c_checkpoint", type=str, default=None)
    parser.add_argument("--stage3_v4_r_checkpoint", type=str, default=None)
    parser.add_argument("--ppo_r_checkpoint", type=str, default=None,
                        help="PPO-R checkpoint used by Compute-Greedy + PPO-R")
    parser.add_argument("--separate_c_checkpoint", type=str, default=None)
    parser.add_argument("--separate_r_checkpoint", type=str, default=None)
    parser.add_argument("--deep_rmsa_checkpoint", type=str, default=None,
                        help="Optional DeepRMSA checkpoint. If provided, adds "
                             "Compute-Greedy + DeepRMSA-A2C/A3C-style to the table.")
    args = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    # ------------------------------------------------------------------
    # Define the 6 methods
    # ------------------------------------------------------------------
    method_specs = [
        {
            "name": "Stage2 Frozen-R+PPO-C",
            "c_type": "ppo",
            "r_type": "ppo",
            "c_ckpt": args.stage2_c_checkpoint or str(ckpt_dir / "agent_c_frozen_r_best.pt"),
            "r_ckpt": args.stage2_r_checkpoint or str(ckpt_dir / "joint_mappo_v2_cshape_metro24_fs002_r_best.pt"),
        },
        {
            "name": "Stage3 v3 MAPPO KL=0.05",
            "c_type": "ppo",
            "r_type": "ppo",
            "c_ckpt": args.stage3_v3_c_checkpoint or str(ckpt_dir / "plan_stage3_mappo_finetune_v3_c_best.pt"),
            "r_ckpt": args.stage3_v3_r_checkpoint or str(ckpt_dir / "plan_stage3_mappo_finetune_v3_r_best.pt"),
        },
        {
            "name": "Stage3 v4 MAPPO KL=0.01",
            "c_type": "ppo",
            "r_type": "ppo",
            "c_ckpt": args.stage3_v4_c_checkpoint or str(ckpt_dir / "plan_stage3_mappo_finetune_v4_c_best.pt"),
            "r_ckpt": args.stage3_v4_r_checkpoint or str(ckpt_dir / "plan_stage3_mappo_finetune_v4_r_best.pt"),
        },
        {
            "name": "Compute-Greedy + PPO-R",
            "c_type": "greedy",
            "r_type": "ppo",
            "c_ckpt": None,
            "r_ckpt": args.ppo_r_checkpoint or str(ckpt_dir / "joint_mappo_v2_cshape_metro24_fs002_r_best.pt"),
        },
        {
            "name": "Separate C+R (DQN)",
            "c_type": "dqn",
            "r_type": "dqn",
            "c_ckpt": args.separate_c_checkpoint or str(ckpt_dir / "separate_agent_c_best.pt"),
            "r_ckpt": args.separate_r_checkpoint or str(ckpt_dir / "separate_agent_r.pt"),
        },
    ]

    if args.deep_rmsa_checkpoint:
        method_specs.append(
            {
                "name": "Compute-Greedy + DeepRMSA-A2C",
                "c_type": "greedy",
                "r_type": "deep_rmsa",
                "c_ckpt": None,
                "r_ckpt": args.deep_rmsa_checkpoint,
            }
        )

    method_specs.append(
        {
            "name": "Compute-Greedy + KSP-BF",
            "c_type": "greedy",
            "r_type": "ksp_bf",
            "c_ckpt": None,
            "r_ckpt": None,
        },
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]

    for topo in topologies:
        print("=" * 120)
        print(f"Unified Evaluation | Topology: {topo}")
        print(f"Seeds: {seeds}  Episodes/seed: {args.episodes}  Requests/episode: {args.requests_per_episode}")
        print("=" * 120)

        # -- Create env (for topology constants) and generate test episodes --
        env = make_env(
            topology=topo,
            num_servers=args.num_servers,
            seed=42,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
        )

        # Pre-generate all episodes for all seeds (ensures identical test set)
        all_episodes: Dict[int, List[List]] = {}
        for seed in seeds:
            rng = np.random.RandomState(seed)
            eps = []
            for _ in range(args.episodes):
                src = rng.randint(0, env.net.NUM_NODES)
                requests = generate_requests(
                    env, rng, src,
                    args.requests_per_episode,
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
                )
                eps.append(requests)
            all_episodes[seed] = eps

        # -- Evaluate each method --
        results: Dict[str, Dict] = {}

        for spec in method_specs:
            name = spec["name"]
            print(f"\n--- {name} ---")
            print(f"    C: {spec['c_type']}  R: {spec['r_type']}")

            # Load agents
            if spec["c_type"] == "ppo":
                agent_c = _load_ppo_c(spec["c_ckpt"], args.device)
            elif spec["c_type"] == "dqn":
                agent_c = _load_dqn_c(spec["c_ckpt"], args.device)
            else:
                agent_c = None  # greedy doesn't need a model

            if spec["r_type"] == "ppo":
                agent_r = _load_ppo_r(spec["r_ckpt"], mod_reg, args.device)
            elif spec["r_type"] == "dqn":
                agent_r = _load_dqn_r(spec["r_ckpt"], mod_reg, args.device)
            elif spec["r_type"] == "deep_rmsa":
                agent_r = _load_deep_rmsa(spec["r_ckpt"], env, mod_reg, args.device)
            else:
                agent_r = None  # ksp_bf doesn't need a model

            # Evaluate per seed
            seed_results = []
            for seed in seeds:
                env_seed = make_env(
                    topology=topo,
                    num_servers=args.num_servers,
                    seed=seed,
                    num_slots=args.num_slots,
                    slot_bw_hz=args.slot_bw_hz,
                    guard_band_fs=args.guard_band_fs,
                    modulation_profile=args.modulation_profile,
                )
                res = evaluate_method(
                    env_seed, agent_c, agent_r,
                    spec["c_type"], spec["r_type"],
                    all_episodes[seed],
                    args.waste_coef,
                )
                seed_results.append(res)

            # Aggregate across seeds
            agg = {}
            scalar_keys = [
                "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
                "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
                "server_overload_ratio", "no_suitable_block_ratio",
            ]
            for k in scalar_keys:
                vals = [r[k] for r in seed_results]
                agg[k] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))

            # Aggregate counters
            agg_mod = Counter()
            agg_split = Counter()
            agg_server = Counter()
            agg_server_success = Counter()
            agg_server_overload = Counter()
            agg_reason = Counter()
            for r in seed_results:
                agg_mod.update(r["mod_counter"])
                agg_split.update(r["split_counter"])
                agg_server.update(r["server_counter"])
                agg_server_success.update(r["server_success_counter"])
                agg_server_overload.update(r["server_overload_counter"])
                agg_reason.update(r["reason_counter"])
            agg["mod_counter"] = dict(agg_mod)
            agg["split_counter"] = dict(agg_split)
            agg["server_counter"] = dict(agg_server)
            agg["server_success_counter"] = dict(agg_server_success)
            agg["server_overload_counter"] = dict(agg_server_overload)
            agg["reason_counter"] = dict(agg_reason)

            results[name] = agg

            # Per-seed blocking summary
            seed_blks = [f"{r['blocking_rate']:.3f}" for r in seed_results]
            print(f"    Seed blocking rates: [{', '.join(seed_blks)}]")

        # ------------------------------------------------------------------
        # Print report
        # ------------------------------------------------------------------
        print("\n" + "=" * 120)
        print("RESULTS SUMMARY")
        print("=" * 120)

        header = (f"{'Method':35s} {'BlkRate':>8s} {'SuccRate':>9s} {'AvgRwd':>8s} "
                  f"{'AvgDelay':>9s} {'DeadSat':>8s} {'AvgFS':>7s} {'Waste':>7s} "
                  f"{'PathKm':>7s} {'SrvOver':>8s} {'NoBlock':>8s}")
        print(header)
        print("-" * 120)

        for name, agg in results.items():
            print(
                f"{name:35s} "
                f"{agg['blocking_rate']:8.3f} "
                f"{agg['success_rate']:9.3f} "
                f"{agg['avg_reward']:8.3f} "
                f"{agg['avg_delay_ms']:9.1f} "
                f"{agg['deadline_sat_rate']:8.3f} "
                f"{agg['avg_fs']:7.2f} "
                f"{agg['avg_waste']:7.3f} "
                f"{agg['avg_path_len_km']:7.1f} "
                f"{agg['server_overload_ratio']:8.3f} "
                f"{agg['no_suitable_block_ratio']:8.3f}"
            )

        # --- Rankings ---
        print("\n" + "-" * 120)
        print("RANKINGS")
        print("-" * 120)

        # 1. Reliability ranking (by blocking rate)
        print("\n[Reliability Ranking] by Blocking Rate (lower is better)")
        by_blk = sorted(results.items(), key=lambda x: x[1]["blocking_rate"])
        for rank, (name, agg) in enumerate(by_blk, 1):
            print(f"  {rank}. {name:35s}  {agg['blocking_rate']:.4f}")

        # 2. QoS ranking (by avg reward)
        print("\n[QoS Ranking] by Avg Reward (higher is better)")
        by_rwd = sorted(results.items(), key=lambda x: x[1]["avg_reward"], reverse=True)
        for rank, (name, agg) in enumerate(by_rwd, 1):
            print(f"  {rank}. {name:35s}  {agg['avg_reward']:+.4f}")

        # 3. Pareto-front analysis
        print("\n[Pareto-Front Analysis] on blocking / reward / delay / FS")
        front = compute_pareto_front(results)
        all_names = list(results.keys())
        for name in all_names:
            dom_status = "PARETO-FRONT" if name in front else "dominated"
            print(f"  {name:35s}  {dom_status}")
        print(f"\n  Non-dominated methods: {len(front)} / {len(all_names)}")

        # --- Distribution details ---
        print("\n" + "-" * 120)
        print("MODULATION DISTRIBUTION")
        print("-" * 120)
        for name, agg in results.items():
            print(f"  {name:35s}  {_fmt_pct(Counter(agg['mod_counter']))}")

        print("\n" + "-" * 120)
        print("SPLIT DISTRIBUTION")
        print("-" * 120)
        for name, agg in results.items():
            print(f"  {name:35s}  {_fmt_pct(Counter(agg['split_counter']))}")

        print("\n" + "-" * 120)
        print("SERVER DISTRIBUTION")
        print("-" * 120)
        for name, agg in results.items():
            print(f"  {name:35s}  {_fmt_pct(Counter(agg['server_counter']))}")

        print("\n" + "-" * 120)
        print("SERVER SUCCESS DISTRIBUTION")
        print("-" * 120)
        for name, agg in results.items():
            print(f"  {name:35s}  {_fmt_pct(Counter(agg['server_success_counter']))}")

        print("\n" + "-" * 120)
        print("SERVER OVERLOAD DISTRIBUTION")
        print("-" * 120)
        for name, agg in results.items():
            print(f"  {name:35s}  {_fmt_cnt(Counter(agg['server_overload_counter']))}")

        print("\n" + "-" * 120)
        print("FAILURE REASONS")
        print("-" * 120)
        for name, agg in results.items():
            print(f"  {name:35s}  {_fmt_cnt(Counter(agg['reason_counter']))}")

        # --- JSON export ---
        if args.output_json:
            out_path = Path(args.output_json)
            # Convert all numpy types for JSON
            clean = {}
            for name, agg in results.items():
                clean[name] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                               for k, v in agg.items()}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(clean, open(str(out_path), "w"), indent=2)
            print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
