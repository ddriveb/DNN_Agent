"""Anti-overfitting regularization sweep for NSFNet extended joint fine-tuning.

Problem: Joint-Finetuned on NSFNet extended heavily overuses PM-BPSK (64.2%),
causing high no_suitable_block and blocking=20.3% (worse than Separate C+R 17.6%).

Goal: Test 3 anti-overfitting mechanisms to reduce PM-BPSK over-reliance.

Configurations:
  A: pm_bpsk_penalty=0.02
  B: pm_bpsk_penalty=0.05
  C: pm_bpsk_penalty=0.10
  D: pm_bpsk_penalty=0.05 + r_bc_coef=0.02  (needs warm-start separate R)
  E: pm_bpsk_penalty=0.05 + pm_bpsk_max_share=0.3

Fixed settings:
  --topologies nsfnet --modulation_profile extended
  --num_slots 32 --num_servers 4 --requests_per_episode 60
  --arrival_interval 0.25 --episodes 1000 --team_coef 0.02

Evaluation: 5 seeds x 20 episodes x 60 requests

Usage (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.experiments.run_nsfnet_extended_reg_sweep
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
import torch
from collections import Counter
from types import SimpleNamespace
from typing import Dict, List, Optional
from datetime import datetime

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_agent_c import train as train_agent_c_separate
from sa_hmarl.training.train_agent_r import train as train_agent_r_separate
from sa_hmarl.training.train_joint_alternating import train as train_joint
from sa_hmarl.training.utils import (
    compute_agent_c_reward,
    compute_reward,
    generate_requests,
    make_env,
)
from sa_hmarl.utils.checkpoint import load_checkpoint

# ---------------------------------------------------------------------------
# Fixed base training arguments for NSFNet extended joint fine-tuning
# ---------------------------------------------------------------------------
BASE_JOINT_ARGS = {
    "topologies": "nsfnet",
    "modulation_profile": "extended",
    "num_slots": 32,
    "num_servers": 4,
    "requests_per_episode": 60,
    "arrival_interval": 0.25,
    "holding_min": 4.0,
    "holding_max": 10.0,
    "deadline_min": 30.0,
    "deadline_max": 100.0,
    "size_min_mb": 5.0,
    "size_max_mb": 30.0,
    "edge_cost_min": 0.5,
    "edge_cost_max": 15.0,
    "waste_coef": 0.8,
    "target_update_freq": 100,
    "batch_size": 32,
    "learning_starts": 100,
    "optimize_freq": 2,
    "buffer_capacity": 20000,
    "epsilon_start": 0.3,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "gamma": 0.95,
    "lr": 1e-3,
    "slot_bw_hz": 1.25e9,
    "guard_band_fs": 1,
    "num_splits": 3,
    "device": "cpu",
    "log_interval": 100,
    "eval_interval": 50,
    "eval_episodes": 10,
    "eval_seed": None,
    "update_ratio_c": 0.5,
    "update_ratio_r": 0.5,
    # team_coef is set per-config below
}

# ---------------------------------------------------------------------------
# Separate training args (for generating warm-start checkpoints)
# ---------------------------------------------------------------------------
BASE_SEPARATE_ARGS = {
    "topology": "nsfnet",
    "modulation_profile": "extended",
    "num_slots": 32,
    "num_servers": 4,
    "requests_per_episode": 60,
    "arrival_interval": 0.25,
    "holding_min": 4.0,
    "holding_max": 10.0,
    "deadline_min": 30.0,
    "deadline_max": 100.0,
    "size_min_mb": 5.0,
    "size_max_mb": 30.0,
    "edge_cost_min": 0.5,
    "edge_cost_max": 15.0,
    "waste_coef": 0.8,
    "target_update_freq": 100,
    "batch_size": 32,
    "learning_starts": 100,
    "slot_bw_hz": 1.25e9,
    "guard_band_fs": 1,
    "device": "cpu",
    "seed": 42,
    "mixed_splits": True,
    "episodes": 500,
    "num_splits": 3,
}

# Sweep configurations: (name, overrides)
SWEEP_CONFIGS = [
    (
        "A_pm002",
        {
            "pm_bpsk_penalty": 0.02,
            "r_bc_coef": 0.0,
            "pm_bpsk_max_share": None,
        },
    ),
    (
        "B_pm005",
        {
            "pm_bpsk_penalty": 0.05,
            "r_bc_coef": 0.0,
            "pm_bpsk_max_share": None,
        },
    ),
    (
        "C_pm010",
        {
            "pm_bpsk_penalty": 0.1,
            "r_bc_coef": 0.0,
            "pm_bpsk_max_share": None,
        },
    ),
    (
        "D_pm005_bc002",
        {
            "pm_bpsk_penalty": 0.05,
            "r_bc_coef": 0.02,
            "pm_bpsk_max_share": None,
        },
    ),
    (
        "E_pm005_max03",
        {
            "pm_bpsk_penalty": 0.05,
            "r_bc_coef": 0.0,
            "pm_bpsk_max_share": 0.3,
        },
    ),
]

SEEDS = [42, 123, 456, 789, 1024]
EVAL_SEEDS = [100, 200, 300, 400, 500]
EVAL_EPISODES = 20
EVAL_REQUESTS = 60


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_joint_agent(
    env: SMDPEnv,
    agent_c: AgentC,
    agent_r: AgentR,
    eval_seed: int,
    num_episodes: int = 20,
    requests_per_episode: int = 60,
) -> Dict:
    """Thorough evaluation of a trained joint agent pair.

    Returns blocking rate, success rate, avg FS, path km, modulation
    distribution, PM-BPSK share, and failure reasons.
    """
    rng = np.random.RandomState(eval_seed)

    total_success = 0
    total_blocked = 0
    total_fs = 0.0
    total_path_km = 0.0
    total_delay = 0.0
    total_reward = 0.0
    mod_counter: Counter = Counter()
    reason_counter: Counter = Counter()

    for ep in range(num_episodes):
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env, rng, src, requests_per_episode,
            arrival_interval=0.25,
            holding_min=4.0, holding_max=10.0,
            deadline_min=30.0, deadline_max=100.0,
            size_min_mb=5.0, size_max_mb=30.0,
            edge_cost_min=0.5, edge_cost_max=15.0,
            num_splits=3,
        )
        env.reset(requests)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            action_idx_c = agent_c.select_action(obs_c, epsilon=0.0)
            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

            split_id, server_id = action_c
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = agent_r.select_action(obs_r, epsilon=0.0)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks,
                )

            _, _, done, info = env.step(action_c, action_r)
            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, 0.8, server.utilization,
            ) + compute_reward(info, 0.8)
            total_reward += reward

            if info.get("success", False):
                total_success += 1
                total_fs += info.get("num_slots", 0)
                total_path_km += info.get("path_dist_km", 0.0)
                total_delay += info.get("delay_ms", 0.0)
                mod_counter[info.get("modulation", "unknown")] += 1
            else:
                total_blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1

        env.drain()

    n = total_success + total_blocked
    total_mods = sum(mod_counter.values())
    pm_bpsk_count = mod_counter.get("PM-BPSK", 0)
    pm_bpsk_share = pm_bpsk_count / total_mods if total_mods > 0 else 0.0

    return {
        "blocking_rate": total_blocked / n if n > 0 else 0.0,
        "success_rate": total_success / n if n > 0 else 0.0,
        "avg_fs": total_fs / total_success if total_success > 0 else 0.0,
        "avg_path_km": total_path_km / total_success if total_success > 0 else 0.0,
        "avg_delay_ms": total_delay / total_success if total_success > 0 else 0.0,
        "avg_reward": total_reward / n if n > 0 else 0.0,
        "pm_bpsk_count": pm_bpsk_count,
        "pm_bpsk_share": pm_bpsk_share,
        "mod_counter": dict(mod_counter),
        "reason_counter": dict(reason_counter),
        "total_requests": n,
        "total_success": total_success,
        "total_blocked": total_blocked,
    }


def _fmt_mods(mod_dict: Dict) -> str:
    if not mod_dict:
        return "N/A"
    total = sum(mod_dict.values())
    parts = [f"{k}={v/total:.1%}" for k, v in sorted(mod_dict.items())]
    return ", ".join(parts)


def _fmt_reasons(reason_dict: Dict) -> str:
    if not reason_dict:
        return "none"
    total = sum(reason_dict.values())
    parts = []
    for r in ["no_suitable_block", "modulation_reach", "deadline_infeasible",
              "server_overload", "fs_too_large", "server_saturated"]:
        cnt = reason_dict.get(r, 0)
        if cnt:
            parts.append(f"{r}={cnt} ({cnt/total:.1%})")
    return ", ".join(parts) if parts else "other"


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="NSFNet extended anti-overfitting regularization sweep"
    )
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only evaluate existing checkpoints")
    parser.add_argument("--skip_separate", action="store_true",
                        help="Skip separate agent pre-training")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names to run (default: all)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 42,123,456,789,1024)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Joint-training episodes per config/seed")
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES,
                        help="Evaluation episodes per seed")
    parser.add_argument("--eval_requests", type=int, default=EVAL_REQUESTS,
                        help="Evaluation requests per episode")
    args = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).resolve().parents[2] / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Filter configs if requested
    configs_to_run = SWEEP_CONFIGS
    if args.configs:
        names = set(args.configs.split(","))
        configs_to_run = [(n, o) for n, o in SWEEP_CONFIGS if n in names]

    seeds_to_use = SEEDS
    if args.seeds:
        seeds_to_use = [int(s.strip()) for s in args.seeds.split(",")]

    # -------------------------------------------------------------------
    # Phase 0: Train separate Agent-C and Agent-R (warm-start prerequisites)
    # -------------------------------------------------------------------
    separate_c_path = ckpt_dir / "separate_nsfnet_extended_c.pt"
    separate_r_path = ckpt_dir / "separate_nsfnet_extended_r.pt"

    if not args.skip_training and not args.skip_separate:
        need_c = not separate_c_path.exists()
        need_r = not separate_r_path.exists()

        if need_c or need_r:
            print("=" * 70)
            print("Phase 0: Training separate Agent-C and Agent-R on NSFNet extended")
            print("=" * 70)

        if need_c:
            print("\n--- Training separate Agent-C ---")
            c_args = SimpleNamespace(**{**BASE_SEPARATE_ARGS, "seed": 42})
            from sa_hmarl.training.train_agent_c import train as _train_c
            _train_c(c_args)
            # Copy best checkpoint to the separate namespace
            default_c = ckpt_dir / "agent_c_best.pt"
            if default_c.exists():
                import shutil
                shutil.copy(default_c, separate_c_path)
                print(f"Saved separate Agent-C to {separate_c_path}")

        if need_r:
            print("\n--- Training separate Agent-R ---")
            r_args = SimpleNamespace(**{**BASE_SEPARATE_ARGS, "seed": 42})
            from sa_hmarl.training.train_agent_r import train as _train_r
            _train_r(r_args)
            # mixed_splits=True saves to agent_r_mixed.pt
            default_r = ckpt_dir / "agent_r_mixed.pt"
            if default_r.exists():
                import shutil
                shutil.copy(default_r, separate_r_path)
                print(f"Saved separate Agent-R to {separate_r_path}")

    warm_start_c = str(separate_c_path) if separate_c_path.exists() else None
    warm_start_r = str(separate_r_path) if separate_r_path.exists() else None

    if args.skip_separate and (warm_start_c is None or warm_start_r is None):
        print(
            "WARNING: --skip_separate was set but separate NSFNet extended "
            "checkpoints are missing. Joint warm-start and Separate_C+R "
            "baseline will not be valid until these files exist:\n"
            f"  {separate_c_path}\n"
            f"  {separate_r_path}"
        )

    if warm_start_c:
        print(f"Using warm-start Agent-C: {warm_start_c}")
    if warm_start_r:
        print(f"Using warm-start Agent-R: {warm_start_r}")

    # -------------------------------------------------------------------
    # Phase 1: Train each configuration with multiple seeds
    # -------------------------------------------------------------------
    all_results: Dict[str, Dict] = {}

    for config_name, overrides in configs_to_run:
        print("\n" + "=" * 70)
        print(f"Configuration: {config_name}")
        print(f"Overrides: {overrides}")
        print("=" * 70)

        config_results_per_seed = []

        for seed_idx, seed in enumerate(seeds_to_use):
            eval_seed = EVAL_SEEDS[seed_idx]
            # Include the seed so checkpoints from different runs do not
            # overwrite one another. This keeps --skip_training re-evaluation
            # meaningful and makes failed/interrupted sweeps easier to resume.
            ckpt_prefix = f"joint_nsfnet_extended_reg_{config_name}_s{seed}"

            if not args.skip_training:
                print(f"\n--- Seed {seed} (eval_seed={eval_seed}) ---")

                joint_args_dict = dict(BASE_JOINT_ARGS)
                joint_args_dict.update(overrides)
                joint_args_dict.update({
                    "seed": seed,
                    "episodes": args.episodes,
                    "ckpt_prefix": ckpt_prefix,
                    "warm_start_c": warm_start_c,
                    "warm_start_r": warm_start_r,
                    "team_reward_success": 0.02,
                    "team_reward_blocking": -0.02,
                })
                joint_args = SimpleNamespace(**joint_args_dict)

                # Train joint model
                agent_c, agent_r, train_metrics = train_joint(joint_args)

                # Load best checkpoint for evaluation
                ckpt_c_best = ckpt_dir / f"{ckpt_prefix}_c_best.pt"
                ckpt_r_best = ckpt_dir / f"{ckpt_prefix}_r_best.pt"
                if not ckpt_c_best.exists():
                    ckpt_c_best = ckpt_dir / f"{ckpt_prefix}_c_last.pt"
                if not ckpt_r_best.exists():
                    ckpt_r_best = ckpt_dir / f"{ckpt_prefix}_r_last.pt"
            else:
                ckpt_c_best = ckpt_dir / f"{ckpt_prefix}_c_best.pt"
                ckpt_r_best = ckpt_dir / f"{ckpt_prefix}_r_best.pt"
                if not ckpt_c_best.exists():
                    ckpt_c_best = ckpt_dir / f"{ckpt_prefix}_c_last.pt"
                if not ckpt_r_best.exists():
                    ckpt_r_best = ckpt_dir / f"{ckpt_prefix}_r_last.pt"

            # Evaluate
            print(f"  Evaluating {config_name} seed={seed}...")
            mod_reg = ModulationRegistry.from_profile("extended")
            eval_env = make_env(
                topology="nsfnet",
                num_servers=4,
                seed=eval_seed,
                num_slots=32,
                modulation_profile="extended",
                slot_bw_hz=1.25e9,
                guard_band_fs=1,
            )

            eval_agent_c = AgentC(
                input_dim=17, hidden_dims=(128, 64), gamma=0.95,
                epsilon=0.0, device="cpu",
            )
            eval_agent_r = AgentR(
                input_dim=11, mod_registry=mod_reg, hidden_dims=(128, 64),
                gamma=0.95, epsilon=0.0, device="cpu",
            )

            if ckpt_c_best.exists():
                ckpt_c = load_checkpoint(ckpt_c_best, map_location="cpu")
                eval_agent_c.q_net.load_state_dict(ckpt_c["model_state"])
                eval_agent_c.target_net.load_state_dict(ckpt_c["target_state"])
            else:
                print(f"  WARNING: Agent-C checkpoint not found: {ckpt_c_best}")

            if ckpt_r_best.exists():
                ckpt_r = load_checkpoint(ckpt_r_best, map_location="cpu")
                eval_agent_r.q_net.load_state_dict(ckpt_r["model_state"])
                eval_agent_r.target_net.load_state_dict(ckpt_r["target_state"])
            else:
                print(f"  WARNING: Agent-R checkpoint not found: {ckpt_r_best}")

            eval_result = evaluate_joint_agent(
                eval_env, eval_agent_c, eval_agent_r,
                eval_seed=eval_seed,
                num_episodes=args.eval_episodes,
                requests_per_episode=args.eval_requests,
            )
            eval_result["seed"] = seed
            eval_result["eval_seed"] = eval_seed
            config_results_per_seed.append(eval_result)

        # Aggregate across seeds
        all_results[config_name] = {
            "per_seed": config_results_per_seed,
            "aggregated": _aggregate_results(config_results_per_seed),
        }

    # -------------------------------------------------------------------
    # Phase 2: Evaluate Separate C+R baseline (for comparison)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Evaluating Separate C+R Baseline")
    print("=" * 70)

    separate_baseline_results = []
    for seed_idx, seed in enumerate(seeds_to_use):
        eval_seed = EVAL_SEEDS[seed_idx]
        mod_reg = ModulationRegistry.from_profile("extended")
        baseline_env = make_env(
            topology="nsfnet", num_servers=4, seed=eval_seed,
            num_slots=32, modulation_profile="extended",
            slot_bw_hz=1.25e9, guard_band_fs=1,
        )

        baseline_c = AgentC(
            input_dim=17, hidden_dims=(128, 64), gamma=0.95,
            epsilon=0.0, device="cpu",
        )
        baseline_r = AgentR(
            input_dim=11, mod_registry=mod_reg, hidden_dims=(128, 64),
            gamma=0.95, epsilon=0.0, device="cpu",
        )

        if separate_c_path.exists():
            ckpt = load_checkpoint(separate_c_path, map_location="cpu")
            baseline_c.q_net.load_state_dict(ckpt["model_state"])
            baseline_c.target_net.load_state_dict(ckpt["target_state"])

        if separate_r_path.exists():
            ckpt = load_checkpoint(separate_r_path, map_location="cpu")
            baseline_r.q_net.load_state_dict(ckpt["model_state"])
            baseline_r.target_net.load_state_dict(ckpt["target_state"])

        result = evaluate_joint_agent(
            baseline_env, baseline_c, baseline_r,
            eval_seed=eval_seed,
            num_episodes=args.eval_episodes,
            requests_per_episode=args.eval_requests,
        )
        result["seed"] = seed
        separate_baseline_results.append(result)

    all_results["Separate_C+R"] = {
        "per_seed": separate_baseline_results,
        "aggregated": _aggregate_results(separate_baseline_results),
    }

    # -------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------
    _print_summary(
        all_results,
        seeds_count=len(seeds_to_use),
        train_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        eval_requests=args.eval_requests,
    )

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"nsfnet_extended_reg_sweep_{timestamp}.json"
    # Convert to JSON-serializable format
    serializable = {}
    for config_name, data in all_results.items():
        serializable[config_name] = {
            "settings": {
                "train_episodes": args.episodes,
                "requests_per_episode": BASE_JOINT_ARGS["requests_per_episode"],
                "eval_episodes": args.eval_episodes,
                "eval_requests": args.eval_requests,
                "seeds": seeds_to_use,
            },
            "aggregated": data["aggregated"],
            "per_seed": [],
        }
        for r in data["per_seed"]:
            rr = dict(r)
            rr["mod_counter"] = dict(rr["mod_counter"])
            rr["reason_counter"] = dict(rr["reason_counter"])
            serializable[config_name]["per_seed"].append(rr)

    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


def _aggregate_results(per_seed: List[Dict]) -> Dict:
    """Aggregate per-seed evaluation results."""
    keys = [
        "blocking_rate", "success_rate", "avg_fs", "avg_path_km",
        "avg_delay_ms", "avg_reward", "pm_bpsk_share",
    ]
    agg = {}
    for key in keys:
        values = [r[key] for r in per_seed]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))

    # Aggregate mod and reason counters across seeds
    total_mods = Counter()
    total_reasons = Counter()
    total_success = 0
    total_blocked = 0
    for r in per_seed:
        total_mods.update(r["mod_counter"])
        total_reasons.update(r["reason_counter"])
        total_success += r.get("total_success", 0)
        total_blocked += r.get("total_blocked", 0)

    mod_total = sum(total_mods.values())
    agg["mod_distribution"] = {
        k: v / mod_total for k, v in total_mods.most_common()
    } if mod_total > 0 else {}

    reason_total = sum(total_reasons.values())
    agg["failure_reasons"] = {
        k: v / reason_total for k, v in total_reasons.most_common()
    } if reason_total > 0 else {}

    return agg


def _print_summary(
    all_results: Dict,
    seeds_count: int,
    train_episodes: int,
    eval_episodes: int,
    eval_requests: int,
):
    """Print human-readable summary tables."""
    print("\n" + "=" * 110)
    print("NSFNet Extended Anti-Overfitting Regularization Sweep — Summary")
    print(
        f"Training: {train_episodes} episodes x "
        f"{BASE_JOINT_ARGS['requests_per_episode']} requests, "
        f"Eval: {eval_episodes} episodes x {eval_requests} requests"
    )
    print(f"Seeds: {seeds_count}")
    print("=" * 110)

    # Main metrics table
    header = (
        f"{'Config':<28} {'BlkRate':>8} {'±std':>6} {'AvgFS':>7} "
        f"{'Path(km)':>9} {'PM-BPSK':>8} {'Delay(ms)':>10}"
    )
    print(header)
    print("-" * 110)

    separate_blk = None
    for config_name in list(all_results.keys()):
        agg = all_results[config_name]["aggregated"]
        blk = agg["blocking_rate_mean"]
        if config_name == "Separate_C+R":
            separate_blk = blk

        pm_bpsk = agg.get("pm_bpsk_share_mean", 0.0)
        print(
            f"{config_name:<28} {blk:>8.4f} {agg['blocking_rate_std']:>6.4f} "
            f"{agg['avg_fs_mean']:>7.2f} {agg['avg_path_km_mean']:>9.1f} "
            f"{pm_bpsk:>7.1%} {agg['avg_delay_ms_mean']:>10.1f}"
        )

    print("-" * 110)

    # Comparison vs Separate C+R
    if separate_blk is not None:
        print(f"\nTarget: Joint Blk < Separate C+R = {separate_blk:.4f}")
        print("-" * 110)
        for config_name in list(all_results.keys()):
            if config_name == "Separate_C+R":
                continue
            agg = all_results[config_name]["aggregated"]
            blk = agg["blocking_rate_mean"]
            diff = blk - separate_blk
            status = "✓ SUCCESS" if diff < 0 else "✗ FAIL"
            print(f"  {config_name:<26} blk={blk:.4f} diff={diff:+.4f}  {status}")
        print("-" * 110)

    # Modulation distribution
    print("\nModulation Distribution (avg across seeds)")
    print("-" * 110)
    for config_name in list(all_results.keys()):
        agg = all_results[config_name]["aggregated"]
        mod_dist = agg.get("mod_distribution", {})
        mod_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(mod_dist.items()))
        pm_bpsk = mod_dist.get("PM-BPSK", 0.0)
        print(f"  {config_name:<26} PM-BPSK={pm_bpsk:.1%} | {mod_str}")

    # Failure reasons
    print("\nFailure Reasons (avg across seeds)")
    print("-" * 110)
    for config_name in list(all_results.keys()):
        agg = all_results[config_name]["aggregated"]
        reasons = agg.get("failure_reasons", {})
        reason_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(reasons.items()))
        print(f"  {config_name:<26} {reason_str}")

    # Per-seed details
    print("\nPer-Seed Blocking Rates")
    print("-" * 110)
    for config_name in list(all_results.keys()):
        seeds_str = ", ".join(
            f"s{r['seed']}={r['blocking_rate']:.4f}"
            for r in all_results[config_name]["per_seed"]
        )
        print(f"  {config_name:<26} {seeds_str}")

    print("=" * 110)


if __name__ == "__main__":
    main()
