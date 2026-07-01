"""Joint ablation experiment: A/B/C/D.

Runs four configurations of stabilized joint alternating training and
evaluates them on the same held-out test set.

Configurations:
  A: random init, team_reward=+0.2/-0.2
  B: random init, team_reward=0.0
  C: warm-start separate ckpt, team_reward=0.0
  D: warm-start separate ckpt, team_reward=+0.05/-0.05

Usage (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.experiments.run_joint_ablation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from collections import Counter
from types import SimpleNamespace
from typing import Dict, List

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.evaluation.eval_joint_alternating import evaluate_method, METHODS
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_joint_alternating import train
from sa_hmarl.training.utils import generate_requests, make_env


BASE_ARGS = {
    "episodes": 400,
    "requests_per_episode": 30,
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
    "num_slots": 32,
    "num_splits": 3,
    "modulation_profile": "default",
    "seed": 42,
    "device": "cpu",
    "log_interval": 100,
    "warm_start_c": None,
    "warm_start_r": None,
    "update_ratio_c": 0.5,
    "update_ratio_r": 0.5,
    "team_reward_success": 0.0,
    "team_reward_blocking": 0.0,
}


def _evaluate_agent_pair(
    env: SMDPEnv,
    agent_c: AgentC,
    agent_r: AgentR,
    eval_episodes: int = 50,
    eval_seed: int = 123,
) -> Dict:
    """Evaluate a trained agent pair on a fixed test set."""
    rng = np.random.RandomState(eval_seed)
    all_results = {m: [] for m in METHODS}

    for ep in range(eval_episodes):
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env,
            rng,
            src,
            num_requests=30,
            arrival_interval=0.25,
            holding_min=4.0,
            holding_max=10.0,
            deadline_min=30.0,
            deadline_max=100.0,
            size_min_mb=5.0,
            size_max_mb=30.0,
            edge_cost_min=0.5,
            edge_cost_max=15.0,
        )
        for method in METHODS:
            result = evaluate_method(env, agent_c, agent_r, requests, method, 0.8)
            all_results[method].append(result)

    target = "Agent-C-DQN + Agent-R-DQN"
    results = all_results[target]

    blocking = float(np.mean([r["blocking_rate"] for r in results]))
    success = float(np.mean([r["success_rate"] for r in results]))
    avg_reward = float(np.mean([r["avg_reward"] for r in results]))
    avg_fs = float(np.mean([r["avg_fs"] for r in results]))

    mod_counter: Counter = Counter()
    reason_counter: Counter = Counter()
    for r in results:
        mod_counter.update(r["mod_counter"])
        reason_counter.update(r["reason_counter"])

    return {
        "blocking": blocking,
        "success": success,
        "avg_reward": avg_reward,
        "avg_fs": avg_fs,
        "mods": dict(mod_counter),
        "reasons": dict(reason_counter),
    }


def run_experiment(name: str, overrides: Dict, ckpt_dir: Path):
    """Run one training + evaluation configuration."""
    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print("=" * 70)

    kwargs = dict(BASE_ARGS)
    kwargs.update(overrides)
    args = SimpleNamespace(**kwargs)

    # Train
    agent_c, agent_r, metrics = train(args)

    # Save uniquely-named checkpoints
    torch.save(
        {
            "model_state": agent_c.q_net.state_dict(),
            "target_state": agent_c.target_net.state_dict(),
            "input_dim": agent_c.input_dim,
            "hidden_dims": (128, 64),
            "gamma": agent_c.gamma,
            "seed": args.seed,
            "training_metrics": metrics,
            "args": vars(args),
        },
        str(ckpt_dir / f"{name}_c.pt"),
    )
    torch.save(
        {
            "model_state": agent_r.q_net.state_dict(),
            "target_state": agent_r.target_net.state_dict(),
            "input_dim": agent_r.input_dim,
            "hidden_dims": (128, 64),
            "gamma": agent_r.gamma,
            "seed": args.seed,
            "training_metrics": metrics,
            "args": vars(args),
        },
        str(ckpt_dir / f"{name}_r.pt"),
    )

    # Evaluate on fresh env with fixed seed
    env = make_env(num_servers=2, seed=123, num_slots=32)
    eval_result = _evaluate_agent_pair(env, agent_c, agent_r, eval_episodes=50, eval_seed=123)
    eval_result["name"] = name
    return eval_result


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
    for r in ["server_overload", "no_suitable_block", "fs_too_large", "deadline_infeasible"]:
        cnt = reason_dict.get(r, 0)
        if cnt:
            parts.append(f"{r}={cnt} ({cnt/total:.1%})")
    return ", ".join(parts) if parts else "other"


def main():
    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        (
            "A_random_tr0.2",
            {"team_reward_success": 0.2, "team_reward_blocking": -0.2},
        ),
        (
            "B_random_tr0.0",
            {"team_reward_success": 0.0, "team_reward_blocking": 0.0},
        ),
        (
            "C_warm_tr0.0",
            {
                "team_reward_success": 0.0,
                "team_reward_blocking": 0.0,
                "warm_start_c": str(ckpt_dir / "separate_agent_c_best.pt"),
                "warm_start_r": str(ckpt_dir / "separate_agent_r.pt"),
            },
        ),
        (
            "D_warm_tr0.05",
            {
                "team_reward_success": 0.05,
                "team_reward_blocking": -0.05,
                "warm_start_c": str(ckpt_dir / "separate_agent_c_best.pt"),
                "warm_start_r": str(ckpt_dir / "separate_agent_r.pt"),
            },
        ),
    ]

    results: List[Dict] = []
    for name, overrides in experiments:
        result = run_experiment(name, overrides, ckpt_dir)
        results.append(result)

    # ---- Summary table ------------------------------------------------
    print("\n" + "=" * 100)
    print("Joint Ablation Experiment Summary (Agent-C-DQN + Agent-R-DQN)")
    print("Eval: 50 episodes x 30 requests, seed=123")
    print("=" * 100)
    header = (
        f"{'Config':<20} {'BlkRate':>8} {'SuccRate':>9} {'AvgRwd':>8} "
        f"{'AvgFS':>7} {'Modulations':>40}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        mod_str = _fmt_mods(r["mods"])
        print(
            f"{r['name']:<20} {r['blocking']:>8.3f} {r['success']:>9.3f} "
            f"{r['avg_reward']:>8.3f} {r['avg_fs']:>7.2f} {mod_str:>40}"
        )
    print("-" * 100)

    print("\nFailure Reasons")
    print("-" * 100)
    for r in results:
        reason_str = _fmt_reasons(r["reasons"])
        print(f"{r['name']:<20} {reason_str}")
    print("=" * 100)

    # Also print baseline heuristics from the last eval for reference
    print("\nReference: Heuristic Baselines (from last eval run)")
    print("-" * 100)
    env = make_env(num_servers=2, seed=123, num_slots=32,
                   modulation_profile=BASE_ARGS["modulation_profile"])
    mod_reg = ModulationRegistry.from_profile(BASE_ARGS["modulation_profile"])
    agent_c = AgentC(input_dim=17, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device="cpu")
    agent_r = AgentR(input_dim=11, mod_registry=mod_reg, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device="cpu")
    rng = np.random.RandomState(123)
    baseline_results = {m: [] for m in METHODS}
    for ep in range(50):
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env, rng, src, 30, arrival_interval=0.25,
            holding_min=4.0, holding_max=10.0, deadline_min=30.0, deadline_max=100.0,
            size_min_mb=5.0, size_max_mb=30.0, edge_cost_min=0.5, edge_cost_max=15.0,
        )
        for method in METHODS:
            if method == "Agent-C-DQN + Agent-R-DQN":
                continue  # Skip, not meaningful with random-init agents
            baseline_results[method].append(
                evaluate_method(env, agent_c, agent_r, requests, method, 0.8)
            )
    for method in METHODS:
        if method == "Agent-C-DQN + Agent-R-DQN":
            continue
        blk = np.mean([r["blocking_rate"] for r in baseline_results[method]])
        succ = np.mean([r["success_rate"] for r in baseline_results[method]])
        rwd = np.mean([r["avg_reward"] for r in baseline_results[method]])
        afs = np.mean([r["avg_fs"] for r in baseline_results[method]])
        print(f"{method:<30} blk={blk:.3f} succ={succ:.3f} rwd={rwd:+.3f} fs={afs:.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
