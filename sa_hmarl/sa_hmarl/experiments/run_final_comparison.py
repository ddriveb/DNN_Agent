"""Final unified comparison across all methods on the exact same test set.

Methods:
  1. Separate Agent-C + Agent-R
  2. Warm-start Joint Alternating (tr=0.05)
  3. Agent-C-DQN + KSP-BF
  4. Compute-Greedy + KSP-BF
  5. Spectrum-Greedy + KSP-BF
  6. Random-valid-C + KSP-BF

Usage (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.experiments.run_final_comparison
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from collections import Counter
from typing import Dict, List

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.evaluation.eval_joint_alternating import evaluate_method, METHODS
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_agent_c(ckpt_path: str, device: str = "cpu") -> AgentC:
    agent = AgentC(
        input_dim=17, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device=device
    )
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent.q_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_state"])
    return agent


def _load_agent_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> AgentR:
    agent = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        gamma=0.95,
        epsilon=0.0,
        device=device,
    )
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent.q_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_state"])
    return agent


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Final comparison experiment")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    args_cli = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    device = "cpu"
    mod_reg = ModulationRegistry.from_profile(args_cli.modulation_profile)

    # ------------------------------------------------------------------
    # Load three agent pairs
    # ------------------------------------------------------------------
    print("Loading checkpoints...")
    separate_c = _load_agent_c(str(ckpt_dir / "separate_agent_c_best.pt"), device)
    separate_r = _load_agent_r(str(ckpt_dir / "separate_agent_r.pt"), mod_reg, device)

    joint_c = _load_agent_c(str(ckpt_dir / "D_warm_tr0.05_c.pt"), device)
    joint_r = _load_agent_r(str(ckpt_dir / "D_warm_tr0.05_r.pt"), mod_reg, device)

    # For "Agent-C only + KSP-BF" we reuse separate_c with a dummy Agent-R
    dummy_r = _load_agent_r(str(ckpt_dir / "separate_agent_r.pt"), mod_reg, device)

    # ------------------------------------------------------------------
    # Generate unified test set
    # ------------------------------------------------------------------
    eval_seed = 123
    eval_episodes = 50
    requests_per_episode = 30
    env = make_env(num_servers=2, seed=eval_seed, num_slots=32,
                   modulation_profile=args_cli.modulation_profile)
    rng = np.random.RandomState(eval_seed)

    test_sets: List[List] = []
    for ep in range(eval_episodes):
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env,
            rng,
            src,
            num_requests=requests_per_episode,
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
        test_sets.append(requests)

    # ------------------------------------------------------------------
    # Evaluate each configuration
    # ------------------------------------------------------------------
    configs = [
        ("Separate Agent-C + Agent-R", separate_c, separate_r),
        ("Warm-start Joint (tr=0.05)", joint_c, joint_r),
        ("Agent-C-DQN + KSP-BF", separate_c, dummy_r),
    ]

    # Baseline methods (3-6) are evaluated via the same evaluate_method,
    # but we only care about Agent-C-DQN + KSP-BF from the 3rd config.
    # Heuristics don't need trained agents, so we can run them with dummy agents.
    print("\nRunning evaluation...")

    # Collect results for the 6 methods
    results_map: Dict[str, List[Dict]] = {}

    for cfg_name, agent_c, agent_r in configs:
        all_results = {m: [] for m in METHODS}
        for requests in test_sets:
            for method in METHODS:
                result = evaluate_method(env, agent_c, agent_r, requests, method, 0.8)
                all_results[method].append(result)

        if cfg_name == "Separate Agent-C + Agent-R":
            results_map["1.Separate C+R"] = all_results["Agent-C-DQN + Agent-R-DQN"]
        elif cfg_name == "Warm-start Joint (tr=0.05)":
            results_map["2.Joint warm tr0.05"] = all_results["Agent-C-DQN + Agent-R-DQN"]
        elif cfg_name == "Agent-C-DQN + KSP-BF":
            results_map["3.Agent-C + KSP-BF"] = all_results["Agent-C-DQN + KSP-BF"]
            # Heuristics are independent of the trained agents; all configs produce
            # the same heuristic results.  We grab them once from this config.
            results_map["4.Compute-Greedy + KSP-BF"] = all_results["Compute-Greedy + KSP-BF"]
            results_map["5.Spectrum-Greedy + KSP-BF"] = all_results["Spectrum-Greedy + KSP-BF"]
            results_map["6.Random-valid-C + KSP-BF"] = all_results["Random-valid-C + KSP-BF"]

    # ------------------------------------------------------------------
    # Aggregate and print
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("Final Unified Comparison (seed=123, 50 eps x 30 req)")
    print("=" * 110)
    header = (
        f"{'Method':<30} {'BlkRate':>8} {'SuccRate':>9} {'AvgRwd':>8} "
        f"{'Delay(ms)':>10} {'Waste':>7} {'AvgFS':>7} {'Path(km)':>9}"
    )
    print(header)
    print("-" * 110)

    for label, results in results_map.items():
        blocking = np.mean([r["blocking_rate"] for r in results])
        success = np.mean([r["success_rate"] for r in results])
        avg_reward = np.mean([r["avg_reward"] for r in results])
        delay = np.mean([r["avg_delay_ms"] for r in results])
        waste = np.mean([r["avg_block_waste"] for r in results])
        avg_fs = np.mean([r["avg_fs"] for r in results])
        path_len = np.mean([r["avg_path_len_km"] for r in results])
        print(
            f"{label:<30} {blocking:>8.3f} {success:>9.3f} {avg_reward:>8.3f} "
            f"{delay:>10.2f} {waste:>7.3f} {avg_fs:>7.2f} {path_len:>9.2f}"
        )
    print("=" * 110)

    # Modulation & Failure details
    print("\nModulation Distribution")
    print("-" * 110)
    for label, results in results_map.items():
        mod_counter: Counter = Counter()
        for r in results:
            mod_counter.update(r["mod_counter"])
        if mod_counter:
            total = sum(mod_counter.values())
            mod_str = ", ".join(
                f"{k}={v/total:.1%}" for k, v in sorted(mod_counter.items())
            )
        else:
            mod_str = "N/A"
        print(f"{label:<30} {mod_str}")

    print("\nFailure Reasons")
    print("-" * 110)
    for label, results in results_map.items():
        reason_counter: Counter = Counter()
        for r in results:
            reason_counter.update(r["reason_counter"])
        if reason_counter:
            total = sum(reason_counter.values())
            parts = []
            for r in ["server_overload", "no_suitable_block", "fs_too_large", "deadline_infeasible"]:
                cnt = reason_counter.get(r, 0)
                if cnt:
                    parts.append(f"{r}={cnt} ({cnt/total:.1%})")
            reason_str = ", ".join(parts) if parts else "none"
        else:
            reason_str = "none"
        print(f"{label:<30} {reason_str}")
    print("=" * 110)


if __name__ == "__main__":
    main()
