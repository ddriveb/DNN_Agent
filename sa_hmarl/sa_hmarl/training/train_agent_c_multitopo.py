"""Agent-C multi-topology DQN training with fixed KSP-BF RMSA.

Each episode samples a topology from {net1, net2, net3} uniformly.
The same Agent-C network is trained across all topologies.

Usage (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.train_agent_c_multitopo \
        --episodes 1000 --requests_per_episode 30
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
from collections import Counter
from typing import Dict, List

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.replay_buffer import ReplayBuffer
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env


def _compute_greedy_action(obs_c, valid, key):
    if len(valid) == 0:
        return None
    best = valid[0]
    best_val = obs_c["candidate_features"][best][key]
    if best_val == float("inf"):
        best_val = 1e9
    for idx in valid[1:]:
        val = obs_c["candidate_features"][idx][key]
        if val == float("inf"):
            val = 1e9
        if val < best_val:
            best_val = val
            best = idx
    return int(best)


def _spectrum_greedy_action(obs_c, valid):
    if len(valid) == 0:
        return None
    best = valid[0]
    best_fs = obs_c["candidate_features"][best]["best_fs_estimate"]
    best_count = obs_c["candidate_features"][best]["feasible_count"]
    for idx in valid[1:]:
        fs = obs_c["candidate_features"][idx]["best_fs_estimate"]
        count = obs_c["candidate_features"][idx]["feasible_count"]
        if fs is not None and (best_fs is None or fs < best_fs):
            best = idx
            best_fs = fs
            best_count = count
        elif fs == best_fs and count > best_count:
            best = idx
            best_count = count
    return int(best)


def evaluate_agent(
    env: SMDPEnv, agent: AgentC, requests: List, waste_coef: float = 0.8
) -> Dict[str, float]:
    env.reset(requests)
    total_reward = 0.0
    blocked = 0
    total_waste = 0.0
    num_success = 0
    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        action_idx_c = agent.select_action(obs_c, epsilon=0.0)
        if action_idx_c is None:
            action_c = (0, 0)
        else:
            action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))
        split_id, server_id = action_c
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        action_idx_r = ksp_bf_action(obs_r)
        if action_idx_r is None:
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )
        _, _, done, info = env.step(action_c, action_r)
        server = env.mec.servers[server_id]
        reward = compute_agent_c_reward(info, req.deadline_ms, waste_coef, server.utilization)
        total_reward += reward
        if not info.get("success", False):
            blocked += 1
        else:
            total_waste += info.get("block_waste", 0.0)
            num_success += 1
    n = len(requests)
    return {
        "avg_reward": total_reward / n,
        "blocking_rate": blocked / n,
        "avg_waste": total_waste / num_success if num_success > 0 else 0.0,
    }


def evaluate_baseline(
    env: SMDPEnv, requests: List, waste_coef: float, baseline_name: str
) -> Dict[str, float]:
    env.reset(requests)
    total_reward = 0.0
    blocked = 0
    total_waste = 0.0
    num_success = 0
    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        mask = obs_c["agent_c_mask"]
        valid = np.where(mask)[0]
        if baseline_name == "Compute-Greedy":
            action_idx_c = _compute_greedy_action(obs_c, valid, "edge_compute_ms")
        elif baseline_name == "Spectrum-Greedy":
            action_idx_c = _spectrum_greedy_action(obs_c, valid)
        elif baseline_name == "Random-valid-C":
            action_idx_c = int(np.random.choice(valid)) if len(valid) > 0 else None
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        if action_idx_c is None:
            action_c = (0, 0)
        else:
            action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))
        split_id, server_id = action_c
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        action_idx_r = ksp_bf_action(obs_r)
        if action_idx_r is None:
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )
        _, _, done, info = env.step(action_c, action_r)
        server = env.mec.servers[server_id]
        reward = compute_agent_c_reward(info, req.deadline_ms, waste_coef, server.utilization)
        total_reward += reward
        if not info.get("success", False):
            blocked += 1
        else:
            total_waste += info.get("block_waste", 0.0)
            num_success += 1
    n = len(requests)
    return {
        "avg_reward": total_reward / n,
        "blocking_rate": blocked / n,
        "avg_waste": total_waste / num_success if num_success > 0 else 0.0,
    }


def train(args):
    agent = AgentC(
        input_dim=17,
        hidden_dims=(128, 64),
        gamma=0.95,
        epsilon=0.3,
        lr=1e-3,
        device="cpu",
    )
    replay_buffer = ReplayBuffer(capacity=20000)
    rng = np.random.RandomState(args.seed)

    topologies = args.topologies.split(",")

    metrics: Dict[str, List[float]] = {
        "episode_rewards": [],
        "episode_blocking": [],
        "losses": [],
        "eval_rewards": [],
        "eval_blocking": [],
    }

    best_eval_reward = -1e9
    best_episode = -1

    package_root = Path(__file__).resolve().parents[2]
    ckpt_dir = package_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = ckpt_dir / "agent_c_multitopo_best.pt"
    ckpt_last = ckpt_dir / "agent_c_multitopo_last.pt"

    # Fixed eval sets on net1/net2/net3 for multi-topology validation
    eval_topologies = [t.strip() for t in args.topologies.split(",")]
    eval_envs = {}
    eval_requests_map = {}
    for topo in eval_topologies:
        eval_env = make_env(
            topology=topo,
            num_servers=2,
            seed=args.seed,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
        )
        eval_src = rng.randint(0, eval_env.net.NUM_NODES)
        eval_reqs = generate_requests(
            eval_env,
            rng,
            eval_src,
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
        eval_envs[topo] = eval_env
        eval_requests_map[topo] = eval_reqs

    print("=" * 70)
    print("Agent-C Multi-Topology DQN Training (fixed KSP-BF RMSA)")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"Topologies: {topologies}")
    print(f"Eval topologies: {list(eval_envs.keys())}")
    print(f"Splits: {args.num_splits}, Slots: {args.num_slots}")
    print("=" * 70)

    for episode in range(args.episodes):
        topo = rng.choice(topologies)
        env = make_env(
            topology=topo.strip(),
            num_servers=2,
            seed=args.seed + episode,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
        )
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env,
            rng,
            src,
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
        env.reset(requests)

        episode_reward = 0.0
        episode_blocked = 0
        episode_fs_values = []
        reason_counter = Counter()
        split_counter = Counter()
        server_counter = Counter()

        for t, req in enumerate(requests):
            obs_c = build_agent_c_observation(env, req)
            action_idx_c = agent.select_action(obs_c)
            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

            split_id, server_id = action_c
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = ksp_bf_action(obs_r)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                )

            _, _, done, info = env.step(action_c, action_r)
            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            episode_reward += reward

            if info.get("success", False):
                episode_fs_values.append(info.get("num_slots", 0))
            else:
                episode_blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1

            split_counter[split_id] += 1
            server_counter[server_id] += 1

            action_features, mask = agent.build_action_features(obs_c)
            if t + 1 < len(requests):
                next_req = requests[t + 1]
                next_obs_c = build_agent_c_observation(env, next_req)
                next_action_features, next_mask = agent.build_action_features(next_obs_c)
                transition_done = False
            else:
                next_action_features = np.zeros_like(action_features)
                next_mask = np.zeros_like(mask)
                transition_done = True

            replay_buffer.push(
                action_features,
                mask,
                action_idx_c if action_idx_c is not None else 0,
                reward,
                next_action_features,
                next_mask,
                transition_done,
            )

            if len(replay_buffer) >= args.learning_starts and t % 2 == 0:
                batch = replay_buffer.sample(args.batch_size)
                if batch is not None:
                    loss = agent.optimize(batch, args.batch_size)
                    if loss is not None:
                        metrics["losses"].append(loss)

            if agent.step_count > 0 and agent.step_count % args.target_update_freq == 0:
                agent.update_target()

        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        metrics["episode_rewards"].append(episode_reward / args.requests_per_episode)
        metrics["episode_blocking"].append(episode_blocked / args.requests_per_episode)

        if episode % 100 == 0 or episode == args.episodes - 1:
            # Multi-topology validation: average reward across all eval topologies
            eval_results = {}
            baseline_results = {}
            for eval_topo, eval_env in eval_envs.items():
                eval_reqs = eval_requests_map[eval_topo]
                eval_results[eval_topo] = evaluate_agent(eval_env, agent, eval_reqs, args.waste_coef)
                baseline_results[eval_topo] = {
                    "comp": evaluate_baseline(eval_env, eval_reqs, args.waste_coef, "Compute-Greedy"),
                    "spec": evaluate_baseline(eval_env, eval_reqs, args.waste_coef, "Spectrum-Greedy"),
                    "rand": evaluate_baseline(eval_env, eval_reqs, args.waste_coef, "Random-valid-C"),
                }

            avg_eval_reward = np.mean([r["avg_reward"] for r in eval_results.values()])
            avg_eval_blocking = np.mean([r["blocking_rate"] for r in eval_results.values()])
            metrics["eval_rewards"].append(avg_eval_reward)
            metrics["eval_blocking"].append(avg_eval_blocking)

            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                best_episode = episode
                torch.save(
                    {
                        "model_state": agent.q_net.state_dict(),
                        "target_state": agent.target_net.state_dict(),
                        "input_dim": agent.input_dim,
                        "hidden_dims": (128, 64),
                        "gamma": agent.gamma,
                        "seed": args.seed,
                        "training_metrics": metrics,
                        "args": vars(args),
                        "best_eval_reward": best_eval_reward,
                        "best_episode": best_episode,
                    },
                    str(ckpt_best),
                )

            avg_fs = np.mean(episode_fs_values) if episode_fs_values else 0.0
            reason_str = ", ".join(f"{k}={v}" for k, v in reason_counter.most_common(5))
            split_str = ", ".join(f"s{k}={v}" for k, v in sorted(split_counter.items()))
            server_str = ", ".join(f"sv{k}={v}" for k, v in sorted(server_counter.items()))
            topo_str = f"[{topo}]"
            # Build per-topology eval summary string
            per_topo_eval = " | ".join(
                f"{t}:r={eval_results[t]['avg_reward']:+.3f}/blk={eval_results[t]['blocking_rate']:.2f}"
                for t in eval_results
            )
            print(
                f"Ep {episode:4d} {topo_str:<8} | "
                f"train_r={metrics['episode_rewards'][-1]:+.3f} "
                f"train_blk={metrics['episode_blocking'][-1]:.2f} "
                f"avg_fs={avg_fs:.1f} "
                f"eps={agent.epsilon:.3f} | "
                f"avg_eval_r={avg_eval_reward:+.3f} "
                f"avg_eval_blk={avg_eval_blocking:.2f} | "
                f"{per_topo_eval} "
                f"| best={best_eval_reward:+.3f}@{best_episode}"
            )
            if reason_str:
                print(f"         Reasons: {reason_str}")
            print(f"         Splits: {split_str} | Servers: {server_str}")

    print("=" * 70)
    print("Final vs Baselines (per-topology eval set):")
    for eval_topo, eval_env in eval_envs.items():
        eval_reqs = eval_requests_map[eval_topo]
        final_eval = evaluate_agent(eval_env, agent, eval_reqs, args.waste_coef)
        final_comp = evaluate_baseline(eval_env, eval_reqs, args.waste_coef, "Compute-Greedy")
        final_spec = evaluate_baseline(eval_env, eval_reqs, args.waste_coef, "Spectrum-Greedy")
        final_rand = evaluate_baseline(eval_env, eval_reqs, args.waste_coef, "Random-valid-C")
        print(f"  [{eval_topo}] Agent-C:   reward={final_eval['avg_reward']:+.3f}, blocking={final_eval['blocking_rate']:.2f}")
        print(f"  [{eval_topo}] Compute-G: reward={final_comp['avg_reward']:+.3f}, blocking={final_comp['blocking_rate']:.2f}")
        print(f"  [{eval_topo}] Spectrum-G: reward={final_spec['avg_reward']:+.3f}, blocking={final_spec['blocking_rate']:.2f}")
        print(f"  [{eval_topo}] Random-C:  reward={final_rand['avg_reward']:+.3f}, blocking={final_rand['blocking_rate']:.2f}")
    print(f"\nBest checkpoint: avg_reward={best_eval_reward:+.3f} at episode {best_episode}")
    print("=" * 70)

    torch.save(
        {
            "model_state": agent.q_net.state_dict(),
            "target_state": agent.target_net.state_dict(),
            "input_dim": agent.input_dim,
            "hidden_dims": (128, 64),
            "gamma": agent.gamma,
            "seed": args.seed,
            "training_metrics": metrics,
            "args": vars(args),
            "best_eval_reward": best_eval_reward,
            "best_episode": best_episode,
        },
        str(ckpt_last),
    )
    print(f"Last checkpoint saved to {ckpt_last}")
    if ckpt_best.exists():
        print(f"Best checkpoint saved to {ckpt_best}")

    return agent, metrics


def main():
    parser = argparse.ArgumentParser(description="Agent-C multi-topology DQN training")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--requests_per_episode", type=int, default=30)
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
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=100)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topologies",
        type=str,
        default="net1,net2,net3",
        help="Comma-separated topology names for multi-topology training",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
