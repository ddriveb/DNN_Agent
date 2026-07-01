"""Agent-R multi-topology DQN training.

Each episode samples a topology from {net1, net2, net3} uniformly.
The same Agent-R network is trained across all topologies.

Usage (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.train_agent_r_multitopo \
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

from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.agents.replay_buffer import ReplayBuffer
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_reward, generate_requests, make_env


def evaluate_agent(
    env: SMDPEnv, agent: AgentR, requests: List, server_id: int, waste_coef: float = 0.8
) -> Dict[str, float]:
    env.reset(requests)
    total_reward = 0.0
    blocked = 0
    total_waste = 0.0
    num_success = 0
    for req in requests:
        obs = build_agent_r_observation(env, req, split_id=0, server_id=server_id)
        action_idx = agent.select_action(obs, epsilon=0.0)
        if action_idx is None:
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(action_idx, len(obs["mod_names"]), env.max_blocks)
        _, _, done, info = env.step((0, server_id), action_r)
        total_reward += compute_reward(info, waste_coef)
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
    env: SMDPEnv, requests: List, server_id: int, waste_coef: float = 0.8
) -> Dict[str, float]:
    env.reset(requests)
    total_reward = 0.0
    blocked = 0
    total_waste = 0.0
    num_success = 0
    for req in requests:
        obs = build_agent_r_observation(env, req, split_id=0, server_id=server_id)
        action_idx = ksp_ff_action(obs)
        if action_idx is None:
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(action_idx, len(obs["mod_names"]), env.max_blocks)
        _, _, done, info = env.step((0, server_id), action_r)
        total_reward += compute_reward(info, waste_coef)
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
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
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

    package_root = Path(__file__).resolve().parents[2]
    ckpt_dir = package_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "agent_r_multitopo.pt"

    # Fixed eval topology (net1) for stable progress tracking
    eval_env = make_env(
        topology="net1",
        num_servers=2,
        seed=args.seed,
        num_slots=args.num_slots,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )
    eval_src = rng.randint(0, eval_env.net.NUM_NODES)
    eval_server = rng.randint(0, len(eval_env.mec.servers))
    eval_requests = generate_requests(
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
    )

    print("=" * 70)
    print("Agent-R Multi-Topology DQN Training")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"Topologies: {topologies}")
    print(f"Slots: {args.num_slots}, Slot BW: {args.slot_bw_hz/1e9:.2f} GHz, Guard: {args.guard_band_fs}")
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
        server_id = rng.randint(0, len(env.mec.servers))
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
        )
        env.reset(requests)

        episode_reward = 0.0
        episode_blocked = 0
        episode_fs_values = []
        reason_counter = Counter()

        for t, req in enumerate(requests):
            split_id = rng.randint(0, len(req.splits)) if args.mixed_splits else 0
            obs = build_agent_r_observation(env, req, split_id=split_id, server_id=server_id)
            action_idx = agent.select_action(obs)

            if action_idx is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx, len(obs["mod_names"]), env.max_blocks
                )

            _, _, done, info = env.step((split_id, server_id), action_r)
            reward = compute_reward(info, args.waste_coef)
            episode_reward += reward

            if info.get("success", False):
                episode_fs_values.append(info.get("num_slots", 0))
            else:
                episode_blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1

            # Build next_obs
            action_features, mask = agent.build_action_features(obs)
            if t + 1 < len(requests):
                next_req = requests[t + 1]
                next_split = rng.randint(0, len(next_req.splits)) if args.mixed_splits else 0
                next_obs = build_agent_r_observation(
                    env, next_req, split_id=next_split, server_id=server_id
                )
                next_action_features, next_mask = agent.build_action_features(next_obs)
                transition_done = False
            else:
                next_action_features = np.zeros_like(action_features)
                next_mask = np.zeros_like(mask)
                transition_done = True

            replay_buffer.push(
                action_features,
                mask,
                action_idx if action_idx is not None else 0,
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
            eval_m = evaluate_agent(
                eval_env, agent, eval_requests, eval_server, args.waste_coef
            )
            base_m = evaluate_baseline(
                eval_env, eval_requests, eval_server, args.waste_coef
            )
            metrics["eval_rewards"].append(eval_m["avg_reward"])
            metrics["eval_blocking"].append(eval_m["blocking_rate"])

            avg_fs = np.mean(episode_fs_values) if episode_fs_values else 0.0
            reason_str = ", ".join(f"{k}={v}" for k, v in reason_counter.most_common(5))
            topo_str = f"[{topo}]"
            print(
                f"Ep {episode:4d} {topo_str:<8} | "
                f"train_r={metrics['episode_rewards'][-1]:+.3f} "
                f"train_blk={metrics['episode_blocking'][-1]:.2f} "
                f"avg_fs={avg_fs:.1f} "
                f"eps={agent.epsilon:.3f} | "
                f"eval_r={eval_m['avg_reward']:+.3f} "
                f"eval_blk={eval_m['blocking_rate']:.2f} | "
                f"base_r={base_m['avg_reward']:+.3f} "
                f"base_blk={base_m['blocking_rate']:.2f}"
            )
            if reason_str:
                print(f"         Reasons: {reason_str}")

    # Save checkpoint
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
        },
        str(ckpt_path),
    )
    print(f"Checkpoint saved to {ckpt_path}")
    print("=" * 70)
    return agent, metrics


def main():
    parser = argparse.ArgumentParser(description="Agent-R multi-topology DQN training")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topologies",
        type=str,
        default="net1,net2,net3",
        help="Comma-separated topology names for multi-topology training",
    )
    parser.add_argument("--mixed_splits", action="store_true", default=True,
                        help="Train with random split_id per request (default: True)")
    parser.add_argument("--no_mixed_splits", action="store_true", default=False,
                        help="Disable mixed splits, use fixed split0")
    args = parser.parse_args()
    if args.no_mixed_splits:
        args.mixed_splits = False
    train(args)


if __name__ == "__main__":
    main()
