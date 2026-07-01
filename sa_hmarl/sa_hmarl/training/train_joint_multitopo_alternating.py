"""Warm-start joint alternating fine-tuning across multiple topologies.

Each episode samples a topology from {net1, net2, net3}.
Episode-level alternating updates:
  - Even episodes: optimize Agent-C, freeze Agent-R
  - Odd episodes:  optimize Agent-R, freeze Agent-C

Agent-R reward = compute_reward + team_coef * team_signal
  team_signal: +1 on success, -1 on failure/blocking.

Checkpoint outputs:
  sa_hmarl/checkpoints/joint_multitopo_c_best.pt
  sa_hmarl/checkpoints/joint_multitopo_r_best.pt
  sa_hmarl/checkpoints/joint_multitopo_c_last.pt
  sa_hmarl/checkpoints/joint_multitopo_r_last.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
from collections import Counter
from typing import Dict, List, Optional

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.agents.joint_replay_buffer import JointReplayBuffer
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import (
    compute_agent_c_reward,
    compute_reward,
    generate_requests,
    make_env,
)
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _warm_start_agent(agent, ckpt_path: str, device: str):
    """Load model weights from a checkpoint file."""
    path = Path(ckpt_path)
    if not path.exists():
        print(f"WARNING: warm-start checkpoint not found: {path}")
        return
    ckpt = load_checkpoint(path, map_location=device)
    agent.q_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_state"])
    print(f"Warm-started from {path}")


def _set_requires_grad(net, requires_grad: bool):
    for p in net.parameters():
        p.requires_grad = requires_grad


def _evaluate_joint(
    agent_c: AgentC,
    agent_r: AgentR,
    topology: str,
    seeds: List[int],
    eval_episodes: int,
    requests_per_episode: int,
    traffic_args: dict,
) -> Dict[str, float]:
    """Return eval metrics for the joint (Agent-C + Agent-R) policy."""
    env = make_env(
        topology=topology,
        num_servers=traffic_args.get("num_servers", 2),
        seed=42,
        num_slots=traffic_args.get("num_slots", 32),
        slot_bw_hz=traffic_args.get("slot_bw_hz", 1.25e9),
        guard_band_fs=traffic_args.get("guard_band_fs", 1),
        modulation_profile=traffic_args.get("modulation_profile", "default"),
    )
    old_eps_c, old_eps_r = agent_c.epsilon, agent_r.epsilon
    agent_c.epsilon = 0.0
    agent_r.epsilon = 0.0

    blocking_rates = []
    rewards = []
    mod_counter: Counter = Counter()

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for _ in range(eval_episodes):
            src = rng.randint(0, env.net.NUM_NODES)
            requests = generate_requests(
                env,
                rng,
                src,
                num_requests=requests_per_episode,
                arrival_interval=traffic_args.get("arrival_interval", 0.25),
                holding_min=traffic_args.get("holding_min", 4.0),
                holding_max=traffic_args.get("holding_max", 10.0),
                deadline_min=traffic_args.get("deadline_min", 30.0),
                deadline_max=traffic_args.get("deadline_max", 100.0),
                size_min_mb=traffic_args.get("size_min_mb", 5.0),
                size_max_mb=traffic_args.get("size_max_mb", 30.0),
                edge_cost_min=traffic_args.get("edge_cost_min", 0.5),
                edge_cost_max=traffic_args.get("edge_cost_max", 15.0),
                num_splits=traffic_args.get("num_splits", 3),
            )
            env.reset(requests)
            blocked = 0
            ep_reward = 0.0
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
                        action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                    )

                _, _, _, info = env.step(action_c, action_r)
                server = env.mec.servers[server_id]
                reward = compute_agent_c_reward(
                    info, req.deadline_ms, traffic_args.get("waste_coef", 0.8), server.utilization
                )
                ep_reward += reward
                if not info.get("success", False):
                    blocked += 1
                else:
                    mod_counter[info.get("modulation", "unknown")] += 1
            blocking_rates.append(blocked / len(requests))
            rewards.append(ep_reward / len(requests))

    agent_c.epsilon = old_eps_c
    agent_r.epsilon = old_eps_r

    total_mods = sum(mod_counter.values())
    return {
        "blocking": float(np.mean(blocking_rates)),
        "reward": float(np.mean(rewards)),
        "mod_16qam": mod_counter.get("16QAM", 0) / total_mods if total_mods > 0 else 0.0,
        "mod_qpsk": mod_counter.get("QPSK", 0) / total_mods if total_mods > 0 else 0.0,
        "mod_bpsk": mod_counter.get("BPSK", 0) / total_mods if total_mods > 0 else 0.0,
        "mod_8qam": mod_counter.get("8QAM", 0) / total_mods if total_mods > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    package_root = Path(__file__).resolve().parents[2]
    ckpt_dir = package_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = AgentC(
        input_dim=17,
        hidden_dims=(128, 64),
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        lr=args.lr_c,
        device=args.device,
    )
    agent_r = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        lr=args.lr_r,
        device=args.device,
    )

    # ---- Warm-start ------------------------------------------------------
    if args.warm_start_c:
        _warm_start_agent(agent_c, args.warm_start_c, args.device)
    if args.warm_start_r:
        _warm_start_agent(agent_r, args.warm_start_r, args.device)

    replay_buffer = JointReplayBuffer(capacity=args.buffer_capacity, seed=args.seed)
    rng = np.random.RandomState(args.seed)

    topologies = args.topologies.split(",")

    metrics: Dict[str, List[float]] = {
        "episode_rewards_c": [],
        "episode_rewards_r": [],
        "episode_rewards_global": [],
        "episode_blocking": [],
        "losses_c": [],
        "losses_r": [],
        "eval_blocking": [],
    }

    # ---- Episode-level update schedule -----------------------------------
    total_ratio = args.update_ratio_c + args.update_ratio_r
    if total_ratio <= 0:
        total_ratio = 1.0
    num_c = int(args.episodes * args.update_ratio_c / total_ratio)
    if args.update_ratio_c == 0.5 and args.update_ratio_r == 0.5:
        # Strict alternating: even episodes update C, odd episodes update R
        update_schedule = ["C" if i % 2 == 0 else "R" for i in range(args.episodes)]
    else:
        update_schedule = ["C"] * num_c + ["R"] * (args.episodes - num_c)
        rng.shuffle(update_schedule)

    # ---- Fixed eval seeds / episodes for periodic validation -------------
    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",")]
    eval_topologies = [t.strip() for t in args.topologies.split(",")]

    best_avg_blocking = float("inf")
    best_episode = -1

    prefix = args.ckpt_prefix
    ckpt_best_c = ckpt_dir / f"{prefix}_c_best.pt"
    ckpt_best_r = ckpt_dir / f"{prefix}_r_best.pt"
    ckpt_last_c = ckpt_dir / f"{prefix}_c_last.pt"
    ckpt_last_r = ckpt_dir / f"{prefix}_r_last.pt"

    # Traffic args shared across training and eval
    traffic_args = {
        "arrival_interval": args.arrival_interval,
        "holding_min": args.holding_min,
        "holding_max": args.holding_max,
        "deadline_min": args.deadline_min,
        "deadline_max": args.deadline_max,
        "size_min_mb": args.size_min_mb,
        "size_max_mb": args.size_max_mb,
        "edge_cost_min": args.edge_cost_min,
        "edge_cost_max": args.edge_cost_max,
        "num_splits": args.num_splits,
        "num_slots": args.num_slots,
        "num_servers": args.num_servers,
        "slot_bw_hz": args.slot_bw_hz,
        "guard_band_fs": args.guard_band_fs,
        "modulation_profile": args.modulation_profile,
    }

    print("=" * 70)
    print("Warm-Start Joint Multi-Topology Alternating Fine-Tuning")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"Topologies: {topologies}")
    print(f"Update schedule: C={num_c}, R={args.episodes - num_c} (per-episode)")
    print(f"Batch size: {args.batch_size}, Buffer capacity: {args.buffer_capacity}")
    print(f"Team coef: {args.team_coef}")
    print(f"LR: C={args.lr_c}, R={args.lr_r}")
    if args.warm_start_c or args.warm_start_r:
        print(f"Warm-start: C={args.warm_start_c or 'N/A'}, R={args.warm_start_r or 'N/A'}")
    print("=" * 70)

    for episode in range(args.episodes):
        topo = rng.choice(topologies)
        env = make_env(
            topology=topo.strip(),
            num_servers=args.num_servers,
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

        episode_reward_c = 0.0
        episode_reward_r = 0.0
        episode_blocked = 0
        reason_counter = Counter()

        update_agent = update_schedule[episode]
        update_c = update_agent == "C"
        _set_requires_grad(agent_c.q_net, update_c)
        _set_requires_grad(agent_r.q_net, not update_c)

        # ---- Pending transition mechanism --------------------------------
        pending: Optional[Dict] = None

        for t, req in enumerate(requests):
            # 1. Agent-C observation & action
            obs_c = build_agent_c_observation(env, req)
            action_features_c, mask_c = agent_c.build_action_features(obs_c)
            action_idx_c = agent_c.select_action(obs_c, epsilon=agent_c.epsilon)
            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

            split_id, server_id = action_c

            # 2. Agent-R observation & action
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_features_r, mask_r = agent_r.build_action_features(obs_r)
            action_idx_r = agent_r.select_action(obs_r, epsilon=agent_r.epsilon)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                )

            # 3. Environment step
            _, _, done, info = env.step(action_c, action_r)

            # 4. Reward computation
            server = env.mec.servers[server_id]
            reward_c = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            reward_r_base = compute_reward(info, args.waste_coef)
            team_signal = 1.0 if info.get("success", False) else -1.0
            reward_r = reward_r_base + args.team_coef * team_signal

            episode_reward_c += reward_c
            episode_reward_r += reward_r

            if not info.get("success", False):
                episode_blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1

            # 5. Resolve pending transition with *real* next obs
            if pending is not None:
                pending["next_obs_c_features"] = action_features_c
                pending["next_mask_c"] = mask_c
                pending["next_obs_r_features"] = action_features_r
                pending["next_mask_r"] = mask_r
                pending["done"] = False
                replay_buffer.push(**pending)

            # 6. Create new pending (current step)
            pending = {
                "obs_c_features": action_features_c,
                "mask_c": mask_c,
                "action_c_idx": action_idx_c if action_idx_c is not None else 0,
                "obs_r_features": action_features_r,
                "mask_r": mask_r,
                "action_r_idx": action_idx_r if action_idx_r is not None else 0,
                "reward_global": reward_c + reward_r,
                "reward_c": reward_c,
                "reward_r": reward_r,
                "done": True,
                "info": info,
            }

            # 7. Optimize (only the scheduled agent for this episode)
            if len(replay_buffer) >= args.learning_starts and t % args.optimize_freq == 0:
                if update_c:
                    batch = replay_buffer.sample_agent_c(args.batch_size)
                    if batch is not None:
                        loss = agent_c.optimize(batch, args.batch_size)
                        if loss is not None:
                            metrics["losses_c"].append(loss)
                else:
                    batch = replay_buffer.sample_agent_r(args.batch_size)
                    if batch is not None:
                        loss = agent_r.optimize(batch, args.batch_size)
                        if loss is not None:
                            metrics["losses_r"].append(loss)

            # 8. Target network update
            if agent_c.step_count > 0 and agent_c.step_count % args.target_update_freq == 0:
                agent_c.update_target()
            if agent_r.step_count > 0 and agent_r.step_count % args.target_update_freq == 0:
                agent_r.update_target()

        # 9. Flush last pending (terminal)
        if pending is not None:
            pending["next_obs_c_features"] = np.zeros_like(pending["obs_c_features"])
            pending["next_mask_c"] = np.zeros_like(pending["mask_c"])
            pending["next_obs_r_features"] = np.zeros_like(pending["obs_r_features"])
            pending["next_mask_r"] = np.zeros_like(pending["mask_r"])
            pending["done"] = True
            replay_buffer.push(**pending)

        # Decay exploration
        agent_c.epsilon = max(args.epsilon_min, agent_c.epsilon * args.epsilon_decay)
        agent_r.epsilon = max(args.epsilon_min, agent_r.epsilon * args.epsilon_decay)

        metrics["episode_rewards_c"].append(episode_reward_c / args.requests_per_episode)
        metrics["episode_rewards_r"].append(episode_reward_r / args.requests_per_episode)
        metrics["episode_rewards_global"].append(
            (episode_reward_c + episode_reward_r) / args.requests_per_episode
        )
        metrics["episode_blocking"].append(episode_blocked / args.requests_per_episode)

        # ---- Logging ------------------------------------------------------
        if episode % args.log_interval == 0 or episode == args.episodes - 1:
            reason_str = ", ".join(f"{k}={v}" for k, v in reason_counter.most_common(5))
            print(
                f"Ep {episode:4d} [{update_agent}] [{topo}] | "
                f"r_c={metrics['episode_rewards_c'][-1]:+.3f} "
                f"r_r={metrics['episode_rewards_r'][-1]:+.3f} "
                f"blk={metrics['episode_blocking'][-1]:.2f} "
                f"eps_c={agent_c.epsilon:.3f} eps_r={agent_r.epsilon:.3f} | "
                f"buf={len(replay_buffer)}"
            )
            if reason_str:
                print(f"         Reasons: {reason_str}")

        # ---- Periodic evaluation & best checkpoint -----------------------
        if (episode > 0 and episode % args.eval_freq == 0) or episode == args.episodes - 1:
            eval_results = {}
            for eval_topo in eval_topologies:
                eval_results[eval_topo] = _evaluate_joint(
                    agent_c, agent_r, eval_topo,
                    eval_seeds, args.eval_episodes, args.eval_requests_per_episode,
                    traffic_args,
                )
            eval_blocking_rates = [eval_results[t]["blocking"] for t in eval_topologies]
            eval_rewards = [eval_results[t]["reward"] for t in eval_topologies]
            avg_blocking = float(np.mean(eval_blocking_rates))
            avg_reward = float(np.mean(eval_rewards))
            metrics["eval_blocking"].append(avg_blocking)

            # Aggregate modulation counts across topologies
            total_mod_16qam = np.mean([eval_results[t]["mod_16qam"] for t in eval_topologies])
            total_mod_qpsk = np.mean([eval_results[t]["mod_qpsk"] for t in eval_topologies])
            total_mod_bpsk = np.mean([eval_results[t]["mod_bpsk"] for t in eval_topologies])
            total_mod_8qam = np.mean([eval_results[t]["mod_8qam"] for t in eval_topologies])

            is_best = avg_blocking < best_avg_blocking
            if is_best:
                best_avg_blocking = avg_blocking
                best_episode = episode

            # ---- CSV logging -------------------------------------------------
            if args.log_csv:
                csv_path = Path(args.log_csv)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not csv_path.exists()
                with open(csv_path, "a") as f:
                    per_topo_blk = ",".join(f"{eval_results[t]['blocking']:.6f}" for t in eval_topologies)
                    per_topo_rwd = ",".join(f"{eval_results[t]['reward']:.6f}" for t in eval_topologies)
                    if write_header:
                        topo_headers = ",".join(f"{t}_blocking" for t in eval_topologies)
                        topo_rwd_headers = ",".join(f"{t}_reward" for t in eval_topologies)
                        f.write(
                            f"episode,avg_blocking,{topo_headers},"
                            f"avg_reward,{topo_rwd_headers},"
                            "mod_16qam,mod_qpsk,mod_bpsk,mod_8qam,best_so_far\n"
                        )
                    f.write(
                        f"{episode},{avg_blocking:.6f},"
                        f"{per_topo_blk},"
                        f"{avg_reward:.6f},"
                        f"{per_topo_rwd},"
                        f"{total_mod_16qam:.6f},"
                        f"{total_mod_qpsk:.6f},"
                        f"{total_mod_bpsk:.6f},"
                        f"{total_mod_8qam:.6f},"
                        f"{int(is_best)}\n"
                    )

            per_topo_str = " | ".join(
                f"{t}:{eval_results[t]['blocking']:.3f}"
                for t in eval_topologies
            )
            print(
                f"  >> EVAL ep {episode} | avg_blk={avg_blocking:.3f} | {per_topo_str}"
            )

            if is_best:
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
                        "best_avg_blocking": best_avg_blocking,
                        "best_episode": best_episode,
                    },
                    str(ckpt_best_c),
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
                        "best_avg_blocking": best_avg_blocking,
                        "best_episode": best_episode,
                    },
                    str(ckpt_best_r),
                )
                print(f"  >> NEW BEST @ ep {best_episode} | avg_blk={best_avg_blocking:.3f}")

    # ---- Save last checkpoints -------------------------------------------
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
        str(ckpt_last_c),
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
        str(ckpt_last_r),
    )
    print(f"\nSaved last checkpoints: {ckpt_last_c}, {ckpt_last_r}")
    print(f"Best checkpoint: avg_blk={best_avg_blocking:.3f} @ ep {best_episode}")
    print("=" * 70)

    return agent_c, agent_r, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Warm-start joint multi-topology alternating fine-tuning"
    )
    parser.add_argument("--episodes", type=int, default=2000)
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
    parser.add_argument("--team_coef", type=float, default=0.02,
                        help="Team reward coefficient for Agent-R (default: 0.02)")
    parser.add_argument("--update_ratio_c", type=float, default=0.5,
                        help="Proportion of episodes updating Agent-C")
    parser.add_argument("--update_ratio_r", type=float, default=0.5,
                        help="Proportion of episodes updating Agent-R")
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=100)
    parser.add_argument("--optimize_freq", type=int, default=2)
    parser.add_argument("--buffer_capacity", type=int, default=20000)
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr_c", type=float, default=1e-4)
    parser.add_argument("--lr_r", type=float, default=5e-5)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_requests_per_episode", type=int, default=20)
    parser.add_argument("--eval_seeds", type=str, default="42,123,456")
    parser.add_argument("--topologies", type=str, default="net1,net2,net3")
    parser.add_argument(
        "--warm_start_c",
        type=str,
        default=None,
        help="Path to pre-trained Agent-C checkpoint (default: agent_c_multitopo_best.pt)",
    )
    parser.add_argument(
        "--warm_start_r",
        type=str,
        default=None,
        help="Path to pre-trained Agent-R checkpoint (default: agent_r_multitopo.pt)",
    )
    parser.add_argument(
        "--ckpt_prefix",
        type=str,
        default="joint_multitopo",
        help="Checkpoint filename prefix (default: joint_multitopo)",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="Path to CSV file for training curve logging",
    )
    args = parser.parse_args()

    # Default warm-start paths if not explicitly provided
    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    if args.warm_start_c is None:
        args.warm_start_c = str(ckpt_dir / "agent_c_multitopo_best.pt")
    if args.warm_start_r is None:
        args.warm_start_r = str(ckpt_dir / "agent_r_multitopo.pt")

    train(args)


if __name__ == "__main__":
    main()
