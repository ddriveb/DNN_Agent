"""CTDE-style joint training with a centralized critic.

The centralized critic is used only during training. Agent-C and Agent-R still
execute hierarchically and independently at evaluation time.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.centralized_critic import CentralizedCritic
from sa_hmarl.agents.joint_replay_buffer import JointReplayBuffer
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_joint_alternating import (
    _build_eval_set,
    _evaluate_on_val_set,
    _get_reference_mod_name,
    _load_frozen_reference,
    _warm_start_agent,
)
from sa_hmarl.training.utils import (
    compute_agent_c_reward,
    compute_reward,
    generate_requests,
    make_env,
)


def _sample_topology(rng: np.random.RandomState, topologies: List[str]) -> str:
    return str(rng.choice(topologies))


def train(args):
    rng = np.random.RandomState(args.seed)
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    if not topologies:
        raise ValueError("--topologies must contain at least one topology")

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
    critic = CentralizedCritic(
        c_dim=agent_c.input_dim,
        r_dim=agent_r.input_dim,
        gamma=args.gamma,
        lr=args.lr_critic,
        device=args.device,
    )

    if args.warm_start_c:
        _warm_start_agent(agent_c, args.warm_start_c, args.device)
    if args.warm_start_r:
        _warm_start_agent(agent_r, args.warm_start_r, args.device)

    ref_agent_r = None
    if args.r_bc_coef > 0 and args.warm_start_r:
        ref_agent_r = _load_frozen_reference(args.warm_start_r, mod_reg, args.device)

    replay_buffer = JointReplayBuffer(capacity=args.buffer_capacity, seed=args.seed)

    package_root = Path(__file__).resolve().parents[2]
    ckpt_dir = package_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_envs = {
        topo: make_env(
            topology=topo,
            num_servers=args.num_servers,
            seed=args.seed + 17,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
        )
        for topo in topologies
    }
    eval_sets = {
        topo: _build_eval_set(
            env,
            np.random.RandomState(args.seed + 1000 + i),
            args,
        )
        for i, (topo, env) in enumerate(eval_envs.items())
    }

    metrics: Dict[str, List[float]] = {
        "episode_blocking": [],
        "losses_c": [],
        "losses_r": [],
        "losses_critic": [],
        "losses_distill": [],
    }
    best_blocking = float("inf")
    best_episode = -1

    print("=" * 72)
    print("CTDE Joint Training with Centralized Critic")
    print(f"Topologies: {topologies}")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"Critic distill coef: {args.ctde_distill_coef}")
    if args.pm_bpsk_penalty > 0:
        print(f"PM-BPSK penalty: {args.pm_bpsk_penalty}")
    if args.r_bc_coef > 0:
        print(f"BC regularization coef: {args.r_bc_coef} "
              f"(ref={'frozen' if ref_agent_r else 'N/A'})")
    print(f"Checkpoint prefix: {args.ckpt_prefix}")
    print("=" * 72)

    for episode in range(args.episodes):
        topo = _sample_topology(rng, topologies)
        env = make_env(
            topology=topo,
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

        episode_blocked = 0
        reason_counter = Counter()
        pending: Optional[Dict] = None

        for t, req in enumerate(requests):
            obs_c = build_agent_c_observation(env, req)
            c_features, mask_c = agent_c.build_action_features(obs_c)
            action_idx_c = agent_c.select_action(obs_c, epsilon=agent_c.epsilon)
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, len(env.mec.servers)
            )
            split_id, server_id = action_c

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_features, mask_r = agent_r.build_action_features(obs_r)
            action_idx_r = agent_r.select_action(obs_r, epsilon=agent_r.epsilon)
            action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )

            _, _, _, info = env.step(action_c, action_r)
            server = env.mec.servers[server_id]
            mod_name = info.get("modulation", None)
            reward_c = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            reward_r = compute_reward(
                info,
                args.waste_coef,
                mod_name=mod_name,
                pm_bpsk_penalty=args.pm_bpsk_penalty,
            )
            if info.get("success", False):
                reward_r += args.team_coef
            else:
                reward_r -= args.team_coef
                episode_blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1

            if (args.r_bc_coef > 0 and ref_agent_r is not None
                    and info.get("success", False)
                    and mod_name == "PM-BPSK"):
                ref_mod = _get_reference_mod_name(
                    ref_agent_r, obs_r, obs_r["mod_names"]
                )
                if ref_mod is not None and ref_mod in ("BPSK", "QPSK", "8QAM"):
                    reward_r -= args.r_bc_coef

            # Keep a single global cooperative signal for the centralized critic.
            reward_global = reward_c + reward_r

            if pending is not None:
                pending["next_obs_c_features"] = c_features
                pending["next_mask_c"] = mask_c
                pending["next_obs_r_features"] = r_features
                pending["next_mask_r"] = mask_r
                pending["done"] = False
                replay_buffer.push(**pending)

            pending = {
                "obs_c_features": c_features,
                "mask_c": mask_c,
                "action_c_idx": action_idx_c if action_idx_c is not None else 0,
                "obs_r_features": r_features,
                "mask_r": mask_r,
                "action_r_idx": action_idx_r if action_idx_r is not None else 0,
                "reward_global": reward_global,
                "reward_c": reward_c,
                "reward_r": reward_r,
                "done": True,
                "info": info,
            }

            if len(replay_buffer) >= args.learning_starts and t % args.optimize_freq == 0:
                batch_joint = replay_buffer.sample(args.batch_size)
                critic_loss = critic.optimize(batch_joint, agent_c, agent_r)
                if critic_loss is not None:
                    metrics["losses_critic"].append(critic_loss)

                batch_c = replay_buffer.sample_agent_c(args.batch_size)
                loss_c = agent_c.optimize(batch_c, args.batch_size)
                if loss_c is not None:
                    metrics["losses_c"].append(loss_c)

                batch_r = replay_buffer.sample_agent_r(args.batch_size)
                loss_r = agent_r.optimize(batch_r, args.batch_size)
                if loss_r is not None:
                    metrics["losses_r"].append(loss_r)

                distill_loss = critic.distill_agents(
                    batch_joint, agent_c, agent_r, args.ctde_distill_coef
                )
                if distill_loss is not None:
                    metrics["losses_distill"].append(distill_loss)

            if agent_c.step_count > 0 and agent_c.step_count % args.target_update_freq == 0:
                agent_c.update_target()
            if agent_r.step_count > 0 and agent_r.step_count % args.target_update_freq == 0:
                agent_r.update_target()
            if critic.step_count > 0 and critic.step_count % args.target_update_freq == 0:
                critic.update_target()

        if pending is not None:
            pending["next_obs_c_features"] = np.zeros_like(pending["obs_c_features"])
            pending["next_mask_c"] = np.zeros_like(pending["mask_c"])
            pending["next_obs_r_features"] = np.zeros_like(pending["obs_r_features"])
            pending["next_mask_r"] = np.zeros_like(pending["mask_r"])
            pending["done"] = True
            replay_buffer.push(**pending)

        agent_c.epsilon = max(args.epsilon_min, agent_c.epsilon * args.epsilon_decay)
        agent_r.epsilon = max(args.epsilon_min, agent_r.epsilon * args.epsilon_decay)
        metrics["episode_blocking"].append(episode_blocked / args.requests_per_episode)

        if (episode > 0 and episode % args.eval_freq == 0) or episode == args.episodes - 1:
            eval_results = {
                topo: _evaluate_on_val_set(env, agent_c, agent_r, eval_sets[topo])
                for topo, env in eval_envs.items()
            }
            avg_blocking = float(np.mean([r["blocking_rate"] for r in eval_results.values()]))
            if avg_blocking < best_blocking:
                best_blocking = avg_blocking
                best_episode = episode
                _save_checkpoints(ckpt_dir, args.ckpt_prefix, "best", agent_c, agent_r, critic, args, metrics)
                per_topo = ", ".join(
                    f"{topo}={res['blocking_rate']:.3f}"
                    for topo, res in eval_results.items()
                )
                print(f"  >> Best CTDE @ ep {episode}: avg_blk={avg_blocking:.3f} | {per_topo}")

        if episode % args.log_interval == 0 or episode == args.episodes - 1:
            reasons = ", ".join(f"{k}={v}" for k, v in reason_counter.most_common(4))
            print(
                f"Ep {episode:4d} [{topo}] blk={metrics['episode_blocking'][-1]:.2f} "
                f"eps_c={agent_c.epsilon:.3f} eps_r={agent_r.epsilon:.3f} "
                f"buf={len(replay_buffer)}"
            )
            if reasons:
                print(f"         Reasons: {reasons}")

    _save_checkpoints(ckpt_dir, args.ckpt_prefix, "last", agent_c, agent_r, critic, args, metrics)
    print(f"Best CTDE checkpoint: avg_blk={best_blocking:.3f} @ ep {best_episode}")
    return agent_c, agent_r, critic, metrics


def _save_checkpoints(ckpt_dir, prefix, suffix, agent_c, agent_r, critic, args, metrics):
    torch.save(
        {
            "model_state": agent_c.q_net.state_dict(),
            "target_state": agent_c.target_net.state_dict(),
            "input_dim": agent_c.input_dim,
            "hidden_dims": (128, 64),
            "gamma": agent_c.gamma,
            "args": vars(args),
            "training_metrics": metrics,
        },
        str(ckpt_dir / f"{prefix}_c_{suffix}.pt"),
    )
    torch.save(
        {
            "model_state": agent_r.q_net.state_dict(),
            "target_state": agent_r.target_net.state_dict(),
            "input_dim": agent_r.input_dim,
            "hidden_dims": (128, 64),
            "gamma": agent_r.gamma,
            "args": vars(args),
            "training_metrics": metrics,
        },
        str(ckpt_dir / f"{prefix}_r_{suffix}.pt"),
    )
    torch.save(
        {
            "model_state": critic.q_net.state_dict(),
            "target_state": critic.target_net.state_dict(),
            "input_dim": critic.input_dim,
            "hidden_dims": critic.hidden_dims,
            "gamma": critic.gamma,
            "args": vars(args),
            "training_metrics": metrics,
        },
        str(ckpt_dir / f"{prefix}_critic_{suffix}.pt"),
    )


def main():
    parser = argparse.ArgumentParser(description="CTDE joint training with centralized critic")
    parser.add_argument("--topologies", type=str, default="net1,net2,net3")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--episodes", type=int, default=400)
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
    parser.add_argument("--team_coef", type=float, default=0.02)
    parser.add_argument("--pm_bpsk_penalty", type=float, default=0.0,
                        help="Penalty subtracted from Agent-R reward when PM-BPSK is selected")
    parser.add_argument("--r_bc_coef", type=float, default=0.0,
                        help="Penalize PM-BPSK if frozen reference Agent-R would choose BPSK/QPSK/8QAM")
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
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--ctde_distill_coef", type=float, default=0.05)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--warm_start_c", type=str, default=None)
    parser.add_argument("--warm_start_r", type=str, default=None)
    parser.add_argument("--ckpt_prefix", type=str, default="joint_ctde")
    args = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    if args.warm_start_c is None:
        args.warm_start_c = str(ckpt_dir / "agent_c_multitopo_best.pt")
    if args.warm_start_r is None:
        args.warm_start_r = str(ckpt_dir / "agent_r_multitopo.pt")

    train(args)


if __name__ == "__main__":
    main()
