"""Stabilized joint alternating co-training for Agent-C and Agent-R.

Improvements over the basic version:
  1. Warm-start: can load pre-trained separate Agent-C / Agent-R checkpoints.
  2. Fine-grained intra-episode update ratio (--update_ratio_c / --update_ratio_r)
     instead of coarse even/odd episode alternating.
  3. Pending-transition mechanism: next_obs_r uses the *real* next-step Agent-C
     action and Agent-R observation, eliminating the extra sampling bias.
  4. Team reward defaults to 0.0 (neutral).

Checkpoint outputs:
  sa_hmarl/checkpoints/joint_alternating_c.pt
  sa_hmarl/checkpoints/joint_alternating_r.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import copy
import numpy as np
import torch
from collections import Counter
from typing import Dict, List, Optional, Tuple

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


def _load_frozen_reference(ckpt_path: str, mod_reg: ModulationRegistry,
                           device: str) -> Optional[AgentR]:
    """Load a frozen reference Agent-R for BC regularization.

    The reference is loaded from the same checkpoint used for warm-start
    and its weights are frozen (no gradients, eval mode).

    Returns None if the checkpoint cannot be loaded.
    """
    path = Path(ckpt_path)
    if not path.exists():
        print(f"WARNING: reference Agent-R checkpoint not found: {path}")
        return None
    ref_agent = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        gamma=0.95,
        epsilon=0.0,  # Greedy
        device=device,
    )
    ckpt = load_checkpoint(path, map_location=device)
    ref_agent.q_net.load_state_dict(ckpt["model_state"])
    ref_agent.target_net.load_state_dict(ckpt["target_state"])
    # Freeze all parameters
    for p in ref_agent.q_net.parameters():
        p.requires_grad = False
    for p in ref_agent.target_net.parameters():
        p.requires_grad = False
    ref_agent.q_net.eval()
    ref_agent.target_net.eval()
    print(f"Loaded frozen reference Agent-R from {path}")
    return ref_agent


def _get_reference_mod_name(ref_agent: AgentR, obs_r: Dict,
                            mod_names: List[str]) -> Optional[str]:
    """Get the modulation name that the reference Agent-R would choose.

    Returns None if no valid action exists.
    """
    action_idx = ref_agent.select_action(obs_r, epsilon=0.0)
    if action_idx is None:
        return None
    _, mod_idx, _ = decode_agent_r_action(
        action_idx, len(mod_names), 5  # max_blocks=5
    )
    if mod_idx < 0 or mod_idx >= len(mod_names):
        return None
    return mod_names[mod_idx]


def _build_eval_set(env: "SMDPEnv", rng: np.random.RandomState,
                    args) -> List[List]:
    """Build a fixed validation set of request episodes for periodic eval."""
    eval_episodes_list = []
    split_profile = getattr(args, "split_profile", "default3")
    for _ in range(args.eval_episodes):
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
            split_profile=split_profile,
            traffic_mode=getattr(args, "traffic_mode", "iid"),
            regime_stay_prob=getattr(args, "regime_stay_prob", 0.9),
        )
        eval_episodes_list.append(requests)
    return eval_episodes_list


def _evaluate_on_val_set(env: "SMDPEnv", agent_c: AgentC, agent_r: AgentR,
                         eval_episodes_list: List[List],
                         pm_bpsk_penalty: float = 0.0) -> Dict:
    """Evaluate current agent pair on a fixed validation set.

    Returns metrics including blocking_rate, pm_bpsk_share, mod distribution,
    and failure reasons.
    """
    env_eval = make_env(
        topology=env.net.topology,
        num_servers=len(env.mec.servers),
        seed=42,
        num_slots=env.net.num_slots,
        slot_bw_hz=env.fs_calc.slot_bw_hz,
        guard_band_fs=env.fs_calc.guard_band_fs,
        modulation_profile="extended" if env.mod_reg.num_formats == 7 else "default",
    )

    total_success = 0
    total_blocked = 0
    total_pm_bpsk = 0
    total_reward = 0.0
    total_fs = 0.0
    mod_counter: Counter = Counter()
    reason_counter: Counter = Counter()

    for requests in eval_episodes_list:
        env_eval.reset(requests)
        for req in requests:
            obs_c = build_agent_c_observation(env_eval, req)
            action_idx_c = agent_c.select_action(obs_c, epsilon=0.0)
            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, len(env_eval.mec.servers))

            split_id, server_id = action_c
            obs_r = build_agent_r_observation(env_eval, req, split_id, server_id)
            action_idx_r = agent_r.select_action(obs_r, epsilon=0.0)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env_eval.max_blocks
                )

            _, _, done, info = env_eval.step(action_c, action_r)
            server = env_eval.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, 0.8, server.utilization
            ) + compute_reward(
                info, 0.8,
                mod_name=info.get("modulation", None),
                pm_bpsk_penalty=pm_bpsk_penalty,
            )
            total_reward += reward

            if info.get("success", False):
                total_success += 1
                total_fs += info.get("num_slots", 0)
                mod_name = info.get("modulation", "unknown")
                mod_counter[mod_name] += 1
                if mod_name == "PM-BPSK":
                    total_pm_bpsk += 1
            else:
                total_blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1

        env_eval.drain()

    n = total_success + total_blocked
    pm_bpsk_share = total_pm_bpsk / total_success if total_success > 0 else 0.0

    return {
        "blocking_rate": total_blocked / n if n > 0 else 0.0,
        "success_rate": total_success / n if n > 0 else 0.0,
        "avg_reward": total_reward / n if n > 0 else 0.0,
        "avg_fs": total_fs / total_success if total_success > 0 else 0.0,
        "pm_bpsk_count": total_pm_bpsk,
        "pm_bpsk_share": pm_bpsk_share,
        "mod_counter": mod_counter,
        "reason_counter": reason_counter,
        "total_requests": n,
    }


def train(args):
    """Run stabilized joint alternating DQN training loop.

    Anti-overfitting extensions (all optional, default off):
      --pm_bpsk_penalty:  subtract penalty when Agent-R selects PM-BPSK
      --r_bc_coef:        penalize deviation from frozen reference Agent-R
      --pm_bpsk_max_share: filter best checkpoints by max PM-BPSK share
    """
    topology = getattr(args, "topologies", "nsfnet")
    env = make_env(
        topology=topology,
        num_servers=args.num_servers,
        seed=args.seed,
        num_slots=args.num_slots,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = AgentC(
        input_dim=17,
        hidden_dims=(128, 64),
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        lr=args.lr,
        device=args.device,
    )
    agent_r = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        lr=args.lr,
        device=args.device,
    )

    # ---- Warm-start --------------------------------------------------
    if args.warm_start_c:
        _warm_start_agent(agent_c, args.warm_start_c, args.device)
    if args.warm_start_r:
        _warm_start_agent(agent_r, args.warm_start_r, args.device)

    # ---- Frozen reference Agent-R (for BC regularization) -----------
    ref_agent_r: Optional[AgentR] = None
    if args.r_bc_coef > 0 and args.warm_start_r:
        ref_agent_r = _load_frozen_reference(
            args.warm_start_r, mod_reg, args.device
        )

    replay_buffer = JointReplayBuffer(capacity=args.buffer_capacity, seed=args.seed)
    rng = np.random.RandomState(args.seed)

    metrics: Dict[str, List[float]] = {
        "episode_rewards_c": [],
        "episode_rewards_r": [],
        "episode_rewards_global": [],
        "episode_blocking": [],
        "losses_c": [],
        "losses_r": [],
    }

    package_root = Path(__file__).resolve().parents[2]
    ckpt_dir = package_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute expected optimize steps per episode for schedule generation
    expected_opt_steps = max(1, args.requests_per_episode // args.optimize_freq)

    # ---- Validation set for periodic eval / best-checkpoint tracking --
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 1000
    eval_rng = np.random.RandomState(eval_seed)
    eval_episodes_list = _build_eval_set(env, eval_rng, args)
    best_blocking = float("inf")
    best_ckpt_metrics = None
    best_saved_ep = -1

    print("=" * 70)
    print("Stabilized Joint Alternating DQN Training")
    print(f"Topology: {topology}, Episodes: {args.episodes}, "
          f"Requests/episode: {args.requests_per_episode}")
    print(
        f"Update ratio: C={args.update_ratio_c:.2f}, R={args.update_ratio_r:.2f} "
        f"(per-episode schedule)"
    )
    print(f"Batch size: {args.batch_size}, Buffer capacity: {args.buffer_capacity}")
    print(
        f"Team reward: success={args.team_reward_success:+g}, "
        f"blocking={args.team_reward_blocking:+g}"
    )
    if args.warm_start_c or args.warm_start_r:
        print(
            f"Warm-start: C={args.warm_start_c or 'N/A'}, "
            f"R={args.warm_start_r or 'N/A'}"
        )
    if args.pm_bpsk_penalty > 0:
        print(f"PM-BPSK penalty: {args.pm_bpsk_penalty}")
    if args.r_bc_coef > 0:
        print(f"BC regularization coef: {args.r_bc_coef} "
              f"(ref={'frozen' if ref_agent_r else 'N/A'})")
    if args.pm_bpsk_max_share is not None:
        print(f"Best-ckpt PM-BPSK max share: {args.pm_bpsk_max_share}")
    print(f"Eval interval: {args.eval_interval} episodes, "
          f"val set: {args.eval_episodes} episodes")
    print(f"Checkpoint prefix: {args.ckpt_prefix}")
    print("=" * 70)

    for episode in range(args.episodes):
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

        # ---- Build intra-episode update schedule ---------------------
        # True  -> update Agent-C, False -> update Agent-R
        total_ratio = args.update_ratio_c + args.update_ratio_r
        if total_ratio <= 0:
            total_ratio = 1.0  # avoid div-by-zero
        num_c = int(expected_opt_steps * args.update_ratio_c / total_ratio)
        update_schedule = [True] * num_c + [False] * (expected_opt_steps - num_c)
        rng.shuffle(update_schedule)
        opt_step_idx = 0

        # ---- Pending transition mechanism ----------------------------
        pending: Optional[Dict] = None

        for t, req in enumerate(requests):
            # ---- 1. Agent-C observation & action ----
            obs_c = build_agent_c_observation(env, req)
            action_features_c, mask_c = agent_c.build_action_features(obs_c)
            action_idx_c = agent_c.select_action(obs_c, epsilon=agent_c.epsilon)
            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

            split_id, server_id = action_c

            # ---- 2. Agent-R observation & action ----
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_features_r, mask_r = agent_r.build_action_features(obs_r)
            action_idx_r = agent_r.select_action(obs_r, epsilon=agent_r.epsilon)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                )

            # ---- 3. Environment step ----
            _, _, done, info = env.step(action_c, action_r)

            # ---- 4. Reward computation ----
            server = env.mec.servers[server_id]
            mod_name = info.get("modulation", None)
            reward_c = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            reward_r = compute_reward(
                info, args.waste_coef,
                mod_name=mod_name,
                pm_bpsk_penalty=args.pm_bpsk_penalty,
            )
            if info.get("success", False):
                reward_r += args.team_reward_success
            else:
                reward_r += args.team_reward_blocking

            # BC regularization: penalize PM-BPSK when reference chooses higher SE
            if (args.r_bc_coef > 0 and ref_agent_r is not None
                    and info.get("success", False)
                    and mod_name == "PM-BPSK"):
                ref_mod = _get_reference_mod_name(
                    ref_agent_r, obs_r, obs_r["mod_names"]
                )
                if ref_mod is not None and ref_mod in ("BPSK", "QPSK", "8QAM"):
                    reward_r -= args.r_bc_coef

            episode_reward_c += reward_c
            episode_reward_r += reward_r

            if not info.get("success", False):
                episode_blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1

            # ---- 5. Resolve pending transition with *real* next obs ----
            if pending is not None:
                pending["next_obs_c_features"] = action_features_c
                pending["next_mask_c"] = mask_c
                pending["next_obs_r_features"] = action_features_r
                pending["next_mask_r"] = mask_r
                pending["done"] = False
                replay_buffer.push(**pending)

            # ---- 6. Create new pending (current step) ----
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
                # next_* will be filled by the next iteration
                "done": True,  # default; overwritten if not last step
                "info": info,
            }

            # ---- 7. Optimize (intra-episode schedule) ----------------
            if len(replay_buffer) >= args.learning_starts and t % args.optimize_freq == 0:
                if opt_step_idx < len(update_schedule):
                    update_c = update_schedule[opt_step_idx]
                else:
                    # Fallback: round-robin if schedule exhausted
                    update_c = (opt_step_idx % 2 == 0)
                opt_step_idx += 1

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

            # ---- 8. Target network update ----------------------------
            if agent_c.step_count > 0 and agent_c.step_count % args.target_update_freq == 0:
                agent_c.update_target()
            if agent_r.step_count > 0 and agent_r.step_count % args.target_update_freq == 0:
                agent_r.update_target()

        # ---- 9. Flush last pending (terminal) ------------------------
        if pending is not None:
            pending["next_obs_c_features"] = np.zeros_like(pending["obs_c_features"])
            pending["next_mask_c"] = np.zeros_like(pending["mask_c"])
            pending["next_obs_r_features"] = np.zeros_like(pending["obs_r_features"])
            pending["next_mask_r"] = np.zeros_like(pending["mask_r"])
            pending["done"] = True
            replay_buffer.push(**pending)

        # Decay exploration for both agents
        agent_c.epsilon = max(args.epsilon_min, agent_c.epsilon * args.epsilon_decay)
        agent_r.epsilon = max(args.epsilon_min, agent_r.epsilon * args.epsilon_decay)

        metrics["episode_rewards_c"].append(episode_reward_c / args.requests_per_episode)
        metrics["episode_rewards_r"].append(episode_reward_r / args.requests_per_episode)
        metrics["episode_rewards_global"].append(
            (episode_reward_c + episode_reward_r) / args.requests_per_episode
        )
        metrics["episode_blocking"].append(episode_blocked / args.requests_per_episode)

        # ---- Periodic evaluation & best-checkpoint tracking ---------
        if (episode > 0 and episode % args.eval_interval == 0) or episode == args.episodes - 1:
            eval_metrics = _evaluate_on_val_set(
                env, agent_c, agent_r, eval_episodes_list,
                pm_bpsk_penalty=args.pm_bpsk_penalty,
            )
            blk = eval_metrics["blocking_rate"]
            pm_bpsk_s = eval_metrics["pm_bpsk_share"]

            # Determine if this is a "best" checkpoint
            is_best = False
            if args.pm_bpsk_max_share is not None:
                # Filter: only consider if PM-BPSK share <= threshold
                if pm_bpsk_s <= args.pm_bpsk_max_share and blk < best_blocking:
                    is_best = True
                    best_blocking = blk
                    best_ckpt_metrics = eval_metrics
                    best_saved_ep = episode
                # Fallback: if this is the lowest blocking overall, track it anyway
                if blk < best_blocking:
                    # Still update best_blocking for the fallback case
                    if pm_bpsk_s > args.pm_bpsk_max_share:
                        pass  # Don't mark as best, but we still track
            else:
                if blk < best_blocking:
                    is_best = True
                    best_blocking = blk
                    best_ckpt_metrics = eval_metrics
                    best_saved_ep = episode

            # Save best checkpoints
            if is_best:
                ckpt_prefix = args.ckpt_prefix
                ckpt_c_best = ckpt_dir / f"{ckpt_prefix}_c_best.pt"
                ckpt_r_best = ckpt_dir / f"{ckpt_prefix}_r_best.pt"
                torch.save(
                    {
                        "model_state": agent_c.q_net.state_dict(),
                        "target_state": agent_c.target_net.state_dict(),
                        "input_dim": agent_c.input_dim,
                        "hidden_dims": (128, 64),
                        "gamma": agent_c.gamma,
                        "seed": args.seed,
                        "training_metrics": metrics,
                        "eval_metrics": eval_metrics,
                        "args": vars(args),
                    },
                    str(ckpt_c_best),
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
                        "eval_metrics": eval_metrics,
                        "args": vars(args),
                    },
                    str(ckpt_r_best),
                )
                mod_str = ", ".join(
                    f"{k}={v/eval_metrics['total_requests']:.1%}"
                    for k, v in eval_metrics["mod_counter"].most_common(4)
                )
                print(f"  >> Best ckpt @ ep {episode}: blk={blk:.3f}, "
                      f"PM-BPSK={pm_bpsk_s:.1%}, mods=[{mod_str}]")

        if episode % args.log_interval == 0 or episode == args.episodes - 1:
            reason_str = ", ".join(f"{k}={v}" for k, v in reason_counter.most_common(5))
            print(
                f"Ep {episode:4d} | "
                f"r_c={metrics['episode_rewards_c'][-1]:+.3f} "
                f"r_r={metrics['episode_rewards_r'][-1]:+.3f} "
                f"blk={metrics['episode_blocking'][-1]:.2f} "
                f"eps_c={agent_c.epsilon:.3f} eps_r={agent_r.epsilon:.3f} | "
                f"buf={len(replay_buffer)}"
            )
            if reason_str:
                print(f"         Reasons: {reason_str}")

    # ---- Save last checkpoints ---------------------------------------
    ckpt_prefix = args.ckpt_prefix
    ckpt_c = ckpt_dir / f"{ckpt_prefix}_c_last.pt"
    ckpt_r = ckpt_dir / f"{ckpt_prefix}_r_last.pt"
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
        str(ckpt_c),
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
        str(ckpt_r),
    )
    print(f"Saved Agent-C last checkpoint to {ckpt_c}")
    print(f"Saved Agent-R last checkpoint to {ckpt_r}")

    # ---- Final summary ------------------------------------------------
    if best_ckpt_metrics is not None:
        print(f"\nBest checkpoint: episode {best_saved_ep}")
        print(f"  Blocking: {best_ckpt_metrics['blocking_rate']:.3f}")
        print(f"  PM-BPSK share: {best_ckpt_metrics['pm_bpsk_share']:.1%}")
        mod_str = ", ".join(
            f"{k}={v/best_ckpt_metrics['total_requests']:.1%}"
            for k, v in best_ckpt_metrics["mod_counter"].most_common(4)
        )
        print(f"  Modulations: [{mod_str}]")
    print("=" * 70)

    return agent_c, agent_r, metrics


def main():
    parser = argparse.ArgumentParser(description="Stabilized joint alternating DQN training")
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
    parser.add_argument(
        "--team_reward_success",
        type=float,
        default=0.0,
        help="Extra team reward added to Agent-R on success (default: 0.0)",
    )
    parser.add_argument(
        "--team_reward_blocking",
        type=float,
        default=0.0,
        help="Extra team reward added to Agent-R on blocking/failure (default: 0.0)",
    )
    parser.add_argument(
        "--update_ratio_c",
        type=float,
        default=0.5,
        help="Per-episode proportion of optimize steps assigned to Agent-C",
    )
    parser.add_argument(
        "--update_ratio_r",
        type=float,
        default=0.5,
        help="Per-episode proportion of optimize steps assigned to Agent-R",
    )
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=100)
    parser.add_argument("--optimize_freq", type=int, default=2)
    parser.add_argument("--buffer_capacity", type=int, default=20000)
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=2,
                        help="Number of MEC servers")
    parser.add_argument("--topologies", type=str, default="net1",
                        help="Topology name (e.g. nsfnet, net1, usnet)")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--warm_start_c",
        type=str,
        default=None,
        help="Path to pre-trained Agent-C checkpoint for warm-start",
    )
    parser.add_argument(
        "--warm_start_r",
        type=str,
        default=None,
        help="Path to pre-trained Agent-R checkpoint for warm-start",
    )

    # ---- Anti-overfitting / regularization switches ------------------
    parser.add_argument(
        "--pm_bpsk_penalty",
        type=float,
        default=0.0,
        help="Penalty subtracted from Agent-R reward when PM-BPSK is selected "
             "(default: 0.0 = off). Suggested: 0.02, 0.05, 0.1",
    )
    parser.add_argument(
        "--r_bc_coef",
        type=float,
        default=0.0,
        help="BC regularization coefficient: penalize Agent-R when it selects "
             "PM-BPSK but frozen reference would use BPSK/QPSK/8QAM "
             "(default: 0.0 = off). Suggested: 0.02, 0.05",
    )
    parser.add_argument(
        "--pm_bpsk_max_share",
        type=float,
        default=None,
        help="If set, best checkpoint selection only considers eval points "
             "with PM-BPSK share <= this threshold (e.g. 0.3). "
             "Falls back to lowest-blocking if all exceed.",
    )

    # ---- Checkpoint & eval settings ----------------------------------
    parser.add_argument(
        "--ckpt_prefix",
        type=str,
        default="joint_alternating",
        help="Prefix for checkpoint filenames",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=50,
        help="Evaluate on validation set every N episodes",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Number of episodes in the validation set",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Seed for validation set generation (default: training_seed + 1000)",
    )

    # ---- Team reward (aliased for compatibility) ---------------------
    parser.add_argument(
        "--team_coef",
        type=float,
        default=None,
        help="Shorthand: sets team_reward_success=+coef, team_reward_blocking=-coef",
    )

    args = parser.parse_args()

    # Handle --team_coef shorthand
    if args.team_coef is not None:
        args.team_reward_success = args.team_coef
        args.team_reward_blocking = -args.team_coef

    train(args)


if __name__ == "__main__":
    main()
