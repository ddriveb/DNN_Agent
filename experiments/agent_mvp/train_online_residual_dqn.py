"""Train an online residual DQN over Top-K offloading candidates.

This is the first "real RL loop" version of the current project:
  - run env online on fixed traces
  - epsilon-greedy exploration
  - replay buffer sampling
  - TD policy updates
  - target network sync

The policy still respects the current engineering decomposition:
Top-K candidate proposal + deterministic optical mapping remain in place,
and RL learns only a residual long-term action value.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dnn_models import get_split_bandwidth_map
from eval_fragmentation_mainline import ensure_trace, load_predictor
from eval_imitation import create_env
from fixed_trace import load_trace
from online_residual_rl import (
    build_q_net,
    compose_action_inputs,
    compose_augmented_q_input,
    OnlineResidualDQNAgent,
    ReplayBuffer,
    candidate_action_ids,
    serialize_state_dict,
    soft_update,
)


def preload_env(env, preload: int, seed: int) -> None:
    """Match the existing evaluation preloading behavior."""
    rng = np.random.RandomState(seed)
    bw_map = get_split_bandwidth_map()
    for _ in range(preload):
        src = rng.randint(0, env.net.NUM_NODES)
        dst = rng.randint(0, env.net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, env.net.NUM_NODES)
        bw = bw_map[rng.randint(0, 3)]
        success, path, start_slot, _ = env.mapper.map(src, dst, bw)
        if success:
            env.active_connections.append((path, start_slot, bw, rng.exponential(10.0), -1, 0.0))


def ensure_requests(trace_dir: Path, num_nodes: int, num_requests: int, arr: float, ht: float, seed: int):
    trace_path = ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed)
    return load_trace(trace_path)


def state_to_input(state_dict: Dict, action_id: int, agent: OnlineResidualDQNAgent) -> np.ndarray:
    candidate_ids = agent._candidate_action_ids(state_dict)
    return compose_augmented_q_input(
        state_dict,
        action_id,
        relative_features=agent.relative_features,
        predictor_input_feature=agent.predictor_input_feature,
        candidate_ids=candidate_ids,
        alpha=agent.alpha,
        beta=agent.beta,
        gamma=agent.gamma,
        delta=agent.delta,
        enforce_mapper_feasibility=agent.enforce_mapper_feasibility,
    )


def compute_training_reward(result, *, frag_coef: float, lfb_coef: float, risk_coef: float) -> float:
    """Optionally shape the raw env reward with fragmentation-aware penalties."""
    reward = float(result.reward)
    info = result.info or {}
    reward -= frag_coef * max(0.0, float(info.get("delta_frag", 0.0)))
    reward -= lfb_coef * max(0.0, float(info.get("delta_lfb", 0.0)))
    reward -= risk_coef * max(0.0, float(info.get("future_risk", 0.0)))
    return reward


def choose_next_action_id(
    q_net,
    agent: OnlineResidualDQNAgent,
    next_state: Dict,
    device: str,
) -> Optional[int]:
    candidate_ids = agent._candidate_action_ids(next_state)
    if candidate_ids.size == 0:
        return None

    if getattr(q_net, "set_aware", False):
        all_inputs = np.stack([state_to_input(next_state, int(a), agent) for a in range(int(next_state["num_actions"]))])
        valid_mask = np.zeros(int(next_state["num_actions"]), dtype=bool)
        valid_mask[candidate_ids] = True
        inputs_t = torch.from_numpy(all_inputs).float().to(device)
        valid_mask_t = torch.from_numpy(valid_mask).bool().to(device)
        with torch.no_grad():
            q_all = q_net(inputs_t, valid_mask_t).cpu().numpy()
        q_values = q_all[candidate_ids]
    else:
        inputs = np.stack([state_to_input(next_state, int(a), agent) for a in candidate_ids])
        inputs_t = torch.from_numpy(inputs).float().to(device)
        with torch.no_grad():
            q_values = q_net(inputs_t).cpu().numpy()

    best_action_id = None
    best_score = -float("inf")
    for idx, action_id in enumerate(candidate_ids.tolist()):
        final_score = agent._final_score(next_state, action_id, float(q_values[idx]))
        if final_score > best_score:
            best_score = final_score
            best_action_id = action_id
    return best_action_id


def compute_rank_aux_loss(
    q_net,
    agent: OnlineResidualDQNAgent,
    batch: List[Dict],
    *,
    margin: float,
    threshold: float,
    device: str,
) -> Optional[torch.Tensor]:
    """Predictor-guided ranking loss for masked-Q training.

    The predictor score is used only as a training-time ordering prior; it is
    never added to the inference-time action score in masked_q mode.
    """
    losses = []
    for item in batch:
        state = item["state"]
        if agent.decision_mode != "masked_q":
            continue
        candidate_ids = candidate_action_ids(
            state,
            p_success_min=agent.p_success_min,
            require_mapper_feasible=agent.enforce_mapper_feasibility,
        )
        if candidate_ids.size < 2:
            continue

        predictor_scores = [agent._base_score(state, int(a)) for a in candidate_ids.tolist()]
        best_idx = int(np.argmax(predictor_scores))
        worst_idx = int(np.argmin(predictor_scores))
        if predictor_scores[best_idx] <= predictor_scores[worst_idx] + threshold:
            continue

        pos_action = int(candidate_ids[best_idx])
        neg_action = int(candidate_ids[worst_idx])
        pos_input = state_to_input(state, pos_action, agent)
        neg_input = state_to_input(state, neg_action, agent)

        pair_inputs = torch.from_numpy(np.stack([pos_input, neg_input])).float().to(device)
        q_pos, q_neg = q_net(pair_inputs)
        losses.append(F.relu(margin - (q_pos - q_neg)))

    if not losses:
        return None
    return torch.stack(losses).mean()


def optimize_step(
    q_net,
    target_net,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    agent: OnlineResidualDQNAgent,
    *,
    batch_size: int,
    gamma: float,
    use_double_dqn: bool,
    rank_aux_alpha: float,
    rank_margin: float,
    rank_threshold: float,
    device: str,
) -> Optional[Dict[str, float]]:
    if len(replay) < batch_size:
        return None

    batch = replay.sample(batch_size)
    if getattr(q_net, "set_aware", False):
        all_inputs = []
        all_masks = []
        actions = []
        for item in batch:
            state = item["state"]
            all_inputs.append(np.stack([state_to_input(state, int(a), agent) for a in range(int(state["num_actions"]))]))
            candidate_ids = candidate_action_ids(
                state,
                p_success_min=agent.p_success_min if agent.decision_mode in {"masked_q", "pure_q_feasible"} else 0.0,
                require_mapper_feasible=agent.enforce_mapper_feasibility if agent.decision_mode in {"masked_q", "pure_q_feasible"} else False,
                use_topk=agent.decision_mode != "pure_q_feasible",
            )
            mask = np.zeros(int(state["num_actions"]), dtype=bool)
            mask[candidate_ids] = True
            all_masks.append(mask)
            actions.append(int(item["action"]))
        curr_inputs_t = torch.from_numpy(np.stack(all_inputs)).float().to(device)
        curr_masks_t = torch.from_numpy(np.stack(all_masks)).bool().to(device)
        q_all = q_net(curr_inputs_t, curr_masks_t)
        batch_indices = torch.arange(len(batch), device=device)
        action_indices = torch.tensor(actions, dtype=torch.long, device=device)
        q_pred = q_all[batch_indices, action_indices]
    else:
        curr_inputs = np.stack([state_to_input(item["state"], int(item["action"]), agent) for item in batch])
        curr_inputs_t = torch.from_numpy(curr_inputs).float().to(device)
        q_pred = q_net(curr_inputs_t)

    targets: List[float] = []
    for item in batch:
        reward = float(item["reward"])
        done = bool(item["done"])
        next_state = item.get("next_state")
        gamma_n = gamma ** int(item.get("n_step", 1))
        if done or next_state is None:
            targets.append(reward)
            continue

        if use_double_dqn:
            next_action_id = choose_next_action_id(q_net, agent, next_state, device)
            if next_action_id is None:
                targets.append(reward)
                continue
            if getattr(target_net, "set_aware", False):
                all_inputs = np.stack([state_to_input(next_state, int(a), agent) for a in range(int(next_state["num_actions"]))])
                candidate_ids = candidate_action_ids(
                    next_state,
                    p_success_min=agent.p_success_min if agent.decision_mode in {"masked_q", "pure_q_feasible"} else 0.0,
                    require_mapper_feasible=agent.enforce_mapper_feasibility if agent.decision_mode in {"masked_q", "pure_q_feasible"} else False,
                    use_topk=agent.decision_mode != "pure_q_feasible",
                )
                valid_mask = np.zeros(int(next_state["num_actions"]), dtype=bool)
                valid_mask[candidate_ids] = True
                next_inputs_t = torch.from_numpy(all_inputs).float().to(device)
                valid_mask_t = torch.from_numpy(valid_mask).bool().to(device)
                with torch.no_grad():
                    q_all = target_net(next_inputs_t, valid_mask_t)
                    next_q = float(q_all[int(next_action_id)].item())
            else:
                next_input = state_to_input(next_state, int(next_action_id), agent)
                next_input_t = torch.from_numpy(next_input).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    next_q = float(target_net(next_input_t).item())
            targets.append(reward + gamma_n * next_q)
        else:
            candidate_ids = candidate_action_ids(
                next_state,
                p_success_min=agent.p_success_min if agent.decision_mode in {"masked_q", "pure_q_feasible"} else 0.0,
                require_mapper_feasible=agent.enforce_mapper_feasibility if agent.decision_mode in {"masked_q", "pure_q_feasible"} else False,
                use_topk=agent.decision_mode != "pure_q_feasible",
            )
            if candidate_ids.size == 0:
                targets.append(reward)
                continue
            if getattr(target_net, "set_aware", False):
                all_inputs = np.stack([state_to_input(next_state, int(a), agent) for a in range(int(next_state["num_actions"]))])
                valid_mask = np.zeros(int(next_state["num_actions"]), dtype=bool)
                valid_mask[candidate_ids] = True
                next_inputs_t = torch.from_numpy(all_inputs).float().to(device)
                valid_mask_t = torch.from_numpy(valid_mask).bool().to(device)
                with torch.no_grad():
                    next_q = float(torch.max(target_net(next_inputs_t, valid_mask_t)).item())
            else:
                next_inputs = np.stack([state_to_input(next_state, int(a), agent) for a in candidate_ids])
                next_inputs_t = torch.from_numpy(next_inputs).float().to(device)
                with torch.no_grad():
                    next_q = float(torch.max(target_net(next_inputs_t)).item())
            targets.append(reward + gamma_n * next_q)

    target_t = torch.tensor(targets, dtype=torch.float32, device=device)
    td_loss = nn.SmoothL1Loss()(q_pred, target_t)
    rank_loss = None
    if rank_aux_alpha > 0.0:
        rank_loss = compute_rank_aux_loss(
            q_net,
            agent,
            batch,
            margin=rank_margin,
            threshold=rank_threshold,
            device=device,
        )
    loss = td_loss
    if rank_loss is not None:
        loss = loss + rank_aux_alpha * rank_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "td_loss": float(td_loss.item()),
        "rank_loss": float(rank_loss.item()) if rank_loss is not None else 0.0,
    }


def run_training_episode(
    agent: OnlineResidualDQNAgent,
    requests,
    env,
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    target_net,
    *,
    preload: int,
    epsilon: float,
    gamma: float,
    batch_size: int,
    device: str,
    train_freq: int,
    learning_starts: int,
    target_update_freq: int,
    use_double_dqn: bool,
    soft_tau: float,
    rank_aux_alpha: float,
    rank_margin: float,
    rank_threshold: float,
    reward_frag_coef: float,
    reward_lfb_coef: float,
    reward_risk_coef: float,
    n_step: int,
    global_step: int,
    rng: np.random.RandomState,
) -> Tuple[Dict, int, List[float]]:
    """Run one online episode over a fixed request trace."""
    agent.bind_runtime(env.encoder, env.mec)
    env.reset()
    preload_env(env, preload=preload, seed=int(rng.randint(0, 10_000_000)))

    rewards = []
    raw_rewards = []
    losses = []
    td_losses = []
    rank_losses = []
    pending_steps = deque()

    def discounted_reward(queue, steps):
        total = 0.0
        for idx, item in enumerate(list(queue)[:steps]):
            total += (gamma ** idx) * float(item["reward"])
        return float(total)

    for idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        state_dict = agent.build_state(req, env)
        state_serialized = serialize_state_dict(state_dict)

        if len(pending_steps) >= n_step:
            reward_n = discounted_reward(pending_steps, n_step)
            oldest = pending_steps.popleft()
            replay.add(
                {
                    "state": oldest["state"],
                    "action": oldest["action"],
                    "reward": reward_n,
                    "done": False,
                    "next_state": state_serialized,
                    "n_step": int(n_step),
                }
            )

        split_id, server_id, _state_dict, chosen = agent.act(req, env, epsilon=epsilon, rng=rng)
        result = env.step(req, split_id, server_id)
        raw_rewards.append(float(result.reward))
        shaped_reward = compute_training_reward(
            result,
            frag_coef=reward_frag_coef,
            lfb_coef=reward_lfb_coef,
            risk_coef=reward_risk_coef,
        )
        rewards.append(shaped_reward)

        pending_steps.append(
            {
            "state": state_serialized,
            "action": int(chosen["action_id"]),
            "reward": float(shaped_reward),
            }
        )

        global_step += 1
        if global_step >= learning_starts and global_step % train_freq == 0:
            loss = optimize_step(
                agent.q_net,
                target_net,
                optimizer,
                replay,
                agent,
                batch_size=batch_size,
                gamma=gamma,
                use_double_dqn=use_double_dqn,
                rank_aux_alpha=rank_aux_alpha,
                rank_margin=rank_margin,
                rank_threshold=rank_threshold,
                device=device,
            )
            if loss is not None:
                losses.append(loss["loss"])
                td_losses.append(loss["td_loss"])
                rank_losses.append(loss["rank_loss"])
        if soft_tau > 0.0:
            soft_update(target_net, agent.q_net, soft_tau)
        elif global_step > 0 and global_step % target_update_freq == 0:
            target_net.load_state_dict(agent.q_net.state_dict())

    while pending_steps:
        steps = len(pending_steps)
        reward_n = discounted_reward(pending_steps, steps)
        oldest = pending_steps.popleft()
        replay.add(
            {
                "state": oldest["state"],
                "action": oldest["action"],
                "reward": reward_n,
                "done": True,
                "next_state": None,
                "n_step": int(steps),
            }
        )
        global_step += 1
        if global_step >= learning_starts and global_step % train_freq == 0:
            loss = optimize_step(
                agent.q_net,
                target_net,
                optimizer,
                replay,
                agent,
                batch_size=batch_size,
                gamma=gamma,
                use_double_dqn=use_double_dqn,
                rank_aux_alpha=rank_aux_alpha,
                rank_margin=rank_margin,
                rank_threshold=rank_threshold,
                device=device,
            )
            if loss is not None:
                losses.append(loss["loss"])
                td_losses.append(loss["td_loss"])
                rank_losses.append(loss["rank_loss"])
        if soft_tau > 0.0:
            soft_update(target_net, agent.q_net, soft_tau)
        elif global_step > 0 and global_step % target_update_freq == 0:
            target_net.load_state_dict(agent.q_net.state_dict())

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards)) if rewards else 0.0
    metrics["avg_env_reward"] = float(np.mean(raw_rewards)) if raw_rewards else 0.0
    metrics["loss_mean"] = float(np.mean(losses)) if losses else None
    metrics["td_loss_mean"] = float(np.mean(td_losses)) if td_losses else None
    metrics["rank_loss_mean"] = float(np.mean(rank_losses)) if rank_losses else None
    return metrics, global_step, losses


def evaluate_agent(agent: OnlineResidualDQNAgent, requests, *, topology: str, num_slots: int, num_servers: int, preload: int, seed: int):
    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    agent.bind_runtime(env.encoder, env.mec)
    env.reset()
    preload_env(env, preload=preload, seed=seed)

    rewards = []
    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, _state_dict, _chosen = agent.act(req, env, epsilon=0.0, rng=np.random.RandomState(seed))
        result = env.step(req, split_id, server_id)
        rewards.append(float(result.reward))

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards)) if rewards else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="nsfnet")
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=5)
    parser.add_argument("--num_requests", type=int, default=500)
    parser.add_argument("--arrival_rate", type=float, default=5.0)
    parser.add_argument("--avg_holding_time", type=float, default=10.0)
    parser.add_argument("--preload", type=int, default=300)
    parser.add_argument("--train_seeds", default="42,123,456")
    parser.add_argument("--eval_seed", type=int, default=789)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--replay_capacity", type=int, default=50000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=500)
    parser.add_argument("--gamma_td", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=10000)
    parser.add_argument("--exploration_mode", choices=["epsilon_greedy", "epsilon_boltzmann"], default="epsilon_greedy")
    parser.add_argument("--exploration_temp", type=float, default=1.0)
    parser.add_argument("--lambda_residual", type=float, default=0.2)
    parser.add_argument("--decision_mode", choices=["residual", "masked_q", "pure_q_feasible"], default="residual")
    parser.add_argument("--p_success_min", type=float, default=0.0)
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--set_aware", action="store_true")
    parser.add_argument("--tspq", action="store_true")
    parser.add_argument("--gated_residual", action="store_true")
    parser.add_argument("--no_double_dqn", action="store_true")
    parser.add_argument("--soft_tau", type=float, default=0.0)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--relative_features", action="store_true")
    parser.add_argument("--predictor_input_feature", action="store_true")
    parser.add_argument("--disable_imitation_mask", action="store_true")
    parser.add_argument("--disable_predictor_action_features", action="store_true")
    parser.add_argument("--drop_predictor_action_features", action="store_true")
    parser.add_argument("--rank_aux_alpha", type=float, default=0.0)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--rank_threshold", type=float, default=0.05)
    parser.add_argument("--reward_frag_coef", type=float, default=0.0)
    parser.add_argument("--reward_lfb_coef", type=float, default=0.0)
    parser.add_argument("--reward_risk_coef", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint_name", default="online_residual_dqn.pt")
    parser.add_argument("--history_name", default="online_residual_dqn_history.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).parent
    trace_dir = root / "traces"
    ckpt_dir = root / "checkpoints"
    result_dir = root / "results"
    ckpt_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    predictor_path = str(root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    imitation_ckpt = str(root / "checkpoints" / "imitation_agent_enhanced.pt")

    num_nodes = 14 if args.topology == "nsfnet" else 28
    train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]

    train_traces = {
        seed: ensure_requests(trace_dir, num_nodes, args.num_requests, args.arrival_rate, args.avg_holding_time, seed)
        for seed in train_seeds
    }
    eval_requests = ensure_requests(
        trace_dir,
        num_nodes,
        args.num_requests,
        args.arrival_rate,
        args.avg_holding_time,
        args.eval_seed,
    )

    env0, encoder0, mec0 = create_env(args.topology, args.num_slots, args.num_servers, train_seeds[0])
    hidden_dims = (64, 64)
    if args.relative_features:
        input_dim = 23 + (1 if args.predictor_input_feature else 0)
    else:
        input_dim = 16 + (1 if args.predictor_input_feature else 0)
    if args.tspq:
        hidden_dims = (128, 128)
    q_net = build_q_net(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dueling=args.dueling,
        set_aware=args.set_aware,
        tspq=args.tspq,
        gated_residual=args.gated_residual,
    ).to(device)
    target_net = build_q_net(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dueling=args.dueling,
        set_aware=args.set_aware,
        tspq=args.tspq,
        gated_residual=args.gated_residual,
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(capacity=args.replay_capacity)
    load_imitation_mask = not args.disable_imitation_mask and args.decision_mode != "pure_q_feasible"
    use_predictor_action_features = not args.disable_predictor_action_features
    agent = OnlineResidualDQNAgent(
        imitation_checkpoint_path=imitation_ckpt,
        predictor=predictor,
        encoder=encoder0,
        mec_cluster=env0.mec,
        q_net=q_net,
        num_servers=args.num_servers,
        top_k=2,
        lambda_residual=args.lambda_residual,
        decision_mode=args.decision_mode,
        p_success_min=args.p_success_min,
        enforce_mapper_feasibility=True,
        use_enhanced_state=True,
        relative_features=args.relative_features,
        predictor_input_feature=args.predictor_input_feature,
        load_imitation_mask=load_imitation_mask,
        use_predictor_action_features=use_predictor_action_features,
        drop_predictor_action_features=args.drop_predictor_action_features,
        exploration_mode=args.exploration_mode,
        exploration_temp=args.exploration_temp,
        device=device,
    )

    rng = np.random.RandomState(2026)
    random.seed(2026)
    torch.manual_seed(2026)

    history = {
        "episodes": [],
        "config": vars(args),
    }
    global_step = 0
    best_eval_blocking = float("inf")
    best_state = None

    print(f"Device: {device}")
    print(f"Training online residual DQN for {args.episodes} episodes")
    print(f"Train seeds: {train_seeds}, eval seed: {args.eval_seed}")

    for episode_idx in range(1, args.episodes + 1):
        seed = train_seeds[(episode_idx - 1) % len(train_seeds)]
        requests = train_traces[seed]
        env, encoder, mec = create_env(args.topology, args.num_slots, args.num_servers, seed)
        agent.bind_runtime(env.encoder, env.mec)

        epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * np.exp(
            -global_step / max(float(args.epsilon_decay_steps), 1.0)
        )
        metrics, global_step, losses = run_training_episode(
            agent,
            requests,
            env,
            replay,
            optimizer,
            target_net,
            preload=args.preload,
            epsilon=float(epsilon),
            gamma=args.gamma_td,
            batch_size=args.batch_size,
            device=device,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            target_update_freq=args.target_update_freq,
            use_double_dqn=not args.no_double_dqn,
            soft_tau=args.soft_tau,
            rank_aux_alpha=args.rank_aux_alpha,
            rank_margin=args.rank_margin,
            rank_threshold=args.rank_threshold,
            reward_frag_coef=args.reward_frag_coef,
            reward_lfb_coef=args.reward_lfb_coef,
            reward_risk_coef=args.reward_risk_coef,
            n_step=max(1, int(args.n_step)),
            global_step=global_step,
            rng=rng,
        )

        eval_metrics = evaluate_agent(
            agent,
            eval_requests,
            topology=args.topology,
            num_slots=args.num_slots,
            num_servers=args.num_servers,
            preload=args.preload,
            seed=args.eval_seed,
        )

        if eval_metrics["blocking_rate"] < best_eval_blocking:
            best_eval_blocking = float(eval_metrics["blocking_rate"])
            best_state = {k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()}

        record = {
            "episode": episode_idx,
            "seed": seed,
            "epsilon": float(epsilon),
            "global_step": int(global_step),
            "replay_size": int(len(replay)),
            "train_blocking_rate": float(metrics["blocking_rate"]),
            "train_avg_reward": float(metrics["avg_reward"]),
            "train_avg_env_reward": float(metrics["avg_env_reward"]),
            "eval_blocking_rate": float(eval_metrics["blocking_rate"]),
            "eval_avg_reward": float(eval_metrics["avg_reward"]),
            "eval_avg_delay_ms": float(eval_metrics["avg_delay_ms"]),
            "loss_mean": float(np.mean(losses)) if losses else None,
            "td_loss_mean": float(metrics["td_loss_mean"]) if metrics["td_loss_mean"] is not None else None,
            "rank_loss_mean": float(metrics["rank_loss_mean"]) if metrics["rank_loss_mean"] is not None else None,
        }
        history["episodes"].append(record)
        print(
            f"ep={episode_idx:03d} seed={seed} eps={epsilon:.3f} "
            f"train_block={metrics['blocking_rate']*100:.2f}% "
            f"eval_block={eval_metrics['blocking_rate']*100:.2f}% "
            f"replay={len(replay):5d} "
            f"loss={record['loss_mean'] if record['loss_mean'] is not None else float('nan'):.4f}"
        )

    if best_state is not None:
        q_net.load_state_dict(best_state)

    ckpt_path = ckpt_dir / args.checkpoint_name
    torch.save(
        {
            "model_state": q_net.state_dict(),
            "target_state": target_net.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "lambda_residual": args.lambda_residual,
            "decision_mode": args.decision_mode,
            "p_success_min": args.p_success_min,
            "exploration_mode": args.exploration_mode,
            "exploration_temp": float(args.exploration_temp),
            "relative_features": bool(args.relative_features),
            "predictor_input_feature": bool(args.predictor_input_feature),
            "disable_imitation_mask": bool(args.disable_imitation_mask),
            "disable_predictor_action_features": bool(args.disable_predictor_action_features),
            "drop_predictor_action_features": bool(args.drop_predictor_action_features),
            "dueling": bool(args.dueling),
            "set_aware": bool(args.set_aware),
            "tspq": bool(args.tspq),
            "gated_residual": bool(args.gated_residual),
            "double_dqn": bool(not args.no_double_dqn),
            "soft_tau": float(args.soft_tau),
            "n_step": int(args.n_step),
            "rank_aux_alpha": float(args.rank_aux_alpha),
            "rank_margin": float(args.rank_margin),
            "rank_threshold": float(args.rank_threshold),
            "best_eval_blocking": best_eval_blocking,
            "history": history,
        },
        ckpt_path,
    )

    history_path = result_dir / args.history_name
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
