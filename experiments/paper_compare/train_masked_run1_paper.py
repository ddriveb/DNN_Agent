"""Fine-tune MaskedDQN run1 on the paper-condition topologies."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_mvp"))

import numpy as np
import torch
import torch.optim as optim

from eval_fragmentation_mainline import load_predictor
from online_residual_rl import OnlineResidualDQNAgent, ReplayBuffer, build_q_net
from paper_env import create_paper_env
from paper_requests import generate_request_stream
from train_online_residual_dqn import run_training_episode


def build_agent_from_checkpoint(
    ckpt_path: Path,
    predictor,
    encoder,
    mec,
    *,
    num_servers: int,
    device: str,
):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = int(ckpt.get("input_dim", 16))
    hidden_dims = tuple(ckpt.get("hidden_dims", (64, 64)))
    q_net = build_q_net(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dueling=bool(ckpt.get("dueling", False)),
        set_aware=bool(ckpt.get("set_aware", False)),
        tspq=bool(ckpt.get("tspq", False)),
    ).to(device)
    q_net.load_state_dict(ckpt["model_state"])

    agent = OnlineResidualDQNAgent(
        imitation_checkpoint_path="__disabled_imitation_checkpoint__.pt",
        predictor=predictor,
        encoder=encoder,
        mec_cluster=mec,
        q_net=q_net,
        num_servers=num_servers,
        top_k=2,
        lambda_residual=float(ckpt.get("lambda_residual", 0.2)),
        decision_mode="masked_q",
        p_success_min=float(ckpt.get("p_success_min", 0.6)),
        alpha=2.0,
        beta=0.3,
        gamma=0.2,
        delta=0.05,
        enforce_mapper_feasibility=True,
        use_enhanced_state=True,
        relative_features=bool(ckpt.get("relative_features", False)),
        predictor_input_feature=bool(ckpt.get("predictor_input_feature", False)),
        device=device,
    )
    return agent, q_net, input_dim, hidden_dims


def evaluate_paper_agent(agent, topology_file: Path, n_requests: int, seed: int, *, arrival_rate: float, avg_holding_time: float):
    env, encoder, mec = create_paper_env(topology_file, load_factor=0.6, fragmentation=0.2, seed=seed)
    agent.bind_runtime(env.encoder, env.mec)
    server_nodes = env.mec.server_node_ids
    requests = generate_request_stream(
        n_requests,
        server_nodes,
        seed=seed,
        arrival_rate=arrival_rate,
        avg_holding_time=avg_holding_time,
    )

    env.reset()
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
    parser.add_argument("--topology", choices=["net1", "net2", "net3"], default="net1")
    parser.add_argument("--num_requests", type=int, default=60)
    parser.add_argument("--arrival_rate", type=float, default=5.0)
    parser.add_argument("--avg_holding_time", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--train_seeds", type=str, default="42,123,456")
    parser.add_argument("--eval_seed", type=int, default=789)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_capacity", type=int, default=50000)
    parser.add_argument("--learning_starts", type=int, default=200)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--target_update_freq", type=int, default=200)
    parser.add_argument("--gamma_td", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=5000)
    parser.add_argument("--soft_tau", type=float, default=0.0)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--checkpoint_out", default="masked_dqn_run1_paper_ft.pt")
    parser.add_argument("--history_out", default="masked_dqn_run1_paper_ft_history.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).parent
    topo_file = root / "topologies" / f"{args.topology}.txt"
    ckpt_dir = root.parent / "agent_mvp" / "checkpoints"
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    predictor_path = str(root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)

    env0, encoder0, mec0 = create_paper_env(topo_file, load_factor=0.6, fragmentation=0.2, seed=42)
    base_ckpt = Path(args.init_ckpt) if args.init_ckpt else (ckpt_dir / "masked_dqn_run1.pt")
    agent, q_net, input_dim, hidden_dims = build_agent_from_checkpoint(
        base_ckpt, predictor, encoder0, mec0, num_servers=len(mec0.servers), device=device
    )
    target_net = build_q_net(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(capacity=args.replay_capacity)

    train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]
    rng = np.random.RandomState(2026)
    random.seed(2026)
    torch.manual_seed(2026)

    history = {"episodes": [], "config": vars(args)}
    global_step = 0
    best_eval_blocking = float("inf")
    best_state = None

    print(f"Device: {device}")
    print(f"Fine-tuning run1 on {args.topology} for {args.episodes} episodes")

    for episode_idx in range(1, args.episodes + 1):
        seed = train_seeds[(episode_idx - 1) % len(train_seeds)]
        env, encoder, mec = create_paper_env(topo_file, load_factor=0.6, fragmentation=0.2, seed=seed)
        agent.bind_runtime(env.encoder, env.mec)
        requests = generate_request_stream(
            args.num_requests,
            env.mec.server_node_ids,
            seed=seed,
            arrival_rate=args.arrival_rate,
            avg_holding_time=args.avg_holding_time,
        )

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
            preload=0,
            epsilon=float(epsilon),
            gamma=args.gamma_td,
            batch_size=args.batch_size,
            device=device,
            train_freq=args.train_freq,
            learning_starts=args.learning_starts,
            target_update_freq=args.target_update_freq,
            use_double_dqn=True,
            soft_tau=args.soft_tau,
            rank_aux_alpha=0.0,
            rank_margin=0.05,
            rank_threshold=0.05,
            reward_frag_coef=0.0,
            reward_lfb_coef=0.0,
            reward_risk_coef=0.0,
            n_step=1,
            global_step=global_step,
            rng=rng,
        )

        eval_metrics = evaluate_paper_agent(
            agent,
            topo_file,
            args.num_requests,
            args.eval_seed,
            arrival_rate=args.arrival_rate,
            avg_holding_time=args.avg_holding_time,
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

    ckpt_path = result_dir / args.checkpoint_out
    torch.save(
        {
            "model_state": q_net.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "decision_mode": "masked_q",
            "p_success_min": 0.6,
            "relative_features": False,
            "predictor_input_feature": False,
            "best_eval_blocking": best_eval_blocking,
            "history": history,
            "paper_topology": args.topology,
        },
        ckpt_path,
    )
    history_path = result_dir / args.history_out
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
