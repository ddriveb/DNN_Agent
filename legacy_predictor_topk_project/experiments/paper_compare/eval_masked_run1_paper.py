"""Evaluate MaskedDQN run1 under the paper-style topology and init conditions.

Supports online adaptation: the Q-network is first warmed up on the target
topology for a configurable number of online episodes before greedy evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_mvp"))

from baselines import YinLikeAgent
from eval_fragmentation_mainline import load_predictor
from online_residual_rl import (
    OnlineResidualDQNAgent,
    ReplayBuffer,
    build_q_net,
    soft_update,
)
from paper_env import create_paper_env
from paper_requests import generate_request_stream
from train_online_residual_dqn import run_training_episode
from yin_baselines import WOAgent, DFAgent, RFAgent


def summarize(metrics):
    return {
        "blocking_rate": float(metrics["blocking_rate"]),
        "acceptance_rate": float(metrics["acceptance_rate"]),
        "avg_delay_ms": float(metrics["avg_delay_ms"]),
        "avg_reward": float(metrics["avg_reward"]),
        "avg_frag_index": float(metrics["avg_frag_index"]),
        "avg_largest_free_block_ratio": float(metrics["avg_largest_free_block_ratio"]),
        "avg_spectrum_utilization": float(metrics["avg_spectrum_utilization"]),
    }


def run_agent_stream(agent, env, requests):
    env.reset()
    if hasattr(agent, "mec"):
        agent.mec = env.mec
    if hasattr(agent, "encoder"):
        agent.encoder = env.encoder
    if hasattr(agent, "bind_runtime"):
        agent.bind_runtime(env.encoder, env.mec)

    rewards = []
    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, _score, _info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards)) if rewards else 0.0
    return metrics


def build_run1_agent(ckpt_path: Path, predictor, encoder, mec, num_servers: int, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    q_net = build_q_net(
        input_dim=ckpt.get("input_dim", 16),
        hidden_dims=tuple(ckpt.get("hidden_dims", (64, 64))),
        dueling=ckpt.get("dueling", False),
        set_aware=ckpt.get("set_aware", False),
        tspq=ckpt.get("tspq", False),
        gated_residual=ckpt.get("gated_residual", False),
    ).to(device)
    q_net.load_state_dict(ckpt["model_state"])

    # Deliberately disable the fixed 5-server imitation top-k proposer.
    # This lets the run1 Q-network score all valid actions on 4/5/6-server topologies.
    return OnlineResidualDQNAgent(
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


def online_adapt_masked_dqn(
    agent: OnlineResidualDQNAgent,
    topology_file: Path,
    n_requests: int,
    n_adapt_episodes: int,
    device: str,
    seed_base: int = 1000,
):
    """Run online adaptation episodes on the target topology.

    The agent's Q-network is updated in-place via TD learning.
    """
    # Fresh optimizer + target net for adaptation only
    target_net = build_q_net(
        input_dim=agent.q_net.input_dim if hasattr(agent.q_net, "input_dim") else 16,
        hidden_dims=tuple(agent.q_net.hidden_dims) if hasattr(agent.q_net, "hidden_dims") else (64, 64),
        dueling=bool(getattr(agent.q_net, "dueling", False)),
        set_aware=bool(getattr(agent.q_net, "set_aware", False)),
        tspq=bool(getattr(agent.q_net, "tspq", False)),
    ).to(device)
    target_net.load_state_dict(agent.q_net.state_dict())
    optimizer = optim.Adam(agent.q_net.parameters(), lr=5e-4)
    replay = ReplayBuffer(capacity=20000)

    rng = np.random.RandomState(seed_base)
    global_step = 0

    print(f"  Online adaptation: {n_adapt_episodes} episodes on target topology")
    for ep in range(n_adapt_episodes):
        env, encoder, mec = create_paper_env(topology_file, load_factor=0.6, fragmentation=0.2, seed=seed_base + ep)
        agent.bind_runtime(env.encoder, env.mec)
        requests = generate_request_stream(
            n_requests,
            env.mec.server_node_ids,
            seed=seed_base + ep,
            arrival_rate=5.0,
            avg_holding_time=10.0,
        )

        # Fast decay: high explore early, then exploit quickly
        epsilon = 0.5 * np.exp(-ep / 2.0) + 0.05  # ep0=0.55, ep2=0.23, ep4=0.11, ep6=0.07

        metrics, global_step, losses = run_training_episode(
            agent,
            requests,
            env,
            replay,
            optimizer,
            target_net,
            preload=0,
            epsilon=float(epsilon),
            gamma=0.95,
            batch_size=32,
            device=device,
            train_freq=1,
            learning_starts=30,
            target_update_freq=100,
            use_double_dqn=True,
            soft_tau=0.0,
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
        loss_str = f"{np.mean(losses):.3f}" if losses else "None"
        print(
            f"    adapt ep={ep + 1:2d} epsilon={epsilon:.3f} "
            f"train_block={metrics['blocking_rate'] * 100:.1f}% loss={loss_str}"
        )

    agent.q_net.eval()
    print("  Adaptation complete.")


def evaluate_one(
    topology_file: Path,
    n_requests: int,
    seed: int,
    device: str,
    ckpt_override: Path | None = None,
    online_adapt_episodes: int = 0,
):
    env, encoder, mec = create_paper_env(topology_file, load_factor=0.6, fragmentation=0.2, seed=seed)
    server_nodes = env.mec.server_node_ids
    requests = generate_request_stream(
        n_requests,
        server_nodes,
        seed=seed,
        arrival_rate=5.0,
        avg_holding_time=10.0,
    )

    predictor_path = str(Path(__file__).parent.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    ckpt_path = ckpt_override or (Path(__file__).parent.parent / "agent_mvp" / "checkpoints" / "masked_dqn_run1.pt")

    methods = {
        "WO": WOAgent(env.mec),
        "DF": DFAgent(env.mec),
        "RF": RFAgent(env.mec),
        "YinLike": YinLikeAgent(env.mec),
    }

    out = {}
    for name, agent in methods.items():
        env, encoder, mec = create_paper_env(topology_file, load_factor=0.6, fragmentation=0.2, seed=seed)
        agent.mec = env.mec
        if hasattr(agent, "encoder"):
            agent.encoder = env.encoder
        metrics = run_agent_stream(agent, env, requests)
        out[name] = summarize(metrics)

    # MaskedDQN run1: build fresh, optionally online-adapt, then evaluate
    env, encoder, mec = create_paper_env(topology_file, load_factor=0.6, fragmentation=0.2, seed=seed)
    agent = build_run1_agent(ckpt_path, predictor, encoder, env.mec, len(env.mec.servers), device)

    if online_adapt_episodes > 0:
        online_adapt_masked_dqn(
            agent,
            topology_file,
            n_requests,
            online_adapt_episodes,
            device,
            seed_base=seed + 10000,
        )

    env, encoder, mec = create_paper_env(topology_file, load_factor=0.6, fragmentation=0.2, seed=seed)
    agent.bind_runtime(env.encoder, env.mec)
    metrics = run_agent_stream(agent, env, requests)
    out["MaskedDQN run1"] = summarize(metrics)

    return out


def aggregate_runs(run_dicts):
    out = {}
    for method, vals in run_dicts.items():
        out[method] = {}
        for metric_name in vals[0].keys():
            series = [float(v[metric_name]) for v in vals]
            out[method][metric_name] = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=["net1", "net2", "net3"], default="net1")
    parser.add_argument("--n_requests", type=int, default=15)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--online_adapt", type=int, default=0, help="Number of online adaptation episodes before eval")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).parent
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    topology_file = root / "topologies" / f"{args.topology}.txt"
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ckpt_override = Path(args.ckpt) if args.ckpt else None

    if args.sweep:
        request_counts = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        all_results = {}
        for n_requests in request_counts:
            per_seed = [
                evaluate_one(topology_file, n_requests, seed, device, ckpt_override, args.online_adapt)
                for seed in seeds
            ]
            by_method = {}
            for method in per_seed[0].keys():
                by_method[method] = [seed_res[method] for seed_res in per_seed]
            all_results[str(n_requests)] = aggregate_runs(by_method)
        adapt_suffix = f"_adapt{args.online_adapt}" if args.online_adapt > 0 else ""
        out_path = out_dir / f"masked_run1_paper_{args.topology}{adapt_suffix}_sweep.json"
        out_path.write_text(json.dumps(all_results, indent=2))
        print(json.dumps(all_results, indent=2))
        print(f"Saved to {out_path}")
        return

    per_seed = [
        evaluate_one(topology_file, args.n_requests, seed, device, ckpt_override, args.online_adapt)
        for seed in seeds
    ]
    if len(per_seed) == 1:
        results = per_seed[0]
    else:
        by_method = {}
        for method in per_seed[0].keys():
            by_method[method] = [seed_res[method] for seed_res in per_seed]
        results = aggregate_runs(by_method)
    out_path = out_dir / f"masked_run1_paper_{args.topology}_n{args.n_requests}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
