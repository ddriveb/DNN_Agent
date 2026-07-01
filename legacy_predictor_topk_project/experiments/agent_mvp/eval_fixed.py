"""Fixed-trace evaluation: all agents see the exact same request sequence.

This ensures fair comparison by eliminating randomness in request generation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import csv
import json
from typing import Dict, List, Tuple
from dataclasses import asdict

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor

from dnn_models import get_split_bandwidth_map
from mec_servers import MECCluster
from traffic_generator import DNNRequest
from env_wrapper import DNNOpticalEnv, StepResult
from rule_agent import RuleAgent
from baselines import RandomAgent, ShortestPathAgent, YinLikeAgent, LoadBalancedAgent
from fixed_trace import generate_and_save_trace, load_trace


def load_pretrained_predictor(checkpoint_path: str, state_dim: int = 10,
                               max_servers: int = 128, device: str = "cpu"):
    predictor = Predictor(state_dim, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def create_env(topology: str, num_slots: int, num_servers: int,
               reward_version: str, seed: int):
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=num_servers, seed=seed)
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version=reward_version)
    return env, encoder, mapper, mec, net


def run_agent_on_trace(
    env: DNNOpticalEnv,
    agent,
    requests: List[DNNRequest],
    preload: int = 300,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Run agent on a fixed request trace."""
    env.reset()
    rng = np.random.RandomState(seed)

    # --- Preload ---
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

    # --- Evaluate on fixed trace ---
    rewards = []
    logs = []

    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

        log = {
            "req_id": req.req_id,
            "model": req.model_name,
            "source_node": req.source_node,
            "deadline_ms": req.deadline_ms,
            "split_id": split_id,
            "server_id": server_id,
            "success": result.success,
            "reward": result.reward,
            "delay_ms": result.real_delay_ms,
            "bw_slots": result.bandwidth_slots,
            "blocking_reason": result.info.get("blocking_reason", "none"),
            "path_length": result.info.get("path_length", 0),
        }
        logs.append(log)

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards))
    metrics["reward_std"] = float(np.std(rewards))
    return metrics, logs


def save_csv(logs: List[Dict], fpath: Path):
    if not logs:
        return
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)


def evaluate_all(
    topology: str = "nsfnet",
    num_slots: int = 32,
    num_servers: int = 5,
    num_requests: int = 2000,
    arrival_rate: float = 5.0,
    avg_holding_time: float = 10.0,
    preload: int = 300,
    reward_version: str = "v1",
    predictor_path: str = None,
    device: str = "cpu",
    seed: int = 42,
    out_dir: Path = None,
    rule_configs: List[Tuple[str, float, float, float, float]] = None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if out_dir is None:
        out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    # --- Ensure trace exists ---
    trace_dir = Path(__file__).parent / "traces"
    trace_fname = f"trace_n{14 if topology=='nsfnet' else 28}_r{num_requests}_a{arrival_rate}_h{avg_holding_time}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
        trace_path = generate_and_save_trace(
            num_nodes=num_nodes,
            num_requests=num_requests,
            arrival_rate=arrival_rate,
            avg_holding_time=avg_holding_time,
            seed=seed,
        )
    requests = load_trace(trace_path)
    print(f"Loaded trace: {len(requests)} requests from {trace_path}")

    # --- Load predictor ---
    predictor = None
    if predictor_path is None:
        candidates = [
            Path(__file__).parent.parent / "predictor_mvp" / "pretrained_mixed_v2b.pt",
            Path(__file__).parent.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt",
        ]
        for p in candidates:
            if p.exists():
                predictor_path = str(p)
                break

    if predictor_path and Path(predictor_path).exists():
        print(f"Loading predictor: {predictor_path}")
        predictor = load_pretrained_predictor(predictor_path, state_dim=10,
                                               max_servers=128, device=device)
        print("Predictor loaded.")
    else:
        print("WARNING: No pretrained predictor found.")

    # --- Agents ---
    base_agents = {
        "Random": RandomAgent,
        "ShortestPath": ShortestPathAgent,
        "YinLike": YinLikeAgent,
        "LoadBalanced": LoadBalancedAgent,
    }

    all_results = {}
    all_logs = {}

    for name, AgentCls in base_agents.items():
        print(f"\nEvaluating {name} ...")
        env, encoder, mapper, mec, net = create_env(topology, num_slots, num_servers, reward_version, seed)
        agent = AgentCls(mec)
        metrics, logs = run_agent_on_trace(env, agent, requests, preload=preload, seed=seed)
        all_results[name] = metrics
        all_logs[name] = logs
        print(f"  Blocking: {metrics['blocking_rate']*100:.2f}% | Accept: {metrics['acceptance_rate']*100:.2f}% | "
              f"Delay: {metrics['avg_delay_ms']:.2f}ms | Reward: {metrics['avg_reward']:.3f}")
        save_csv(logs, out_dir / f"{topology}_{name}_log.csv")

    if predictor is not None and rule_configs:
        for name, alpha, beta, gamma, delta in rule_configs:
            print(f"\nEvaluating {name} ...")
            env, encoder, mapper, mec, net = create_env(topology, num_slots, num_servers, reward_version, seed)
            agent = RuleAgent(predictor, encoder, mec,
                              alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                              device=device)
            metrics, logs = run_agent_on_trace(env, agent, requests, preload=preload, seed=seed)
            all_results[name] = metrics
            all_logs[name] = logs
            print(f"  Blocking: {metrics['blocking_rate']*100:.2f}% | Accept: {metrics['acceptance_rate']*100:.2f}% | "
                  f"Delay: {metrics['avg_delay_ms']:.2f}ms | Reward: {metrics['avg_reward']:.3f}")
            save_csv(logs, out_dir / f"{topology}_{name.replace(' ', '_')}_log.csv")

    # --- Print table ---
    print("\n" + "=" * 100)
    print(f"RESULTS: {topology} | slots={num_slots} | requests={num_requests} | preload={preload}")
    print("=" * 100)
    print(f"{'Agent':<30} {'Blocking%':>10} {'Accept%':>10} {'AvgDelay(ms)':>14} {'DL-Viol%':>10} {'AvgReward':>10}")
    print("-" * 100)
    for name in sorted(all_results.keys(), key=lambda x: (0 if "RuleAgent" in x else 1, x)):
        m = all_results[name]
        print(f"{name:<30} {m['blocking_rate']*100:>9.2f}% {m['acceptance_rate']*100:>9.2f}% "
              f"{m['avg_delay_ms']:>13.2f} {m['deadline_violation_rate']*100:>9.2f}% {m['avg_reward']:>9.3f}")
    print("=" * 100)

    # --- Save JSON ---
    with open(out_dir / f"fixed_{topology}_{num_slots}s_{num_requests}r.json", "w") as f:
        serializable = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}
        json.dump(serializable, f, indent=2)

    return all_results, all_logs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="nsfnet")
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=5)
    parser.add_argument("--num_requests", type=int, default=2000)
    parser.add_argument("--arrival_rate", type=float, default=5.0)
    parser.add_argument("--avg_holding_time", type=float, default=10.0)
    parser.add_argument("--preload", type=int, default=300)
    parser.add_argument("--reward_version", type=str, default="v1")
    parser.add_argument("--predictor", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Default rule configs to test
    rule_configs = [
        ("RuleAgent-default", 1.0, 0.5, 0.3, 0.1),
        ("RuleAgent-tuned", 2.0, 0.3, 0.2, 0.05),
    ]

    evaluate_all(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        avg_holding_time=args.avg_holding_time,
        preload=args.preload,
        reward_version=args.reward_version,
        predictor_path=args.predictor,
        device=args.device,
        seed=args.seed,
        rule_configs=rule_configs,
    )
