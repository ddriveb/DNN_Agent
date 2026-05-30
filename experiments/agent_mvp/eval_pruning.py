"""Evaluate action pruning impact on RuleAgent performance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import time
from typing import Dict

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dnn_models import get_split_bandwidth_map
from mec_servers import MECCluster
from env_wrapper import DNNOpticalEnv
from rule_agent import RuleAgent
from pruned_rule_agent import PrunedRuleAgent
from baselines import YinLikeAgent
from fixed_trace import load_trace, generate_and_save_trace


def load_predictor(path: str):
    predictor = Predictor(10, num_partitions=3, max_servers=128, hidden_dim=64)
    state = torch.load(path, map_location="cpu", weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    return predictor


def create_env(topology: str, num_slots: int, num_servers: int, seed: int):
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=num_servers, seed=seed)
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version="v1")
    return env, encoder, mec


def evaluate_agent(agent, requests, topology, num_slots, num_servers, preload, seed):
    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    env.reset()
    # CRITICAL: agent must share the same encoder (network state) and MEC as env
    if hasattr(agent, 'mec'):
        agent.mec = env.mec
    if hasattr(agent, 'encoder'):
        agent.encoder = env.encoder
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

    rewards = []
    prune_stats_agg = {
        "total_candidates": 0,
        "pruned_deadline": 0,
        "pruned_overload": 0,
        "pruned_bw_risk": 0,
        "evaluated": 0,
        "mispruned_count": 0,
    }
    t0 = time.time()

    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

        if hasattr(agent, 'prune_stats'):
            for k in prune_stats_agg:
                if k in agent.prune_stats:
                    prune_stats_agg[k] += agent.prune_stats[k]
            if "mispruned_count" in info:
                prune_stats_agg["mispruned_count"] += info["mispruned_count"]

    elapsed = time.time() - t0
    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards))
    metrics["runtime_sec"] = elapsed
    metrics["req_per_sec"] = len(requests) / elapsed
    if hasattr(agent, 'prune_stats'):
        metrics["prune_stats"] = prune_stats_agg
        total = prune_stats_agg["total_candidates"]
        if total > 0:
            metrics["prune_ratio"] = (total - prune_stats_agg["evaluated"]) / total
            metrics["misprune_ratio"] = prune_stats_agg["mispruned_count"] / total
    return metrics


def main(
    topology: str = "nsfnet",
    num_slots: int = 32,
    num_servers: int = 5,
    num_requests: int = 2000,
    arrival_rate: float = 5.0,
    avg_holding_time: float = 10.0,
    preload: int = 300,
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load trace
    num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
    trace_dir = Path(__file__).parent / "traces"
    trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arrival_rate}_h{avg_holding_time}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(num_nodes, num_requests, arrival_rate, avg_holding_time, seed)
    requests = load_trace(trace_path)

    predictor = load_predictor("../predictor_mvp/pretrained_mixed_v2b.pt")

    print(f"\n{'='*90}")
    print(f"ACTION PRUNING EVALUATION")
    print(f"Scenario: {topology} | slots={num_slots} | requests={num_requests}")
    print(f"{'='*90}")

    configs = [
        ("YinLike", None),
        ("RuleAgent-no-prune", None),
        ("RuleAgent-prune-light", {"prune_deadline_ms": 60, "prune_server_util": 0.95,
                                    "prune_bw_threshold": 8, "prune_min_p_success": 0.3}),
        ("RuleAgent-prune-medium", {"prune_deadline_ms": 80, "prune_server_util": 0.90,
                                     "prune_bw_threshold": 6, "prune_min_p_success": 0.5}),
        ("RuleAgent-prune-aggressive", {"prune_deadline_ms": 100, "prune_server_util": 0.85,
                                         "prune_bw_threshold": 4, "prune_min_p_success": 0.6}),
    ]

    results = {}
    for name, prune_kwargs in configs:
        print(f"\nEvaluating {name} ...")
        env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
        if name == "YinLike":
            agent = YinLikeAgent(mec)
        elif name == "RuleAgent-no-prune":
            agent = RuleAgent(predictor, encoder, mec,
                              alpha=2.0, beta=0.3, gamma=0.2, delta=0.05)
        else:
            agent = PrunedRuleAgent(predictor, encoder, mec,
                                    alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                                    **prune_kwargs)

        metrics = evaluate_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
        results[name] = metrics

        print(f"  Blocking: {metrics['blocking_rate']*100:.2f}% | Accept: {metrics['acceptance_rate']*100:.2f}% | "
              f"Delay: {metrics['avg_delay_ms']:.2f}ms | Reward: {metrics['avg_reward']:.3f} | "
              f"Runtime: {metrics['runtime_sec']:.2f}s")
        if "prune_ratio" in metrics:
            print(f"  Prune ratio: {metrics['prune_ratio']*100:.1f}% | "
                  f"Misprune ratio: {metrics['misprune_ratio']*100:.2f}%")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Agent':<30} {'Blocking%':>10} {'Accept%':>10} {'Delay(ms)':>10} {'Reward':>10} {'Req/s':>8} {'Prune%':>8}")
    print("-" * 90)
    for name in results:
        m = results[name]
        prune_str = f"{m.get('prune_ratio', 0)*100:.1f}%" if "prune_ratio" in m else "-"
        print(f"{name:<30} {m['blocking_rate']*100:>9.2f}% {m['acceptance_rate']*100:>9.2f}% "
              f"{m['avg_delay_ms']:>9.2f} {m['avg_reward']:>10.3f} {m['req_per_sec']:>7.1f} {prune_str:>8}")
    print("=" * 90)

    # Save
    import json
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"pruning_{topology}_{num_slots}s_{num_requests}r.json", "w") as f:
        serializable = {k: {kk: float(vv) for kk, vv in v.items() if isinstance(vv, (int, float, bool, str))}
                        for k, v in results.items()}
        json.dump(serializable, f, indent=2)

    return results


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        avg_holding_time=args.avg_holding_time,
        preload=args.preload,
        seed=args.seed,
    )
