"""Comprehensive evaluation: all baselines + all RuleAgent variants + all predictors.

This is the final Phase A validation script that produces the key comparison table.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json
from typing import Dict, List, Tuple

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dnn_models import get_split_bandwidth_map
from mec_servers import MECCluster
from env_wrapper import DNNOpticalEnv
from rule_agent import RuleAgent
from pruned_rule_agent import PrunedRuleAgent
from baselines import RandomAgent, ShortestPathAgent, YinLikeAgent, LoadBalancedAgent
from fixed_trace import load_trace, generate_and_save_trace


def load_predictor(path: str, max_servers: int = 128):
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location="cpu", weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    return predictor


def create_env(topology, num_slots, num_servers, seed):
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=num_servers, seed=seed)
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version="v1")
    return env, encoder, mec


def run_agent(agent, requests, topology, num_slots, num_servers, preload, seed):
    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    env.reset()
    # Ensure agent shares env's encoder and mec
    if hasattr(agent, 'mec'):
        agent.mec = env.mec
    if hasattr(agent, 'encoder'):
        agent.encoder = env.encoder
    if hasattr(agent, 'reset_prune_stats'):
        agent.reset_prune_stats()

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
    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards))
    if hasattr(agent, 'prune_stats'):
        total = agent.prune_stats.get("total_candidates", 1)
        metrics["prune_ratio"] = (total - agent.prune_stats.get("evaluated", 0)) / total if total > 0 else 0
    return metrics


def main(
    scenarios: List[Tuple[str, int, int, int, float, float]] = None,
    predictors: List[Tuple[str, str, int]] = None,
    seed: int = 42,
):
    if scenarios is None:
        scenarios = [
            ("nsfnet", 32, 5, 2000, 5.0, 10.0),
            ("nsfnet", 64, 5, 2000, 8.0, 12.0),
            ("usnet", 64, 5, 2000, 8.0, 12.0),
        ]

    if predictors is None:
        base = Path(__file__).parent.parent / "predictor_mvp"
        predictors = [
            ("mixed-v2b", str(base / "pretrained_mixed_v2b.pt"), 128),
            ("nsfnet-v2b", str(base / "pretrained_nsfnet_v2b.pt"), 128),
        ]

    # Load traces
    traces = {}
    for topology, num_slots, num_servers, num_requests, arr, ht in scenarios:
        num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
        trace_dir = Path(__file__).parent / "traces"
        trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
        trace_path = trace_dir / trace_fname
        if not trace_path.exists():
            trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
        traces[(topology, num_slots)] = load_trace(trace_path)
        print(f"Loaded trace: {topology} {num_slots}s -> {len(traces[(topology, num_slots)])} requests")

    all_results = {}

    for topology, num_slots, num_servers, num_requests, arr, ht in scenarios:
        key = f"{topology}_{num_slots}s"
        requests = traces[(topology, num_slots)]
        preload = 300

        print(f"\n{'='*90}")
        print(f"SCENARIO: {key} | requests={num_requests} | preload={preload}")
        print(f"{'='*90}")

        # Baselines (don't need predictor)
        baseline_agents = [
            ("Random", RandomAgent),
            ("ShortestPath", ShortestPathAgent),
            ("YinLike", YinLikeAgent),
            ("LoadBalanced", LoadBalancedAgent),
        ]

        for name, AgentCls in baseline_agents:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = AgentCls(mec)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results[(key, name)] = metrics
            print(f"  {name:<20}: Blocking={metrics['blocking_rate']*100:.2f}% "
                  f"Accept={metrics['acceptance_rate']*100:.2f}% "
                  f"Reward={metrics['avg_reward']:.3f}")

        # RuleAgent variants for each predictor
        for pname, ppath, max_srv in predictors:
            if not Path(ppath).exists():
                continue
            predictor = load_predictor(ppath, max_servers=max_srv)

            # RuleAgent with default weights
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = RuleAgent(predictor, encoder, mec, alpha=1.0, beta=0.5, gamma=0.3, delta=0.1)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results[(key, f"RA-default+{pname}")] = metrics
            print(f"  RA-default+{pname:<12}: Blocking={metrics['blocking_rate']*100:.2f}% "
                  f"Accept={metrics['acceptance_rate']*100:.2f}% "
                  f"Reward={metrics['avg_reward']:.3f}")

            # RuleAgent with tuned weights
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = RuleAgent(predictor, encoder, mec, alpha=2.0, beta=0.3, gamma=0.2, delta=0.05)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results[(key, f"RA-tuned+{pname}")] = metrics
            print(f"  RA-tuned+{pname:<12}: Blocking={metrics['blocking_rate']*100:.2f}% "
                  f"Accept={metrics['acceptance_rate']*100:.2f}% "
                  f"Reward={metrics['avg_reward']:.3f}")

            # RuleAgent with pruning (light)
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = PrunedRuleAgent(predictor, encoder, mec,
                                    alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                                    prune_deadline_ms=60, prune_server_util=0.95,
                                    prune_bw_threshold=8, prune_min_p_success=0.3)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results[(key, f"RA-pruned+{pname}")] = metrics
            pr_str = f"prune={metrics.get('prune_ratio', 0)*100:.0f}%" if "prune_ratio" in metrics else ""
            print(f"  RA-pruned+{pname:<12}: Blocking={metrics['blocking_rate']*100:.2f}% "
                  f"Accept={metrics['acceptance_rate']*100:.2f}% "
                  f"Reward={metrics['avg_reward']:.3f} {pr_str}")

    # Final summary table
    print("\n" + "=" * 110)
    print("FINAL COMPREHENSIVE RESULTS TABLE")
    print("=" * 110)
    print(f"{'Scenario':<18} {'Agent':<25} {'Blocking%':>10} {'Accept%':>10} {'Delay(ms)':>10} {'Reward':>10} {'vs YinLike':>12}")
    print("-" * 110)

    for key in sorted(set(k for k, _ in all_results.keys())):
        yin_key = (key, "YinLike")
        yin_br = all_results[yin_key]["blocking_rate"] if yin_key in all_results else 1.0

        # Sort agents: baselines first, then RuleAgent variants
        agents_in_scenario = [a for k, a in all_results.keys() if k == key]
        agents_in_scenario.sort(key=lambda x: (
            0 if x in ["Random", "ShortestPath", "YinLike", "LoadBalanced"] else 1,
            x
        ))

        for agent_name in agents_in_scenario:
            metrics = all_results[(key, agent_name)]
            reduction = (yin_br - metrics["blocking_rate"]) / yin_br * 100 if yin_br > 0 else 0
            print(f"{key:<18} {agent_name:<25} {metrics['blocking_rate']*100:>9.2f}% "
                  f"{metrics['acceptance_rate']*100:>9.2f}% {metrics['avg_delay_ms']:>9.2f} "
                  f"{metrics['avg_reward']:>10.3f} {reduction:>+11.1f}%")
    print("=" * 110)

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "comprehensive_results.json", "w") as f:
        serializable = {}
        for (key, agent), metrics in all_results.items():
            serializable[f"{key}__{agent}"] = {k: float(v) for k, v in metrics.items()}
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_dir / 'comprehensive_results.json'}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
