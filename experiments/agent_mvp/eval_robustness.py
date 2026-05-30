"""Robustness evaluation: multi-seed + load sweep.

Two modes:
  1. Multi-seed: fix scenario, run N seeds → report mean ± std
  2. Load sweep: fix seed, vary arrival_rate → report trend
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass

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
    return metrics


def multi_seed_eval(
    topology: str,
    num_slots: int,
    num_servers: int,
    num_requests: int,
    arrival_rate: float,
    avg_holding_time: float,
    preload: int,
    seeds: List[int],
    predictor_configs: List[Tuple[str, str, int]],
) -> Dict:
    """Run multiple seeds and report mean ± std."""
    num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
    trace_dir = Path(__file__).parent / "traces"

    all_results = {}

    for seed in seeds:
        trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arrival_rate}_h{avg_holding_time}_s{seed}.pkl"
        trace_path = trace_dir / trace_fname
        if not trace_path.exists():
            trace_path = generate_and_save_trace(num_nodes, num_requests, arrival_rate, avg_holding_time, seed)
        requests = load_trace(trace_path)

        # YinLike baseline
        env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
        agent = YinLikeAgent(mec)
        metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
        all_results.setdefault("YinLike", []).append(metrics)

        for pname, ppath, max_srv in predictor_configs:
            if not Path(ppath).exists():
                continue
            predictor = load_predictor(ppath, max_servers=max_srv)

            # Default weights
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = RuleAgent(predictor, encoder, mec, alpha=1.0, beta=0.5, gamma=0.3, delta=0.1)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault(f"RA-default+{pname}", []).append(metrics)

            # Tuned weights
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = RuleAgent(predictor, encoder, mec, alpha=2.0, beta=0.3, gamma=0.2, delta=0.05)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault(f"RA-tuned+{pname}", []).append(metrics)

            # Pruned
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = PrunedRuleAgent(predictor, encoder, mec,
                                    alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                                    prune_deadline_ms=60, prune_server_util=0.95,
                                    prune_bw_threshold=8, prune_min_p_success=0.3)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault(f"RA-pruned+{pname}", []).append(metrics)

    # Aggregate: mean ± std
    aggregated = {}
    for agent_name, runs in all_results.items():
        blocking_rates = [r["blocking_rate"] for r in runs]
        acceptance_rates = [r["acceptance_rate"] for r in runs]
        rewards = [r["avg_reward"] for r in runs]
        delays = [r["avg_delay_ms"] for r in runs]

        aggregated[agent_name] = {
            "blocking_mean": float(np.mean(blocking_rates)),
            "blocking_std": float(np.std(blocking_rates)),
            "accept_mean": float(np.mean(acceptance_rates)),
            "accept_std": float(np.std(acceptance_rates)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "delay_mean": float(np.mean(delays)),
            "delay_std": float(np.std(delays)),
            "n_seeds": len(seeds),
        }

    return aggregated


def load_sweep(
    topology: str,
    num_slots: int,
    num_servers: int,
    num_requests: int,
    arrival_rates: List[float],
    avg_holding_time: float,
    preload: int,
    seed: int,
    predictor_configs: List[Tuple[str, str, int]],
) -> Dict:
    """Sweep arrival rate and report trend."""
    num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
    trace_dir = Path(__file__).parent / "traces"

    all_results = {}

    for arr in arrival_rates:
        trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{avg_holding_time}_s{seed}.pkl"
        trace_path = trace_dir / trace_fname
        if not trace_path.exists():
            trace_path = generate_and_save_trace(num_nodes, num_requests, arr, avg_holding_time, seed)
        requests = load_trace(trace_path)

        # YinLike
        env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
        agent = YinLikeAgent(mec)
        metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
        all_results.setdefault("YinLike", {})[arr] = metrics

        for pname, ppath, max_srv in predictor_configs:
            if not Path(ppath).exists():
                continue
            predictor = load_predictor(ppath, max_servers=max_srv)

            for label, alpha, beta, gamma, delta in [
                ("default", 1.0, 0.5, 0.3, 0.1),
                ("tuned", 2.0, 0.3, 0.2, 0.05),
            ]:
                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
                agent = RuleAgent(predictor, encoder, mec, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
                metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
                all_results.setdefault(f"RA-{label}+{pname}", {})[arr] = metrics

            # Pruned
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = PrunedRuleAgent(predictor, encoder, mec,
                                    alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                                    prune_deadline_ms=60, prune_server_util=0.95,
                                    prune_bw_threshold=8, prune_min_p_success=0.3)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault(f"RA-pruned+{pname}", {})[arr] = metrics

    return all_results


def print_multi_seed_table(aggregated: Dict, scenario_label: str):
    print(f"\n{'='*100}")
    print(f"MULTI-SEED RESULTS: {scenario_label} (n={aggregated[list(aggregated.keys())[0]]['n_seeds']} seeds)")
    print(f"{'='*100}")
    print(f"{'Agent':<25} {'Blocking%':>18} {'Accept%':>18} {'Reward':>18} {'Delay(ms)':>18}")
    print("-" * 100)

    for name in sorted(aggregated.keys(), key=lambda x: (0 if x == "YinLike" else (1 if "default" in x else 2), x)):
        m = aggregated[name]
        print(f"{name:<25} {m['blocking_mean']*100:>8.2f}±{m['blocking_std']*100:<5.2f}% "
              f"{m['accept_mean']*100:>8.2f}±{m['accept_std']*100:<5.2f}% "
              f"{m['reward_mean']:>8.3f}±{m['reward_std']:<5.3f} "
              f"{m['delay_mean']:>8.2f}±{m['delay_std']:<5.2f}")
    print("=" * 100)


def print_load_sweep_table(sweep_results: Dict, arrival_rates: List[float]):
    print(f"\n{'='*100}")
    print(f"LOAD SWEEP RESULTS")
    print(f"{'='*100}")

    for agent_name in sorted(sweep_results.keys()):
        print(f"\n{agent_name}:")
        print(f"  {'ArrRate':>8} {'Blocking%':>10} {'Accept%':>10} {'Reward':>10}")
        for arr in arrival_rates:
            if arr in sweep_results[agent_name]:
                m = sweep_results[agent_name][arr]
                print(f"  {arr:>8.1f} {m['blocking_rate']*100:>9.2f}% {m['acceptance_rate']*100:>9.2f}% {m['avg_reward']:>10.3f}")


def main():
    predictor_configs = [
        ("mixed-v2b", str(Path(__file__).parent.parent / "predictor_mvp" / "pretrained_mixed_v2b.pt"), 128),
        ("nsfnet-v2b", str(Path(__file__).parent.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt"), 128),
    ]

    # ========== MODE 1: Multi-seed robustness ==========
    print("\n" + "=" * 100)
    print("MODE 1: MULTI-SEED ROBUSTNESS CHECK")
    print("=" * 100)

    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard"),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load"),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo"),
    ]

    seeds = [42, 123, 456, 789, 2024]
    all_multi_seed = {}

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label in scenarios:
        print(f"\nRunning {label} with {len(seeds)} seeds ...")
        agg = multi_seed_eval(topology, num_slots, num_servers, num_requests, arr, ht, preload, seeds, predictor_configs)
        all_multi_seed[label] = agg
        print_multi_seed_table(agg, label)

    # ========== MODE 2: Load sweep ==========
    print("\n" + "=" * 100)
    print("MODE 2: LOAD SWEEP (NSFNET, 32 slots)")
    print("=" * 100)

    arrival_rates = [2.0, 4.0, 6.0, 8.0, 10.0]
    sweep = load_sweep(
        topology="nsfnet",
        num_slots=32,
        num_servers=5,
        num_requests=2000,
        arrival_rates=arrival_rates,
        avg_holding_time=10.0,
        preload=300,
        seed=42,
        predictor_configs=predictor_configs,
    )
    print_load_sweep_table(sweep, arrival_rates)

    # Save all results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "robustness_multi_seed.json", "w") as f:
        serializable = {}
        for scenario, agents in all_multi_seed.items():
            serializable[scenario] = {a: {k: float(v) for k, v in m.items()} for a, m in agents.items()}
        json.dump(serializable, f, indent=2)

    with open(out_dir / "robustness_load_sweep.json", "w") as f:
        serializable = {}
        for agent, data in sweep.items():
            serializable[agent] = {arr: {k: float(v) for k, v in m.items()} for arr, m in data.items()}
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir}/robustness_multi_seed.json and robustness_load_sweep.json")


if __name__ == "__main__":
    main()
