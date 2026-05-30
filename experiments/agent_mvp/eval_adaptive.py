"""Evaluate AdaptiveRuleAgent against fixed-strategy baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json
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
from adaptive_rule_agent import AdaptiveRuleAgent
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
    for sub_name, sub_agent in getattr(agent, '_sub_agents', {}).items():
        sub_agent.mec = env.mec
        sub_agent.encoder = env.encoder
    if hasattr(agent, 'reset'):
        agent.reset()

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
    strategy_counts = {}
    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

        if hasattr(agent, 'update_history'):
            agent.update_history(result.success, result.info.get("blocking_reason"))

        if "adaptive_strategy" in info:
            strategy_counts[info["adaptive_strategy"]] = strategy_counts.get(info["adaptive_strategy"], 0) + 1

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards))
    if strategy_counts:
        metrics["strategy_distribution"] = {k: v / len(requests) for k, v in strategy_counts.items()}
    return metrics


def main(
    scenarios=None,
    seeds=None,
    predictor_configs=None,
):
    if scenarios is None:
        scenarios = [
            ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard"),
            ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load"),
            ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo"),
        ]

    if seeds is None:
        seeds = [42, 123, 456, 789, 2024]

    if predictor_configs is None:
        base = Path(__file__).parent.parent / "predictor_mvp"
        predictor_configs = [
            ("mixed-v2b", str(base / "pretrained_mixed_v2b.pt"), 128),
            ("nsfnet-v2b", str(base / "pretrained_nsfnet_v2b.pt"), 128),
        ]

    all_results = {}

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label in scenarios:
        print(f"\n{'='*90}")
        print(f"ADAPTIVE AGENT EVAL: {label}")
        print(f"{'='*90}")

        num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
        trace_dir = Path(__file__).parent / "traces"

        for seed in seeds:
            trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
            trace_path = trace_dir / trace_fname
            if not trace_path.exists():
                trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
            requests = load_trace(trace_path)

            # Load predictors and encoders for this seed
            predictors = {}
            encoders = {}
            for pname, ppath, max_srv in predictor_configs:
                if not Path(ppath).exists():
                    continue
                predictors[pname] = load_predictor(ppath, max_servers=max_srv)
                # Each predictor needs its own encoder tied to the env network
                # But we'll create fresh envs per run, so we pass encoders as templates
                # and let AdaptiveRuleAgent clone them... actually, easier:
                # We create one env to get its encoder, then pass that encoder
                env_tmp, enc_tmp, mec_tmp = create_env(topology, num_slots, num_servers, seed)
                encoders[pname] = enc_tmp

            # YinLike baseline
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = YinLikeAgent(mec)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "YinLike"), []).append(metrics)

            # Fixed strategies
            fixed_strategies = [
                ("RA-default+nsfnet", "nsfnet-v2b", 1.0, 0.5, 0.3, 0.1, False),
                ("RA-pruned+mixed", "mixed-v2b", 2.0, 0.3, 0.2, 0.05, True),
            ]
            for name, pname, alpha, beta, gamma, delta, use_prune in fixed_strategies:
                if pname not in predictors:
                    continue
                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
                if use_prune:
                    agent = PrunedRuleAgent(
                        predictors[pname], encoders[pname], mec,
                        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                        prune_deadline_ms=60, prune_server_util=0.95,
                        prune_bw_threshold=8, prune_min_p_success=0.3,
                    )
                else:
                    agent = RuleAgent(
                        predictors[pname], encoders[pname], mec,
                        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                    )
                metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
                all_results.setdefault((label, name), []).append(metrics)

            # Adaptive agent
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = AdaptiveRuleAgent(predictors, encoders, mec, history_window=100)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "AdaptiveRA"), []).append(metrics)

    # Aggregate and print
    print("\n" + "=" * 100)
    print("ADAPTIVE RULE AGENT: MULTI-SEED RESULTS")
    print("=" * 100)

    for label in sorted(set(l for l, _ in all_results.keys())):
        print(f"\n--- {label} ---")
        print(f"{'Agent':<25} {'Blocking%':>18} {'Accept%':>18} {'Reward':>18}")
        print("-" * 80)

        for agent_name in sorted(set(a for l2, a in all_results.keys() if l2 == label)):
            runs = all_results[(label, agent_name)]
            blocking_rates = [r["blocking_rate"] for r in runs]
            acceptance_rates = [r["acceptance_rate"] for r in runs]
            rewards = [r["avg_reward"] for r in runs]

            b_mean = np.mean(blocking_rates)
            b_std = np.std(blocking_rates)
            a_mean = np.mean(acceptance_rates)
            a_std = np.std(acceptance_rates)
            r_mean = np.mean(rewards)
            r_std = np.std(rewards)

            print(f"{agent_name:<25} {b_mean*100:>8.2f}±{b_std*100:<5.2f}% "
                  f"{a_mean*100:>8.2f}±{a_std*100:<5.2f}% "
                  f"{r_mean:>8.3f}±{r_std:<5.3f}")

            # Print strategy distribution for AdaptiveRA
            if agent_name == "AdaptiveRA" and runs and "strategy_distribution" in runs[0]:
                print(f"  Strategy dist (seed={seeds[0]}): {runs[0]['strategy_distribution']}")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "adaptive_agent_results.json", "w") as f:
        serializable = {}
        for (label, agent), runs in all_results.items():
            key = f"{label}__{agent}"
            serializable[key] = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                   for k, v in r.items()} for r in runs]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir / 'adaptive_agent_results.json'}")
    return all_results


if __name__ == "__main__":
    main()
