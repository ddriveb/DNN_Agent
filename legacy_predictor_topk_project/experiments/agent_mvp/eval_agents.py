"""Phase A Evaluation: Compare Rule Agent against baselines.

Run each agent on the same sequence of requests and report:
- Blocking rate
- Average delay
- Acceptance rate
- Deadline violation rate
- Resource utilization
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json
from typing import Dict, List

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor

from dnn_models import get_split_bandwidth_map
from mec_servers import MECCluster
from traffic_generator import TrafficGenerator
from env_wrapper import DNNOpticalEnv
from rule_agent import RuleAgent
from baselines import RandomAgent, ShortestPathAgent, YinLikeAgent, LoadBalancedAgent


def load_pretrained_predictor(checkpoint_path: str, state_dim: int = 10,
                               max_servers: int = 128, device: str = "cpu"):
    """Load a pretrained predictor from checkpoint."""
    predictor = Predictor(state_dim, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def create_env(topology: str, num_slots: int, num_servers: int,
               reward_version: str, seed: int):
    """Create a fully consistent env where all components share the same network."""
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)  # mapper shares net
    encoder = Encoder(net, k=3)   # encoder shares net
    encoder.encode = encoder.encode_v2b
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=num_servers, seed=seed)
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version=reward_version)
    return env, encoder, mapper, mec, net


def run_agent_episode(env: DNNOpticalEnv,
                      agent,
                      traffic_gen: TrafficGenerator,
                      num_requests: int = 2000,
                      preload: int = 300,
                      seed: int = 42) -> Dict:
    """Run one evaluation episode with a given agent."""
    env.reset()
    traffic_gen.reset()
    rng = np.random.RandomState(seed)

    # --- Preload network with random connections ---
    bw_map = get_split_bandwidth_map()
    for _ in range(preload):
        src = rng.randint(0, env.net.NUM_NODES)
        dst = rng.randint(0, env.net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, env.net.NUM_NODES)
        part = rng.randint(0, 3)
        bw = bw_map[part]
        success, path, start_slot, _ = env.mapper.map(src, dst, bw)
        if success:
            ht = rng.exponential(10.0)
            env.active_connections.append((path, start_slot, bw, ht, -1, 0.0))

    # --- Now evaluate agent on fresh requests ---
    current_time = 0.0
    rewards = []
    decisions_log = []

    for _ in range(num_requests):
        req = traffic_gen.next_request(current_time)
        env.advance_time(req.arrival_time)
        current_time = req.arrival_time

        # Agent decides (pass env so agent can read correct network state)
        split_id, server_id, score, info = agent.decide(req, env)

        # Execute
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

        decisions_log.append({
            "req_id": req.req_id,
            "model": req.model_name,
            "split_id": split_id,
            "server_id": server_id,
            "success": result.success,
            "reward": result.reward,
            "delay_ms": result.real_delay_ms,
        })

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards))
    metrics["reward_std"] = float(np.std(rewards))
    return metrics, decisions_log


def print_comparison_table(all_results: Dict[str, Dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("PHASE A: RULE AGENT vs BASELINES")
    print("=" * 90)
    print(f"{'Agent':<25} {'Blocking%':>10} {'Accept%':>10} {'AvgDelay(ms)':>14} "
          f"{'DL-Viol%':>10} {'AvgReward':>10}")
    print("-" * 90)

    for name, metrics in all_results.items():
        print(f"{name:<25} "
              f"{metrics['blocking_rate']*100:>9.2f}% "
              f"{metrics['acceptance_rate']*100:>9.2f}% "
              f"{metrics['avg_delay_ms']:>13.2f} "
              f"{metrics['deadline_violation_rate']*100:>9.2f}% "
              f"{metrics['avg_reward']:>9.3f}")
    print("=" * 90)


def main(
    topology: str = "nsfnet",
    num_slots: int = 32,
    num_servers: int = 5,
    num_requests: int = 2000,
    arrival_rate: float = 5.0,
    avg_holding_time: float = 10.0,
    preload: int = 300,
    predictor_path: str = None,
    device: str = "cpu",
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*70}")
    print(f"Phase A Evaluation Setup")
    print(f"  Topology: {topology} | Slots: {num_slots} | Servers: {num_servers}")
    print(f"  Requests: {num_requests} | ArrRate: {arrival_rate} | HT: {avg_holding_time}")
    print(f"{'='*70}")

    # --- Build shared infrastructure (for reference) ---
    ref_env, ref_encoder, ref_mapper, ref_mec, ref_net = create_env(
        topology, num_slots, num_servers, "v1", seed
    )

    print(f"  Network nodes: {ref_net.NUM_NODES}")
    print(f"  MEC servers at nodes: {ref_mec.server_node_ids}")

    # --- Load predictor (if available) ---
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
        print(f"  Loading predictor: {predictor_path}")
        predictor = load_pretrained_predictor(predictor_path, state_dim=10,
                                               max_servers=128, device=device)
        print(f"  Predictor loaded successfully.")
    else:
        print("  WARNING: No pretrained predictor found. RuleAgent will fallback to heuristics.")

    # --- Build agents that don't need predictor ---
    base_agents = {
        "Random": RandomAgent,
        "ShortestPath": ShortestPathAgent,
        "YinLike": YinLikeAgent,
        "LoadBalanced": LoadBalancedAgent,
    }

    # --- Run evaluation ---
    all_results = {}
    all_logs = {}

    for name, AgentCls in base_agents.items():
        print(f"\n  Evaluating {name} ...")
        env, encoder, mapper, mec, net = create_env(topology, num_slots, num_servers, "v1", seed)
        agent = AgentCls(mec)
        traffic_gen = TrafficGenerator(
            num_nodes=net.NUM_NODES,
            arrival_rate=arrival_rate,
            avg_holding_time=avg_holding_time,
            seed=seed,
        )
        metrics, logs = run_agent_episode(env, agent, traffic_gen,
                                          num_requests=num_requests,
                                          preload=preload, seed=seed)
        all_results[name] = metrics
        all_logs[name] = logs
        print(f"    Blocking: {metrics['blocking_rate']*100:.2f}% | "
              f"Accept: {metrics['acceptance_rate']*100:.2f}% | "
              f"AvgDelay: {metrics['avg_delay_ms']:.2f}ms | "
              f"AvgReward: {metrics['avg_reward']:.3f}")

    # --- Rule agents (need predictor + encoder) ---
    if predictor is not None:
        rule_configs = [
            ("RuleAgent (alpha=1.0)", 1.0, 0.5, 0.3, 0.1),
            ("RuleAgent (alpha=2.0)", 2.0, 0.3, 0.2, 0.05),
            ("RuleAgent (delay-focused)", 0.5, 1.0, 0.1, 0.2),
        ]
        for name, alpha, beta, gamma, delta in rule_configs:
            print(f"\n  Evaluating {name} ...")
            env, encoder, mapper, mec, net = create_env(topology, num_slots, num_servers, "v1", seed)
            agent = RuleAgent(predictor, encoder, mec,
                              alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                              device=device)
            traffic_gen = TrafficGenerator(
                num_nodes=net.NUM_NODES,
                arrival_rate=arrival_rate,
                avg_holding_time=avg_holding_time,
                seed=seed,
            )
            metrics, logs = run_agent_episode(env, agent, traffic_gen,
                                              num_requests=num_requests,
                                              preload=preload, seed=seed)
            all_results[name] = metrics
            all_logs[name] = logs
            print(f"    Blocking: {metrics['blocking_rate']*100:.2f}% | "
                  f"Accept: {metrics['acceptance_rate']*100:.2f}% | "
                  f"AvgDelay: {metrics['avg_delay_ms']:.2f}ms | "
                  f"AvgReward: {metrics['avg_reward']:.3f}")

    # --- Print summary ---
    print_comparison_table(all_results)

    # --- Save results ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"phase_a_{topology}_{num_slots}s_{num_requests}r.json", "w") as f:
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {kk: float(vv) for kk, vv in v.items()}
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {out_dir}")

    return all_results, all_logs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase A: Rule Agent Evaluation")
    parser.add_argument("--topology", type=str, default="nsfnet")
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=5)
    parser.add_argument("--num_requests", type=int, default=2000)
    parser.add_argument("--arrival_rate", type=float, default=5.0)
    parser.add_argument("--avg_holding_time", type=float, default=10.0)
    parser.add_argument("--preload", type=int, default=300)
    parser.add_argument("--predictor", type=str, default=None,
                        help="Path to pretrained predictor .pt file")
    parser.add_argument("--device", type=str, default="cpu")
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
        predictor_path=args.predictor,
        device=args.device,
        seed=args.seed,
    )
