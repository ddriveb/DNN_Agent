"""Load sweep for TopK2-30D and CorrectionNet λ=0.2.

Tests blocking rate across arrival rates [2.0, 4.0, 6.0, 8.0, 10.0].
Single seed (42) for speed; trend is what matters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json

from eval_imitation import create_env
from fixed_trace import load_trace, generate_and_save_trace
from topk_selector_agent import TopKSelectorAgent
from correction_agent import TopK2CorrectionAgent


def load_predictor(path: str, max_servers: int = 128, device: str = "cpu"):
    from predictor import Predictor
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def run_agent(agent, requests, topology, num_slots, num_servers, preload, seed):
    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    env.reset()
    if hasattr(agent, 'mec'):
        agent.mec = env.mec
    if hasattr(agent, 'encoder'):
        agent.encoder = env.encoder

    rng = np.random.RandomState(seed)
    from dnn_models import get_split_bandwidth_map
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


def main():
    device = "cpu"
    topology, num_slots, num_servers = "nsfnet", 32, 5
    num_requests, ht, preload, seed = 2000, 10.0, 300, 42
    arrival_rates = [2.0, 4.0, 6.0, 8.0, 10.0]

    base = Path(__file__).parent.parent / "predictor_mvp"
    predictor_path = str(base / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    correction_ckpt = str(Path(__file__).parent / "checkpoints" / "correction_net.pt")

    results = {
        "TopK2-30D": {},
        "CorrNet-λ0.2": {},
    }

    trace_dir = Path(__file__).parent / "traces"
    trace_dir.mkdir(exist_ok=True)

    for arr in arrival_rates:
        print(f"\nArrival rate = {arr}")
        trace_fname = f"trace_n14_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
        trace_path = trace_dir / trace_fname
        if not trace_path.exists():
            trace_path = generate_and_save_trace(14, num_requests, arr, ht, seed)
        requests = load_trace(trace_path)

        env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

        # TopK2
        agent_topk = TopKSelectorAgent(
            "checkpoints/imitation_agent_enhanced.pt",
            predictor, encoder, mec,
            num_servers=num_servers, top_k=2,
            alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
            use_enhanced_state=True, device=device,
        )
        m = run_agent(agent_topk, requests, topology, num_slots, num_servers, preload, seed)
        results["TopK2-30D"][str(arr)] = m
        print(f"  TopK2:    blocking={m['blocking_rate']*100:.2f}% reward={m['avg_reward']:.3f}")

        # CorrectionNet
        agent_corr = TopK2CorrectionAgent(
            correction_ckpt, predictor, encoder, mec,
            num_servers=num_servers, top_k=2, lambda_corr=0.2,
            alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
            use_enhanced_state=True, device=device,
        )
        m = run_agent(agent_corr, requests, topology, num_slots, num_servers, preload, seed)
        results["CorrNet-λ0.2"][str(arr)] = m
        print(f"  CorrNet:  blocking={m['blocking_rate']*100:.2f}% reward={m['avg_reward']:.3f}")

    out_path = Path(__file__).parent / "results" / "load_sweep_topk_corrnet.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("LOAD SWEEP SUMMARY (NSFNET, 32 slots, seed=42)")
    print("=" * 60)
    print(f"{'Arrival Rate':>12} {'TopK2 Block%':>14} {'CorrNet Block%':>16} {'Improvement':>12}")
    print("-" * 60)
    for arr in arrival_rates:
        topk_br = results["TopK2-30D"][str(arr)]["blocking_rate"] * 100
        corr_br = results["CorrNet-λ0.2"][str(arr)]["blocking_rate"] * 100
        print(f"{arr:>12.1f} {topk_br:>13.2f}% {corr_br:>15.2f}% {topk_br-corr_br:>10.2f}pp")


if __name__ == "__main__":
    main()
