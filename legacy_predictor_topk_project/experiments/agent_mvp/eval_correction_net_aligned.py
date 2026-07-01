"""Aligned CorrectionNet evaluation — λ=0.0 guaranteed equivalent to TopK2-30D.

Tests λ ∈ {0.0, 0.1, 0.2} across all scenarios and 5 seeds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json

from baselines import YinLikeAgent
from fixed_trace import load_trace, generate_and_save_trace
from eval_imitation import create_env
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


def run_agent_inline(agent, requests, topology, num_slots, num_servers, preload, seed):
    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    env.reset()
    agent.mec = env.mec
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard", 14),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load", 14),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo", 28),
    ]
    seeds = [42, 123, 456, 789, 2024]

    base = Path(__file__).parent.parent / "predictor_mvp"
    predictor_path = str(base / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)

    correction_ckpt = str(Path(__file__).parent / "checkpoints" / "correction_net.pt")

    trace_dir = Path(__file__).parent / "traces"

    configs = [
        ("YinLike", None, None),
        ("TopK2-30D", None, None),
        ("CorrNet-λ0.0", "correction", 0.0),
        ("CorrNet-λ0.1", "correction", 0.1),
        ("CorrNet-λ0.2", "correction", 0.2),
    ]

    all_results = {}

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {label}")
        print(f"{'='*80}")

        for cfg_name, agent_type, lam in configs:
            print(f"\n  Testing: {cfg_name}")
            for seed in seeds:
                trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
                trace_path = trace_dir / trace_fname
                if not trace_path.exists():
                    trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
                requests = load_trace(trace_path)

                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

                if agent_type is None and cfg_name == "YinLike":
                    agent = YinLikeAgent(env.mec)
                elif agent_type is None:
                    agent = TopKSelectorAgent(
                        "checkpoints/imitation_agent_enhanced.pt",
                        predictor, encoder, env.mec,
                        num_servers=num_servers, top_k=2,
                        alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                        use_enhanced_state=True, device=device,
                    )
                else:
                    agent = TopK2CorrectionAgent(
                        correction_ckpt, predictor, encoder, env.mec,
                        num_servers=num_servers, top_k=2, lambda_corr=lam,
                        alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                        use_enhanced_state=True, device=device,
                    )

                metrics = run_agent_inline(agent, requests, topology, num_slots, num_servers, preload, seed)
                key = (label, cfg_name)
                all_results.setdefault(key, []).append(metrics)
                print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

    # Summary
    print("\n" + "=" * 120)
    print("ALIGNED CORRECTIONNET EVALUATION (5 seeds)")
    print("=" * 120)
    print(f"{'Scenario':<25} {'Agent':<18} {'Blocking%':>18} {'Accept%':>18} {'Reward':>18}")
    print("-" * 120)

    for label in sorted(set(l for l, _ in all_results.keys())):
        for cfg_name in ["YinLike", "TopK2-30D", "CorrNet-λ0.0", "CorrNet-λ0.1", "CorrNet-λ0.2"]:
            key = (label, cfg_name)
            if key not in all_results:
                continue
            runs = all_results[key]
            br = [r["blocking_rate"] for r in runs]
            ar = [r["acceptance_rate"] for r in runs]
            rw = [r["avg_reward"] for r in runs]
            print(f"{label:<25} {cfg_name:<18} {np.mean(br)*100:>8.2f}±{np.std(br)*100:<6.2f}% "
                  f"{np.mean(ar)*100:>8.2f}±{np.std(ar)*100:<6.2f}% {np.mean(rw):>10.3f}±{np.std(rw):<6.3f}")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "correction_net_eval_aligned.json", "w") as f:
        serializable = {}
        for (label, cfg_name), runs in all_results.items():
            serializable[f"{label}__{cfg_name}"] = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                                       for k, v in r.items()} for r in runs]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir / 'correction_net_eval_aligned.json'}")


if __name__ == "__main__":
    main()
