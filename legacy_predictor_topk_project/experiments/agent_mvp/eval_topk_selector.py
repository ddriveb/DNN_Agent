"""Evaluate Top-K Selector Agent in closed loop.

Compares:
  - YinLike baseline
  - Original ImitationAgent (21D)
  - Enhanced ImitationAgent (30D)
  - TopK-Original (K=3, predictor re-rank)
  - TopK-Enhanced (K=3, predictor re-rank)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from mec_servers import MECCluster
from env_wrapper import DNNOpticalEnv
from baselines import YinLikeAgent
from fixed_trace import load_trace, generate_and_save_trace
from eval_imitation import create_env, run_agent, ImitationAgentWrapper
from enhance_state_and_retrain import EnhancedImitationAgentWrapper
from topk_selector_agent import TopKSelectorAgent


def load_predictor(path: str, max_servers: int = 128, device: str = "cpu"):
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard"),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load"),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo"),
    ]
    seeds = [42, 123, 456, 789, 2024]

    base = Path(__file__).parent.parent / "predictor_mvp"
    predictor_path = str(base / "pretrained_nsfnet_v2b.pt")

    all_results = {}

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label in scenarios:
        print(f"\n{'='*100}")
        print(f"EVALUATION: {label}")
        print(f"{'='*100}")

        num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
        trace_dir = Path(__file__).parent / "traces"

        # Load predictor once per topology
        predictor = load_predictor(predictor_path, max_servers=128, device=device)

        for seed in seeds:
            trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
            trace_path = trace_dir / trace_fname
            if not trace_path.exists():
                trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
            requests = load_trace(trace_path)

            # --- YinLike ---
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = YinLikeAgent(mec)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "YinLike"), []).append(metrics)

            # --- Original Imitation (21D) ---
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = ImitationAgentWrapper("checkpoints/imitation_agent.pt", num_servers=num_servers)
            agent.mec = env.mec
            agent.encoder = env.encoder
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "Imitation-21D"), []).append(metrics)

            # --- Enhanced Imitation (30D) ---
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = EnhancedImitationAgentWrapper("checkpoints/imitation_agent_enhanced.pt", num_servers=num_servers)
            agent.mec = env.mec
            agent.encoder = env.encoder
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "Imitation-30D"), []).append(metrics)

            # --- TopK-Original (K=3) ---
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = TopKSelectorAgent(
                "checkpoints/imitation_agent.pt",
                predictor, encoder, env.mec,
                num_servers=num_servers, top_k=3,
                alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                use_enhanced_state=False, device=device,
            )
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "TopK3-21D"), []).append(metrics)

            # --- TopK-Enhanced (K=3) ---
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = TopKSelectorAgent(
                "checkpoints/imitation_agent_enhanced.pt",
                predictor, encoder, env.mec,
                num_servers=num_servers, top_k=3,
                alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                use_enhanced_state=True, device=device,
            )
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "TopK3-30D"), []).append(metrics)

            print(f"  seed={seed} done")

    # Print summary
    print("\n" + "=" * 120)
    print("TOP-K SELECTOR CLOSED-LOOP RESULTS (5 seeds, 2000 requests)")
    print("=" * 120)
    print(f"{'Scenario':<25} {'Agent':<18} {'Blocking%':>15} {'Accept%':>15} {'Reward':>15}")
    print("-" * 120)

    for label in sorted(set(l for l, _ in all_results.keys())):
        for agent_name in ["YinLike", "Imitation-21D", "Imitation-30D", "TopK3-21D", "TopK3-30D"]:
            if (label, agent_name) not in all_results:
                continue
            runs = all_results[(label, agent_name)]
            br = [r["blocking_rate"] for r in runs]
            ar = [r["acceptance_rate"] for r in runs]
            rw = [r["avg_reward"] for r in runs]
            print(f"{label:<25} {agent_name:<18} {np.mean(br)*100:>7.2f}±{np.std(br)*100:<5.2f}% "
                  f"{np.mean(ar)*100:>7.2f}±{np.std(ar)*100:<5.2f}% {np.mean(rw):>8.3f}±{np.std(rw):<5.3f}")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "topk_selector_eval.json", "w") as f:
        serializable = {}
        for (label, agent), runs in all_results.items():
            serializable[f"{label}__{agent}"] = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                                   for k, v in r.items()} for r in runs]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir / 'topk_selector_eval.json'}")

    return all_results


if __name__ == "__main__":
    main()
