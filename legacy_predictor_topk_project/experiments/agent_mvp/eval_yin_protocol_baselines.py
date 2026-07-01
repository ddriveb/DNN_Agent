"""Unified evaluation of Yin-protocol baselines + our methods.

Evaluates on 3 scenarios × 5 seeds:
- New baselines: WO, DF, RF, IWD-Approx
- Existing: YinLike, TopK2-30D, CorrectionNet-λ0.2

Uses fixed traces for fair comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json

from eval_imitation import create_env
from fixed_trace import load_trace, generate_and_save_trace

from baselines import YinLikeAgent
from topk_selector_agent import TopKSelectorAgent
from correction_agent import TopK2CorrectionAgent
from yin_baselines import WOAgent, DFAgent, RFAgent, IWDApproxAgent


def load_predictor(path: str, max_servers: int = 128, device: str = "cpu"):
    from predictor import Predictor
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def run_agent_inline(agent, requests, topology, num_slots, num_servers, preload, seed):
    """Run a single agent on a fixed request trace."""
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

    all_results = {}

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\n{'='*90}")
        print(f"SCENARIO: {label}")
        print(f"{'='*90}")

        # Pre-load/generate traces for all seeds
        traces = {}
        for seed in seeds:
            trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
            trace_path = trace_dir / trace_fname
            if not trace_path.exists():
                trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
            traces[seed] = load_trace(trace_path)

        # ------------------------------------------------------------------
        # 1. WO
        # ------------------------------------------------------------------
        print("\n  Testing: WO")
        for seed in seeds:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = WOAgent(env.mec)
            metrics = run_agent_inline(agent, traces[seed], topology, num_slots, num_servers, preload, seed)
            key = (label, "WO")
            all_results.setdefault(key, []).append(metrics)
            print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

        # ------------------------------------------------------------------
        # 2. DF
        # ------------------------------------------------------------------
        print("\n  Testing: DF")
        for seed in seeds:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = DFAgent(env.mec)
            metrics = run_agent_inline(agent, traces[seed], topology, num_slots, num_servers, preload, seed)
            key = (label, "DF")
            all_results.setdefault(key, []).append(metrics)
            print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

        # ------------------------------------------------------------------
        # 3. RF
        # ------------------------------------------------------------------
        print("\n  Testing: RF")
        for seed in seeds:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = RFAgent(env.mec)
            metrics = run_agent_inline(agent, traces[seed], topology, num_slots, num_servers, preload, seed)
            key = (label, "RF")
            all_results.setdefault(key, []).append(metrics)
            print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

        # ------------------------------------------------------------------
        # 4. IWD-Approx
        # ------------------------------------------------------------------
        print("\n  Testing: IWD-Approx")
        for seed in seeds:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = IWDApproxAgent(env.mec, seed=seed)
            metrics = run_agent_inline(agent, traces[seed], topology, num_slots, num_servers, preload, seed)
            key = (label, "IWD-Approx")
            all_results.setdefault(key, []).append(metrics)
            print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

        # ------------------------------------------------------------------
        # 5. YinLike
        # ------------------------------------------------------------------
        print("\n  Testing: YinLike")
        for seed in seeds:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = YinLikeAgent(env.mec)
            metrics = run_agent_inline(agent, traces[seed], topology, num_slots, num_servers, preload, seed)
            key = (label, "YinLike")
            all_results.setdefault(key, []).append(metrics)
            print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

        # ------------------------------------------------------------------
        # 6. TopK2-30D
        # ------------------------------------------------------------------
        print("\n  Testing: TopK2-30D")
        for seed in seeds:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = TopKSelectorAgent(
                "checkpoints/imitation_agent_enhanced.pt",
                predictor, encoder, env.mec,
                num_servers=num_servers, top_k=2,
                alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                use_enhanced_state=True, device=device,
            )
            metrics = run_agent_inline(agent, traces[seed], topology, num_slots, num_servers, preload, seed)
            key = (label, "TopK2-30D")
            all_results.setdefault(key, []).append(metrics)
            print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

        # ------------------------------------------------------------------
        # 7. CorrectionNet λ=0.2
        # ------------------------------------------------------------------
        print("\n  Testing: CorrectionNet-λ0.2")
        for seed in seeds:
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = TopK2CorrectionAgent(
                correction_ckpt, predictor, encoder, env.mec,
                num_servers=num_servers, top_k=2, lambda_corr=0.2,
                alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                use_enhanced_state=True, device=device,
            )
            metrics = run_agent_inline(agent, traces[seed], topology, num_slots, num_servers, preload, seed)
            key = (label, "CorrectionNet-λ0.2")
            all_results.setdefault(key, []).append(metrics)
            print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

    # =====================================================================
    # Summary tables
    # =====================================================================
    agent_order = [
        "WO", "DF", "RF", "IWD-Approx",
        "YinLike", "TopK2-30D", "CorrectionNet-λ0.2",
    ]

    print("\n" + "=" * 130)
    print("YIN PROTOCOL BASELINES + OUR METHODS (5 seeds)")
    print("=" * 130)
    print(f"{'Scenario':<25} {'Agent':<22} {'Blocking%':>18} {'Accept%':>18} {'AvgDelay(ms)':>18} {'Reward':>18}")
    print("-" * 130)

    for label in sorted(set(l for l, _ in all_results.keys())):
        for cfg_name in agent_order:
            key = (label, cfg_name)
            if key not in all_results:
                continue
            runs = all_results[key]
            br = [r["blocking_rate"] for r in runs]
            ar = [r["acceptance_rate"] for r in runs]
            dl = [r["avg_delay_ms"] for r in runs]
            rw = [r["avg_reward"] for r in runs]
            print(f"{label:<25} {cfg_name:<22} "
                  f"{np.mean(br)*100:>8.2f}±{np.std(br)*100:<6.2f}% "
                  f"{np.mean(ar)*100:>8.2f}±{np.std(ar)*100:<6.2f}% "
                  f"{np.mean(dl):>8.2f}±{np.std(dl):<6.2f} "
                  f"{np.mean(rw):>8.3f}±{np.std(rw):<6.3f}")
    print("=" * 130)

    # =====================================================================
    # Compact table (blocking only, for quick paper insertion)
    # =====================================================================
    print("\n" + "=" * 100)
    print("COMPACT TABLE: Blocking Rate (%)")
    print("=" * 100)
    header = f"{'Method':<22}"
    for label in ["NSFNET standard", "NSFNET high load", "USNET cross-topo"]:
        header += f" {label:>22}"
    print(header)
    print("-" * 100)
    for cfg_name in agent_order:
        row = f"{cfg_name:<22}"
        for label in ["NSFNET standard", "NSFNET high load", "USNET cross-topo"]:
            key = (label, cfg_name)
            if key in all_results:
                runs = all_results[key]
                br = [r["blocking_rate"] for r in runs]
                row += f" {np.mean(br)*100:>8.2f}±{np.std(br)*100:<10.2f}%"
            else:
                row += f" {'N/A':>22}"
        print(row)
    print("=" * 100)

    # =====================================================================
    # Save JSON
    # =====================================================================
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "yin_protocol_baselines.json", "w") as f:
        serializable = {}
        for (label, cfg_name), runs in all_results.items():
            serializable[f"{label}__{cfg_name}"] = [
                {k: float(v) if isinstance(v, (np.floating, float)) else v
                 for k, v in r.items()}
                for r in runs
            ]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir / 'yin_protocol_baselines.json'}")
    return all_results


if __name__ == "__main__":
    main()
