"""Yin 2024 simulation evaluation.

Runs WO, DF, RF, IWD-Approx, YinLike, TopK2-30D, CorrectionNet-λ0.2
on Net-1/2/3 with request batches [15, 20, ..., 60].

TopK2/CorrectionNet are run in zero-shot mode (checkpoints trained on
NSFNET/USNET) as a cross-topology generalization probe.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_mvp"))

import numpy as np
import torch
import json

from yin2024_env import create_yin2024_env
from yin2024_requests import generate_requests

from baselines import YinLikeAgent
from yin_baselines import WOAgent, DFAgent, RFAgent, IWDApproxAgent


def load_predictor(path: str, max_servers: int = 128, device: str = "cpu"):
    from predictor import Predictor
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def run_agent_online(agent, env, requests):
    """Run agent in event-driven mode with arrival times and holding times.

    Spectrum and compute resources are released automatically by
    env.advance_time() when their holding periods expire.
    """
    env.reset()
    if hasattr(agent, 'mec'):
        agent.mec = env.mec
    if hasattr(agent, 'encoder'):
        agent.encoder = env.encoder

    rewards = []
    for req in sorted(requests, key=lambda r: r.arrival_time):
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards)) if rewards else 0.0
    return metrics


def run_scenario(topology, frag_level, n_req_list, seeds):
    results = {}
    for n_req in n_req_list:
        print(f"\n  Requests={n_req}")
        for seed in seeds:
            requests = generate_requests(n_req, _get_server_nodes(topology), seed=seed)

            env, encoder, mec = create_yin2024_env(topology, frag_level, seed=seed)

            # --- WO ---
            agent = WOAgent(env.mec)
            m = run_agent_online(agent, env, requests)
            _store(results, topology, frag_level, n_req, seed, "WO", m)

            # --- DF ---
            env, encoder, mec = create_yin2024_env(topology, frag_level, seed=seed)
            agent = DFAgent(env.mec)
            m = run_agent_online(agent, env, requests)
            _store(results, topology, frag_level, n_req, seed, "DF", m)

            # --- RF ---
            env, encoder, mec = create_yin2024_env(topology, frag_level, seed=seed)
            agent = RFAgent(env.mec)
            m = run_agent_online(agent, env, requests)
            _store(results, topology, frag_level, n_req, seed, "RF", m)

            # --- IWD-Approx ---
            env, encoder, mec = create_yin2024_env(topology, frag_level, seed=seed)
            agent = IWDApproxAgent(env.mec, seed=seed)
            m = run_agent_online(agent, env, requests)
            _store(results, topology, frag_level, n_req, seed, "IWD-Approx", m)

            # --- YinLike ---
            env, encoder, mec = create_yin2024_env(topology, frag_level, seed=seed)
            agent = YinLikeAgent(env.mec)
            m = run_agent_online(agent, env, requests)
            _store(results, topology, frag_level, n_req, seed, "YinLike", m)

            # Note: TopK2/CorrectionNet require re-training on Yin2024 topologies.
            # Skipped in this baseline-focused round.

    return results


def _get_server_nodes(topology):
    from yin2024_network import NET1_SERVERS, NET2_SERVERS, NET3_SERVERS
    if topology == "net1":
        return NET1_SERVERS
    elif topology == "net2":
        return NET2_SERVERS
    else:
        return NET3_SERVERS


def _store(results, topology, frag, n_req, seed, method, metrics):
    key = (topology, frag, n_req, method)
    results.setdefault(key, []).append({k: float(v) if isinstance(v, (np.floating, float)) else v
                                         for k, v in metrics.items()})
    print(f"    {method:<18} seed={seed} block={metrics['blocking_rate']*100:.2f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Config
    n_req_list = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    seeds = [42, 123, 456, 789, 2024]
    frag_levels = [0.2, 0.5]
    topologies = ["net1", "net2", "net3"]

    # ckpt_base = Path(__file__).parent.parent / "agent_mvp" / "checkpoints"
    # predictor_path = str(Path(__file__).parent.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")

    all_results = {}

    for topology in topologies:
        for frag in frag_levels:
            print(f"\n{'='*80}")
            print(f"TOPOLOGY: {topology} | frag={frag}")
            print(f"{'='*80}")
            res = run_scenario(topology, frag, n_req_list, seeds)
            all_results.update(res)

    # =====================================================================
    # Summary
    # =====================================================================
    agent_order = ["WO", "DF", "RF", "IWD-Approx", "YinLike"]

    # =====================================================================
    # Summary per topology/frag
    # =====================================================================
    for topology in topologies:
        for frag in frag_levels:
            print("\n" + "=" * 120)
            print(f"YIN 2024 SIMULATION RESULTS ({topology}, frag={frag}, 5 seeds)")
            print("=" * 120)
            print(f"{'N_req':>6} {'Agent':<22} {'Blocking%':>18} {'Accept%':>18} {'AvgDelay(ms)':>18} {'Reward':>18}")
            print("-" * 120)

            for n_req in n_req_list:
                for method in agent_order:
                    key = (topology, frag, n_req, method)
                    if key not in all_results:
                        continue
                    runs = all_results[key]
                    br = [r["blocking_rate"] for r in runs]
                    ar = [r["acceptance_rate"] for r in runs]
                    dl = [r["avg_delay_ms"] for r in runs]
                    rw = [r["avg_reward"] for r in runs]
                    print(f"{n_req:>6} {method:<22} "
                          f"{np.mean(br)*100:>8.2f}±{np.std(br)*100:<6.2f}% "
                          f"{np.mean(ar)*100:>8.2f}±{np.std(ar)*100:<6.2f}% "
                          f"{np.mean(dl):>8.2f}±{np.std(dl):<6.2f} "
                          f"{np.mean(rw):>8.3f}±{np.std(rw):<6.3f}")
            print("=" * 120)

            # Compact table (blocking only)
            print("\n" + "=" * 100)
            print(f"COMPACT TABLE: Blocking Rate vs Number of Requests (%) — {topology} frag={frag}")
            print("=" * 100)
            header = f"{'Method':<22}"
            for n in n_req_list:
                header += f" {n:>7}"
            print(header)
            print("-" * 100)
            for method in agent_order:
                row = f"{method:<22}"
                for n in n_req_list:
                    key = (topology, frag, n, method)
                    if key in all_results:
                        runs = all_results[key]
                        br = [r["blocking_rate"] for r in runs]
                        row += f" {np.mean(br)*100:>6.1f}%"
                    else:
                        row += f" {'--':>7}"
                print(row)
            print("=" * 100)

    # =====================================================================
    # Save JSON
    # =====================================================================
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    serializable = {}
    for (topology, frag, n_req, method), runs in all_results.items():
        serializable[f"{topology}_f{frag}_n{n_req}_{method}"] = runs
    out_path = out_dir / "yin2024_all_results.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")
    return all_results


if __name__ == "__main__":
    main()
