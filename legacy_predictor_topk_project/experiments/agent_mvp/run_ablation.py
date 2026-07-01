"""Module ablation study.

Compares:
- Random
- ShortestPath
- YinLike
- Imitation-30D (neural only, no predictor reranking)
- TopK2-30D (neural + predictor, no correction)
- CorrectionNet λ=0.2 (full method)
- Direct Offline DQN (RL replaces selector — failure case)

Uses same traces as correction_net_eval_aligned.json for fair comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json

from eval_imitation import create_env
from fixed_trace import load_trace, generate_and_save_trace
from baselines import RandomAgent, ShortestPathAgent, YinLikeAgent
from train_imitation import ImitationAgent
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


class ImitationOnlyAgent:
    """Neural agent without predictor reranking."""

    def __init__(self, checkpoint_path, mec_cluster, num_servers=5,
                 use_enhanced_state=True, device="cpu"):
        self.mec = mec_cluster
        self.num_servers = num_servers
        self.use_enhanced_state = use_enhanced_state
        self.device = device

        from eval_imitation import build_state_vector_for_imitation
        from enhance_state_and_retrain import build_enhanced_state
        self.build_state = build_state_vector_for_imitation
        self.build_enhanced = build_enhanced_state

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = ImitationAgent(
            ckpt["state_dim"], ckpt["num_actions"],
            hidden_dims=ckpt.get("hidden_dims", (128, 128)),
            dropout=0.0,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.model.to(device)

    def decide(self, request, env):
        state = self.build_state(request, env, env.encoder, self.num_servers)
        if self.use_enhanced_state:
            state = self.build_enhanced(state, request.model_name)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(state_t)
            action_id = int(torch.argmax(logits, dim=1).item())
        split_id = action_id // self.num_servers
        server_id = action_id % self.num_servers
        # Validate
        if split_id >= len(request.model.splits):
            split_id = 0
        if server_id >= len(self.mec.servers):
            server_id = 0
        return split_id, server_id, 0.0, {"type": "imitation_only"}


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
    trace_dir.mkdir(exist_ok=True)

    results = {}

    configs = [
        ("Random", "random"),
        ("ShortestPath", "shortestpath"),
        ("YinLike", "yinlike"),
        ("Imitation-30D", "imitation"),
    ]

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {label}")
        print(f"{'='*80}")

        for cfg_name, cfg_type in configs:
            print(f"\n  Testing: {cfg_name}")
            for seed in seeds:
                trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
                trace_path = trace_dir / trace_fname
                if not trace_path.exists():
                    trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
                requests = load_trace(trace_path)

                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

                if cfg_type == "random":
                    agent = RandomAgent(env.mec, seed=seed)
                elif cfg_type == "shortestpath":
                    agent = ShortestPathAgent(env.mec)
                elif cfg_type == "yinlike":
                    agent = YinLikeAgent(env.mec)
                elif cfg_type == "imitation":
                    agent = ImitationOnlyAgent(
                        "checkpoints/imitation_agent_enhanced.pt",
                        env.mec, num_servers=num_servers,
                        use_enhanced_state=True, device=device,
                    )

                metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
                key = (label, cfg_name)
                results.setdefault(key, []).append(metrics)
                print(f"    seed={seed} block={metrics['blocking_rate']*100:.2f}%")

    # Save
    out_path = Path(__file__).parent / "results" / "ablation_study.json"
    with open(out_path, "w") as f:
        serializable = {}
        for (label, cfg_name), runs in results.items():
            serializable[f"{label}__{cfg_name}"] = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                                       for k, v in r.items()} for r in runs]
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("ABLATION STUDY (5 seeds)")
    print("=" * 100)
    print(f"{'Scenario':<25} {'Agent':<18} {'Blocking%':>18}")
    print("-" * 100)
    for label in ["NSFNET standard", "NSFNET high load", "USNET cross-topo"]:
        for cfg_name in ["Random", "ShortestPath", "YinLike", "Imitation-30D"]:
            key = (label, cfg_name)
            if key not in results:
                continue
            runs = results[key]
            br = [r["blocking_rate"] for r in runs]
            print(f"{label:<25} {cfg_name:<18} {np.mean(br)*100:>8.2f}±{np.std(br)*100:<6.2f}%")


if __name__ == "__main__":
    main()
