"""Phase 1: Top-K ablation study.

Tests K ∈ {1, 2, 3, 5, all} to confirm K=3 is optimal.
Also tests re-ranking variants:
  - random: random pick from top-K
  - neural: use neural score (logit) only
  - predictor: use predictor P_success only
  - full: predictor + delay + load (default)
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


class RandomTopKAgent(TopKSelectorAgent):
    """Pick randomly from top-K (ablation: is re-ranking necessary?)."""

    def decide(self, request, env):
        state = build_state_vector_for_imitation(request, env, self.encoder, self.num_servers)
        if self.use_enhanced_state:
            from enhance_state_and_retrain import build_enhanced_state
            state = build_enhanced_state(state, request.model_name)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.imitation_model(state_t)
            top_k_indices = torch.topk(logits, k=min(self.top_k, logits.shape[1]), dim=1)[1]

        valid_actions = []
        for action_id in top_k_indices[0].cpu().tolist():
            split_id = action_id // self.num_servers
            server_id = action_id % self.num_servers
            if split_id < len(request.model.splits) and server_id < len(self.mec.servers):
                srv = self.mec.get_server_by_id(server_id)
                if srv is not None and srv.node_id != request.source_node:
                    valid_actions.append((split_id, server_id))

        if valid_actions:
            split_id, server_id = valid_actions[np.random.randint(len(valid_actions))]
            return split_id, server_id, 0.0, {"type": "random_topk", "top_k": self.top_k}
        return 0, 0, 0.0, {"type": "random_topk_fallback"}


class NeuralOnlyTopKAgent(TopKSelectorAgent):
    """Re-rank by neural logit score only (ablation: is predictor necessary?)."""

    def decide(self, request, env):
        state = build_state_vector_for_imitation(request, env, self.encoder, self.num_servers)
        if self.use_enhanced_state:
            from enhance_state_and_retrain import build_enhanced_state
            state = build_enhanced_state(state, request.model_name)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.imitation_model(state_t)
            top_k_values, top_k_indices = torch.topk(logits, k=min(self.top_k, logits.shape[1]), dim=1)

        best_action = None
        best_score = -float('inf')

        for idx, action_id in enumerate(top_k_indices[0].cpu().tolist()):
            split_id = action_id // self.num_servers
            server_id = action_id % self.num_servers
            if split_id >= len(request.model.splits) or server_id >= len(self.mec.servers):
                continue
            srv = self.mec.get_server_by_id(server_id)
            if srv is None or srv.node_id == request.source_node:
                continue

            score = float(top_k_values[0][idx].cpu())
            if score > best_score:
                best_score = score
                best_action = (split_id, server_id)

        if best_action is None and len(top_k_indices[0]) > 0:
            action_id = int(top_k_indices[0][0].item())
            best_action = (action_id // self.num_servers, action_id % self.num_servers)

        if best_action is None:
            best_action = (0, 0)

        return best_action[0], best_action[1], best_score, {"type": "neural_only_topk", "top_k": self.top_k}


class PredictorOnlyTopKAgent(TopKSelectorAgent):
    """Re-rank by predictor P_success only (ablation: are delay/load necessary?)."""

    def _score_candidate(self, request, split_id, server_id):
        src = request.source_node
        srv = self.mec.get_server_by_id(server_id)
        if srv is None or srv.node_id == src:
            return None

        dst = srv.node_id
        z = self.encoder.encode(src, dst)
        z_t = torch.from_numpy(z).unsqueeze(0).float().to(self.device)

        partition = torch.tensor([split_id], dtype=torch.long, device=self.device)
        src_t = torch.tensor([src], dtype=torch.long, device=self.device)
        dst_t = torch.tensor([dst], dtype=torch.long, device=self.device)

        with torch.no_grad():
            success_logit, _ = self.predictor(partition, src_t, dst_t, z_t)
            p_success = torch.sigmoid(success_logit).item()

        return {"score": p_success, "p_success": p_success}


def build_state_vector_for_imitation(req, env, encoder, num_servers):
    """Import from eval_imitation to avoid circular import issues."""
    from eval_imitation import build_state_vector_for_imitation as _build
    return _build(req, env, encoder, num_servers)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scenario = ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard")
    topology, num_slots, num_servers, num_requests, arr, ht, preload, label = scenario
    seeds = [42, 123, 456, 789, 2024]

    base = Path(__file__).parent.parent / "predictor_mvp"
    predictor_path = str(base / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)

    # Configurations to test
    configs = [
        ("YinLike", None, None),
        ("Imitation-21D", None, None),
        ("Imitation-30D", None, None),
    ]

    # TopK ablation: K = 1, 2, 3, 5, all(15)
    for k in [1, 2, 3, 5, 15]:
        configs.append((f"TopK{k}-30D-full", k, "full"))

    # Re-ranking ablation (all with K=3)
    configs.append(("TopK3-30D-random", 3, "random"))
    configs.append(("TopK3-30D-neural", 3, "neural"))
    configs.append(("TopK3-30D-predictor", 3, "predictor"))

    all_results = {}
    trace_dir = Path(__file__).parent / "traces"
    num_nodes = 14

    for cfg_name, k, rerank in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {cfg_name}")
        print(f"{'='*80}")

        for seed in seeds:
            trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
            trace_path = trace_dir / trace_fname
            if not trace_path.exists():
                trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
            requests = load_trace(trace_path)

            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

            if cfg_name == "YinLike":
                agent = YinLikeAgent(mec)
            elif cfg_name == "Imitation-21D":
                agent = ImitationAgentWrapper("checkpoints/imitation_agent.pt", num_servers=num_servers)
                agent.mec = env.mec
                agent.encoder = env.encoder
            elif cfg_name == "Imitation-30D":
                agent = EnhancedImitationAgentWrapper("checkpoints/imitation_agent_enhanced.pt", num_servers=num_servers)
                agent.mec = env.mec
                agent.encoder = env.encoder
            elif rerank == "random":
                agent = RandomTopKAgent(
                    "checkpoints/imitation_agent_enhanced.pt",
                    predictor, encoder, env.mec,
                    num_servers=num_servers, top_k=k,
                    use_enhanced_state=True, device=device,
                )
            elif rerank == "neural":
                agent = NeuralOnlyTopKAgent(
                    "checkpoints/imitation_agent_enhanced.pt",
                    predictor, encoder, env.mec,
                    num_servers=num_servers, top_k=k,
                    use_enhanced_state=True, device=device,
                )
            elif rerank == "predictor":
                agent = PredictorOnlyTopKAgent(
                    "checkpoints/imitation_agent_enhanced.pt",
                    predictor, encoder, env.mec,
                    num_servers=num_servers, top_k=k,
                    use_enhanced_state=True, device=device,
                )
            else:  # full
                agent = TopKSelectorAgent(
                    "checkpoints/imitation_agent_enhanced.pt",
                    predictor, encoder, env.mec,
                    num_servers=num_servers, top_k=k,
                    alpha=2.0, beta=0.3, gamma=0.2, delta=0.05,
                    use_enhanced_state=True, device=device,
                )

            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault(cfg_name, []).append(metrics)
            print(f"  seed={seed} block={metrics['blocking_rate']*100:.2f}%")

    # Summary
    print("\n" + "=" * 100)
    print("TOP-K ABLATION RESULTS (NSFNET standard, 5 seeds)")
    print("=" * 100)
    print(f"{'Config':<30} {'Blocking%':>15} {'Accept%':>15} {'Reward':>15}")
    print("-" * 100)

    for cfg_name in [c[0] for c in configs]:
        runs = all_results[cfg_name]
        br = [r["blocking_rate"] for r in runs]
        ar = [r["acceptance_rate"] for r in runs]
        rw = [r["avg_reward"] for r in runs]
        print(f"{cfg_name:<30} {np.mean(br)*100:>7.2f}±{np.std(br)*100:<5.2f}% "
              f"{np.mean(ar)*100:>7.2f}±{np.std(ar)*100:<5.2f}% {np.mean(rw):>8.3f}±{np.std(rw):<5.3f}")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "topk_ablation.json", "w") as f:
        serializable = {}
        for cfg_name, runs in all_results.items():
            serializable[cfg_name] = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                       for k, v in r.items()} for r in runs]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir / 'topk_ablation.json'}")


if __name__ == "__main__":
    main()
