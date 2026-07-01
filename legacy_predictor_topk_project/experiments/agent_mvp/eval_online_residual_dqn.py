"""Evaluate an online residual DQN checkpoint against current baselines."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import torch

from correction_agent import TopK2CorrectionAgent
from eval_fragmentation_mainline import build_topk_agent, load_predictor
from eval_imitation import create_env, run_agent
from fixed_trace import generate_and_save_trace, load_trace
from online_residual_rl import OnlineResidualDQNAgent, build_q_net


def ensure_trace_local(trace_dir, num_nodes, num_requests, arr, ht, seed):
    trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
    return trace_path


def summarize(metric_dict):
    return {
        "blocking_rate": metric_dict["blocking_rate"],
        "acceptance_rate": metric_dict["acceptance_rate"],
        "avg_delay_ms": metric_dict["avg_delay_ms"],
        "avg_reward": metric_dict["avg_reward"],
        "avg_frag_index": metric_dict["avg_frag_index"],
        "avg_largest_free_block_ratio": metric_dict["avg_largest_free_block_ratio"],
        "avg_spectrum_utilization": metric_dict["avg_spectrum_utilization"],
    }


def build_online_agent(ckpt_path, predictor, encoder, mec, num_servers, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    q_net = build_q_net(
        input_dim=ckpt.get("input_dim", 16),
        hidden_dims=tuple(ckpt.get("hidden_dims", (64, 64))),
        dueling=ckpt.get("dueling", False),
        set_aware=ckpt.get("set_aware", False),
        tspq=ckpt.get("tspq", False),
        gated_residual=ckpt.get("gated_residual", False),
    ).to(device)
    q_net.load_state_dict(ckpt["model_state"])
    q_net.eval()

    return OnlineResidualDQNAgent(
        imitation_checkpoint_path=str(Path(__file__).parent / "checkpoints" / "imitation_agent_enhanced.pt"),
        predictor=predictor,
        encoder=encoder,
        mec_cluster=mec,
        q_net=q_net,
        num_servers=num_servers,
        top_k=2,
        lambda_residual=float(ckpt.get("lambda_residual", 0.2)),
        decision_mode=ckpt.get("decision_mode", "residual"),
        p_success_min=float(ckpt.get("p_success_min", 0.0)),
        enforce_mapper_feasibility=True,
        use_enhanced_state=True,
        relative_features=bool(ckpt.get("relative_features", False)),
        predictor_input_feature=bool(ckpt.get("predictor_input_feature", False)),
        device=device,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).parent
    trace_dir = root / "traces"
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)

    predictor_path = str(root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    imitation_ckpt = str(root / "checkpoints" / "imitation_agent_enhanced.pt")
    old_corr_ckpt = str(root / "checkpoints" / "correction_net.pt")
    online_ckpt = str(root / "checkpoints" / "online_residual_dqn_run2.pt")

    topology = "nsfnet"
    num_slots = 32
    num_servers = 5
    num_requests = 500
    arr = 5.0
    ht = 10.0
    preload = 300
    seed = 789
    num_nodes = 14

    trace_path = ensure_trace_local(trace_dir, num_nodes, num_requests, arr, ht, seed)
    requests = load_trace(trace_path)

    methods = [
        ("TopK2+Feasibility", "topk", {"enforce_mapper_feasibility": True}),
        ("Old CorrectionNet λ=0.2", "corr_old", {"lambda_corr": 0.2, "ckpt": old_corr_ckpt}),
        ("OnlineResidualDQN run2", "online", {"ckpt": online_ckpt}),
    ]

    results = {}
    for name, kind, cfg in methods:
        env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
        if kind == "topk":
            agent = build_topk_agent(
                predictor,
                encoder,
                env.mec,
                imitation_ckpt,
                num_servers,
                device,
                cfg,
            )
        elif kind == "corr_old":
            agent = TopK2CorrectionAgent(
                cfg["ckpt"],
                predictor,
                encoder,
                env.mec,
                num_servers=num_servers,
                top_k=2,
                lambda_corr=cfg["lambda_corr"],
                alpha=2.0,
                beta=0.3,
                gamma=0.2,
                delta=0.05,
                use_enhanced_state=True,
                device=device,
            )
        else:
            agent = build_online_agent(cfg["ckpt"], predictor, encoder, env.mec, num_servers, device)
        metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
        results[name] = summarize(metrics)
        print(
            f"{name}: block={metrics['blocking_rate']*100:.2f}% "
            f"accept={metrics['acceptance_rate']*100:.2f}% "
            f"delay={metrics['avg_delay_ms']:.2f} "
            f"frag={metrics['avg_frag_index']:.4f} "
            f"lfb={metrics['avg_largest_free_block_ratio']:.4f} "
            f"util={metrics['avg_spectrum_utilization']:.4f}"
        )

    out_path = out_dir / "online_residual_dqn_compare.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
