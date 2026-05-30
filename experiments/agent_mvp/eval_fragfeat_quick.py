"""Quick evaluation for FragFeature-CorrectionNet checkpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import json
import torch

from correction_agent import TopK2CorrectionAgent
from eval_fragmentation_mainline import build_topk_agent, load_predictor
from eval_imitation import create_env, run_agent
from fixed_trace import load_trace, generate_and_save_trace


def ensure_trace_local(trace_dir, num_nodes, num_requests, arr, ht, seed):
    trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
    return trace_path


def summarize(metric_dict):
    return {
        "blocking_rate": metric_dict["blocking_rate"],
        "avg_delay_ms": metric_dict["avg_delay_ms"],
        "avg_frag_index": metric_dict["avg_frag_index"],
        "avg_largest_free_block_ratio": metric_dict["avg_largest_free_block_ratio"],
        "avg_spectrum_utilization": metric_dict["avg_spectrum_utilization"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).parent
    trace_dir = root / "traces"
    predictor_path = str(root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)

    topology = "nsfnet"
    num_slots = 32
    num_servers = 5
    num_requests = 500
    arr = 5.0
    ht = 10.0
    preload = 300
    seed = 42
    num_nodes = 14

    trace_path = ensure_trace_local(trace_dir, num_nodes, num_requests, arr, ht, seed)
    requests = load_trace(trace_path)

    methods = [
        ("TopK2+Feasibility", "topk", {"enforce_mapper_feasibility": True}),
        ("Old CorrectionNet λ=0.2", "corr_old", {"lambda_corr": 0.2, "ckpt": "checkpoints/correction_net.pt"}),
        ("FragFeature CorrectionNet λ=0.1", "corr_new", {"lambda_corr": 0.1, "ckpt": "checkpoints/correction_net_fragfeat_v1.pt"}),
        ("FragFeature CorrectionNet λ=0.2", "corr_new", {"lambda_corr": 0.2, "ckpt": "checkpoints/correction_net_fragfeat_v1.pt"}),
        ("FragFeature CorrectionNet λ=0.3", "corr_new", {"lambda_corr": 0.3, "ckpt": "checkpoints/correction_net_fragfeat_v1.pt"}),
    ]

    results = {}
    for name, kind, cfg in methods:
        env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
        if kind == "topk":
            agent = build_topk_agent(
                predictor,
                encoder,
                env.mec,
                "checkpoints/imitation_agent_enhanced.pt",
                num_servers,
                device,
                cfg,
            )
        else:
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
        metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
        results[name] = summarize(metrics)
        print(
            f"{name}: block={metrics['blocking_rate']*100:.2f}% "
            f"delay={metrics['avg_delay_ms']:.2f} "
            f"frag={metrics['avg_frag_index']:.4f} "
            f"lfb={metrics['avg_largest_free_block_ratio']:.4f} "
            f"util={metrics['avg_spectrum_utilization']:.4f}"
        )

    out_path = root / "results" / "fragfeat_quick_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved quick eval to {out_path}")


if __name__ == "__main__":
    main()
