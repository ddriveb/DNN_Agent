"""Targeted lambda sweep for FragFeature-CorrectionNet on hard scenarios."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import json
import numpy as np
import torch

from baselines import YinLikeAgent
from correction_agent import TopK2CorrectionAgent
from eval_fragmentation_mainline import build_topk_agent, load_predictor
from eval_imitation import create_env, run_agent
from fixed_trace import load_trace, generate_and_save_trace


def ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed):
    trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
    return trace_path


def summarize_runs(runs):
    out = {}
    for metric in runs[0].keys():
        values = [float(r[metric]) for r in runs if isinstance(r[metric], (float, int, np.floating))]
        if values:
            out[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return out


def fmt_pct(summary, key):
    return f"{summary[key]['mean']*100:.2f}±{summary[key]['std']*100:.2f}%"


def build_corr_agent(ckpt, predictor, encoder, mec, num_servers, device, lam):
    return TopK2CorrectionAgent(
        ckpt,
        predictor,
        encoder,
        mec,
        num_servers=num_servers,
        top_k=2,
        lambda_corr=lam,
        alpha=2.0,
        beta=0.3,
        gamma=0.2,
        delta=0.05,
        use_enhanced_state=True,
        device=device,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    scenarios = [
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load", 14),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo", 28),
    ]
    seeds = [42, 123, 456, 789, 2024]
    lambdas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    root = Path(__file__).parent
    trace_dir = root / "traces"
    predictor_path = str(root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)

    old_ckpt = "checkpoints/correction_net.pt"
    new_ckpt = "checkpoints/correction_net_fragfeat_v1.pt"

    methods = [
        ("TopK2+Feasibility", "topk", {"enforce_mapper_feasibility": True}),
        ("Old CorrectionNet λ=0.2", "corr_old", {"lambda_corr": 0.2}),
    ] + [
        (f"FragFeature λ={lam:.2f}", "corr_new", {"lambda_corr": lam})
        for lam in lambdas
    ]

    raw_results = {"scenarios": {}}

    print("\n" + "=" * 100)
    print("FRAGFEATURE LAMBDA SWEEP")
    print("=" * 100)
    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\nScenario: {label}")
        for method_name, method_type, cfg in methods:
            runs = []
            print(f"  {method_name:<28}", end="", flush=True)
            for seed in seeds:
                trace_path = ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed)
                requests = load_trace(trace_path)
                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

                if method_type == "topk":
                    agent = build_topk_agent(
                        predictor,
                        encoder,
                        env.mec,
                        "checkpoints/imitation_agent_enhanced.pt",
                        num_servers,
                        device,
                        cfg,
                    )
                elif method_type == "corr_old":
                    agent = build_corr_agent(old_ckpt, predictor, encoder, env.mec, num_servers, device, cfg["lambda_corr"])
                else:
                    agent = build_corr_agent(new_ckpt, predictor, encoder, env.mec, num_servers, device, cfg["lambda_corr"])

                metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
                runs.append(metrics)

            summary = summarize_runs(runs)
            raw_results["scenarios"][f"{label}__{method_name}"] = {
                "config": cfg,
                "runs": runs,
                "summary": summary,
            }
            print(
                f" block={fmt_pct(summary, 'blocking_rate')} "
                f"frag={summary['avg_frag_index']['mean']:.4f} "
                f"lfb={summary['avg_largest_free_block_ratio']['mean']:.4f}"
            )

    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "fragfeat_lambda_sweep.json"
    with open(out_json, "w") as f:
        json.dump(raw_results, f, indent=2)

    md_lines = []
    md_lines.append("# FragFeature Lambda Sweep\n")
    md_lines.append("| Scenario | Method | Blocking | Avg Delay (ms) | Avg Frag | Avg LFB | Avg Util |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for label in [s[7] for s in scenarios]:
        for method_name, _, _ in methods:
            summary = raw_results["scenarios"][f"{label}__{method_name}"]["summary"]
            md_lines.append(
                f"| {label} | {method_name} | "
                f"{fmt_pct(summary, 'blocking_rate')} | "
                f"{summary['avg_delay_ms']['mean']:.2f} | "
                f"{summary['avg_frag_index']['mean']:.4f} | "
                f"{summary['avg_largest_free_block_ratio']['mean']:.4f} | "
                f"{summary['avg_spectrum_utilization']['mean']:.4f} |"
            )

    out_md = out_dir / "fragfeat_lambda_sweep.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nSaved JSON to {out_json}")
    print(f"Saved markdown to {out_md}")


if __name__ == "__main__":
    main()
