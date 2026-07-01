"""Formal multi-seed evaluation for FragFeature-CorrectionNet."""
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
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard", 14),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load", 14),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo", 28),
    ]
    seeds = [42, 123, 456, 789, 2024]
    root = Path(__file__).parent
    trace_dir = root / "traces"

    predictor_path = str(root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)

    old_ckpt = "checkpoints/correction_net.pt"
    new_ckpt = "checkpoints/correction_net_fragfeat_v1.pt"
    new_ckpt_meta = torch.load(new_ckpt, map_location=device, weights_only=False)

    methods = [
        ("YinLike", "yinlike", {}),
        ("TopK2-30D", "topk", {}),
        ("TopK2+Feasibility", "topk", {"enforce_mapper_feasibility": True}),
        ("Old CorrectionNet λ=0.2", "corr_old", {"lambda_corr": 0.2}),
        ("FragFeature CorrectionNet λ=0.1", "corr_new", {"lambda_corr": 0.1}),
        ("FragFeature CorrectionNet λ=0.2", "corr_new", {"lambda_corr": 0.2}),
        ("FragFeature CorrectionNet λ=0.3", "corr_new", {"lambda_corr": 0.3}),
    ]

    raw_results = {
        "metadata": {
            "device": device,
            "old_checkpoint": old_ckpt,
            "new_checkpoint": new_ckpt,
            "new_checkpoint_input_dim": int(new_ckpt_meta.get("input_dim", 0)),
            "new_action_feat_dim": int(new_ckpt_meta.get("action_feat_dim", 0)),
            "replay_size": int(new_ckpt_meta.get("dataset_size", 0)),
            "best_val_loss": float(min(new_ckpt_meta.get("history", {}).get("val_loss", [0.0]))),
            "action_feature_names": new_ckpt_meta.get("action_feature_names", []),
        },
        "scenarios": {},
    }

    print("\n" + "=" * 100)
    print("FRAGFEATURE FORMAL EVALUATION")
    print("=" * 100)

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\nScenario: {label}")
        for method_name, method_type, cfg in methods:
            runs = []
            print(f"  {method_name:<32}", end="", flush=True)
            for seed in seeds:
                trace_path = ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed)
                requests = load_trace(trace_path)
                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

                if method_type == "yinlike":
                    agent = YinLikeAgent(env.mec)
                elif method_type == "topk":
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
                    agent = build_corr_agent(
                        old_ckpt, predictor, encoder, env.mec, num_servers, device, cfg["lambda_corr"]
                    )
                else:
                    agent = build_corr_agent(
                        new_ckpt, predictor, encoder, env.mec, num_servers, device, cfg["lambda_corr"]
                    )

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
    out_json = out_dir / "fragfeat_formal_eval.json"
    with open(out_json, "w") as f:
        json.dump(raw_results, f, indent=2)

    perf_lines = []
    perf_lines.append("# FragFeature Formal Evaluation\n")
    perf_lines.append("## Table 1: Main Performance\n")
    perf_lines.append("| Scenario | Method | Blocking | Avg Delay (ms) | Acceptance |")
    perf_lines.append("|---|---|---:|---:|---:|")

    spectrum_lines = []
    spectrum_lines.append("# FragFeature Formal Evaluation\n")
    spectrum_lines.append("## Table 2: Spectrum State\n")
    spectrum_lines.append("| Scenario | Method | Avg Frag | Avg LFB | Avg Util |")
    spectrum_lines.append("|---|---|---:|---:|---:|")

    for label in [s[7] for s in scenarios]:
        for method_name, _, _ in methods:
            summary = raw_results["scenarios"][f"{label}__{method_name}"]["summary"]
            perf_lines.append(
                f"| {label} | {method_name} | "
                f"{fmt_pct(summary, 'blocking_rate')} | "
                f"{summary['avg_delay_ms']['mean']:.2f} | "
                f"{fmt_pct(summary, 'acceptance_rate')} |"
            )
            spectrum_lines.append(
                f"| {label} | {method_name} | "
                f"{summary['avg_frag_index']['mean']:.4f} | "
                f"{summary['avg_largest_free_block_ratio']['mean']:.4f} | "
                f"{summary['avg_spectrum_utilization']['mean']:.4f} |"
            )

    out_perf = out_dir / "fragfeat_formal_table_main.md"
    out_spectrum = out_dir / "fragfeat_formal_table_spectrum.md"
    out_perf.write_text("\n".join(perf_lines) + "\n", encoding="utf-8")
    out_spectrum.write_text("\n".join(spectrum_lines) + "\n", encoding="utf-8")

    print(f"\nSaved JSON to {out_json}")
    print(f"Saved main table to {out_perf}")
    print(f"Saved spectrum table to {out_spectrum}")


if __name__ == "__main__":
    main()
