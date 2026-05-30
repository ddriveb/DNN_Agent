"""Evaluate the fragmentation-aware mainline.

This script compares:
  - YinLike
  - TopK2 baseline
  - Mapper-feasibility-gated TopK2
  - Fragmentation-aware TopK2
  - CorrectionNet baseline
  - Fragmentation-aware CorrectionNet

It first tunes explicit fragmentation weights on one development trace,
then evaluates the chosen setting across the standard paper scenarios.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import json
import numpy as np
import torch

from baselines import YinLikeAgent
from correction_agent import TopK2CorrectionAgent
from eval_imitation import create_env, run_agent
from fixed_trace import load_trace, generate_and_save_trace
from topk_selector_agent import TopKSelectorAgent


def load_predictor(path: str, max_servers: int = 128, device: str = "cpu"):
    from predictor import Predictor

    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location=device, weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    predictor.to(device)
    return predictor


def ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed):
    trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
    return trace_path


def build_topk_agent(predictor, encoder, mec, imitation_ckpt, num_servers, device, cfg):
    return TopKSelectorAgent(
        imitation_ckpt,
        predictor,
        encoder,
        mec,
        num_servers=num_servers,
        top_k=2,
        alpha=2.0,
        beta=0.3,
        gamma=0.2,
        delta=0.05,
        frag_weight=cfg.get("frag_weight", 0.0),
        lfb_weight=cfg.get("lfb_weight", 0.0),
        risk_weight=cfg.get("risk_weight", 0.0),
        enforce_mapper_feasibility=cfg.get("enforce_mapper_feasibility", False),
        use_fragmentation_tiebreaker=cfg.get("use_fragmentation_tiebreaker", False),
        p_success_min=cfg.get("p_success_min", 0.0),
        p_success_tie_epsilon=cfg.get("p_success_tie_epsilon", 0.0),
        use_enhanced_state=True,
        device=device,
    )


def build_corr_agent(predictor, encoder, mec, correction_ckpt, num_servers, device, cfg):
    return TopK2CorrectionAgent(
        correction_ckpt,
        predictor,
        encoder,
        mec,
        num_servers=num_servers,
        top_k=2,
        lambda_corr=cfg.get("lambda_corr", 0.2),
        alpha=2.0,
        beta=0.3,
        gamma=0.2,
        delta=0.05,
        frag_weight=cfg.get("frag_weight", 0.0),
        lfb_weight=cfg.get("lfb_weight", 0.0),
        risk_weight=cfg.get("risk_weight", 0.0),
        enforce_mapper_feasibility=cfg.get("enforce_mapper_feasibility", False),
        use_fragmentation_tiebreaker=cfg.get("use_fragmentation_tiebreaker", False),
        p_success_min=cfg.get("p_success_min", 0.0),
        p_success_tie_epsilon=cfg.get("p_success_tie_epsilon", 0.0),
        use_enhanced_state=True,
        device=device,
    )


def evaluate_cfg(predictor, imitation_ckpt, trace_dir, device, cfg):
    """Evaluate one TopK config on the development trace."""
    topology, num_slots, num_servers = "nsfnet", 32, 5
    num_requests, arr, ht, preload, seed, num_nodes = 2000, 5.0, 10.0, 300, 42, 14
    trace_path = ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed)
    requests = load_trace(trace_path)

    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    agent = build_topk_agent(
        predictor,
        encoder,
        env.mec,
        imitation_ckpt,
        num_servers=num_servers,
        device=device,
        cfg=cfg,
    )
    return run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)


def tune_weight_family(predictor, imitation_ckpt, trace_dir, device, family_name, cfgs):
    """Tune a family of fragmentation configs on one development trace."""
    best_cfg = None
    best_key = None
    trials = []

    print("\n" + "=" * 80)
    print(f"TUNING {family_name.upper()}")
    print("=" * 80)

    for cfg in cfgs:
        metrics = evaluate_cfg(predictor, imitation_ckpt, trace_dir, device, cfg)
        key = (
            metrics["blocking_rate"],
            metrics["avg_frag_index"],
            -metrics["avg_largest_free_block_ratio"],
            metrics["avg_future_risk_on_accept"],
        )
        trials.append({"cfg": dict(cfg), "metrics": metrics})
        print(
            f"w={cfg['frag_weight']:<4.2f} "
            f"mode={'tiebreak' if cfg.get('use_fragmentation_tiebreaker', False) else 'explicit':<9} "
            f"pmin={cfg.get('p_success_min', 0.0):<4.2f} "
            f"eps={cfg.get('p_success_tie_epsilon', 0.0):<4.2f} "
            f"-> block={metrics['blocking_rate']*100:5.2f}% "
            f"frag={metrics['avg_frag_index']:.4f} "
            f"lfb={metrics['avg_largest_free_block_ratio']:.4f}"
        )
        if best_key is None or key < best_key:
            best_key = key
            best_cfg = dict(cfg)

    print(f"\nChosen {family_name}: {best_cfg}")
    return best_cfg, trials


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


def fmt_pct(mean, std):
    return f"{mean*100:.2f}±{std*100:.2f}%"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    trace_dir = Path(__file__).parent / "traces"
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    predictor_path = str(Path(__file__).parent.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    imitation_ckpt = "checkpoints/imitation_agent_enhanced.pt"
    correction_ckpt = "checkpoints/correction_net.pt"

    unified_weights = [0.05, 0.10, 0.20, 0.30, 0.50]
    explicit_cfgs = [
        {
            "frag_weight": w,
            "lfb_weight": w,
            "risk_weight": w,
            "enforce_mapper_feasibility": True,
        }
        for w in unified_weights
    ]
    tie_epsilons = [0.0, 0.01, 0.03, 0.05]
    tiebreak_cfgs = [
        {
            "frag_weight": 1.0,
            "lfb_weight": 1.0,
            "risk_weight": 1.0,
            "enforce_mapper_feasibility": True,
            "use_fragmentation_tiebreaker": True,
            "p_success_tie_epsilon": eps,
        }
        for eps in tie_epsilons
    ]

    tuned_explicit_cfg, explicit_trials = tune_weight_family(
        predictor, imitation_ckpt, trace_dir, device, "explicit_frag", explicit_cfgs
    )
    tuned_tiebreak_cfg, tiebreak_trials = tune_weight_family(
        predictor, imitation_ckpt, trace_dir, device, "tiebreak_frag", tiebreak_cfgs
    )

    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard", 14),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load", 14),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo", 28),
    ]
    seeds = [42, 123, 456, 789, 2024]

    method_cfgs = [
        ("YinLike", "yinlike", {}),
        ("TopK2-30D", "topk", {}),
        ("TopK2+Feasibility", "topk", {"enforce_mapper_feasibility": True}),
        ("FragTopK2-Explicit", "topk", tuned_explicit_cfg),
        ("FragTopK2-TieBreak", "topk", tuned_tiebreak_cfg),
        ("CorrectionNet λ=0.2", "corr", {"lambda_corr": 0.2}),
        ("FragCorrection-Explicit λ=0.2", "corr", dict(tuned_explicit_cfg, lambda_corr=0.2)),
        ("FragCorrection-TieBreak λ=0.2", "corr", dict(tuned_tiebreak_cfg, lambda_corr=0.2)),
    ]

    raw_results = {
        "tuned_cfgs": {
            "explicit": tuned_explicit_cfg,
            "tiebreak": tuned_tiebreak_cfg,
        },
        "tuning_trials": {
            "explicit": explicit_trials,
            "tiebreak": tiebreak_trials,
        },
        "scenarios": {},
    }

    print("\n" + "=" * 100)
    print("MAINLINE COMPARISON")
    print("=" * 100)

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\nScenario: {label}")
        for method_name, method_type, cfg in method_cfgs:
            runs = []
            print(f"  {method_name:<24}", end="", flush=True)
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
                        imitation_ckpt,
                        num_servers=num_servers,
                        device=device,
                        cfg=cfg,
                    )
                else:
                    agent = build_corr_agent(
                        predictor,
                        encoder,
                        env.mec,
                        correction_ckpt,
                        num_servers=num_servers,
                        device=device,
                        cfg=cfg,
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
                f" block={fmt_pct(summary['blocking_rate']['mean'], summary['blocking_rate']['std'])} "
                f"frag={summary['avg_frag_index']['mean']:.4f} "
                f"lfb={summary['avg_largest_free_block_ratio']['mean']:.4f}"
            )

    out_json = out_dir / "fragmentation_mainline_eval.json"
    with open(out_json, "w") as f:
        json.dump(raw_results, f, indent=2)

    md = []
    md.append("# Fragmentation-Aware Mainline Evaluation\n")
    md.append(f"- Device: `{device}`")
    md.append(f"- Tuned explicit weights: `{tuned_explicit_cfg}`")
    md.append(f"- Tuned tie-break weights: `{tuned_tiebreak_cfg}`\n")
    md.append("## Main Results\n")
    md.append("| Scenario | Method | Blocking | Avg Delay (ms) | Avg Frag | Avg LFB | Avg Util |")
    md.append("|---|---|---:|---:|---:|---:|---:|")

    for label in [s[7] for s in scenarios]:
        for method_name, _, _ in method_cfgs:
            summary = raw_results["scenarios"][f"{label}__{method_name}"]["summary"]
            md.append(
                f"| {label} | {method_name} | "
                f"{fmt_pct(summary['blocking_rate']['mean'], summary['blocking_rate']['std'])} | "
                f"{summary['avg_delay_ms']['mean']:.2f} | "
                f"{summary['avg_frag_index']['mean']:.4f} | "
                f"{summary['avg_largest_free_block_ratio']['mean']:.4f} | "
                f"{summary['avg_spectrum_utilization']['mean']:.4f} |"
            )

    out_md = out_dir / "fragmentation_mainline_summary.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nSaved JSON to {out_json}")
    print(f"Saved summary to {out_md}")


if __name__ == "__main__":
    main()
