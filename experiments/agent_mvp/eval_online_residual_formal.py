"""Formal multi-seed evaluation for online residual DQN checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch

from correction_agent import TopK2CorrectionAgent
from baselines import YinLikeAgent
from eval_fragmentation_mainline import build_topk_agent, load_predictor
from eval_imitation import create_env, run_agent
from fixed_trace import generate_and_save_trace, load_trace
from online_residual_rl import OnlineResidualDQNAgent, build_q_net
from yin_baselines import DFAgent, IWDApproxAgent, RFAgent


def ensure_trace_local(trace_dir, num_nodes, num_requests, arr, ht, seed):
    trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
    return trace_path


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
        load_imitation_mask=not bool(ckpt.get("disable_imitation_mask", False)),
        use_predictor_action_features=not bool(ckpt.get("disable_predictor_action_features", False)),
        drop_predictor_action_features=bool(ckpt.get("drop_predictor_action_features", False)),
        device=device,
    )


def summarize_runs(runs):
    out = {}
    for metric in runs[0].keys():
        values = [float(r[metric]) for r in runs if isinstance(r[metric], (float, int, np.floating))]
        if values:
            out[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/online_residual_dqn_run2.pt")
    parser.add_argument("--pure_q_checkpoint", default="checkpoints/pure_q_feasible_long.pt")
    parser.add_argument("--include_baselines", action="store_true")
    parser.add_argument("--output_json", default="results/dqn_standardization_formal.json")
    parser.add_argument("--output_md", default="results/dqn_standardization_formal.md")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).parent
    trace_dir = root / "traces"
    predictor_path = str(root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    imitation_ckpt = str(root / "checkpoints" / "imitation_agent_enhanced.pt")
    correction_ckpt = str(root / "checkpoints" / "correction_net.pt")
    online_ckpt = str((root / args.checkpoint).resolve()) if not Path(args.checkpoint).is_absolute() else args.checkpoint
    pure_q_ckpt = str((root / args.pure_q_checkpoint).resolve()) if not Path(args.pure_q_checkpoint).is_absolute() else args.pure_q_checkpoint

    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard", 14),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load", 14),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo", 28),
    ]
    seeds = [42, 123, 456, 789, 2024]

    methods = [
        ("OnlineResidualDQN", "online", {"ckpt": online_ckpt}),
        ("PureQFeasible", "online", {"ckpt": pure_q_ckpt}),
    ]
    if args.include_baselines:
        methods = [
            ("RF", "rf", {}),
            ("DF", "df", {}),
            ("IWD-Approx", "iwd", {}),
            ("YinLike", "yin", {}),
            ("TopK2+Feasibility", "topk", {"enforce_mapper_feasibility": True}),
            ("Old CorrectionNet λ=0.2", "corr_old", {"lambda_corr": 0.2, "ckpt": correction_ckpt}),
            ("OnlineResidualDQN", "online", {"ckpt": online_ckpt}),
            ("PureQFeasible", "online", {"ckpt": pure_q_ckpt}),
        ]

    raw = {}
    print(f"Device: {device}")
    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\nScenario: {label}")
        raw[label] = {}
        for method_name, kind, cfg in methods:
            runs = []
            print(f"  {method_name:<24}", end="", flush=True)
            for seed in seeds:
                trace_path = ensure_trace_local(trace_dir, num_nodes, num_requests, arr, ht, seed)
                requests = load_trace(trace_path)
                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

                if kind == "rf":
                    agent = RFAgent(env.mec)
                elif kind == "df":
                    agent = DFAgent(env.mec)
                elif kind == "iwd":
                    agent = IWDApproxAgent(env.mec, seed=seed)
                elif kind == "yin":
                    agent = YinLikeAgent(env.mec)
                elif kind == "topk":
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
                runs.append(metrics)
                print(".", end="", flush=True)
            print()
            raw[label][method_name] = {"runs": runs, "summary": summarize_runs(runs)}

    out_json = root / args.output_json
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(raw, f, indent=2)

    lines = ["# DQN Standardization Formal Evaluation", "", "| Scenario | Method | Blocking | Avg Delay (ms) | Acceptance | Avg Frag | Avg LFB |", "|---|---|---:|---:|---:|---:|---:|"]
    for label, methods_dict in raw.items():
        for method_name, payload in methods_dict.items():
            s = payload["summary"]
            lines.append(
                f"| {label} | {method_name} | "
                f"{s['blocking_rate']['mean']*100:.2f}±{s['blocking_rate']['std']*100:.2f}% | "
                f"{s['avg_delay_ms']['mean']:.2f} | "
                f"{s['acceptance_rate']['mean']*100:.2f}±{s['acceptance_rate']['std']*100:.2f}% | "
                f"{s['avg_frag_index']['mean']:.4f} | "
                f"{s['avg_largest_free_block_ratio']['mean']:.4f} |"
            )
    out_md = root / args.output_md
    out_md.write_text("\n".join(lines))
    print(f"\nSaved formal evaluation to {out_json}")
    print(f"Saved markdown table to {out_md}")


if __name__ == "__main__":
    main()
