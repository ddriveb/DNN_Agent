"""Small adaptive-lambda evaluation for FragFeature-CorrectionNet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import json
import numpy as np
import torch

from correction_agent import TopK2CorrectionAgent
from eval_fragmentation_mainline import load_predictor
from eval_imitation import create_env, run_agent
from fixed_trace import generate_and_save_trace, load_trace


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


class OracleScenarioAdaptiveFragAgent:
    """Oracle-style scenario adaptive lambda using the best fixed lambda per scenario."""

    def __init__(self, predictor, encoder, mec, num_servers, device, scenario_label):
        self.inner = TopK2CorrectionAgent(
            "checkpoints/correction_net_fragfeat_v1.pt",
            predictor,
            encoder,
            mec,
            num_servers=num_servers,
            top_k=2,
            lambda_corr=0.2,
            alpha=2.0,
            beta=0.3,
            gamma=0.2,
            delta=0.05,
            use_enhanced_state=True,
            device=device,
        )
        self.scenario_label = scenario_label

    @property
    def mec(self):
        return self.inner.mec

    @mec.setter
    def mec(self, value):
        self.inner.mec = value

    @property
    def encoder(self):
        return self.inner.encoder

    @encoder.setter
    def encoder(self, value):
        self.inner.encoder = value

    def _select_lambda(self):
        if self.scenario_label == "NSFNET standard":
            return 0.20
        if self.scenario_label == "NSFNET high load":
            return 0.05
        return 0.15

    def decide(self, request, env):
        self.inner.lambda_corr = self._select_lambda()
        split_id, server_id, score, info = self.inner.decide(request, env)
        info["adaptive_lambda"] = self.inner.lambda_corr
        info["adaptive_mode"] = "oracle_scenario"
        return split_id, server_id, score, info


class StateAdaptiveFragAgent:
    """Deployable lightweight adaptive lambda based on runtime system state."""

    def __init__(self, predictor, encoder, mec, num_servers, device):
        self.inner = TopK2CorrectionAgent(
            "checkpoints/correction_net_fragfeat_v1.pt",
            predictor,
            encoder,
            mec,
            num_servers=num_servers,
            top_k=2,
            lambda_corr=0.2,
            alpha=2.0,
            beta=0.3,
            gamma=0.2,
            delta=0.05,
            use_enhanced_state=True,
            device=device,
        )

    @property
    def mec(self):
        return self.inner.mec

    @mec.setter
    def mec(self, value):
        self.inner.mec = value

    @property
    def encoder(self):
        return self.inner.encoder

    @encoder.setter
    def encoder(self, value):
        self.inner.encoder = value

    def _select_lambda(self, env):
        spectrum = env.net.get_global_spectrum_stats()
        util = float(spectrum["spectrum_utilization"])
        frag = float(spectrum["avg_frag_index"])
        avg_srv_load = float(np.mean(env.mec.server_load_vector(normalize=True)))

        if env.net.NUM_NODES > 14:
            return 0.15
        if avg_srv_load >= 0.55 or util >= 0.30:
            return 0.05
        if frag <= 0.30:
            return 0.20
        return 0.15

    def decide(self, request, env):
        self.inner.lambda_corr = self._select_lambda(env)
        split_id, server_id, score, info = self.inner.decide(request, env)
        info["adaptive_lambda"] = self.inner.lambda_corr
        info["adaptive_mode"] = "state_rule"
        return split_id, server_id, score, info


def build_old_agent(predictor, encoder, mec, num_servers, device):
    return TopK2CorrectionAgent(
        "checkpoints/correction_net.pt",
        predictor,
        encoder,
        mec,
        num_servers=num_servers,
        top_k=2,
        lambda_corr=0.2,
        alpha=2.0,
        beta=0.3,
        gamma=0.2,
        delta=0.05,
        use_enhanced_state=True,
        device=device,
    )


def build_new_fixed_agent(predictor, encoder, mec, num_servers, device):
    return TopK2CorrectionAgent(
        "checkpoints/correction_net_fragfeat_v1.pt",
        predictor,
        encoder,
        mec,
        num_servers=num_servers,
        top_k=2,
        lambda_corr=0.2,
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

    methods = [
        ("Old CorrectionNet λ=0.2", "old"),
        ("FragFeature λ=0.2", "fixed"),
        ("Oracle Adaptive λ", "oracle"),
        ("State Adaptive λ", "state"),
    ]

    raw_results = {"scenarios": {}}

    print("\n" + "=" * 100)
    print("FRAGFEATURE ADAPTIVE-LAMBDA EVALUATION")
    print("=" * 100)
    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\nScenario: {label}")
        for method_name, method_type in methods:
            runs = []
            print(f"  {method_name:<24}", end="", flush=True)
            for seed in seeds:
                trace_path = ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed)
                requests = load_trace(trace_path)
                env, encoder, mec = create_env(topology, num_slots, num_servers, seed)

                if method_type == "old":
                    agent = build_old_agent(predictor, encoder, env.mec, num_servers, device)
                elif method_type == "fixed":
                    agent = build_new_fixed_agent(predictor, encoder, env.mec, num_servers, device)
                elif method_type == "oracle":
                    agent = OracleScenarioAdaptiveFragAgent(predictor, encoder, env.mec, num_servers, device, label)
                else:
                    agent = StateAdaptiveFragAgent(predictor, encoder, env.mec, num_servers, device)

                metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
                runs.append(metrics)

            summary = summarize_runs(runs)
            raw_results["scenarios"][f"{label}__{method_name}"] = {
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
    out_json = out_dir / "fragfeat_adaptive_lambda_eval.json"
    with open(out_json, "w") as f:
        json.dump(raw_results, f, indent=2)

    md = []
    md.append("# FragFeature Adaptive Lambda Evaluation\n")
    md.append("| Scenario | Method | Blocking | Avg Delay (ms) | Avg Frag | Avg LFB | Avg Util |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for label in [s[7] for s in scenarios]:
        for method_name, _ in methods:
            summary = raw_results["scenarios"][f"{label}__{method_name}"]["summary"]
            md.append(
                f"| {label} | {method_name} | "
                f"{fmt_pct(summary, 'blocking_rate')} | "
                f"{summary['avg_delay_ms']['mean']:.2f} | "
                f"{summary['avg_frag_index']['mean']:.4f} | "
                f"{summary['avg_largest_free_block_ratio']['mean']:.4f} | "
                f"{summary['avg_spectrum_utilization']['mean']:.4f} |"
            )

    out_md = out_dir / "fragfeat_adaptive_lambda_eval.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nSaved JSON to {out_json}")
    print(f"Saved markdown to {out_md}")


if __name__ == "__main__":
    main()
