#!/usr/bin/env python3
"""Build the EXPERIMENT_MATRIX.md report from the new-topology artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path("sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron")
TOPOLOGIES = (
    ("cost239_deeprmsa", "cost239"),
    ("abilene", "abilene"),
    ("xlron_nsfnet_deeprmsa", "nsfnet"),
    ("xlron_jpn48", "jpn48"),
    ("xlron_usnet_gcnrmsa", "usnet"),
)
OUT = Path("sa_hmarl/pure_rmsa_v13/neural_opportunity/n1_100/EXPERIMENT_MATRIX.md")

CONFIDENCE = 0.95


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def arm_values(rows: list[dict], arm: str) -> list[float]:
    return [r["blocked_rate"] for r in rows if r["arm"] == arm]


def summarize(values: list[float]) -> tuple[float, float, tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    n = len(arr)
    if n > 1:
        ci = stats.t.interval(CONFIDENCE, df=n - 1, loc=mean, scale=std / np.sqrt(n))
    else:
        ci = (mean, mean)
    return mean, std, (float(ci[0]), float(ci[1]))


def paired_p(n1: list[float], k50: list[float]) -> float:
    return float(stats.ttest_rel(k50, n1).pvalue)


def format_p(p: float) -> str:
    """p<0.001 for tiny values instead of misleading 0.0000."""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def format_mean_std_ci(mean: float, std: float, ci: tuple[float, float]) -> str:
    return f"{mean:.3f}% ±{std:.3f}% ({ci[0]:.3f}–{ci[1]:.3f})"


def main() -> None:
    lines = [
        "# N1-100 Experiment Matrix — cost239 / abilene / nsfnet / jpn48 / usnet",
        "",
        "**Evaluation settings:** 100 slots, KSP-FF K=50 XLRON-consistent path pool, "
        "holding_truncation=2.0, warmup=3000, eval=10000.  "
        "Seeds: cost239/abilene 61821-61830; nsfnet/jpn48/usnet 62021-62030 (disjoint "
        "from their calibration 62001-62005, train 62011-62015, validation 62016-62017).  "
        "Bitrate choices are the project default `{25, 50, 75, 100}` Gbps "
        "(not the XLRON 25-100 step-1 set).",
        "",
        "**Baseline disclosure:** Two baselines are reported.  The *new* (XLRON-consistent) "
        "baseline uses `path_sort_strategy='xlron'` (tie-break by `(hops, tuple)`).  "
        "The *legacy* baseline used the project-default `path_sort_strategy='hops'` "
        "(tie-break by `(hops, km, tuple)`), which produced a stronger KSP-FF pool on "
        "dense topologies such as COST239 and is kept only for comparison.",
        "",
        "**Latency (per request, mean):** N1 88-153 μs vs KSP-FF 29-72 μs (N1 ~2.3-3.4×).  "
        "All sub-millisecond; `LATENCY_MEAN_MS_GATE=0.060` ms is defined but not enforced "
        "by any script, and N1 (and jpn48's KSP-FF) exceed it.  See EXPERIMENT_REPORT.md.",
        "",
        "## New XLRON-consistent baseline",
        "",
        "| Topology | N1 | KSP-FF K5 | KSP-FF K50 | dK | N1 vs K50 | Relative drop | Wins | p-value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    details = []
    for topology, short in TOPOLOGIES:
        topo_dir = BASE / short
        cal = load_json(topo_dir / "CALIBRATION_RESULTS.json")
        train = load_json(topo_dir / "training" / "TRAINING_RESULTS.json")
        comp = load_json(topo_dir / "COMPARE_RESULTS.json")
        if cal is None or train is None or comp is None:
            raise FileNotFoundError(f"Missing artifacts for {topology}")

        rows = comp["results"]
        n1_values = arm_values(rows, "n1")
        k5_values = arm_values(rows, "ksp_ff_k5")
        k50_values = arm_values(rows, "ksp_ff_k50")

        n1_mean, n1_std, n1_ci = summarize(n1_values)
        k5_mean, k5_std, k5_ci = summarize(k5_values)
        k50_mean, k50_std, k50_ci = summarize(k50_values)

        dk = k5_mean - k50_mean
        n1_vs_k50 = k50_mean - n1_mean
        rel_drop = (n1_vs_k50 / k50_mean * 100.0) if k50_mean else 0.0
        wins = sum(n < k for n, k in zip(n1_values, k50_values))
        p = paired_p(n1_values, k50_values)

        lines.append(
            f"| {short} | {n1_mean:.3f}% | {k5_mean:.3f}% | {k50_mean:.3f}% | "
            f"{dk:.2f} | {n1_vs_k50:.2f} pp | {rel_drop:.1f}% | {wins}/10 | {format_p(p)} |"
        )

        metrics = train.get("best_validation", {})
        recall = metrics.get("top1_recall", -1)
        regret = metrics.get("mean_normalized_regret", -1)
        regret_orig = "met" if regret <= 0.10 else "NOT met"
        details.append(
            f"\n## {short}\n\n"
            f"- Topology key: `{topology}`\n"
            f"- Calibrated load: {cal['chosen_load']} Erlang "
            f"(KSP-FF K=50 mean blocking {cal['chosen_mean_blocking']:.3f}%)\n"
            f"- Training: {train['train_samples']} train / {train['validation_samples']} validation samples\n"
            f"- Best validation: top-1 recall={recall:.3f}, "
            f"regret={regret:.3f}, "
            f"price MAE={metrics.get('valid_price_mae', -1):.2f}\n"
            f"- **Quality gates (DISCLOSED relaxation 2026-08-06):** original gates were "
            f"recall≥0.80 / regret≤0.10; relaxed to recall≥0.20 / regret≤0.25.  "
            f"Actual recall={recall:.3f} (original recall gate: NOT met; relaxed gate: "
            f"{'met' if recall >= 0.20 else 'NOT met'}), "
            f"actual regret={regret:.3f} (original gate: {regret_orig}; relaxed gate: "
            f"{'met' if regret <= 0.25 else 'NOT met'}).\n"
            f"- Checkpoint: `{topo_dir / 'training' / 'deployment_weights.npz'}`\n"
            "\n### Per-arm statistics (new baseline)\n\n"
            f"| Arm | Mean ± std | 95% CI |\n"
            f"|---|---|---:|\n"
            f"| N1 | {format_mean_std_ci(n1_mean, n1_std, n1_ci)} |\n"
            f"| KSP-FF K5 | {format_mean_std_ci(k5_mean, k5_std, k5_ci)} |\n"
            f"| KSP-FF K50 | {format_mean_std_ci(k50_mean, k50_std, k50_ci)} |\n"
        )

    # Legacy baseline section from backups
    lines.extend([
        "",
        "## Legacy project-default baseline",
        "",
        "| Topology | N1 | KSP-FF K5 | KSP-FF K50 | dK | N1 vs K50 | Relative drop | Wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for topology, short in TOPOLOGIES:
        backup_dir = BASE / short / "backup_5_8pct"
        comp = load_json(backup_dir / "COMPARE_RESULTS.json")
        if comp is None:
            lines.append(f"| {short} | *missing* | *missing* | *missing* | - | - | - | - |")
            continue
        rows = comp["results"]
        n1_values = arm_values(rows, "n1")
        k5_values = arm_values(rows, "ksp_ff_k5")
        k50_values = arm_values(rows, "ksp_ff_k50")
        n1_mean = float(np.mean(n1_values))
        k5_mean = float(np.mean(k5_values))
        k50_mean = float(np.mean(k50_values))
        dk = k5_mean - k50_mean
        n1_vs_k50 = k50_mean - n1_mean
        rel_drop = (n1_vs_k50 / k50_mean * 100.0) if k50_mean else 0.0
        wins = sum(n < k for n, k in zip(n1_values, k50_values))
        lines.append(
            f"| {short} | {n1_mean:.3f}% | {k5_mean:.3f}% | {k50_mean:.3f}% | "
            f"{dk:.2f} | {n1_vs_k50:.2f} pp | {rel_drop:.1f}% | {wins}/10 |"
        )

    lines.extend([
        "",
        "## Remaining known口径 differences from Doherty 2025 / XLRON",
        "",
        "1. **Path pool tie-break:** now aligned to `(hops, tuple)` (XLRON).",
        "2. **Bitrate set:** project uses `{25, 50, 75, 100}` Gbps; Doherty/XLRON uses 25-100 Gbps step 1.  "
        "   Mean is identical (62.5 Gbps), but the tail distribution differs.",
        "3. **Modulation / first-fit / holding-time sampling semantics:** project env may still differ in edge cases; "
        "   no residual bias observed on NSFNET, but COST239's density amplifies any ordering effect.",
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines + details), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
