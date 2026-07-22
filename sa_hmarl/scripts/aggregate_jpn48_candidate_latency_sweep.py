#!/usr/bin/env python3
"""Aggregate individual JPN48 candidate-latency sweep runs into a single report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_aggregate(outdir: Path, label: str) -> dict:
    path = outdir / f"{label}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    method = next(iter(data["methods"].values()))
    return dict(method["aggregate"])


def main() -> None:
    outdir = Path("sa_hmarl/experiments/jpn48_candidate_latency_sweep")
    labels = [
        "baseline_v12",
        "v12_k20_hops",
        "v12_k30_hops",
        "v12_k50_hops_all_legal",
        "v12_k50_hops_legalctx48",
        "v12_k50_hops_legalctx48_ensure_ksp",
    ]
    display = {
        "baseline_v12": "baseline_v12",
        "v12_k20_hops": "v12_k20_hops",
        "v12_k30_hops": "v12_k30_hops",
        "v12_k50_hops_all_legal": "v12_k50_hops_all_legal",
        "v12_k50_hops_legalctx48": "v12_k50_hops_legalctx48",
        "v12_k50_hops_legalctx48_ensure_ksp": "v12_k50_hops_legalctx48_ensure_ksp",
    }

    rows = []
    for label in labels:
        agg = load_aggregate(outdir, label)
        rows.append({
            "label": display[label],
            "blocking": agg["blocking_rate"],
            "raw_empty": agg["raw_mask_empty_rate"],
            "nsb": agg["no_suitable_block_rate"],
            "overload": agg["server_overload_rate"],
            "deadline": agg.get("deadline_failure_rate", 0.0),
            "other": agg.get("other_failure_rate", 0.0),
            "delay_mean": agg["mean_delay_ms"],
            "delay_p95": agg["p95_delay_ms"],
            "decision_mean": agg["mean_decision_time_ms"],
            "decision_p95": agg["p95_decision_time_ms"],
        })

    base = rows[0]
    for r in rows:
        r["delta_blocking_pp"] = (r["blocking"] - base["blocking"]) * 100
        r["delta_nsb_pp"] = (r["nsb"] - base["nsb"]) * 100
        r["delta_overload_pp"] = (r["overload"] - base["overload"]) * 100
        r["delta_decision_mean"] = r["decision_mean"] - base["decision_mean"]
        r["delta_decision_p95"] = r["decision_p95"] - base["decision_p95"]
        blocking_drop_pp = (base["blocking"] - r["blocking"]) * 100
        if blocking_drop_pp > 1e-6:
            r["mean_ms_per_pp"] = r["delta_decision_mean"] / blocking_drop_pp
            r["p95_ms_per_pp"] = r["delta_decision_p95"] / blocking_drop_pp
        else:
            r["mean_ms_per_pp"] = np.nan
            r["p95_ms_per_pp"] = np.nan

    # Build markdown
    lines = [
        "# JPN48 Online Candidate-Set / Latency Sweep (v1.3 Selection)",
        "",
        "## Table 1: Full Results",
        "",
        "| Method | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['blocking']:.2%} | {r['raw_empty']:.2%} | "
            f"{r['nsb']:.2%} | {r['overload']:.2%} | {r['deadline']:.2%} | {r['other']:.2%} | "
            f"{r['delay_mean']:.3f}/{r['delay_p95']:.3f} ms | "
            f"{r['decision_mean']:.3f}/{r['decision_p95']:.3f} ms |"
        )

    lines += ["", "## Table 2: Delta vs baseline_v12", ""]
    lines.append("| Method | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) | Δ Decision mean (ms) | Δ Decision P95 (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['delta_blocking_pp']:+.2f} | {r['delta_nsb_pp']:+.2f} | "
            f"{r['delta_overload_pp']:+.2f} | {r['delta_decision_mean']:+.3f} | {r['delta_decision_p95']:+.3f} |"
        )

    lines += ["", "## Table 3: Blocking-Latency Trade-off", ""]
    lines.append("| Method | Blocking | Decision mean | Decision P95 | mean ms / 1pp blocking | P95 ms / 1pp blocking |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['blocking']:.2%} | {r['decision_mean']:.3f} ms | {r['decision_p95']:.3f} ms | "
            f"{r['mean_ms_per_pp']:.2f} | {r['p95_ms_per_pp']:.2f} |"
        )

    # Recommendations
    lines += ["", "## Questions & Recommendations", ""]
    lines.append("1. **k=20 -> 30 -> 50 收益递减？** 观察 blocking 下降幅度和时延增幅。")
    lines.append("2. **legalctx48 是否显著降低时延？** 对比 `v12_k50_hops_all_legal` 与 `v12_k50_hops_legalctx48` 的 decision mean/P95。")
    lines.append("3. **ensure_ksp_action 是否几乎不损失 blocking？** 对比两个 legalctx48 配置。")
    lines.append("4. **推荐配置**：根据 blocking 改善和时延代价综合判断。")
    lines.append("5. **推荐方案相对 baseline_v12 的收益/代价**：见 Table 2/3。")

    md_path = Path("sa_hmarl/experiments/jpn48_candidate_latency_sweep_full.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = Path("sa_hmarl/experiments/jpn48_candidate_latency_sweep_full.json")
    json_path.write_text(json.dumps({
        "labels": labels,
        "rows": rows,
        "baseline": "baseline_v12",
    }, indent=2), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
