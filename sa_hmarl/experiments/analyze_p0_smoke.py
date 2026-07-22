#!/usr/bin/env python3
"""Quick comparison of P0 COST239 smoke datasets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def summarize_run(root: Path, label: str) -> dict:
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        return {"label": label, "missing": True}
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    diag = meta.get("diagnostics", {}).get("train", {})
    coefs = meta.get("return_coefs", {})

    extra: dict = {}
    npz_path = root / "train.npz"
    if npz_path.exists():
        data = dict(np.load(npz_path, allow_pickle=False))
        returns = data["returns"]
        mask = data["mask"]
        ranges = []
        overload_pos = []
        blocked_pos = []
        top1_overload = []
        for i in range(returns.shape[0]):
            valid = mask[i]
            ret = returns[i, valid]
            if ret.size == 0:
                continue
            ranges.append(float(ret.max() - ret.min()))
            best_idx = int(ret.argmax())
            ol = data.get("future_server_overload_counts")
            bc = data.get("future_blocked_counts")
            if ol is not None:
                ol_row = ol[i, valid]
                overload_pos.append(float((ol_row > 0).mean()))
                top1_overload.append(float(ol_row[best_idx] > 0))
            if bc is not None:
                blocked_pos.append(float((bc[i, valid] > 0).mean()))
        if ranges:
            extra["mean_return_range"] = float(np.mean(ranges))
            extra["nonzero_range_rate"] = float(np.mean(np.asarray(ranges) > 1e-6))
        if overload_pos:
            extra["future_overload_positive_rate"] = float(np.mean(overload_pos))
            extra["top1_future_overload_rate"] = float(np.mean(top1_overload))
        if blocked_pos:
            extra["future_blocked_positive_rate"] = float(np.mean(blocked_pos))

    return {
        "label": label,
        "missing": False,
        "groups": diag.get("groups", 0),
        "avg_candidates": diag.get("avg_candidates", 0.0),
        "nonzero_range_rate": diag.get("nonzero_return_range_rate", 0.0),
        "mean_return_range": diag.get("mean_return_range", 0.0),
        "oracle_headroom_pp": diag.get("oracle_headroom_pp", 0.0),
        "future_blocked_var": diag.get("future_blocked_variance", 0.0),
        "future_overload_var": diag.get("future_server_overload_variance", 0.0),
        "future_overload_positive_rate": diag.get("future_server_overload_positive_rate", 0.0),
        "future_nsb_positive_rate": diag.get("future_nsb_positive_rate", 0.0),
        "top1_top2_gap": diag.get("top1_top2_return_gap", 0.0),
        "ppo_top1_rate": diag.get("ppo_top1_rate", 0.0),
        "return_overload_coef": coefs.get("future_server_overload", 0.0),
        **extra,
    }


def main() -> None:
    base = Path(__file__).parent / "p0_cost239_smoke"
    runs = [
        (base / "a_baseline", "A_baseline_ppo_r_no_ol"),
        (base / "b_strong_ppo_r_overload8", "B_strong_ppo_r_ol8"),
        (base / "c_ranker_future_overload8", "C_ranker_future_ol8"),
    ]
    rows = [summarize_run(p, l) for p, l in runs]

    # Print markdown table.
    headers = [
        "Run", "groups", "avg_cand", "nonzero_range", "mean_range",
        "oracle_headroom_pp", "blocked_var", "overload_var",
        "overload+", "blocked+", "top1_overload+", "NSB+",
        "top1_top2_gap", "ppo_top1", "ol_coef",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        if r.get("missing"):
            print(f"| {r['label']} | (missing) |" + " |" * (len(headers) - 2))
            continue
        vals = [
            r["label"],
            f"{r['groups']}",
            f"{r['avg_candidates']:.2f}",
            f"{r['nonzero_range_rate']:.2%}",
            f"{r['mean_return_range']:.4f}",
            f"{r['oracle_headroom_pp']:.4f}",
            f"{r['future_blocked_var']:.4f}",
            f"{r['future_overload_var']:.4f}",
            f"{r['future_overload_positive_rate']:.2%}",
            f"{r.get('future_blocked_positive_rate', 0):.2%}",
            f"{r.get('top1_future_overload_rate', 0):.2%}",
            f"{r['future_nsb_positive_rate']:.2%}",
            f"{r['top1_top2_gap']:.4f}",
            f"{r['ppo_top1_rate']:.2%}",
            f"{r['return_overload_coef']:.1f}",
        ]
        print("| " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
