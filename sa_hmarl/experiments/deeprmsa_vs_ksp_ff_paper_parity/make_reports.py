#!/usr/bin/env python3
"""Final aggregation — per-seed CSV, final comparison, training report, run log.

Reads artifacts produced by Stages A-D and writes:
- per_seed_results.csv
- FINAL_COMPARISON.json / FINAL_COMPARISON.md
- DEEPRMSA_TRAINING_REPORT.md
- RUN_LOG.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

from paper_rmsa_core import load_run_config

CFG = load_run_config()
DCFG = CFG["methods"]["deeprmsa_local"]
MEAS = CFG["measurement"]
LEGACY_SNAP24 = {
    "label": "Legacy masked DeepRMSA-style K=3/M=1 on custom SNAP24",
    "blocking_rate": 0.257067,
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{x:.4%}"


def main() -> int:
    # --------------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------------
    unit = _load(EXP_DIR / "results" / "unit_tests.json")
    smoke = _load(EXP_DIR / "results" / "smoke.json")
    ksp_parity = _load(EXP_DIR / "KSP_PARITY_RESULTS.json")
    ksp_raw = _load(EXP_DIR / "results" / "ksp_parity_raw.json")["runs"]
    dr_eval = _load(EXP_DIR / "results" / "deeprmsa_eval.json")

    train_summaries: Dict[str, Dict[str, Any]] = {}
    train_histories: Dict[str, List[Dict[str, Any]]] = {}
    for topology in CFG["topologies"]:
        for seed in DCFG["train_seeds"]:
            sdir = EXP_DIR / "checkpoints" / topology / f"seed_{seed}"
            train_summaries[f"{topology}/{seed}"] = _load(sdir / "training_summary.json")
            train_histories[f"{topology}/{seed}"] = _load(sdir / "history.json")

    # --------------------------------------------------------------
    # per_seed_results.csv
    # --------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for run in ksp_raw:
        for method, res in run["results"].items():
            rows.append({
                "topology": run["topology"], "method": method, "train_seed": "",
                "seed": run["seed"], "warmup": res["warmup"], "evaluated": res["evaluated"],
                "admitted": res["admitted"], "blocked": res["blocked"],
                "blocking_rate": res["blocking_rate"],
                "mean_utilization": res["mean_utilization"],
                "avg_fs": res["avg_fs"], "avg_hop_count": res["avg_hop_count"],
            })
    for entry in dr_eval["results"]:
        for seed, res in entry["per_seed"].items():
            rows.append({
                "topology": entry["topology"], "method": "deeprmsa_local",
                "train_seed": entry["training_seed"], "seed": seed,
                "warmup": res["warmup"], "evaluated": res["evaluated"],
                "admitted": res["admitted"], "blocked": res["blocked"],
                "blocking_rate": res["blocking_rate"],
                "mean_utilization": res["mean_utilization"],
                "avg_fs": res["avg_fs"], "avg_hop_count": res["avg_hop_count"],
            })
    rows.sort(key=lambda r: (r["topology"], r["method"], str(r["train_seed"]), r["seed"]))
    with (EXP_DIR / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------------
    # DeepRMSA aggregation across training seeds
    # --------------------------------------------------------------
    dr_summary: Dict[str, Any] = {}
    for topology in CFG["topologies"]:
        entries = [e for e in dr_eval["results"] if e["topology"] == topology]
        entries.sort(key=lambda e: e["training_seed"])
        train_means = [e["mean_blocking_rate"] for e in entries]
        # Test-seed variance pooled across all (train_seed, test_seed) rates.
        pooled = [r["blocking_rate"] for e in entries for r in e["per_seed"].values()]
        dr_summary[topology] = {
            "per_train_seed": {
                str(e["training_seed"]): {
                    "mean_blocking_rate": e["mean_blocking_rate"],
                    "std_blocking_rate_test_seeds": e["std_blocking_rate"],
                    "checkpoint_meta": e["checkpoint_meta"],
                }
                for e in entries
            },
            "grand_mean": float(np.mean(train_means)),
            "train_seed_std": float(np.std(train_means, ddof=1)) if len(train_means) > 1 else 0.0,
            "test_seed_std_pooled": float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
            "paper_published": CFG["paper"]["reference_values"][topology]["deeprmsa_published"],
        }

    # --------------------------------------------------------------
    # FINAL_COMPARISON.json
    # --------------------------------------------------------------
    final: Dict[str, Any] = {
        "protocol_id": CFG["protocol_id"],
        "paper": CFG["paper"],
        "ksp_ff": {t: ksp_parity["aggregated"][t] for t in CFG["topologies"]},
        "deeprmsa_local": dr_summary,
        "parity": {
            "ksp_ff_k5_km_within_paper_2std": ksp_parity["parity_pass"],
        },
        "appendix_legacy_snap24": LEGACY_SNAP24,
    }
    (EXP_DIR / "FINAL_COMPARISON.json").write_text(
        json.dumps(final, indent=2, default=str), encoding="utf-8"
    )

    # --------------------------------------------------------------
    # FINAL_COMPARISON.md
    # --------------------------------------------------------------
    lines: List[str] = [
        "# FINAL COMPARISON — DeepRMSA vs KSP-FF (paper-standard pure RMSA)",
        "",
        f"* Protocol: `{CFG['protocol_id']}` (see PROTOCOL_LOCK.md)",
        f"* Warmup {MEAS['warmup_requests']}, measured {MEAS['evaluated_requests']} requests; "
        f"10 shared test seeds {MEAS['test_seeds']}",
        "* KSP-FF std = across 10 test seeds. DeepRMSA local: mean over 3 training "
        "seeds; std shown as (training-seed std / pooled test-seed std).",
        "",
    ]
    for topo_name, topo_cfg in CFG["topologies"].items():
        paper = CFG["paper"]["reference_values"][topo_name]
        agg = ksp_parity["aggregated"][topo_name]
        dr = dr_summary[topo_name]
        lines += [
            f"## {topo_cfg['label']} — {topo_cfg['load_erlang']} Erlang, 100 slots",
            "",
            "| Method | Ours | Paper | Within paper ±2σ |",
            "|---|---|---|---|",
        ]
        for method, label in (("ksp_ff_k5_km", "KSP-FF K=5 km"),
                              ("ksp_ff_k5_hops", "KSP-FF K=5 hops"),
                              ("ksp_ff_k50_hops", "KSP-FF K=50 hops")):
            a = agg[method]
            lines.append(
                f"| {label} | {_pct(a['mean'])} ± {_pct(a['std'])} "
                f"| {_pct(a['paper_mean'])} ± {_pct(a['paper_std'])} "
                f"| {'YES' if a['within_paper_2std'] else 'NO'} |"
            )
        lines.append(
            f"| DeepRMSA (local, K=5) | {_pct(dr['grand_mean'])} "
            f"(±{_pct(dr['train_seed_std'])} train / ±{_pct(dr['test_seed_std_pooled'])} test) "
            f"| {_pct(paper['deeprmsa_published'])} (published) | — |"
        )
        lines.append("")
        lines.append("Per-training-seed DeepRMSA means: " + ", ".join(
            f"seed {s}: {_pct(v['mean_blocking_rate'])}"
            for s, v in dr["per_train_seed"].items()
        ))
        lines.append("")

    lines += [
        "## Appendix (excluded from the main comparison)",
        "",
        f"* {LEGACY_SNAP24['label']}: blocking {_pct(LEGACY_SNAP24['blocking_rate'])} "
        "(different topology, masked action space, K=3/M=1 — not comparable; "
        "kept for provenance only).",
        "",
    ]
    (EXP_DIR / "FINAL_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")

    # --------------------------------------------------------------
    # DEEPRMSA_TRAINING_REPORT.md
    # --------------------------------------------------------------
    tlines: List[str] = [
        "# DEEPRMSA TRAINING REPORT — Stage D",
        "",
        "Local source-semantic DeepRMSA: K=5 km-ordered paths, M=1, 5-dim unmasked",
        "policy, 5×128 ELU actor/critic, upstream overlapping-window A2C",
        f"(batch {DCFG['batch_size']}, window {2 * DCFG['batch_size'] - 1}, γ={DCFG['gamma']}, "
        f"entropy={DCFG['entropy_coef']}).",
        "",
        "**Training config (disclosed deviations, see PREFLIGHT_AUDIT.md D6/D7):** "
        f"lr={DCFG['lr']} (upstream 1e-5), grad-clip {DCFG['max_grad_norm']} (upstream 40), "
        f"raw advantages (normalize_advantages={DCFG.get('normalize_advantages', False)}), "
        f"{DCFG['total_train_requests']} requests per run. Deviations were driven by measured "
        "evidence on the worst-behaved seed: lr=1e-5 stalls at the SP-FF collapse at this "
        "compute budget; advantage normalization prevents escape; tighter grad-clip restores "
        "stable escape. DeepRMSA training was seed-unstable throughout — one NSFNET seed "
        "(43) sat at the SP-FF collapse for 525k requests, diverged to 32% blocking around "
        "625k, then recovered and reached its best validation at 1M. This instability is "
        "consistent with the paper's thesis and is reported per-seed below.",
        "",
    ]
    for topology, topo_cfg in CFG["topologies"].items():
        tlines.append(f"## {topo_cfg['label']}")
        tlines.append("")
        for seed in DCFG["train_seeds"]:
            key = f"{topology}/{seed}"
            summ = train_summaries[key]
            hist = train_histories[key]
            val_curve = ", ".join(f"{h['requests'] // 1000}k:{h['val_blocking']:.3%}" for h in hist)
            dr = dr_summary[topology]["per_train_seed"][str(seed)]
            tlines += [
                f"### Training seed {seed}",
                "",
                f"* Requests: {summ['total_requests']}, gradient updates: {summ['gradient_updates']}, "
                f"final ε: {summ['final_epsilon']:.3f}",
                f"* Train blocking (post-warmup stream): {_pct(summ['train_blocking_rate'])} "
                f"(invalid unmasked choices: {summ['train_invalid_choice']})",
                f"* Best validation blocking: {_pct(summ['best_validation_blocking_rate'])} "
                f"at {summ['best_meta'].get('requests', '?')} requests",
                f"* Test (10 seeds, argmax): {_pct(dr['mean_blocking_rate'])} ± {_pct(dr['std_blocking_rate_test_seeds'])}",
                f"* Validation curve (requests:val_blocking): {val_curve}",
                "",
            ]
        dr = dr_summary[topology]
        tlines += [
            f"**{topo_cfg['label']} summary:** grand mean {_pct(dr['grand_mean'])}; "
            f"training-seed variance (std) {_pct(dr['train_seed_std'])}; "
            f"test-seed variance (pooled std) {_pct(dr['test_seed_std_pooled'])}; "
            f"paper published {_pct(dr['paper_published'])}.",
            "",
        ]
    (EXP_DIR / "DEEPRMSA_TRAINING_REPORT.md").write_text("\n".join(tlines), encoding="utf-8")

    # --------------------------------------------------------------
    # RUN_LOG.md
    # --------------------------------------------------------------
    rlines: List[str] = [
        "# RUN LOG — DeepRMSA vs KSP-FF paper parity pipeline",
        "",
        "All stages executed 2026-07-17 on this machine (12 CPU workers).",
        "",
        "| Stage | Command | Outcome | Elapsed |",
        "|---|---|---|---|",
        f"| A unit tests | `python run_unit_tests.py` | "
        f"{'PASS' if unit['all_pass'] else 'FAIL'} ({sum(r['pass'] for r in unit['results'])}/{len(unit['results'])}) "
        f"| {unit['elapsed_seconds']:.1f} s |",
        f"| B smoke | `python run_smoke.py` | {'OK' if smoke['all_ok'] else 'FAIL'} | {smoke['elapsed_seconds']:.1f} s |",
        f"| C KSP parity | `python run_ksp_parity.py` | parity "
        f"{'PASS' if ksp_parity['parity_pass'] else 'FAIL'} | {ksp_parity['elapsed_seconds']:.1f} s |",
    ]
    for topology in CFG["topologies"]:
        for seed in DCFG["train_seeds"]:
            summ = train_summaries[f"{topology}/{seed}"]
            rlines.append(
                f"| D train {topology} s{seed} | `python train_deeprmsa_paper.py --topology {topology} --seed {seed}` "
                f"| best val {_pct(summ['best_validation_blocking_rate'])} | {summ['elapsed_seconds']:.0f} s |"
            )
    rlines.append(f"| D eval | `python eval_deeprmsa_paper.py` | done | {dr_eval['elapsed_seconds']:.1f} s |")
    rlines += [
        "",
        "## Notable events",
        "",
        "* Stage C first run used `xlron_cost239_ptrnet_real` for COST239 and FAILED parity",
        "  (our 3.18% vs paper 6.69±0.35%). Root cause: wrong COST239 variant. Switched to",
        "  upstream/XLRON `cost239_deeprmsa` (2× distances) and reran — see PREFLIGHT_AUDIT.md",
        "  addendum. NSFNET was unaffected and matched on the first run.",
        "* Stage D training required a measured hyperparameter investigation (all deviations",
        "  disclosed in PREFLIGHT_AUDIT.md D6/D7): lr=1e-5 stalls at the SP-FF collapse at this",
        "  compute budget; lr=1e-4 + grad-clip 40 diverged one seed; advantage normalization",
        "  prevents collapse escape; final config lr=1e-4 + grad-clip 5 + raw advantages let all",
        "  seeds converge (one NSFNET seed via a collapse→diverge→recover trajectory).",
        "* Final COST239/NSFNET parity status is recorded in KSP_PARITY_RESULTS.md.",
        "",
    ]
    (EXP_DIR / "RUN_LOG.md").write_text("\n".join(rlines), encoding="utf-8")

    print("[reports] wrote per_seed_results.csv, FINAL_COMPARISON.{json,md}, "
          "DEEPRMSA_TRAINING_REPORT.md, RUN_LOG.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
