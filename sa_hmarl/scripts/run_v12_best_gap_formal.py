"""Stage 2 formal validation for the best v1.2 vs DeepRMSA gap scenarios.

Reads a Stage 1 summary JSON (e.g. from run_v12_split_pressure_grid.py),
selects the top PASS/MARGINAL scenarios (or a specific scenario), and reruns
each with a larger sample (5 seeds x 20 episodes by default).  A paired t-test
across seeds is reported for the blocking-rate gap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import evaluate


def _parse_csv_ints(text: str) -> List[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _metric(report: Dict[str, Any], method: str, key: str, seed: str = None) -> float:
    spec = report["methods"][method]
    if seed is None:
        return float(spec["aggregate"][key])
    return float(spec["per_seed"][seed][key])


def _seed_values(report: Dict[str, Any], method: str, key: str) -> np.ndarray:
    return np.array([float(v[key]) for v in report["methods"][method]["per_seed"].values()])


def _build_args(stage_config: Dict[str, Any], row: Dict[str, Any], formal_args: argparse.Namespace) -> SimpleNamespace:
    # Derive num_splits from profile name if not in row.
    profile = row["split_profile"]
    if profile.startswith("default"):
        num_splits = 3
    elif profile.startswith("complex5"):
        num_splits = 5
    else:
        num_splits = row.get("num_splits", 3)

    return SimpleNamespace(
        agent_c_checkpoint=stage_config.get("agent_c_checkpoint", formal_args.agent_c_checkpoint),
        agent_r_checkpoint=stage_config.get("agent_r_checkpoint", formal_args.agent_r_checkpoint),
        ranking_checkpoint=stage_config.get("ranking_checkpoint", formal_args.ranking_checkpoint),
        supervised_checkpoint=stage_config.get("supervised_checkpoint", formal_args.supervised_checkpoint),
        deep_rmsa_checkpoint=stage_config.get("deep_rmsa_checkpoint", formal_args.deep_rmsa_checkpoint),
        methods=formal_args.methods,
        lambda_values=stage_config.get("lambda_values", "0.5"),
        seeds=",".join(str(s) for s in formal_args.seeds),
        episodes=formal_args.episodes,
        requests_per_episode=formal_args.requests_per_episode,
        topology=row.get("topology", stage_config.get("topology", formal_args.topology)),
        num_slots=int(row.get("num_slots", stage_config.get("num_slots", formal_args.num_slots))),
        num_servers=int(stage_config.get("num_servers", formal_args.num_servers)),
        k_paths=int(stage_config.get("k_paths", formal_args.k_paths)),
        max_blocks=int(stage_config.get("max_blocks", formal_args.max_blocks)),
        block_sort_strategy=stage_config.get("block_sort_strategy", formal_args.block_sort_strategy),
        split_profile=profile,
        num_splits=num_splits,
        arrival_interval=float(row["arrival_interval"]),
        holding_min=float(stage_config.get("holding_min", formal_args.holding_min)),
        holding_max=float(row["holding_max"]),
        deadline_min=float(stage_config.get("deadline_min", formal_args.deadline_min)),
        deadline_max=float(stage_config.get("deadline_max", formal_args.deadline_max)),
        size_min_mb=float(stage_config.get("size_min_mb", formal_args.size_min_mb)),
        size_max_mb=float(row["size_max_mb"]),
        edge_cost_min=float(stage_config.get("edge_cost_min", formal_args.edge_cost_min)),
        edge_cost_max=float(stage_config.get("edge_cost_max", formal_args.edge_cost_max)),
        modulation_profile=stage_config.get("modulation_profile", formal_args.modulation_profile),
        device=formal_args.device,
        output_json="",  # not used by evaluate; we write ourselves
        output_md="",
    )


def _paired_ttest(v12: np.ndarray, deep: np.ndarray) -> Tuple[float, float]:
    delta = deep - v12  # positive means v1.2 better
    if len(delta) < 2:
        return float(np.mean(delta)), math.nan
    t, p = stats.ttest_rel(deep, v12)
    return float(np.mean(delta)), float(p)


def _run_one(row: Dict[str, Any], stage_config: Dict[str, Any], formal_args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    sid = row["scenario_id"]
    eval_args = _build_args(stage_config, row, formal_args)
    print(f"\n[formal] {sid}: seeds={formal_args.seeds}, episodes={formal_args.episodes}", flush=True)
    try:
        report = evaluate(eval_args)
    except Exception as exc:
        print(f"[error] {sid}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return {"scenario_id": sid, "error": f"{type(exc).__name__}: {exc}"}

    out_path = out_dir / f"{sid}_formal.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}", flush=True)

    v12 = _seed_values(report, "counterfactual_rank_only", "blocking_rate")
    deep = _seed_values(report, "deep_rmsa", "blocking_rate")
    gap_mean, gap_p = _paired_ttest(v12, deep)

    v12_delay = _metric(report, "counterfactual_rank_only", "mean_delay_ms")
    deep_delay = _metric(report, "deep_rmsa", "mean_delay_ms")
    v12_overload = _metric(report, "counterfactual_rank_only", "server_overload_rate")
    deep_overload = _metric(report, "deep_rmsa", "server_overload_rate")

    result = {
        "scenario_id": sid,
        "split_profile": row["split_profile"],
        "num_splits": eval_args.num_splits,
        "arrival_interval": eval_args.arrival_interval,
        "size_max_mb": eval_args.size_max_mb,
        "holding_max": eval_args.holding_max,
        "v12_blocking_mean": float(np.mean(v12)),
        "v12_blocking_std": float(np.std(v12, ddof=1)) if len(v12) > 1 else 0.0,
        "deep_blocking_mean": float(np.mean(deep)),
        "deep_blocking_std": float(np.std(deep, ddof=1)) if len(deep) > 1 else 0.0,
        "deep_minus_v12_pp_mean": gap_mean * 100.0,
        "deep_minus_v12_pp_pvalue": gap_p,
        "v12_delay": v12_delay,
        "deep_delay": deep_delay,
        "delay_delta_ms": v12_delay - deep_delay,
        "v12_overload": v12_overload,
        "deep_overload": deep_overload,
        "overload_delta_pp": (v12_overload - deep_overload) * 100.0,
        "formal_json": str(out_path),
    }
    return result


def _write_summary(summary_path: Path, md_path: Path, config: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    summary_path.write_text(json.dumps({"config": config, "rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# v1.2 Formal Validation (Stage 2)",
        "",
        "Positive `Deep-v1.2` means v1.2 has lower blocking than DeepRMSA.",
        "",
        "| Scenario | Profile | AI | SizeMax | HoldMax | v1.2 Blk | Deep Blk | Deep-v1.2 | p-value | Delay Δ ms | Overload Δ pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['scenario_id']} | - | - | - | - | ERROR | - | - | - | - | - |")
            continue
        lines.append(
            f"| {r['scenario_id']} | {r['split_profile']} | "
            f"{r['arrival_interval']:.2f} | {r['size_max_mb']:.1f} | {r['holding_max']:.1f} | "
            f"{r['v12_blocking_mean']:.2%} ± {r['v12_blocking_std']:.2%} | "
            f"{r['deep_blocking_mean']:.2%} ± {r['deep_blocking_std']:.2%} | "
            f"{r['deep_minus_v12_pp_mean']:+.2f}pp | {r['deep_minus_v12_pp_pvalue']:.3f} | "
            f"{r['delay_delta_ms']:+.2f} | {r['overload_delta_pp']:+.2f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_json", required=True, help="Stage 1 summary JSON.")
    parser.add_argument("--scenario_id", default=None, help="Validate a specific scenario ID.")
    parser.add_argument("--top_n", type=int, default=2, help="Validate top N PASS/MARGINAL scenarios if --scenario_id not given.")
    parser.add_argument("--include_fail", action="store_true", help="Also include top FAIL scenarios if no PASS/MARGINAL.")
    parser.add_argument("--seeds", type=_parse_csv_ints, default=[3030, 4040, 5050, 6060, 7070])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--methods", default="ppo_r,counterfactual_rank_only,deep_rmsa,ksp_bf")
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v12_formal_validation")
    # Fallback checkpoint paths (used only if missing from summary config).
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--supervised_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_formal/c_post_decision_joint.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"[abort] Summary not found: {summary_path}", flush=True)
        sys.exit(1)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    stage_config = payload.get("config", {})
    rows = payload.get("rows", [])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenario_id:
        chosen = [r for r in rows if r["scenario_id"] == args.scenario_id]
        if not chosen:
            print(f"[abort] Scenario {args.scenario_id} not found in summary.", flush=True)
            sys.exit(1)
    else:
        ranked = sorted(rows, key=lambda r: r.get("deep_minus_v12_pp", -1e9), reverse=True)
        pass_marg = [r for r in ranked if r.get("verdict") in ("PASS", "MARGINAL")]
        chosen = pass_marg[: args.top_n]
        if not chosen and args.include_fail:
            chosen = ranked[: args.top_n]
        if not chosen:
            print("[abort] No PASS/MARGINAL scenarios found. Use --include_fail to validate top FAIL scenarios.", flush=True)
            sys.exit(1)

    results: List[Dict[str, Any]] = []
    for row in chosen:
        result = _run_one(row, stage_config, args, out_dir)
        results.append(result)
        # Update running summary after each scenario so progress is visible.
        _write_summary(
            out_dir / "v12_formal_validation_summary.json",
            out_dir / "v12_formal_validation_summary.md",
            vars(args),
            results,
        )

    _write_summary(
        out_dir / "v12_formal_validation_summary.json",
        out_dir / "v12_formal_validation_summary.md",
        vars(args),
        results,
    )
    print("\nSaved v12_formal_validation_summary.json/.md")


if __name__ == "__main__":
    main()
