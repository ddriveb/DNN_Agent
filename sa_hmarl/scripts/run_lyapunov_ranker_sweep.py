"""Screen and formally validate Lyapunov drift-plus-penalty inference-time reranking.

Compares `lyapunov_rank_only` against the fixed `counterfactual_rank_only` (v1.2)
ranker on the same request traces.  No retraining is performed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import evaluate


SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario_id": "snap24_gnutella_reach_s24_default3_ai0p09_sz30p0_h14p0",
        "split_profile": "default3",
        "num_splits": 3,
        "arrival_interval": 0.09,
        "size_max_mb": 30.0,
        "holding_max": 14.0,
    },
    {
        "scenario_id": "snap24_gnutella_reach_s24_complex5_v2_lite_ai0p15_sz30p0_h14p0",
        "split_profile": "complex5_v2_lite",
        "num_splits": 5,
        "arrival_interval": 0.15,
        "size_max_mb": 30.0,
        "holding_max": 14.0,
    },
]


def _build_eval_args(
    scenario: Dict[str, Any],
    lyap_args: Dict[str, Any],
    run_args: argparse.Namespace,
    seeds: str,
    episodes: int,
    output_json: Path,
    output_md: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        agent_c_checkpoint=run_args.agent_c_checkpoint,
        agent_r_checkpoint=run_args.agent_r_checkpoint,
        ranking_checkpoint=run_args.ranking_checkpoint,
        supervised_checkpoint=run_args.supervised_checkpoint,
        deep_rmsa_checkpoint=run_args.deep_rmsa_checkpoint,
        methods="ppo_r,counterfactual_rank_only,lyapunov_rank_only,deep_rmsa,ksp_bf",
        lambda_values="0.5",
        seeds=seeds,
        episodes=episodes,
        requests_per_episode=run_args.requests_per_episode,
        topology=run_args.topology,
        num_slots=run_args.num_slots,
        num_servers=run_args.num_servers,
        k_paths=run_args.k_paths,
        max_blocks=run_args.max_blocks,
        block_sort_strategy=run_args.block_sort_strategy,
        split_profile=scenario["split_profile"],
        num_splits=scenario["num_splits"],
        arrival_interval=scenario["arrival_interval"],
        holding_min=run_args.holding_min,
        holding_max=scenario["holding_max"],
        deadline_min=run_args.deadline_min,
        deadline_max=run_args.deadline_max,
        size_min_mb=run_args.size_min_mb,
        size_max_mb=scenario["size_max_mb"],
        edge_cost_min=run_args.edge_cost_min,
        edge_cost_max=run_args.edge_cost_max,
        modulation_profile=run_args.modulation_profile,
        device=run_args.device,
        output_json=str(output_json),
        output_md=str(output_md),
        lyap_lambda_spec=lyap_args["lambda_spec"],
        lyap_lambda_srv=run_args.lyap_lambda_srv,
        lyap_epsilon_spec=lyap_args["epsilon_spec"],
        lyap_epsilon_srv=run_args.lyap_epsilon_srv,
        lyap_queue_clip=run_args.lyap_queue_clip,
        lyap_spec_damage_mode=lyap_args["damage_mode"],
        lyap_srv_damage_mode=run_args.lyap_srv_damage_mode,
        lyap_update_mode=run_args.lyap_update_mode,
    )


def _metric(report: Dict[str, Any], method: str, key: str) -> Optional[float]:
    val = report["methods"][method]["aggregate"].get(key)
    return None if val is None else float(val)


def _extract_row(
    scenario: Dict[str, Any],
    lyap_args: Dict[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    v12_block = _metric(report, "counterfactual_rank_only", "blocking_rate")
    lyap_block = _metric(report, "lyapunov_rank_only", "blocking_rate")
    v12_overload = _metric(report, "counterfactual_rank_only", "server_overload_rate")
    lyap_overload = _metric(report, "lyapunov_rank_only", "server_overload_rate")
    v12_delay = _metric(report, "counterfactual_rank_only", "mean_delay_ms")
    lyap_delay = _metric(report, "lyapunov_rank_only", "mean_delay_ms")

    return {
        "scenario_id": scenario["scenario_id"],
        "split_profile": scenario["split_profile"],
        "arrival_interval": scenario["arrival_interval"],
        "size_max_mb": scenario["size_max_mb"],
        "holding_max": scenario["holding_max"],
        "lambda_spec": lyap_args["lambda_spec"],
        "epsilon_spec": lyap_args["epsilon_spec"],
        "damage_mode": lyap_args["damage_mode"],
        "blocking": lyap_block,
        "raw_empty": _metric(report, "lyapunov_rank_only", "raw_mask_empty_rate"),
        "nsb": _metric(report, "lyapunov_rank_only", "no_suitable_block_rate"),
        "overload": lyap_overload,
        "delay": lyap_delay,
        "mean_H_spec": _metric(report, "lyapunov_rank_only", "mean_H_spec"),
        "p95_H_spec": _metric(report, "lyapunov_rank_only", "p95_H_spec"),
        "mean_adjustment_abs": _metric(report, "lyapunov_rank_only", "mean_adjustment_abs"),
        "changed_action_rate_vs_v12": _metric(report, "lyapunov_rank_only", "changed_action_rate_vs_v12"),
        "delta_blocking_vs_v12": (v12_block - lyap_block) * 100.0 if v12_block is not None and lyap_block is not None else None,
        "delta_overload_vs_v12": (lyap_overload - v12_overload) * 100.0 if v12_overload is not None and lyap_overload is not None else None,
        "delta_delay_vs_v12": (lyap_delay - v12_delay) if v12_delay is not None and lyap_delay is not None else None,
    }


def _classify(row: Dict[str, Any]) -> str:
    db = row.get("delta_blocking_vs_v12")
    do = row.get("delta_overload_vs_v12")
    dd = row.get("delta_delay_vs_v12")
    if db is None or do is None or dd is None:
        return "FAIL"
    if db >= 1.0 and do <= 0.0 and dd <= 0.5:
        return "STRONG_PASS"
    if db >= 0.5 and do <= 0.3 and dd <= 0.5:
        return "PASS"
    return "FAIL"


def _write_summary(path_json: Path, path_md: Path, rows: List[Dict[str, Any]], title: str) -> None:
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("delta_blocking_vs_v12") if r.get("delta_blocking_vs_v12") is not None else -1e9),
        reverse=True,
    )
    for row in rows_sorted:
        row["verdict"] = _classify(row)
    path_json.write_text(json.dumps({"rows": rows_sorted}, indent=2), encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        "Positive `Δ blocking vs v1.2` means Lyapunov reduced blocking. "
        "Positive `Δ overload` means Lyapunov increased overload.",
        "",
        "| Scenario | λ_spec | ε_spec | Mode | Lyap Blk | v1.2 Blk | Δ blk pp | Δ ovld pp | Δ delay ms | mean H | p95 H | adj abs | changed% | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        lines.append(
            f"| {r['scenario_id']} | {r['lambda_spec']:.2f} | {r['epsilon_spec']:.2f} | {r['damage_mode']} | "
            f"{r['blocking']:.2%} | {(r['blocking'] + r['delta_blocking_vs_v12'] / 100.0):.2%} | "
            f"{r['delta_blocking_vs_v12']:+.2f} | {r['delta_overload_vs_v12']:+.2f} | {r['delta_delay_vs_v12']:+.2f} | "
            f"{r['mean_H_spec']:.2f} | {r['p95_H_spec']:.2f} | {r['mean_adjustment_abs']:.3f} | "
            f"{r['changed_action_rate_vs_v12'] * 100.0:.1f}% | {r['verdict']} |"
        )
    path_md.write_text("\n".join(lines), encoding="utf-8")


def _screening(run_args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = Path(run_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    lambdas = [float(v) for v in run_args.lambda_specs.split(",")]
    epsilons = [float(v) for v in run_args.epsilon_specs.split(",")]
    modes = run_args.damage_modes.split(",")

    for scenario in SCENARIOS:
        for lam in lambdas:
            for eps in epsilons:
                for mode in modes:
                    lyap_args = {"lambda_spec": lam, "epsilon_spec": eps, "damage_mode": mode}
                    sid = scenario["scenario_id"]
                    run_name = f"{sid}_l{lam:g}_e{eps:g}_{mode}"
                    out_json = out_dir / f"{run_name}_screen.json"
                    out_md = out_dir / f"{run_name}_screen.md"
                    print(f"\n[screen] {run_name}", flush=True)
                    eval_args = _build_eval_args(
                        scenario, lyap_args, run_args,
                        run_args.screen_seeds, run_args.screen_episodes,
                        out_json, out_md,
                    )
                    try:
                        report = evaluate(eval_args)
                    except Exception as exc:
                        print(f"[error] {run_name}: {type(exc).__name__}: {exc}", flush=True)
                        continue
                    row = _extract_row(scenario, lyap_args, report)
                    rows.append(row)
                    print(
                        f"[done] {run_name}: Δblk={row['delta_blocking_vs_v12']:+.2f}pp, "
                        f"Δovld={row['delta_overload_vs_v12']:+.2f}pp, "
                        f"Δdelay={row['delta_delay_vs_v12']:+.2f}ms, "
                        f"changed={row['changed_action_rate_vs_v12'] * 100.0:.1f}%, "
                        f"Hmean={row['mean_H_spec']:.2f}",
                        flush=True,
                    )
                    _write_summary(
                        Path(run_args.screening_json),
                        Path(run_args.screening_md),
                        rows,
                        "Lyapunov Ranker Screening",
                    )
    return rows


def _select_top(rows: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    pass_rows = [r for r in rows if _classify(r) in ("PASS", "STRONG_PASS")]
    ranked = sorted(
        pass_rows if pass_rows else rows,
        key=lambda r: r.get("delta_blocking_vs_v12") if r.get("delta_blocking_vs_v12") is not None else -1e9,
        reverse=True,
    )
    return ranked[:n]


def _formal(run_args: argparse.Namespace, selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_dir = Path(run_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for row in selected:
        scenario = next(s for s in SCENARIOS if s["scenario_id"] == row["scenario_id"])
        lyap_args = {
            "lambda_spec": row["lambda_spec"],
            "epsilon_spec": row["epsilon_spec"],
            "damage_mode": row["damage_mode"],
        }
        sid = scenario["scenario_id"]
        run_name = f"{sid}_l{row['lambda_spec']:g}_e{row['epsilon_spec']:g}_{row['damage_mode']}"
        out_json = out_dir / f"{run_name}_formal.json"
        out_md = out_dir / f"{run_name}_formal.md"
        print(f"\n[formal] {run_name}", flush=True)
        eval_args = _build_eval_args(
            scenario, lyap_args, run_args,
            run_args.formal_seeds, run_args.formal_episodes,
            out_json, out_md,
        )
        try:
            report = evaluate(eval_args)
        except Exception as exc:
            print(f"[error] {run_name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        frow = _extract_row(scenario, lyap_args, report)
        rows.append(frow)
        print(
            f"[done] {run_name}: Δblk={frow['delta_blocking_vs_v12']:+.2f}pp, "
            f"Δovld={frow['delta_overload_vs_v12']:+.2f}pp, "
            f"Δdelay={frow['delta_delay_vs_v12']:+.2f}ms, "
            f"changed={frow['changed_action_rate_vs_v12'] * 100.0:.1f}%, "
            f"verdict={_classify(frow)}",
            flush=True,
        )
        _write_summary(
            Path(run_args.formal_json),
            Path(run_args.formal_md),
            rows,
            "Lyapunov Ranker Formal Validation",
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/lyapunov_ranker_sweep")
    parser.add_argument("--screening_json", default="sa_hmarl/experiments/lyapunov_ranker_sweep_screening.json")
    parser.add_argument("--screening_md", default="sa_hmarl/experiments/lyapunov_ranker_sweep_screening.md")
    parser.add_argument("--formal_json", default="sa_hmarl/experiments/lyapunov_ranker_sweep_formal.json")
    parser.add_argument("--formal_md", default="sa_hmarl/experiments/lyapunov_ranker_sweep_formal.md")

    # Scenario / env defaults
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--supervised_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_formal/c_post_decision_joint.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--device", default="cpu")

    # Screening grid
    parser.add_argument("--lambda_specs", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--epsilon_specs", default="0.02,0.05,0.10")
    parser.add_argument("--damage_modes", default="fs_path,fs_only,path_only")
    parser.add_argument("--screen_seeds", default="3030,4040,5050")
    parser.add_argument("--screen_episodes", type=int, default=5)
    parser.add_argument("--formal_seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--formal_episodes", type=int, default=20)
    parser.add_argument("--top_n", type=int, default=3)

    # Fixed Lyapunov hyperparameters
    parser.add_argument("--lyap_lambda_srv", type=float, default=0.0)
    parser.add_argument("--lyap_epsilon_srv", type=float, default=0.05)
    parser.add_argument("--lyap_queue_clip", type=float, default=20.0)
    parser.add_argument("--lyap_srv_damage_mode", default="util_after", choices=["util_after", "margin_after"])
    parser.add_argument("--lyap_update_mode", default="binary_raw_empty", choices=["binary_raw_empty", "continuous_pressure"])

    args = parser.parse_args()

    print("[stage] screening", flush=True)
    screen_rows = _screening(args)
    _write_summary(Path(args.screening_json), Path(args.screening_md), screen_rows, "Lyapunov Ranker Screening")
    print(f"\n[saved] {args.screening_json}")
    print(f"[saved] {args.screening_md}")

    selected = _select_top(screen_rows, args.top_n)
    if not selected:
        print("[abort] No configurations selected for formal validation.", flush=True)
        sys.exit(0)

    print(f"\n[stage] formal validation of top {len(selected)} config(s)", flush=True)
    formal_rows = _formal(args, selected)
    _write_summary(Path(args.formal_json), Path(args.formal_md), formal_rows, "Lyapunov Ranker Formal Validation")
    print(f"\n[saved] {args.formal_json}")
    print(f"[saved] {args.formal_md}")


if __name__ == "__main__":
    main()
