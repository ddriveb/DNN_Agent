"""Screen scenarios where v1.2 R-ranker separates from DeepRMSA.

The script intentionally keeps topology/k_paths/max_blocks compatible with the
existing DeepRMSA checkpoint by default, and sweeps load/resource pressure
parameters that do not change model dimensions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np

from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import evaluate


def _parse_csv_floats(text: str) -> List[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _scenario_id(topology: str, slots: int, arrival: float, size_max: float, holding_max: float) -> str:
    return (
        f"{topology}_s{slots}_ai{str(arrival).replace('.', 'p')}"
        f"_sz{str(size_max).replace('.', 'p')}_h{str(holding_max).replace('.', 'p')}"
    )


def _make_eval_args(
    args: argparse.Namespace,
    slots: int,
    arrival: float,
    size_max: float,
    holding_max: float,
    out_dir: Path,
) -> SimpleNamespace:
    sid = _scenario_id(args.topology, slots, arrival, size_max, holding_max)
    return SimpleNamespace(
        agent_c_checkpoint=args.agent_c_checkpoint,
        agent_r_checkpoint=args.agent_r_checkpoint,
        ranking_checkpoint=args.ranking_checkpoint,
        supervised_checkpoint=args.supervised_checkpoint,
        deep_rmsa_checkpoint=args.deep_rmsa_checkpoint,
        methods=args.methods,
        lambda_values=args.lambda_values,
        seeds=args.seeds,
        episodes=args.episodes,
        requests_per_episode=args.requests_per_episode,
        topology=args.topology,
        num_slots=slots,
        num_servers=args.num_servers,
        k_paths=args.k_paths,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        split_profile=args.split_profile,
        num_splits=args.num_splits,
        arrival_interval=arrival,
        holding_min=args.holding_min,
        holding_max=holding_max,
        deadline_min=args.deadline_min,
        deadline_max=args.deadline_max,
        size_min_mb=args.size_min_mb,
        size_max_mb=size_max,
        edge_cost_min=args.edge_cost_min,
        edge_cost_max=args.edge_cost_max,
        modulation_profile=args.modulation_profile,
        device=args.device,
        output_json=str(out_dir / f"{sid}.json"),
        output_md=str(out_dir / f"{sid}.md"),
    )


def _metric(report: Dict[str, Any], method: str, key: str) -> float:
    return float(report["methods"][method]["aggregate"][key])


def _summarize_one(
    sid: str,
    slots: int,
    arrival: float,
    size_max: float,
    holding_max: float,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    rank = _metric(report, "counterfactual_rank_only", "blocking_rate")
    deep = _metric(report, "deep_rmsa", "blocking_rate")
    ppo = _metric(report, "ppo_r", "blocking_rate") if "ppo_r" in report["methods"] else float("nan")
    row = {
        "scenario_id": sid,
        "num_slots": slots,
        "arrival_interval": arrival,
        "size_max_mb": size_max,
        "holding_max": holding_max,
        "rank_blocking": rank,
        "deep_blocking": deep,
        "ppo_blocking": ppo,
        "rank_minus_deep_pp": (rank - deep) * 100.0,
        "deep_minus_rank_pp": (deep - rank) * 100.0,
        "rank_minus_ppo_pp": (rank - ppo) * 100.0 if np.isfinite(ppo) else None,
        "rank_raw_empty": _metric(report, "counterfactual_rank_only", "raw_mask_empty_rate"),
        "deep_raw_empty": _metric(report, "deep_rmsa", "raw_mask_empty_rate"),
        "rank_nsb": _metric(report, "counterfactual_rank_only", "no_suitable_block_rate"),
        "deep_nsb": _metric(report, "deep_rmsa", "no_suitable_block_rate"),
        "rank_overload": _metric(report, "counterfactual_rank_only", "server_overload_rate"),
        "deep_overload": _metric(report, "deep_rmsa", "server_overload_rate"),
        "rank_delay": _metric(report, "counterfactual_rank_only", "mean_delay_ms"),
        "deep_delay": _metric(report, "deep_rmsa", "mean_delay_ms"),
    }
    return row


def _write_summary(path_json: Path, path_md: Path, config: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["deep_minus_rank_pp"], reverse=True)
    payload = {"config": config, "rows": rows_sorted}
    path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v1.2 Scenario Grid Screening",
        "",
        "Positive `Deep-v1.2` means v1.2 has lower blocking than DeepRMSA.",
        "",
        "| Scenario | Slots | Arrival | SizeMax | HoldMax | v1.2 Blk | DeepRMSA Blk | Deep-v1.2 | PPO Blk | RawEmpty v1.2/Deep | NSB v1.2/Deep | Delay v1.2/Deep |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted:
        ppo = row["ppo_blocking"]
        ppo_text = f"{ppo:.2%}" if np.isfinite(ppo) else "-"
        lines.append(
            f"| {row['scenario_id']} | {row['num_slots']} | {row['arrival_interval']:.2f} | "
            f"{row['size_max_mb']:.1f} | {row['holding_max']:.1f} | "
            f"{row['rank_blocking']:.2%} | {row['deep_blocking']:.2%} | "
            f"{row['deep_minus_rank_pp']:+.2f}pp | {ppo_text} | "
            f"{row['rank_raw_empty']:.2%}/{row['deep_raw_empty']:.2%} | "
            f"{row['rank_nsb']:.2%}/{row['deep_nsb']:.2%} | "
            f"{row['rank_delay']:.2f}/{row['deep_delay']:.2f} |"
        )
    path_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--supervised_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_formal/c_post_decision_joint.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="ppo_r,counterfactual_rank_only,deep_rmsa")
    parser.add_argument("--lambda_values", default="0.5")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument(
        "--num_slots_list",
        default="24",
        help=(
            "DeepRMSA checkpoints are slot-count specific. The default keeps "
            "the known-compatible S24 setting. Other values are skipped if the "
            "checkpoint metadata rejects them."
        ),
    )
    parser.add_argument("--arrival_intervals", default="0.12,0.15,0.18,0.20")
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--holding_max_list", default=None)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--size_max_mb_list", default=None)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v12_scenario_grid")
    parser.add_argument("--summary_json", default="sa_hmarl/experiments/v12_scenario_grid_summary.json")
    parser.add_argument("--summary_md", default="sa_hmarl/experiments/v12_scenario_grid_summary.md")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    slots_values = _parse_csv_ints(args.num_slots_list)
    arrival_values = _parse_csv_floats(args.arrival_intervals)
    size_max_values = _parse_csv_floats(args.size_max_mb_list) if args.size_max_mb_list else [args.size_max_mb]
    holding_max_values = _parse_csv_floats(args.holding_max_list) if args.holding_max_list else [args.holding_max]

    for slots in slots_values:
        for arrival in arrival_values:
            for size_max in size_max_values:
                for holding_max in holding_max_values:
                    sid = _scenario_id(args.topology, slots, arrival, size_max, holding_max)
                    print(f"\n[scenario] {sid}", flush=True)
                    eval_args = _make_eval_args(args, slots, arrival, size_max, holding_max, out_dir)
                    try:
                        report = evaluate(eval_args)
                    except ValueError as exc:
                        print(f"[skip] {sid}: {exc}", flush=True)
                        continue
                    Path(eval_args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
                    # Reuse the evaluator's markdown writer would require a private import;
                    # the per-scenario JSON plus the grid summary are the canonical outputs.
                    row = _summarize_one(sid, slots, arrival, size_max, holding_max, report)
                    rows.append(row)
                    print(
                        f"[done] {sid}: v1.2={row['rank_blocking']:.2%}, "
                        f"deep={row['deep_blocking']:.2%}, Deep-v1.2={row['deep_minus_rank_pp']:+.2f}pp",
                        flush=True,
                    )
                    _write_summary(
                        Path(args.summary_json),
                        Path(args.summary_md),
                        vars(args),
                        rows,
                    )

    _write_summary(Path(args.summary_json), Path(args.summary_md), vars(args), rows)
    print(f"Saved {args.summary_json}")
    print(f"Saved {args.summary_md}")


if __name__ == "__main__":
    main()
