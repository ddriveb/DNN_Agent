"""Evaluate a matrix of Agent-C checkpoints and Agent-R backends.

This is a thin orchestrator around ``eval_r_counterfactual_ranking_closed_loop``.
It keeps the request traces and environment settings fixed while swapping the
C-side checkpoint and R-side method set, then writes one compact comparison
table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    build_parser as build_base_parser,
    evaluate as evaluate_r_backends,
)


DEFAULT_C_CHECKPOINTS = [
    (
        "C-default",
        "sa_hmarl/checkpoints/agent_c_default_s123_s20_r80_best.pt",
    ),
    (
        "C-r-feasibility",
        "sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt",
    ),
    (
        "C-delay-aware",
        "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt",
    ),
]


def _parse_c_checkpoints(value: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                "--c_checkpoints entries must use label=path, separated by commas"
            )
        label, path = part.split("=", 1)
        items.append((label.strip(), path.strip()))
    return items


def _base_namespace(args: argparse.Namespace) -> argparse.Namespace:
    parser = build_base_parser()
    base = parser.parse_args([])
    for key, value in vars(args).items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base


def run(args: argparse.Namespace) -> Dict[str, Any]:
    c_items = _parse_c_checkpoints(args.c_checkpoints)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    for c_label, c_path in c_items:
        eval_args = _base_namespace(args)
        eval_args.agent_c_checkpoint = c_path
        eval_args.output_json = str(out_dir / f"{args.prefix}_{c_label}.json")
        eval_args.output_md = str(out_dir / f"{args.prefix}_{c_label}.md")
        print(f"[combo] C={c_label} checkpoint={c_path}", flush=True)
        report = evaluate_r_backends(eval_args)
        details[c_label] = report
        Path(eval_args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

        for r_method, spec in report["methods"].items():
            agg = spec["aggregate"]
            rows.append({
                "agent_c": c_label,
                "agent_r": r_method,
                "blocking_rate": float(agg["blocking_rate"]),
                "raw_mask_empty_rate": float(agg["raw_mask_empty_rate"]),
                "no_suitable_block_rate": float(agg["no_suitable_block_rate"]),
                "server_overload_rate": float(agg["server_overload_rate"]),
                "mean_delay_ms": float(agg["mean_delay_ms"]),
                "p95_delay_ms": float(agg["p95_delay_ms"]),
                "mean_decision_time_ms": float(agg["mean_decision_time_ms"]),
                "p95_decision_time_ms": float(agg["p95_decision_time_ms"]),
            })

    return {
        "config": vars(args),
        "rows": rows,
        "details": details,
    }


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    rows = sorted(report["rows"], key=lambda r: (r["agent_c"], r["blocking_rate"]))
    lines = [
        "# Agent-C / Agent-R Combination Matrix",
        "",
        "| Agent-C | Agent-R | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['agent_c']} | {row['agent_r']} | "
            f"{row['blocking_rate']:.2%} | "
            f"{row['raw_mask_empty_rate']:.2%} | "
            f"{row['no_suitable_block_rate']:.2%} | "
            f"{row['server_overload_rate']:.2%} | "
            f"{row['mean_delay_ms']:.3f}/{row['p95_delay_ms']:.3f} ms | "
            f"{row['mean_decision_time_ms']:.3f}/{row['p95_decision_time_ms']:.3f} ms |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    base = build_base_parser()
    parser = argparse.ArgumentParser(description=__doc__)
    for action in base._actions:
        if not action.option_strings or action.dest in {"help", "output_json", "output_md"}:
            continue
        kwargs = {
            "default": action.default,
            "help": action.help,
        }
        if action.type is not None:
            kwargs["type"] = action.type
        if action.choices is not None:
            kwargs["choices"] = action.choices
        if isinstance(action, argparse._StoreTrueAction):
            parser.add_argument(*action.option_strings, action="store_true", default=action.default, help=action.help)
        else:
            parser.add_argument(*action.option_strings, **kwargs)
    parser.add_argument(
        "--c_checkpoints",
        default=",".join(f"{label}={path}" for label, path in DEFAULT_C_CHECKPOINTS),
        help="Comma-separated label=checkpoint entries for Agent-C.",
    )
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/c_r_combo_matrix")
    parser.add_argument("--prefix", default="s100_low_blocking")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/c_r_combo_matrix_s100_low_blocking.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/c_r_combo_matrix_s100_low_blocking.md")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    result = run(parsed)
    out_json = Path(parsed.output_json)
    out_md = Path(parsed.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_md(out_md, result)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
