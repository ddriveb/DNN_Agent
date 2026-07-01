"""Screen split complexity and pressure scenarios where v1.2 separates from DeepRMSA.

Stage 1 of the v1.2 scenario-selection pipeline.  We sweep:
  - split_profile / num_splits
  - arrival_interval
  - size_max_mb
  - holding_max
while keeping topology, num_slots, k_paths, max_blocks compatible with the
existing DeepRMSA checkpoint.

Each scenario is evaluated with ppo_r, counterfactual_rank_only (v1.2),
deep_rmsa, and ksp_bf.  Per-scenario JSONs plus a sorted summary are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import evaluate
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _parse_csv_floats(text: str) -> List[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_csv_strings(text: str) -> List[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def _split_num_splits(profile: str) -> int:
    """Return default num_splits for a named split profile."""
    if profile.startswith("default"):
        return 3
    if profile.startswith("complex5"):
        return 5
    raise ValueError(f"Cannot infer num_splits for profile {profile}")


def _scenario_id(
    topology: str,
    slots: int,
    profile: str,
    arrival: float,
    size_max: float,
    holding_max: float,
) -> str:
    return (
        f"{topology}_s{slots}_{profile}"
        f"_ai{str(arrival).replace('.', 'p')}"
        f"_sz{str(size_max).replace('.', 'p')}"
        f"_h{str(holding_max).replace('.', 'p')}"
    )


def _check_agent_c_compatibility(
    ckpt_path: str,
    topology: str,
    num_slots: int,
    num_servers: int,
    num_splits: int,
    split_profile: str,
    k_paths: int,
    max_blocks: int,
    device: str,
) -> Tuple[bool, str]:
    """Try to run Agent-C through one observation for the given split config.

    Returns (ok, reason).  If ok is False the scenario must be skipped.
    """""
    try:
        env = make_env(
            topology, num_slots, num_servers, 42,
            modulation_profile="default",
            max_blocks=max_blocks,
            block_sort_strategy="mixed",
            k=k_paths,
        )
        rng = np.random.RandomState(42)
        src = int(rng.randint(0, env.net.NUM_NODES))
        requests = generate_requests(
            env, rng, src, 5, 0.15, 4.0, 10.0, 30.0, 100.0,
            5.0, 30.0, 0.5, 15.0, num_splits, split_profile,
        )
        env.reset(requests)
        agent_c = _load_ppo_c(ckpt_path, device)
        for req in requests:
            env.advance_time(req.arrival_time)
            obs_c = build_agent_c_observation(env, req)
            _ = agent_c.select_action(obs_c, deterministic=True)
            break
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _make_eval_args(
    args: argparse.Namespace,
    profile: str,
    num_splits: int,
    arrival: float,
    size_max: float,
    holding_max: float,
    out_dir: Path,
) -> SimpleNamespace:
    sid = _scenario_id(args.topology, args.num_slots, profile, arrival, size_max, holding_max)
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
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        k_paths=args.k_paths,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        split_profile=profile,
        num_splits=num_splits,
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
    profile: str,
    num_splits: int,
    slots: int,
    arrival: float,
    size_max: float,
    holding_max: float,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    v12 = _metric(report, "counterfactual_rank_only", "blocking_rate")
    deep = _metric(report, "deep_rmsa", "blocking_rate")
    ppo = _metric(report, "ppo_r", "blocking_rate")
    ksp = _metric(report, "ksp_bf", "blocking_rate")
    row = {
        "scenario_id": sid,
        "split_profile": profile,
        "num_splits": num_splits,
        "num_slots": slots,
        "arrival_interval": arrival,
        "size_max_mb": size_max,
        "holding_max": holding_max,
        "v12_blocking": v12,
        "deep_rmsa_blocking": deep,
        "ppo_r_blocking": ppo,
        "ksp_bf_blocking": ksp,
        "deep_minus_v12_pp": (deep - v12) * 100.0,
        "v12_raw_empty": _metric(report, "counterfactual_rank_only", "raw_mask_empty_rate"),
        "deep_raw_empty": _metric(report, "deep_rmsa", "raw_mask_empty_rate"),
        "v12_nsb": _metric(report, "counterfactual_rank_only", "no_suitable_block_rate"),
        "deep_nsb": _metric(report, "deep_rmsa", "no_suitable_block_rate"),
        "v12_overload": _metric(report, "counterfactual_rank_only", "server_overload_rate"),
        "deep_overload": _metric(report, "deep_rmsa", "server_overload_rate"),
        "v12_delay": _metric(report, "counterfactual_rank_only", "mean_delay_ms"),
        "deep_delay": _metric(report, "deep_rmsa", "mean_delay_ms"),
    }
    return row


def _classify(row: Dict[str, Any], max_overload_delta_pp: float = 0.5) -> Tuple[str, str]:
    deep = row["deep_rmsa_blocking"]
    gap = row["deep_minus_v12_pp"]
    delay_delta = row["v12_delay"] - row["deep_delay"]
    overload_delta = (row["v12_overload"] - row["deep_overload"]) * 100.0

    if not (0.05 <= deep <= 0.60):
        return "OUT_OF_RANGE", "deep_blocking_out_of_range"
    if delay_delta > 1.0:
        return "FAIL", f"delay_delta_{delay_delta:.2f}ms"
    if overload_delta > max_overload_delta_pp:
        return "FAIL", f"overload_delta_{overload_delta:.2f}pp"
    if gap >= 2.0:
        return "PASS", "ok"
    if gap >= 1.0:
        return "MARGINAL", "ok"
    return "FAIL", f"gap_{gap:+.2f}pp"


def _write_summary(
    path_json: Path,
    path_md: Path,
    config: Dict[str, Any],
    rows: List[Dict[str, Any]],
    max_overload_delta_pp: float = 0.5,
) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["deep_minus_v12_pp"], reverse=True)
    for row in rows_sorted:
        verdict, reason = _classify(row, max_overload_delta_pp)
        row["verdict"] = verdict
        row["verdict_reason"] = reason
    payload = {"config": config, "rows": rows_sorted}
    path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v1.2 Split-Pressure Grid Screening",
        "",
        "Positive `Deep-v1.2` means v1.2 has lower blocking than DeepRMSA.",
        "",
        "| Scenario | Profile | Splits | AI | SizeMax | HoldMax | v1.2 Blk | Deep Blk | Deep-v1.2 | PPO Blk | KSP Blk | v1.2 Raw/Deep | v1.2 NSB/Deep | v1.2 Delay/Deep | Verdict | Reason |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted:
        lines.append(
            f"| {row['scenario_id']} | {row['split_profile']} | {row['num_splits']} | "
            f"{row['arrival_interval']:.2f} | {row['size_max_mb']:.1f} | {row['holding_max']:.1f} | "
            f"{row['v12_blocking']:.2%} | {row['deep_rmsa_blocking']:.2%} | "
            f"{row['deep_minus_v12_pp']:+.2f}pp | {row['ppo_r_blocking']:.2%} | {row['ksp_bf_blocking']:.2%} | "
            f"{row['v12_raw_empty']:.2%}/{row['deep_raw_empty']:.2%} | "
            f"{row['v12_nsb']:.2%}/{row['deep_nsb']:.2%} | "
            f"{row['v12_delay']:.2f}/{row['deep_delay']:.2f} | {row['verdict']} | {row.get('verdict_reason','')} |"
        )
    path_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--supervised_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_formal/c_post_decision_joint.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="ppo_r,counterfactual_rank_only,deep_rmsa,ksp_bf")
    parser.add_argument("--lambda_values", default="0.5")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profiles", default="default3,complex5_v2_lite")
    parser.add_argument("--arrival_intervals", default="0.10,0.12,0.15")
    parser.add_argument("--size_max_mb_list", default="30,40,50")
    parser.add_argument("--holding_max_list", default="10,14")
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v12_split_pressure_grid")
    parser.add_argument("--summary_json", default="sa_hmarl/experiments/v12_split_pressure_grid_summary.json")
    parser.add_argument("--summary_md", default="sa_hmarl/experiments/v12_split_pressure_grid_summary.md")
    parser.add_argument("--skip_compatibility_check", action="store_true")
    parser.add_argument("--max_overload_delta_pp", type=float, default=0.5,
                        help="Maximum allowable v1.2-overload excess over DeepRMSA in percentage points.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    profile_values = _parse_csv_strings(args.split_profiles)
    arrival_values = _parse_csv_floats(args.arrival_intervals)
    size_max_values = _parse_csv_floats(args.size_max_mb_list)
    holding_max_values = _parse_csv_floats(args.holding_max_list)

    # Pre-check Agent-C compatibility for each requested split profile.
    compatible_profiles = []
    skipped_profiles = []
    for profile in profile_values:
        num_splits = _split_num_splits(profile)
        if args.skip_compatibility_check:
            compatible_profiles.append((profile, num_splits))
            continue
        ok, reason = _check_agent_c_compatibility(
            args.agent_c_checkpoint,
            args.topology,
            args.num_slots,
            args.num_servers,
            num_splits,
            profile,
            args.k_paths,
            args.max_blocks,
            args.device,
        )
        if ok:
            compatible_profiles.append((profile, num_splits))
            print(f"[compat] {profile} (num_splits={num_splits}): OK", flush=True)
        else:
            skipped_profiles.append((profile, num_splits, reason))
            print(f"[skip] {profile} (num_splits={num_splits}): Agent-C incompatible: {reason}", flush=True)

    if not compatible_profiles:
        print("[abort] No compatible split profiles. Exiting.", flush=True)
        sys.exit(1)

    for profile, num_splits in compatible_profiles:
        for arrival in arrival_values:
            for size_max in size_max_values:
                for holding_max in holding_max_values:
                    sid = _scenario_id(args.topology, args.num_slots, profile, arrival, size_max, holding_max)
                    print(f"\n[scenario] {sid}", flush=True)
                    eval_args = _make_eval_args(args, profile, num_splits, arrival, size_max, holding_max, out_dir)
                    try:
                        report = evaluate(eval_args)
                    except ValueError as exc:
                        print(f"[skip] {sid}: {exc}", flush=True)
                        continue
                    except Exception as exc:
                        print(f"[error] {sid}: {type(exc).__name__}: {exc}", flush=True)
                        traceback.print_exc()
                        continue
                    Path(eval_args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
                    row = _summarize_one(sid, profile, num_splits, args.num_slots, arrival, size_max, holding_max, report)
                    rows.append(row)
                    print(
                        f"[done] {sid}: v1.2={row['v12_blocking']:.2%}, "
                        f"deep={row['deep_rmsa_blocking']:.2%}, Deep-v1.2={row['deep_minus_v12_pp']:+.2f}pp, "
                        f"verdict={_classify(row, args.max_overload_delta_pp)[0]}",
                        flush=True,
                    )
                    _write_summary(
                        Path(args.summary_json),
                        Path(args.summary_md),
                        vars(args),
                        rows,
                        args.max_overload_delta_pp,
                    )

    _write_summary(Path(args.summary_json), Path(args.summary_md), vars(args), rows, args.max_overload_delta_pp)
    print(f"\nSaved {args.summary_json}")
    print(f"Saved {args.summary_md}")
    if skipped_profiles:
        print("\nSkipped split profiles:")
        for profile, num_splits, reason in skipped_profiles:
            print(f"  - {profile} (num_splits={num_splits}): {reason}")


if __name__ == "__main__":
    main()
