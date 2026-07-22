"""Optical-regime calibration for corrected SA-HMARL v1.35.

Sweeps traffic parameters and measures, for each scenario:
  - baseline PPO-R blocking rate and reason breakdown
  - how often R-side candidates differ in future optical blocking (oracle headroom)
  - the magnitude of that headroom

A scenario is flagged as "optical-dominated" when:
  - optical blocking accounts for a sizeable share of total blocking, AND
  - >= 30% of decision groups have nonzero future-optical range among candidates,
    with positive mean oracle headroom.

Only when such a scenario is found should expensive Label-B generation be enabled.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.v135_corrected_dataset_generator import (
    _current_optical_blocked,
    _execute_fixed_r,
    _future_optical_blocked_probe,
    _rollout_future_probe,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")

# Fixed COST239 v1.35 protocol.
NUM_SLOTS = 320
NUM_SERVERS = 4
K_PATHS_C = 5
K_PATHS_R = 50
PATH_SORT = "hops"
BLOCK_SORT = "start_asc"
MAX_CANDIDATES = 30
PROBE_HORIZON = 5


def _scenario_label(params: Dict[str, Any]) -> str:
    parts = [
        f"arr{params['arrival_interval']:.4f}",
        f"hold{params['holding_min']:.0f}_{params['holding_max']:.0f}",
        f"size{params['size_min_mb']:.0f}_{params['size_max_mb']:.0f}",
    ]
    return "_".join(parts)


def _run_scenario(
    params: Dict[str, Any],
    agent_c,
    agent_r,
    seed: int,
    n_requests: int,
    warmup: int,
    max_candidates: int,
    horizon: int,
    probe_stride: int = 1,
) -> Dict[str, Any]:
    env = make_env(
        "xlron_cost239_ptrnet_real", NUM_SLOTS, NUM_SERVERS, seed,
        modulation_profile="default", max_blocks=10,
        block_sort_strategy=BLOCK_SORT, path_sort_strategy=PATH_SORT, k=K_PATHS_R,
    )
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env.net.NUM_NODES))
    requests = generate_requests(
        env, rng, src, n_requests,
        params["arrival_interval"], params["holding_min"], params["holding_max"],
        params["deadline_min"], params["deadline_max"],
        params["size_min_mb"], params["size_max_mb"],
        params["edge_cost_min"], params["edge_cost_max"],
        params["num_splits"], params["split_profile"],
    )
    env.reset(requests)
    num_servers = len(env.mec.servers)

    groups: List[Dict[str, Any]] = []
    baseline = {
        "total": 0,
        "blocked": 0,
        "server_overload": 0,
        "no_suitable_block": 0,
        "allocation_failed": 0,
        "other": 0,
    }

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)

        env.k = K_PATHS_C
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if int(raw_c_mask.sum()) == 0:
            baseline["total"] += 1
            baseline["blocked"] += 1
            baseline["other"] += 1
            env.step((0, 0), (0, 0, 0))
            continue

        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), num_servers)

        env.k = K_PATHS_R
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if int(raw_r_mask.sum()) == 0:
            baseline["total"] += 1
            baseline["blocked"] += 1
            baseline["no_suitable_block"] += 1
            env.step((split_id, server_id), (0, 0, 0))
            continue

        legal = np.flatnonzero(raw_r_mask).tolist()
        candidates = legal[:max_candidates]

        # Baseline PPO-R action (do not step env yet; we need a pre-decision snapshot).
        ppo_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        if (
            request_index >= warmup
            and len(candidates) >= 2
            and (request_index - warmup) % probe_stride == 0
        ):
            # Counterfactual optical probe for this decision.
            snapshot = _snapshot_before_r_decision(env, req.req_id)
            optical_returns: List[float] = []
            for action_idx in candidates:
                branch = copy.deepcopy(snapshot)
                branch_info = _execute_fixed_r(branch, req, split_id, server_id, int(action_idx), obs_r)
                current_opt = _current_optical_blocked(branch_info)
                probe = _rollout_future_probe(
                    branch, requests, request_index + 1, [horizon],
                    agent_c, agent_r, num_servers,
                )
                future_opt = _future_optical_blocked_probe(probe[horizon])
                # Negative because blocking is a cost; higher = better.
                optical_returns.append(-float(current_opt + future_opt))

            vals = np.asarray(optical_returns, dtype=np.float32)
            headroom = float(vals.max() - vals.min())
            groups.append({
                "headroom": headroom,
                "nonzero_range": headroom > 1e-8,
                "positive_oracle_headroom": headroom > 1e-8 and vals.max() > vals.min(),
                "min_return": float(vals.min()),
                "max_return": float(vals.max()),
                "n_candidates": len(candidates),
            })

        # Step the real environment with the PPO-R baseline action.
        r_action = decode_agent_r_action(int(ppo_r_idx), len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)
        baseline["total"] += 1
        if not info.get("success", False):
            baseline["blocked"] += 1
            reason = info.get("reason", "")
            if reason == "server_overload":
                baseline["server_overload"] += 1
            elif reason == "no_suitable_block":
                baseline["no_suitable_block"] += 1
            elif reason == "allocation_failed":
                baseline["allocation_failed"] += 1
            else:
                baseline["other"] += 1

    total = max(baseline["total"], 1)
    blocked = max(baseline["blocked"], 1)
    metrics = {
        "params": params,
        "n_groups": len(groups),
        "blocking_rate": baseline["blocked"] / total,
        "server_overload_rate": baseline["server_overload"] / total,
        "no_suitable_block_rate": baseline["no_suitable_block"] / total,
        "allocation_failed_rate": baseline["allocation_failed"] / total,
        "other_block_rate": baseline["other"] / total,
        "optical_block_rate": (baseline["no_suitable_block"] + baseline["allocation_failed"] + baseline["other"]) / total,
        "overload_share_of_blocking": baseline["server_overload"] / blocked,
        "optical_share_of_blocking": (baseline["no_suitable_block"] + baseline["allocation_failed"] + baseline["other"]) / blocked,
    }

    if groups:
        headrooms = np.array([g["headroom"] for g in groups])
        metrics["nonzero_optical_range_rate"] = float(np.mean([g["nonzero_range"] for g in groups]))
        metrics["mean_oracle_headroom"] = float(headrooms.mean())
        metrics["median_oracle_headroom"] = float(np.median(headrooms))
        metrics["max_oracle_headroom"] = float(headrooms.max())
        nonzero = headrooms[headrooms > 1e-8]
        metrics["nonzero_headroom_mean"] = float(nonzero.mean()) if nonzero.size else 0.0
    else:
        metrics["nonzero_optical_range_rate"] = 0.0
        metrics["mean_oracle_headroom"] = 0.0
        metrics["median_oracle_headroom"] = 0.0
        metrics["max_oracle_headroom"] = 0.0
        metrics["nonzero_headroom_mean"] = 0.0

    gate = (
        metrics["nonzero_optical_range_rate"] >= 0.30
        and metrics["mean_oracle_headroom"] > 0.0
        and metrics["optical_share_of_blocking"] >= 0.10
    )
    metrics["passes_optical_gate"] = gate
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3030, 4040])
    parser.add_argument("--n_requests", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max_candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--horizon", type=int, default=PROBE_HORIZON)
    parser.add_argument("--arrival_intervals", type=float, nargs="+", default=[0.03, 0.0625, 0.125, 0.25],
                        help="Subset of arrival intervals to sweep (default: full grid).")
    parser.add_argument("--probe_stride", type=int, default=1,
                        help="Probe every N-th post-warmup decision (default: 1).")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c("sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt", "cpu")
    agent_r = _load_ppo_r("sa_hmarl/checkpoints/agent_r_mixed.pt", mod_reg, "cpu")

    # Grid: vary load and holding/size to shift balance between overload and optical.
    scenarios: List[Dict[str, Any]] = []
    for arrival_interval in args.arrival_intervals:
        for holding_min, holding_max in ((20.0, 30.0), (40.0, 60.0)):
            for size_min_mb, size_max_mb in ((5.0, 30.0), (20.0, 80.0)):
                scenarios.append({
                    "arrival_interval": arrival_interval,
                    "holding_min": holding_min,
                    "holding_max": holding_max,
                    "deadline_min": 30.0,
                    "deadline_max": 100.0,
                    "size_min_mb": size_min_mb,
                    "size_max_mb": size_max_mb,
                    "edge_cost_min": 0.1,
                    "edge_cost_max": 2.2,
                    "num_splits": 3,
                    "split_profile": "default3",
                })

    results: List[Dict[str, Any]] = []
    for params in scenarios:
        label = _scenario_label(params)
        print(f"[calibration] scenario {label}", flush=True)
        seed_metrics: List[Dict[str, Any]] = []
        for seed in args.seeds:
            m = _run_scenario(
                params, agent_c, agent_r, seed, args.n_requests,
                args.warmup, args.max_candidates, args.horizon,
                args.probe_stride,
            )
            seed_metrics.append(m)
        # Aggregate across seeds.
        numeric_keys = [
            "blocking_rate", "server_overload_rate", "no_suitable_block_rate",
            "allocation_failed_rate", "other_block_rate", "optical_block_rate",
            "overload_share_of_blocking", "optical_share_of_blocking",
            "nonzero_optical_range_rate", "mean_oracle_headroom",
            "median_oracle_headroom", "max_oracle_headroom", "nonzero_headroom_mean",
        ]
        agg = {}
        for k in numeric_keys:
            vals = [m[k] for m in seed_metrics]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        agg["n_groups_mean"] = float(np.mean([m["n_groups"] for m in seed_metrics]))
        agg["passes_optical_gate"] = all(m["passes_optical_gate"] for m in seed_metrics)
        results.append({
            "label": label,
            "params": params,
            "per_seed": seed_metrics,
            "aggregate": agg,
        })
        print(
            f"[calibration] {label} blocking={agg['blocking_rate']['mean']:.2%} "
            f"optical_share={agg['optical_share_of_blocking']['mean']:.2%} "
            f"nonzero_range={agg['nonzero_optical_range_rate']['mean']:.2%} "
            f"headroom={agg['mean_oracle_headroom']['mean']:.3f} "
            f"gate={agg['passes_optical_gate']}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "OPTICAL_REGIME_CALIBRATION.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Optical Regime Calibration",
        "",
        "Scenarios are evaluated under the COST239 v1.35 protocol (320 slots, 4 servers, C-K=5, R-K=50).",
        f"Probe horizon H={args.horizon}, max candidates={args.max_candidates}, seeds={args.seeds}.",
        "",
        "| Scenario | Blocking | Overload share | Optical share | Nonzero optical range | Mean headroom | Passes gate |",
        "|---|---:|---:|---:|---:|---:|:---:|"]
    for r in results:
        agg = r["aggregate"]
        lines.append(
            f"| {r['label']} | "
            f"{agg['blocking_rate']['mean']:.2%} ± {agg['blocking_rate']['std']:.2%} | "
            f"{agg['overload_share_of_blocking']['mean']:.2%} | "
            f"{agg['optical_share_of_blocking']['mean']:.2%} | "
            f"{agg['nonzero_optical_range_rate']['mean']:.2%} ± {agg['nonzero_optical_range_rate']['std']:.2%} | "
            f"{agg['mean_oracle_headroom']['mean']:.3f} ± {agg['mean_oracle_headroom']['std']:.3f} | "
            f"{'Yes' if agg['passes_optical_gate'] else 'No'} |"
        )
    lines.append("")
    passing = [r for r in results if r["aggregate"]["passes_optical_gate"]]
    if passing:
        lines.append("## Passing scenarios (enable Label B here)")
        for r in passing:
            lines.append(f"- `{r['label']}`")
    else:
        lines.append("No scenario passed the optical gate. Label-B generation should remain disabled for this protocol.")
    lines.append("")
    (output_dir / "OPTICAL_REGIME_CALIBRATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("Calibration report:", output_dir / "OPTICAL_REGIME_CALIBRATION.md")


if __name__ == "__main__":
    main()
