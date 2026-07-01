"""Plot cumulative blocking curves for v1.2 ranker vs DeepRMSA.

This follows the common "number of processed requests" convention used in
RMSA learning-curve figures: x is cumulative requests, and y is cumulative
blocking probability up to that point.  It is not a load-sensitivity curve.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

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
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    _load_deep_rmsa,
    _select_rank_only_r_action,
    load_ranking_checkpoint,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _parse_seeds(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _generate_episode_traces(args: argparse.Namespace, env_proto) -> Dict[int, List[List[Any]]]:
    traces: Dict[int, List[List[Any]]] = {}
    for seed in _parse_seeds(args.seeds):
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(
                generate_requests(
                    env_proto,
                    rng,
                    src,
                    args.requests_per_episode,
                    args.arrival_interval,
                    args.holding_min,
                    args.holding_max,
                    args.deadline_min,
                    args.deadline_max,
                    args.size_min_mb,
                    args.size_max_mb,
                    args.edge_cost_min,
                    args.edge_cost_max,
                    args.num_splits,
                    args.split_profile,
                )
            )
        traces[seed] = episodes
    return traces


def _select_r_idx(
    method: str,
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    deep_rmsa,
    rank_model,
    rank_mean,
    rank_std,
    split_id: int,
    server_id: int,
    args: argparse.Namespace,
) -> int | None:
    if method == "v1.2":
        return _select_rank_only_r_action(
            env,
            req,
            obs_c,
            obs_r,
            agent_r,
            rank_model,
            rank_mean,
            rank_std,
            split_id,
            server_id,
            args.device,
        )
    if method == "DeepRMSA":
        return deep_rmsa.select_action(obs_r)
    raise ValueError(f"Unknown method: {method}")


def _run_method(
    method: str,
    args: argparse.Namespace,
    traces: Dict[int, List[List[Any]]],
    agent_c,
    agent_r,
    deep_rmsa,
    rank_model,
    rank_mean,
    rank_std,
) -> Dict[str, Any]:
    blocked_flags: List[int] = []
    reasons: Dict[str, int] = {}

    for seed, episodes in traces.items():
        for requests in episodes:
            env = make_env(
                args.topology,
                args.num_slots,
                args.num_servers,
                seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            for req in requests:
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                c_action, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
                split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                r_idx = _select_r_idx(
                    method,
                    env,
                    req,
                    obs_c,
                    obs_r,
                    agent_r,
                    deep_rmsa,
                    rank_model,
                    rank_mean,
                    rank_std,
                    split_id,
                    server_id,
                    args,
                )
                if r_idx is None:
                    r_action = (0, 0, 0)
                else:
                    r_action = decode_agent_r_action(
                        int(r_idx), len(obs_r["mod_names"]), env.max_blocks
                    )
                _, _, _, info = env.step((split_id, server_id), r_action)
                blocked = int(not bool(info.get("success", False)))
                blocked_flags.append(blocked)
                if blocked:
                    reason = str(info.get("reason", "unknown"))
                    reasons[reason] = reasons.get(reason, 0) + 1

    flags = np.asarray(blocked_flags, dtype=np.float64)
    cumulative = np.cumsum(flags) / np.arange(1, len(flags) + 1)
    return {
        "method": method,
        "total_requests": int(len(flags)),
        "blocked": int(flags.sum()),
        "final_blocking_rate": float(cumulative[-1]) if len(cumulative) else 0.0,
        "reasons": reasons,
        "cumulative": cumulative.tolist(),
    }


def _sample_curve(values: Iterable[float], every: int) -> List[float]:
    arr = list(values)
    if not arr:
        return []
    idx = list(range(every - 1, len(arr), every))
    if not idx or idx[-1] != len(arr) - 1:
        idx.append(len(arr) - 1)
    return [float(arr[i]) for i in idx]


def _write_csv(path: Path, results: Dict[str, Dict[str, Any]], every: int) -> None:
    sampled = {name: _sample_curve(row["cumulative"], every) for name, row in results.items()}
    max_len = max(len(values) for values in sampled.values())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["requests"] + list(sampled.keys()))
        for i in range(max_len):
            x = min((i + 1) * every, next(iter(results.values()))["total_requests"])
            writer.writerow([x] + [sampled[name][i] if i < len(sampled[name]) else "" for name in sampled])


def _plot(path: Path, results: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.linewidth": 1.2,
    })
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    styles = {
        "v1.2": {"color": "#d95f02", "marker": "o", "label": "v1.2 Ranker"},
        "DeepRMSA": {"color": "#1b9e77", "marker": "s", "label": "DeepRMSA"},
    }
    for name, row in results.items():
        y = np.asarray(_sample_curve(row["cumulative"], args.sample_every), dtype=float)
        x = np.minimum(
            np.arange(1, len(y) + 1) * args.sample_every,
            row["total_requests"],
        )
        style = styles.get(name, {})
        ax.plot(
            x / 1000.0,
            y * 100.0,
            linewidth=2.2,
            markersize=4.5,
            markevery=max(1, len(x) // 12),
            **style,
        )
    ax.set_xlabel("Number of DNN inference requests ($\\times 10^3$)")
    ax.set_ylabel("Blocking probability (%)")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    env_proto = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    traces = _generate_episode_traces(args, env_proto)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

    results = {}
    for method in ["v1.2", "DeepRMSA"]:
        print(f"[run] {method}", flush=True)
        results[method] = _run_method(
            method,
            args,
            traces,
            agent_c,
            agent_r,
            deep_rmsa,
            rank_model,
            rank_mean,
            rank_std,
        )
        print(
            f"[done] {method}: {results[method]['final_blocking_rate']:.2%}",
            flush=True,
        )
    return {"config": vars(args), "methods": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.09)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=14.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--title", default="v1.2 vs DeepRMSA")
    parser.add_argument("--output_png", default="sa_hmarl/experiments/v12_vs_deeprmsa_cumulative_blocking.png")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/v12_vs_deeprmsa_cumulative_blocking.json")
    parser.add_argument("--output_csv", default="sa_hmarl/experiments/v12_vs_deeprmsa_cumulative_blocking.csv")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    report = run(args)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_png = Path(args.output_png)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(out_csv, report["methods"], args.sample_every)
    _plot(out_png, report["methods"], args)
    print(f"Saved {out_json}")
    print(f"Saved {out_csv}")
    print(f"Saved {out_png}")
