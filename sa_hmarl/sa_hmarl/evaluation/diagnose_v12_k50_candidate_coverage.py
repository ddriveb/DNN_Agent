"""Phase-1 probe: is v1.2's gap on COST239 mainly coverage or scoring?

Runs four R-backend configurations on identical request traces:

1. old v1.2      (k=5, km, mixed, checkpoint default all_legal)
2. k50_all_legal (k=50, hops, start_asc, all_legal)
3. k50_ensure_ksp(k=50, hops, start_asc, legalctx48 + ensure KSP injection)
4. ksp_ff_k50_hops (tuned heuristic baseline)

Outputs aggregate blocking metrics plus selected-action feature summaries so we
can judge whether simply enlarging the candidate set / guaranteeing KSP coverage
closes the gap to KSP-FF.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
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
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import (
    R_FEATURE_ORDER,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


RANKER_CKPT = "sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt"


def _r_backend_env_config(r_mode: str) -> tuple[int, str, str]:
    if r_mode in ("k50_all_legal", "k50_ensure_ksp", "ksp_ff_k50_hops"):
        return 50, "hops", "start_asc"
    return 5, "km", "mixed"


def _selected_action_features(obs_r: Dict[str, Any], r_features: np.ndarray, action_idx: int):
    if r_features is None or action_idx < 0 or action_idx >= len(r_features):
        return None
    num_mods = len(obs_r["mod_names"])
    max_blocks = len(obs_r["agent_r_mask"]) // (len(obs_r["candidate_paths"]) * num_mods)
    path_idx, mod_idx, block_idx = decode_agent_r_action(action_idx, num_mods, max_blocks)
    feat = r_features[action_idx]
    return {
        "path_idx": int(path_idx),
        "mod_idx": int(mod_idx),
        "block_idx": int(block_idx),
        "path_length_km": float(feat[R_FEATURE_ORDER["path_length_km"]]),
        "hop_count": float(feat[R_FEATURE_ORDER["hop_count"]]),
        "spectral_efficiency": float(feat[R_FEATURE_ORDER["spectral_efficiency"]]),
        "required_fs": float(feat[R_FEATURE_ORDER["required_fs"]]),
        "block_size": float(feat[R_FEATURE_ORDER["block_size"]]),
        "block_waste": float(feat[R_FEATURE_ORDER["block_waste"]]),
    }


def _select_r_action(
    r_mode: str,
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    rank_policy: CounterfactualRRankerPolicy,
    split_id: int,
    server_id: int,
):
    r_features, _ = agent_r.build_action_features(obs_r)
    if r_mode == "ksp_ff_k50_hops":
        r_idx = ksp_ff_highest_mod_action(obs_r)
        r_action = (0, 0, 0) if r_idx is None else decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
    else:
        r_idx = rank_policy.select_action(env, req, obs_c, obs_r, agent_r, split_id, server_id)
        r_action = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks) if r_idx is not None else (0, 0, 0)
    return r_idx, r_action, r_features


def _run_episode(
    env,
    requests,
    agent_c,
    agent_r,
    rank_policy,
    r_mode: str,
    args,
    metrics: PerMethodMetrics,
    records: List[Dict[str, Any]],
):
    rng = np.random.RandomState(0)  # placeholder, not used for policy
    server_selected_count = np.zeros(args.num_servers, dtype=int)
    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)

        # C-side always uses main config.
        env.k, env.path_sort_strategy, env.block_sort_strategy = (
            args.k_paths, args.path_sort_strategy, args.block_sort_strategy,
        )
        obs_c = build_agent_c_observation(env, req)
        c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
        server_selected_count[server_id] += 1

        # R-side backend config.
        env.k, env.path_sort_strategy, env.block_sort_strategy = _r_backend_env_config(r_mode)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_idx, r_action, r_features = _select_r_action(
            r_mode, env, req, obs_c, obs_r, agent_r, rank_policy, split_id, server_id
        )
        decision_ms = (time.perf_counter() - started) * 1000.0

        _, _, _, info = env.step((split_id, server_id), r_action)
        same_as_ppo = True
        if r_idx is not None:
            ppo_idx = agent_r.select_action(obs_r, deterministic=True)
            same_as_ppo = int(r_idx) == int(ppo_idx if ppo_idx is not None else 0)

        _record_outcome(
            metrics, info, int(raw_c_mask.sum()) == 0,
            obs_c, split_id, server_id, decision_ms, same_as_ppo,
        )
        metrics.active_connections[-1] = len(env.active_connections)

        records.append({
            "r_mode": r_mode,
            "req_id": int(req.req_id),
            "r_idx": int(r_idx) if r_idx is not None else -1,
            "success": bool(info.get("success", False)),
            "reason": info.get("reason", ""),
            "action_features": _selected_action_features(obs_r, r_features, int(r_idx)) if r_idx is not None else None,
        })


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        path_sort_strategy=args.path_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, rank_ckpt = load_ranking_checkpoint(
        args.ranking_checkpoint, args.device
    )
    rank_policy = CounterfactualRRankerPolicy(
        rank_model,
        rank_mean,
        rank_std,
        device=args.device,
        feature_names=rank_ckpt.get("feature_names"),
        candidate_mode=args.ranker_candidate_mode if args.ranker_candidate_mode is not None else rank_ckpt.get("candidate_mode", "all_legal"),
        max_candidates=args.ranker_max_candidates if args.ranker_max_candidates is not None else rank_ckpt.get("max_candidates", 48),
        ppo_top_k=args.ranker_ppo_top_k if args.ranker_ppo_top_k is not None else rank_ckpt.get("ppo_top_k", 8),
        num_random_candidates=args.ranker_num_random_candidates if args.ranker_num_random_candidates is not None else rank_ckpt.get("num_random_candidates", 5),
        min_candidates=args.ranker_min_candidates if args.ranker_min_candidates is not None else rank_ckpt.get("min_candidates", 15),
        candidate_seed=rank_ckpt.get("candidate_seed", 12345),
        ensure_ksp_action=args.ranker_ensure_ksp,
    )

    episodes_by_seed = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max, args.size_min_mb,
                args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile,
            ))
        episodes_by_seed[seed] = episodes

    probes = {
        "old_v12": {"label": "v1.2 (k=5 km mixed, all_legal)"},
        "k50_all_legal": {"label": "v1.2 k=50 hops all_legal"},
        "k50_ensure_ksp": {"label": "v1.2 k=50 hops legalctx48 + ensure KSP"},
        "ksp_ff_k50_hops": {"label": "KSP-FF K=50 hops"},
    }

    all_records: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {"config": vars(args), "probes": {}}
    for r_mode, probe in probes.items():
        per_seed = {}
        for seed in seeds:
            metrics = PerMethodMetrics()
            for requests in episodes_by_seed[seed]:
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                    path_sort_strategy=args.path_sort_strategy,
                    k=args.k_paths,
                )
                env.reset(requests)
                _run_episode(env, requests, agent_c, agent_r, rank_policy, r_mode, args, metrics, all_records)
            per_seed[str(seed)] = _aggregate_metrics(metrics)
            print(f"[{r_mode}] completed seed {seed}", flush=True)
        aggregate = {
            key: float(np.mean([row[key] for row in per_seed.values()]))
            for key in next(iter(per_seed.values()))
            if isinstance(next(iter(per_seed.values()))[key], (int, float))
        }
        report["probes"][r_mode] = {
            "label": probe["label"],
            "per_seed": per_seed,
            "aggregate": aggregate,
        }

    report["action_feature_summary"] = _action_feature_summary(all_records)
    return report


def _action_feature_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    feature_names = ["path_length_km", "hop_count", "spectral_efficiency", "required_fs", "block_size", "block_waste"]
    for r_mode in sorted({r["r_mode"] for r in records}):
        feats = [r["action_features"] for r in records if r["r_mode"] == r_mode and r["action_features"] is not None]
        if not feats:
            continue
        for name in feature_names:
            vals = [f[name] for f in feats]
            summary.append({
                "r_mode": r_mode,
                "feature": name,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
            })
    return summary


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# v1.2 K=50 Candidate Coverage Probe (COST239 S100)", "",
        "## Configuration", "",
    ]
    cfg = report["config"]
    for key in [
        "topology", "num_slots", "seeds", "episodes", "requests_per_episode",
        "k_paths", "path_sort_strategy", "block_sort_strategy",
        "arrival_interval", "holding_min", "holding_max", "size_min_mb", "size_max_mb",
    ]:
        lines.append(f"- `{key}`: `{cfg[key]}`")

    lines += ["", "## Blocking results", "",
        "| Probe | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r_mode, spec in report["probes"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {spec['label']} | {agg['blocking_rate']:.2%} | "
            f"{agg['raw_mask_empty_rate']:.2%} | {agg['no_suitable_block_rate']:.2%} | "
            f"{agg['server_overload_rate']:.2%} | "
            f"{agg.get('deadline_failure_rate', 0.0):.2%} | "
            f"{agg.get('other_failure_rate', 0.0):.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{agg['mean_decision_time_ms']:.3f}/{agg['p95_decision_time_ms']:.3f} ms |"
        )

    base = report["probes"]["old_v12"]["aggregate"]
    ksp = report["probes"]["ksp_ff_k50_hops"]["aggregate"]
    lines += ["", "## Delta vs old v1.2", "",
        "| Probe | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) |",
        "|---|---:|---:|---:|"]
    for r_mode, spec in report["probes"].items():
        if r_mode == "old_v12":
            continue
        agg = spec["aggregate"]
        lines.append(
            f"| {spec['label']} | "
            f"{(agg['blocking_rate'] - base['blocking_rate']) * 100:+.2f} | "
            f"{(agg['no_suitable_block_rate'] - base['no_suitable_block_rate']) * 100:+.2f} | "
            f"{(agg['server_overload_rate'] - base['server_overload_rate']) * 100:+.2f} |"
        )

    lines += ["", "## Selected action feature summary", "",
        "| Probe | Feature | Mean | Median | Std |",
        "|---|---|---:|---:|---:|"]
    for row in report["action_feature_summary"]:
        label = report["probes"][row["r_mode"]]["label"]
        lines.append(
            f"| {label} | {row['feature']} | {row['mean']:.4f} | {row['median']:.4f} | {row['std']:.4f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default=RANKER_CKPT)
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=100)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--path_sort_strategy", default="km", choices=["km", "hops"])
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.15)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/v12_k50_candidate_coverage_probe.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/v12_k50_candidate_coverage_probe.md")
    parser.add_argument("--ranker_candidate_mode", default=None)
    parser.add_argument("--ranker_max_candidates", type=int, default=None)
    parser.add_argument("--ranker_ppo_top_k", type=int, default=None)
    parser.add_argument("--ranker_num_random_candidates", type=int, default=None)
    parser.add_argument("--ranker_min_candidates", type=int, default=None)
    parser.add_argument("--ranker_ensure_ksp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(out_md, report)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
