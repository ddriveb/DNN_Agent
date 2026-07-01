"""Closed-loop evaluation of counterfactual R-side ranking reranking."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import CounterfactualActionValueRanker
from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
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
    _select_r_action,
    _zscore,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES, _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def load_ranking_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Ranking checkpoint feature schema mismatch")
    model = CounterfactualActionValueRanker(
        ckpt["input_dim"], ckpt["hidden_dims"], ckpt["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return (
        model,
        np.asarray(ckpt["feature_mean"], dtype=np.float32),
        np.asarray(ckpt["feature_std"], dtype=np.float32),
        ckpt,
    )


def load_supervised_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Supervised checkpoint feature schema mismatch")
    model = CounterfactualActionValueRanker(
        ckpt["input_dim"], ckpt["hidden_dims"], ckpt.get("dropout", 0.0)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return (
        model,
        np.asarray(ckpt["feature_mean"], dtype=np.float32),
        np.asarray(ckpt["feature_std"], dtype=np.float32),
    )


def _get_r_logits(agent_r, obs_r: Dict[str, Any]) -> np.ndarray:
    features, mask = agent_r.build_action_features(obs_r)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    return logits


def _select_rank_only_r_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std,
    split_id: int, server_id: int, device: str,
) -> int:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        scores = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy()
    return int(legal[int(np.argmax(scores))])


def _compute_D_spec(r_features: np.ndarray, legal: List[int], mode: str) -> np.ndarray:
    """Compute per-candidate spectral damage proxy among legal actions."""
    feats = r_features[legal]
    path_km = feats[:, 0]
    required_fs = feats[:, 7]
    block_waste = feats[:, 9]

    def _zs(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if len(values) <= 1:
            return np.zeros_like(values)
        std = float(np.std(values))
        if std == 0.0:
            return np.zeros_like(values)
        return (values - np.mean(values)) / std

    if mode == "fs_only":
        return _zs(required_fs)
    if mode == "path_only":
        return _zs(path_km)
    # fs_path
    return _zs(required_fs) + 0.5 * _zs(path_km) + 0.25 * _zs(block_waste)


def _select_lyapunov_rank_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std,
    split_id: int, server_id: int, device: str,
    H_spec: float, H_srv: np.ndarray, args: argparse.Namespace,
) -> Tuple[int, int, float]:
    """Select R action with Lyapunov drift-plus-penalty reranking.

    Returns (lyapunov_action_idx, v12_action_idx, mean_abs_adjustment).
    """
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0, 0, 0.0
    if len(legal) == 1:
        return int(legal[0]), int(legal[0]), 0.0

    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        base_scores = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy().ravel()

    v12_idx = int(legal[int(np.argmax(base_scores))])

    D_spec = _compute_D_spec(r_features, legal, args.lyap_spec_damage_mode)
    adjusted = base_scores - args.lyap_lambda_spec * H_spec * D_spec

    lyap_idx = int(legal[int(np.argmax(adjusted))])
    mean_abs_adjustment = float(np.mean(np.abs(args.lyap_lambda_spec * H_spec * D_spec)))
    return lyap_idx, v12_idx, mean_abs_adjustment


def _select_ppo_rank_rerank_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std,
    split_id: int, server_id: int, lambda_value: float, device: str,
) -> int:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        scores = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy()
    logits = _get_r_logits(agent_r, obs_r)[legal]
    combined = _zscore(logits) + lambda_value * _zscore(scores)
    best = min(
        range(len(legal)),
        key=lambda index: (-float(combined[index]), -float(logits[index]), legal[index]),
    )
    return int(legal[best])


def _run_episode(
    env, requests, agent_c, agent_r,
    rank_model, rank_mean, rank_std,
    rank_policy,
    sup_model, sup_mean, sup_std,
    args, metrics: PerMethodMetrics, mode: str, lambda_value: float = 1.0,
    deep_rmsa=None,
) -> None:
    H_spec = 0.0
    H_srv = np.zeros(args.num_servers, dtype=float)
    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_action, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)

        if mode == "deep_rmsa":
            r_idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
            r_action = (0, 0, 0) if r_idx is None else decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
            r_idx = r_idx if r_idx is not None else -1
        elif mode == "ksp_bf":
            r_idx = ksp_bf_action(obs_r)
            r_action = (0, 0, 0) if r_idx is None else decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
            r_idx = r_idx if r_idx is not None else -1
        elif mode == "ppo_r":
            r_action = _select_r_action(agent_r, obs_r, env.max_blocks)
            r_idx = None
        elif mode == "supervised_r_post":
            r_idx = _select_rank_only_r_action(
                env, req, obs_c, obs_r, agent_r, sup_model, sup_mean, sup_std,
                split_id, server_id, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        elif mode == "counterfactual_rank_only":
            r_idx = rank_policy.select_action(
                env, req, obs_c, obs_r, agent_r, split_id, server_id
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        elif mode == "ppo_rank_rerank":
            r_idx = _select_ppo_rank_rerank_action(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, lambda_value, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        elif mode == "lyapunov_rank_only":
            r_idx, v12_idx, mean_adj = _select_lyapunov_rank_action(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, args.device,
                H_spec, H_srv, args,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        decision_ms = (time.perf_counter() - started) * 1000.0
        _, _, _, info = env.step((split_id, server_id), r_action)
        same_as_ppo = True
        if r_idx is not None:
            ppo_idx = agent_r.select_action(obs_r, deterministic=True)
            same_as_ppo = int(r_idx) == int(ppo_idx if ppo_idx is not None else 0)
        _record_outcome(
            metrics, info, int(np.asarray(raw_c_mask, dtype=bool).sum()) == 0,
            obs_c, split_id, server_id, decision_ms, same_as_ppo,
        )
        metrics.active_connections[-1] = len(env.active_connections)

        if mode == "lyapunov_rank_only":
            metrics.lyap_H_spec.append(float(H_spec))
            metrics.lyap_adjustment_abs.append(float(mean_adj))
            metrics.lyap_changed_vs_v12.append(int(r_idx) != int(v12_idx))

            # Update H_spec based on pre-decision Agent-C observation.
            raw_c_valid = int(np.asarray(obs_c["agent_c_mask"], dtype=bool).sum())
            num_c_actions = max(len(obs_c["agent_c_mask"]), 1)
            if args.lyap_update_mode == "binary_raw_empty":
                pressure_spec = 1.0 if raw_c_valid == 0 else 0.0
            else:
                pressure_spec = 1.0 - raw_c_valid / num_c_actions
            H_spec = max(H_spec + pressure_spec - args.lyap_epsilon_spec, 0.0)
            H_spec = min(H_spec, args.lyap_queue_clip)

            # Update H_srv for the selected server using post-step utilization.
            server = env.mec.servers[server_id]
            util_after = server.utilization
            if args.lyap_srv_damage_mode == "util_after":
                pressure_srv = max(util_after - 0.8, 0.0)
            else:
                margin = 1.0 - util_after
                pressure_srv = max(0.2 - margin, 0.0)
            H_srv[server_id] = max(H_srv[server_id] + pressure_srv - args.lyap_epsilon_srv, 0.0)
            H_srv = np.clip(H_srv, 0.0, args.lyap_queue_clip)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, rank_ckpt = load_ranking_checkpoint(args.ranking_checkpoint, args.device)
    rank_policy = CounterfactualRRankerPolicy(
        rank_model, rank_mean, rank_std, device=args.device
    )

    sup_model = sup_mean = sup_std = None
    if "supervised_r_post" in [item.strip() for item in args.methods.split(",")]:
        sup_model, sup_mean, sup_std = load_supervised_checkpoint(args.supervised_checkpoint, args.device)

    deep_rmsa = None
    if "deep_rmsa" in [item.strip() for item in args.methods.split(",")]:
        deep_rmsa = _load_deep_rmsa(
            args.deep_rmsa_checkpoint,
            env_proto,
            ModulationRegistry.from_profile(args.modulation_profile),
            args.device,
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

    requested = [item.strip() for item in args.methods.split(",") if item.strip()]
    methods = {"ppo_r": {"per_seed": {}}}
    if "supervised_r_post" in requested:
        methods["supervised_r_post"] = {"per_seed": {}}
    if "counterfactual_rank_only" in requested:
        methods["counterfactual_rank_only"] = {"per_seed": {}}
    if "lyapunov_rank_only" in requested:
        methods["lyapunov_rank_only"] = {"per_seed": {}}
    if "ppo_rank_rerank" in requested:
        for value in [float(v) for v in args.lambda_values.split(",") if v.strip()]:
            methods[f"ppo_rank_rerank_{value:g}"] = {"per_seed": {}, "lambda": value}
    if "deep_rmsa" in requested:
        methods["deep_rmsa"] = {"per_seed": {}}
    if "ksp_bf" in requested:
        methods["ksp_bf"] = {"per_seed": {}}

    for method_name, spec in methods.items():
        for seed in seeds:
            metrics = PerMethodMetrics()
            for requests in episodes_by_seed[seed]:
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                env.reset(requests)
                if method_name == "ppo_r":
                    mode = "ppo_r"
                elif method_name == "supervised_r_post":
                    mode = "supervised_r_post"
                elif method_name == "counterfactual_rank_only":
                    mode = "counterfactual_rank_only"
                elif method_name == "lyapunov_rank_only":
                    mode = "lyapunov_rank_only"
                elif method_name == "deep_rmsa":
                    mode = "deep_rmsa"
                elif method_name == "ksp_bf":
                    mode = "ksp_bf"
                else:
                    mode = "ppo_rank_rerank"
                _run_episode(
                    env, requests, agent_c, agent_r,
                    rank_model, rank_mean, rank_std,
                    rank_policy,
                    sup_model, sup_mean, sup_std,
                    args, metrics, mode, spec.get("lambda", 1.0), deep_rmsa,
                )
            spec["per_seed"][str(seed)] = _aggregate_metrics(metrics)
            print(f"[{method_name}] completed seed {seed}", flush=True)
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key in next(iter(spec["per_seed"].values()))
            if isinstance(next(iter(spec["per_seed"].values()))[key], (int, float))
        }
    return {
        "config": vars(args),
        "methods": methods,
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Counterfactual R-Side Ranking Closed-Loop Evaluation", "",
        "| Method | Blocking | Delay mean/P95 | Raw empty | NSB | Overload | Decision mean/P95 | Same as PPO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in report["methods"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {name} | {agg['blocking_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{agg['raw_mask_empty_rate']:.2%} | {agg['no_suitable_block_rate']:.2%} | "
            f"{agg['server_overload_rate']:.2%} | "
            f"{agg['mean_decision_time_ms']:.3f}/{agg['p95_decision_time_ms']:.3f} | "
            f"{agg['agreement_with_ppo_c']:.2%} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--supervised_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_formal/c_post_decision_joint.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="ppo_r,counterfactual_rank_only,deep_rmsa,ksp_bf")
    parser.add_argument("--lambda_values", default="0.5")
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_counterfactual_ranking_k5m10_h5_eval.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_counterfactual_ranking_k5m10_h5_eval.md")
    # Lyapunov drift-plus-penalty inference-time reranking
    parser.add_argument("--lyap_lambda_spec", type=float, default=0.0)
    parser.add_argument("--lyap_lambda_srv", type=float, default=0.0)
    parser.add_argument("--lyap_epsilon_spec", type=float, default=0.05)
    parser.add_argument("--lyap_epsilon_srv", type=float, default=0.05)
    parser.add_argument("--lyap_queue_clip", type=float, default=20.0)
    parser.add_argument("--lyap_spec_damage_mode", default="fs_path", choices=["fs_path", "fs_only", "path_only"])
    parser.add_argument("--lyap_srv_damage_mode", default="util_after", choices=["util_after", "margin_after"])
    parser.add_argument("--lyap_update_mode", default="binary_raw_empty", choices=["binary_raw_empty", "continuous_pressure"])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = evaluate(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(out_md, result)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
