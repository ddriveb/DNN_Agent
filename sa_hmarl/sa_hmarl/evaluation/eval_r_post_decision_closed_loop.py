"""Closed-loop evaluation of R-side learned post-decision reranking."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.agents.post_decision_value import PostDecisionValueNetwork
from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _select_c_action_from_obs,
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
    _select_r_action,
    _zscore,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES, _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def load_post_checkpoint(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("feature_names") != FEATURE_NAMES:
        raise ValueError("R post-decision checkpoint feature schema mismatch")
    model = PostDecisionValueNetwork(
        checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (
        model,
        np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(checkpoint["feature_std"], dtype=np.float32),
        checkpoint,
    )


def _load_deep_rmsa(ckpt_path: str, env, mod_reg: ModulationRegistry, device: str = "cpu"):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_nodes = ckpt.get("num_nodes")
    if ckpt_nodes is not None and ckpt_nodes != env.net.NUM_NODES:
        raise ValueError("DeepRMSA checkpoint topology mismatch")
    ckpt_slots = ckpt.get("num_slots")
    if ckpt_slots is not None and ckpt_slots != env.net.num_slots:
        raise ValueError("DeepRMSA checkpoint slot mismatch")
    agent = DeepRMSAAgent(
        num_nodes=env.net.NUM_NODES,
        num_slots=env.net.num_slots,
        k_path=ckpt.get("k_path", env.k),
        m_blocks=ckpt.get("m_blocks", env.max_blocks),
        mod_registry=mod_reg,
        gamma=ckpt.get("gamma", 0.95),
        device=device,
    )
    agent.load_state_dict(ckpt)
    agent.eval()
    return agent


def _get_r_logits(agent_r, obs_r: Dict[str, Any]) -> np.ndarray:
    features, mask = agent_r.build_action_features(obs_r)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    return logits


def _select_post_r_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std,
    split_id: int, server_id: int, mode: str, lambda_value: float, device: str,
) -> int:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(action), split_id, server_id)
        for action in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        values = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy()
    logits = _get_r_logits(agent_r, obs_r)[legal]
    if mode in ("post_only", "td_post_only"):
        scores = values
    elif mode in ("ppo_post_rerank", "ppo_td_post_rerank"):
        scores = _zscore(logits) + lambda_value * _zscore(values)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    best = min(
        range(len(legal)),
        key=lambda index: (-float(scores[index]), -float(logits[index]), legal[index]),
    )
    return int(legal[best])


def _run_episode(
    env, requests, agent_c, agent_r, model, mean, std,
    args, metrics: PerMethodMetrics, mode: str, lambda_value: float = 1.0,
    deep_rmsa=None,
) -> None:
    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_action, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        if mode == "deep_rmsa":
            r_idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
            if r_idx is None:
                r_action = (0, 0, 0)
            else:
                r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        elif mode == "ppo_r":
            r_action = _select_r_action(agent_r, obs_r, env.max_blocks)
            r_idx = None
        else:
            r_idx = _select_post_r_action(
                env, req, obs_c, obs_r, agent_r, model, mean, std,
                split_id, server_id, mode, lambda_value, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
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


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, ModulationRegistry.from_profile(args.modulation_profile), args.device)
    model, mean, std, checkpoint = load_post_checkpoint(args.post_checkpoint, args.device)
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
    if "ppo_post_rerank" in requested:
        for value in [float(v) for v in args.lambda_values.split(",") if v.strip()]:
            methods[f"ppo_post_rerank_{value:g}"] = {"per_seed": {}, "lambda": value}
    if "post_only" in requested:
        methods["post_only"] = {"per_seed": {}}
    if "deep_rmsa" in requested:
        methods["deep_rmsa"] = {"per_seed": {}}

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
                elif method_name == "post_only":
                    mode = "post_only"
                elif method_name == "deep_rmsa":
                    mode = "deep_rmsa"
                else:
                    mode = "ppo_post_rerank"
                _run_episode(
                    env, requests, agent_c, agent_r, model, mean, std,
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
        "post_checkpoint_mode": checkpoint.get("mode"),
        "methods": methods,
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# R-Side Post-Decision Closed-Loop Evaluation", "",
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
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_reach_mixed.pt")
    parser.add_argument("--post_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_smoke/c_post_decision_joint.pt")
    parser.add_argument("--methods", default="ppo_r,post_only,ppo_post_rerank,deep_rmsa")
    parser.add_argument("--lambda_values", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=5)
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_post_decision_closed_loop.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_post_decision_closed_loop.md")
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
