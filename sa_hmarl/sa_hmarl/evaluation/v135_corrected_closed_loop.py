"""Closed-loop evaluation of corrected compact-v2 R-side ranking (COST239).

Compares the frozen PPO-R baseline against the corrected counterfactual ranker in:
  - rank-only mode
  - PPO+logit rerank mode with a lambda sweep
  - margin-based override mode

All methods share the same C-side action and request stream so that differences
are purely R-side.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
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
)
from sa_hmarl.evaluation.r_poststate_features_v2 import (
    POSTSTATE_COMPACT_V2_FEATURE_NAMES,
    build_poststate_compact_v2_feature,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")

# COST239 v1.35 protocol.
NUM_SLOTS = 320
NUM_SERVERS = 4
K_PATHS_C = 5
K_PATHS_R = 50
PATH_SORT = "hops"
BLOCK_SORT = "start_asc"
MAX_BLOCKS = 10
MAX_CANDIDATES = 30


def load_ranker(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"),
        int(ckpt["input_dim"]),
        tuple(ckpt["hidden_dims"]),
        float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    feature_names = ckpt.get("feature_names", POSTSTATE_COMPACT_V2_FEATURE_NAMES)
    return model, mean, std, list(feature_names)


def _get_r_logits(agent_r, obs_r: Dict[str, Any]) -> np.ndarray:
    features, mask = agent_r.build_action_features(obs_r)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    return logits


def _ranker_features(
    env, req, obs_c, obs_r, r_features, candidates, split_id, server_id, feature_names
) -> np.ndarray:
    return np.stack([
        build_poststate_compact_v2_feature(
            env, req, obs_c, obs_r, r_features, int(a),
            split_id, server_id, feature_names=feature_names,
        )
        for a in candidates
    ])


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x)) + 1e-8
    return (x - np.mean(x)) / std


def _rank_only_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std, feature_names,
    split_id, server_id, device: str,
) -> Tuple[int, np.ndarray, np.ndarray]:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0, r_features, r_mask
    if len(legal) == 1:
        return int(legal[0]), r_features, r_mask
    candidates = legal[:MAX_CANDIDATES]
    X = _ranker_features(env, req, obs_c, obs_r, r_features, candidates, split_id, server_id, feature_names)
    Xn = (X - mean) / std
    with torch.no_grad():
        scores = model(torch.as_tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    best_local = int(np.argmax(scores))
    return int(candidates[best_local]), r_features, r_mask


def _rerank_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std, feature_names,
    split_id, server_id, lambda_value: float, device: str,
) -> int:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])
    candidates = legal[:MAX_CANDIDATES]
    X = _ranker_features(env, req, obs_c, obs_r, r_features, candidates, split_id, server_id, feature_names)
    Xn = (X - mean) / std
    with torch.no_grad():
        scores = model(torch.as_tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    logits = _get_r_logits(agent_r, obs_r)[candidates]
    combined = _zscore(logits) + lambda_value * _zscore(scores)
    best_local = int(np.argmax(combined))
    return int(candidates[best_local])


def _margin_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std, feature_names,
    split_id, server_id, delta: float, device: str,
) -> int:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])
    candidates = legal[:MAX_CANDIDATES]
    X = _ranker_features(env, req, obs_c, obs_r, r_features, candidates, split_id, server_id, feature_names)
    Xn = (X - mean) / std
    with torch.no_grad():
        scores = model(torch.as_tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    logits = _get_r_logits(agent_r, obs_r)[candidates]
    ppo_local = int(np.argmax(logits))
    rank_local = int(np.argmax(scores))
    margin = float(scores[rank_local] - scores[ppo_local])
    chosen = rank_local if margin > delta else ppo_local
    return int(candidates[chosen])


class _Metrics:
    def __init__(self):
        self.total = 0
        self.blocked = 0
        self.server_overload = 0
        self.no_suitable_block = 0
        self.allocation_failed = 0
        self.other = 0
        self.delay_ms: List[float] = []
        self.same_as_ppo = 0
        self.r_decisions = 0

    def record(self, info: Dict[str, Any], decision_ms: float, ppo_r_idx: int, chosen_r_idx: int):
        self.total += 1
        self.delay_ms.append(decision_ms)
        if chosen_r_idx is not None and ppo_r_idx is not None:
            self.r_decisions += 1
            if int(chosen_r_idx) == int(ppo_r_idx):
                self.same_as_ppo += 1
        if not info.get("success", False):
            self.blocked += 1
            reason = info.get("reason", "")
            if reason == "server_overload":
                self.server_overload += 1
            elif reason == "no_suitable_block":
                self.no_suitable_block += 1
            elif reason == "allocation_failed":
                self.allocation_failed += 1
            else:
                self.other += 1

    def aggregate(self) -> Dict[str, Any]:
        total = max(self.total, 1)
        r_dec = max(self.r_decisions, 1)
        arr = np.array(self.delay_ms)
        return {
            "total": self.total,
            "blocking_rate": self.blocked / total,
            "server_overload_rate": self.server_overload / total,
            "no_suitable_block_rate": self.no_suitable_block / total,
            "allocation_failed_rate": self.allocation_failed / total,
            "other_block_rate": self.other / total,
            "optical_block_rate": (self.no_suitable_block + self.allocation_failed + self.other) / total,
            "overload_share_of_blocking": self.server_overload / max(self.blocked, 1),
            "optical_share_of_blocking": (self.no_suitable_block + self.allocation_failed + self.other) / max(self.blocked, 1),
            "mean_decision_time_ms": float(arr.mean()),
            "p95_decision_time_ms": float(np.percentile(arr, 95)),
            "agreement_with_ppo_r": self.same_as_ppo / r_dec,
        }


def _run_episode(env, requests, agent_c, agent_r, model, mean, std, feature_names, mode, lambda_or_delta, device):
    metrics = _Metrics()
    num_servers = len(env.mec.servers)
    for request_index, req in enumerate(requests):
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        env.k = K_PATHS_C
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), num_servers)

        env.k = K_PATHS_R
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_r = build_agent_r_observation(env, req, split_id, server_id)

        # PPO-R reference action for agreement stats.
        ppo_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        if mode == "ppo_r":
            chosen_r_idx = ppo_r_idx
        elif mode == "rank_only":
            chosen_r_idx, _, _ = _rank_only_action(
                env, req, obs_c, obs_r, agent_r, model, mean, std, feature_names,
                split_id, server_id, device,
            )
        elif mode == "rerank":
            chosen_r_idx = _rerank_action(
                env, req, obs_c, obs_r, agent_r, model, mean, std, feature_names,
                split_id, server_id, lambda_or_delta, device,
            )
        elif mode == "margin":
            chosen_r_idx = _margin_action(
                env, req, obs_c, obs_r, agent_r, model, mean, std, feature_names,
                split_id, server_id, lambda_or_delta, device,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        decision_ms = (time.perf_counter() - started) * 1000.0
        r_action = decode_agent_r_action(int(chosen_r_idx), len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)
        metrics.record(info, decision_ms, ppo_r_idx, chosen_r_idx)
    return metrics.aggregate()


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    model, mean, std, feature_names = load_ranker(args.ranker_checkpoint, args.device)

    env_proto = make_env(
        "xlron_cost239_ptrnet_real", NUM_SLOTS, NUM_SERVERS, 42,
        modulation_profile="default", max_blocks=MAX_BLOCKS,
        block_sort_strategy=BLOCK_SORT, path_sort_strategy=PATH_SORT, k=K_PATHS_R,
    )

    episodes_by_seed: Dict[int, List[List[Any]]] = {}
    for seed in args.seeds:
        episodes = []
        rng = np.random.RandomState(seed)
        for _ in range(args.episodes_per_seed):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max,
                args.size_min_mb, args.size_max_mb,
                args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile,
            ))
        episodes_by_seed[seed] = episodes

    methods: Dict[str, Dict[str, Any]] = {}
    methods["ppo_r"] = {"mode": "ppo_r", "lambda_or_delta": 0.0, "per_seed": {}}
    methods["rank_only"] = {"mode": "rank_only", "lambda_or_delta": 0.0, "per_seed": {}}
    for lam in args.lambda_values:
        methods[f"rerank_l{lam:g}"] = {"mode": "rerank", "lambda_or_delta": lam, "per_seed": {}}
    for delta in args.margin_values:
        methods[f"margin_d{delta:g}"] = {"mode": "margin", "lambda_or_delta": delta, "per_seed": {}}

    for method_name, spec in methods.items():
        for seed in args.seeds:
            seed_results = []
            for requests in episodes_by_seed[seed]:
                env = make_env(
                    "xlron_cost239_ptrnet_real", NUM_SLOTS, NUM_SERVERS, seed,
                    modulation_profile="default", max_blocks=MAX_BLOCKS,
                    block_sort_strategy=BLOCK_SORT, path_sort_strategy=PATH_SORT, k=K_PATHS_R,
                )
                env.reset(requests)
                seed_results.append(_run_episode(
                    env, requests, agent_c, agent_r,
                    model, mean, std, feature_names,
                    spec["mode"], spec["lambda_or_delta"], args.device,
                ))
            # Aggregate episodes within seed.
            agg_seed = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
            spec["per_seed"][str(seed)] = agg_seed
        # Aggregate across seeds.
        keys = list(next(iter(spec["per_seed"].values())).keys())
        spec["aggregate"] = {k: float(np.mean([row[k] for row in spec["per_seed"].values()])) for k in keys}

    return {
        "config": vars(args),
        "methods": {k: {"mode": v["mode"], "lambda_or_delta": v["lambda_or_delta"],
                       "per_seed": v["per_seed"], "aggregate": v["aggregate"]}
                    for k, v in methods.items()},
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Corrected R-Side Ranker Closed-Loop Evaluation (COST239)",
        "",
        f"Ranker: `{report['config']['ranker_checkpoint']}`",
        f"Seeds: {report['config']['seeds']}, episodes/seed: {report['config']['episodes_per_seed']}, requests/episode: {report['config']['requests_per_episode']}",
        "",
        "| Method | Blocking | Overload share | Optical share | Decision mean/P95 ms | Agreement with PPO-R |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, spec in report["methods"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {name} | {agg['blocking_rate']:.2%} | "
            f"{agg['overload_share_of_blocking']:.2%} | "
            f"{agg['optical_share_of_blocking']:.2%} | "
            f"{agg['mean_decision_time_ms']:.2f}/{agg['p95_decision_time_ms']:.2f} | "
            f"{agg['agreement_with_ppo_r']:.2%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker_checkpoint", required=True)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3030, 4040, 5050])
    parser.add_argument("--episodes_per_seed", type=int, default=3)
    parser.add_argument("--requests_per_episode", type=int, default=3000)
    parser.add_argument("--lambda_values", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--margin_values", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    report = evaluate(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "CLOSED_LOOP_EVAL.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(output_dir / "CLOSED_LOOP_EVAL.md", report)
    print("Closed-loop report:", output_dir / "CLOSED_LOOP_EVAL.md")


if __name__ == "__main__":
    main()
