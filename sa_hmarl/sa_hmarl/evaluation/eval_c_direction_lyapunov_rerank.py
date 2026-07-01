"""Inference-time C-side direction-level Lyapunov rerank diagnostic.

This is a low-cost probe for whether Agent-C can reduce downstream RMSA
blocking by keeping a per-server "risk debt" queue.  It does not train or
modify checkpoints.

At each request, PPO-C proposes a top-K set of legal (split, server)
candidates.  For each candidate we estimate downstream R survivability with
the frozen v1.2 R-ranker:

    D_C(c, j) = risk(legal_R_count, best_R_score, top3_R_score)

The C score is then:

    score(c, j) = z(PPO-C logit) - lambda * H[j] * z(D_C(c, j))

where H[j] is updated online from the selected server direction's observed
downstream risk.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
    _select_ppo_c_action,
    _zscore,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import load_ranking_checkpoint
from sa_hmarl.evaluation.generate_r_post_decision_dataset import _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _risk_mask_for_agent_c(agent_c, obs_c: Dict[str, Any], num_slots: int) -> np.ndarray:
    """Return the deploy-time C mask after checkpoint risk filtering."""
    _, mask = agent_c.build_action_features(obs_c)
    mask = np.asarray(mask, dtype=bool)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        if risk_kwargs:
            min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
            mask = apply_agent_c_risk_mask(
                obs_c,
                mask,
                num_slots_total=num_slots,
                min_valid_after_mask=min_valid,
                **risk_kwargs,
            )
    return np.asarray(mask, dtype=bool)


def _get_c_logits_all(agent_c, obs_c: Dict[str, Any], mask: np.ndarray) -> np.ndarray:
    features, _ = agent_c.build_action_features(obs_c)
    if features.size == 0:
        return np.array([], dtype=float)
    x = torch.as_tensor(features, dtype=torch.float32, device=agent_c.device).unsqueeze(0)
    with torch.no_grad():
        logits = agent_c.policy_net(x).squeeze(0).cpu().numpy().astype(float)
    masked = logits.copy()
    masked[~np.asarray(mask, dtype=bool)] = -np.inf
    return masked


def _rank_r_candidates(
    env,
    req,
    obs_c: Dict[str, Any],
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    split_id: int,
    server_id: int,
    device: str,
) -> Dict[str, Any]:
    """Return R legal-count and v1.2 ranker score diagnostics for one C action."""
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return {
            "legal_count": 0,
            "best_score": -1e9,
            "top3_score": -1e9,
            "r_idx": 0,
            "r_action": (0, 0, 0),
        }

    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - rank_mean) / rank_std
    with torch.no_grad():
        scores = rank_model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy().ravel()

    order = np.argsort(-scores, kind="stable")
    best_local = int(order[0])
    best_idx = int(legal[best_local])
    top3 = scores[order[: min(3, len(order))]]
    return {
        "legal_count": int(len(legal)),
        "best_score": float(scores[best_local]),
        "top3_score": float(np.mean(top3)),
        "r_idx": best_idx,
        "r_action": decode_agent_r_action(best_idx, len(obs_r["mod_names"]), env.max_blocks),
    }


def _candidate_risk(diag: Dict[str, Any], legal_scale: float) -> float:
    """Positive risk: larger means worse downstream R survivability."""
    legal_count = float(diag["legal_count"])
    # The score part is z-scored across top-K later; keep this term bounded.
    legal_risk = 1.0 / (1.0 + legal_count / max(legal_scale, 1.0))
    return float(legal_risk)


def _select_direction_lyapunov_c_action(
    agent_c,
    agent_r,
    env,
    req,
    obs_c: Dict[str, Any],
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    H_srv: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    """Select C action with server-direction Lyapunov reranking."""
    ppo_action, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
    mask = _risk_mask_for_agent_c(agent_c, obs_c, env.net.num_slots)
    legal = np.flatnonzero(mask).tolist()
    if not legal:
        return int(ppo_action), {
            "evaluated": 0,
            "changed": False,
            "selected_diag": {"legal_count": 0, "best_score": -1e9, "top3_score": -1e9},
            "mean_adjustment": 0.0,
        }

    logits_all = _get_c_logits_all(agent_c, obs_c, mask)
    if args.top_k_c <= 0 or args.top_k_c >= len(legal):
        topk = legal
    else:
        topk = sorted(legal, key=lambda idx: (-float(logits_all[idx]), int(idx)))[: args.top_k_c]

    rows = []
    for c_idx in topk:
        split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
        diag = _rank_r_candidates(
            env, req, obs_c, agent_r, rank_model, rank_mean, rank_std,
            split_id, server_id, args.device,
        )
        rows.append({
            "c_idx": int(c_idx),
            "split_id": int(split_id),
            "server_id": int(server_id),
            "logit": float(logits_all[int(c_idx)]),
            "diag": diag,
            "legal_risk": _candidate_risk(diag, args.legal_scale),
        })

    logits = np.asarray([row["logit"] for row in rows], dtype=float)
    legal_risks = np.asarray([row["legal_risk"] for row in rows], dtype=float)
    best_scores = np.asarray([row["diag"]["best_score"] for row in rows], dtype=float)
    top3_scores = np.asarray([row["diag"]["top3_score"] for row in rows], dtype=float)
    # Higher ranker score is better, so negative z-score is risk.
    risk = (
        args.w_legal * _zscore(legal_risks)
        - args.w_best * _zscore(best_scores)
        - args.w_top3 * _zscore(top3_scores)
    )
    base = _zscore(logits)
    H_for_rows = np.asarray([H_srv[row["server_id"]] for row in rows], dtype=float)
    adjustment = args.lambda_dir * H_for_rows * risk
    scores = base - adjustment

    best_local = min(
        range(len(rows)),
        key=lambda i: (-float(scores[i]), -float(logits[i]), rows[i]["c_idx"]),
    )
    selected = rows[best_local]
    return int(selected["c_idx"]), {
        "evaluated": len(rows),
        "changed": int(selected["c_idx"]) != int(ppo_action),
        "selected_diag": selected["diag"],
        "selected_server": int(selected["server_id"]),
        "selected_risk": float(risk[best_local]),
        "selected_H": float(H_for_rows[best_local]),
        "mean_adjustment": float(np.mean(np.abs(adjustment))) if len(adjustment) else 0.0,
    }


def _update_server_queue(
    H_srv: np.ndarray,
    server_id: int,
    selected_diag: Dict[str, Any],
    info: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Decay all queues and add risk debt to the selected server direction."""
    H_srv[:] = np.maximum(H_srv - args.epsilon_dir, 0.0)
    legal_count = float(selected_diag.get("legal_count", 0))
    legal_deficit = max(0.0, args.k_min_r - legal_count) / max(args.k_min_r, 1.0)
    outcome_risk = 0.0
    if not bool(info.get("success", False)):
        reason = info.get("reason", "")
        if reason == "no_suitable_block":
            outcome_risk += args.nsb_risk
        elif reason in ("server_overload", "server_saturated"):
            outcome_risk += args.overload_risk
        else:
            outcome_risk += args.block_risk
    H_srv[server_id] = min(
        args.queue_clip,
        H_srv[server_id] + args.legal_deficit_risk * legal_deficit + outcome_risk,
    )


def _run_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    mode: str,
) -> None:
    H_srv = np.zeros(args.num_servers, dtype=float)
    c_changed = 0
    c_evaluated_total = 0
    lyap_adjustments: List[float] = []
    queue_max: List[float] = []
    queue_mean: List[float] = []

    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_empty = int(np.asarray(obs_c["agent_c_mask"], dtype=bool).sum()) == 0

        if mode == "counterfactual_rank_only":
            c_idx, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
            selected_diag = _rank_r_candidates(
                env, req, obs_c, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, args.device,
            )
        elif mode == "direction_lyapunov":
            c_idx, diag = _select_direction_lyapunov_c_action(
                agent_c, agent_r, env, req, obs_c,
                rank_model, rank_mean, rank_std, H_srv, args,
            )
            c_changed += int(diag.get("changed", False))
            c_evaluated_total += int(diag.get("evaluated", 0))
            lyap_adjustments.append(float(diag.get("mean_adjustment", 0.0)))
            split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
            selected_diag = diag["selected_diag"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        r_action = selected_diag.get("r_action", (0, 0, 0))
        decision_ms = (time.perf_counter() - started) * 1000.0
        _, _, _, info = env.step((split_id, server_id), r_action)
        _record_outcome(
            metrics, info, raw_empty, obs_c, split_id, server_id, decision_ms,
            same_as_ppo=(mode == "counterfactual_rank_only"),
        )
        metrics.active_connections[-1] = len(env.active_connections)

        if mode == "direction_lyapunov":
            _update_server_queue(H_srv, server_id, selected_diag, info, args)
            queue_max.append(float(np.max(H_srv)))
            queue_mean.append(float(np.mean(H_srv)))

    metrics.c_changed_count = c_changed
    metrics.c_evaluated_avg = c_evaluated_total / max(metrics.total, 1)
    metrics.direction_queue_max_mean = float(np.mean(queue_max)) if queue_max else 0.0
    metrics.direction_queue_mean = float(np.mean(queue_mean)) if queue_mean else 0.0
    metrics.direction_adjustment_abs_mean = float(np.mean(lyap_adjustments)) if lyap_adjustments else 0.0


def _make_episodes(args: argparse.Namespace, env_proto) -> Dict[int, List[List[Any]]]:
    episodes_by_seed: Dict[int, List[List[Any]]] = {}
    for seed in [int(s.strip()) for s in args.seeds.split(",") if s.strip()]:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
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
    return episodes_by_seed


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)
    episodes_by_seed = _make_episodes(args, env_proto)

    methods: Dict[str, Dict[str, Any]] = {
        "v12_rank_only": {"mode": "counterfactual_rank_only", "per_seed": {}},
    }
    for lam in [float(v) for v in args.lambda_values.split(",") if v.strip()]:
        methods[f"dir_lyap_lambda{lam:g}"] = {
            "mode": "direction_lyapunov",
            "lambda_dir": lam,
            "per_seed": {},
        }

    original_lambda = args.lambda_dir
    for method_name, spec in methods.items():
        args.lambda_dir = float(spec.get("lambda_dir", original_lambda))
        for seed, episodes in episodes_by_seed.items():
            metrics = PerMethodMetrics()
            for requests in episodes:
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks,
                    block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                env.reset(requests)
                _run_episode(
                    env, requests, agent_c, agent_r,
                    rank_model, rank_mean, rank_std,
                    args, metrics, spec["mode"],
                )
            agg = _aggregate_metrics(metrics)
            n = max(metrics.total, 1)
            agg["c_action_changed_rate"] = float(getattr(metrics, "c_changed_count", 0)) / n
            agg["c_evaluated_avg"] = float(getattr(metrics, "c_evaluated_avg", 0.0))
            agg["direction_queue_max_mean"] = float(getattr(metrics, "direction_queue_max_mean", 0.0))
            agg["direction_queue_mean"] = float(getattr(metrics, "direction_queue_mean", 0.0))
            agg["direction_adjustment_abs_mean"] = float(getattr(metrics, "direction_adjustment_abs_mean", 0.0))
            spec["per_seed"][str(seed)] = agg
            print(f"[{method_name}] completed seed {seed}", flush=True)
        first = next(iter(spec["per_seed"].values()))
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key, value in first.items()
            if isinstance(value, (int, float))
        }
    args.lambda_dir = original_lambda
    return {"config": vars(args), "methods": methods}


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    def _selected_r_key(agg: Dict[str, Any]) -> float:
        return float(
            agg.get(
                "mean_selected_valid_r_actions",
                agg.get("avg_selected_valid_r_actions", 0.0),
            )
        )

    lines = [
        "# C-Side Direction-Level Lyapunov Rerank Diagnostic",
        "",
        "This is an inference-time diagnostic only: no training and no checkpoint updates.",
        "",
        "| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Avg selected R | C changed | C eval avg | H max mean | H mean | Adj mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = report["methods"]["v12_rank_only"]["aggregate"]
    for name, spec in report["methods"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {name} | {agg['blocking_rate']:.2%} | "
            f"{agg['raw_mask_empty_rate']:.2%} | {agg['no_suitable_block_rate']:.2%} | "
            f"{agg['server_overload_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{_selected_r_key(agg):.2f} | "
            f"{agg.get('c_action_changed_rate', 0.0):.2%} | "
            f"{agg.get('c_evaluated_avg', 0.0):.1f} | "
            f"{agg.get('direction_queue_max_mean', 0.0):.2f} | "
            f"{agg.get('direction_queue_mean', 0.0):.2f} | "
            f"{agg.get('direction_adjustment_abs_mean', 0.0):.3f} |"
        )
    lines.extend(["", "## Delta vs v1.2", ""])
    lines.append("| Method | Δ Blocking | Δ Raw empty | Δ NSB | Δ Overload | Δ Delay mean | Verdict hint |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for name, spec in report["methods"].items():
        if name == "v12_rank_only":
            continue
        agg = spec["aggregate"]
        d_block = baseline["blocking_rate"] - agg["blocking_rate"]
        d_raw = baseline["raw_mask_empty_rate"] - agg["raw_mask_empty_rate"]
        d_nsb = baseline["no_suitable_block_rate"] - agg["no_suitable_block_rate"]
        d_ov = agg["server_overload_rate"] - baseline["server_overload_rate"]
        d_delay = agg["mean_delay_ms"] - baseline["mean_delay_ms"]
        verdict = "PASS-candidate" if d_block >= 0.005 and d_ov <= 0.005 else "FAIL/MARGINAL"
        lines.append(
            f"| {name} | {d_block*100:+.2f} pp | {d_raw*100:+.2f} pp | "
            f"{d_nsb*100:+.2f} pp | {d_ov*100:+.2f} pp | {d_delay:+.3f} ms | {verdict} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=80)
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
    parser.add_argument("--size_max_mb", type=float, default=40.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top_k_c", type=int, default=3)
    parser.add_argument("--lambda_values", default="0.1,0.3,0.5,1.0")
    parser.add_argument("--lambda_dir", type=float, default=0.3)
    parser.add_argument("--w_legal", type=float, default=1.0)
    parser.add_argument("--w_best", type=float, default=1.0)
    parser.add_argument("--w_top3", type=float, default=0.5)
    parser.add_argument("--legal_scale", type=float, default=20.0)
    parser.add_argument("--k_min_r", type=float, default=10.0)
    parser.add_argument("--epsilon_dir", type=float, default=0.05)
    parser.add_argument("--queue_clip", type=float, default=10.0)
    parser.add_argument("--legal_deficit_risk", type=float, default=1.0)
    parser.add_argument("--nsb_risk", type=float, default=1.0)
    parser.add_argument("--overload_risk", type=float, default=0.5)
    parser.add_argument("--block_risk", type=float, default=0.5)
    parser.add_argument("--output_json", default="sa_hmarl/experiments/c_direction_lyapunov_rerank.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/c_direction_lyapunov_rerank.md")
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
