"""Diagnose candidate-conditioned resource impact over a short horizon.

Each raw-mask-valid Agent-C candidate is executed from the same environment
snapshot.  A fixed reference C policy and frozen PPO-R then handle the next H
requests from the same request trace.  This isolates the controllable effect
of the current split/server decision from traffic and state differences.

The script also measures whether a historical request-demand mean field adds
out-of-seed predictive value beyond candidate and resource-state features.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from sa_hmarl.env.multi_resource_mean_field import (
    multi_resource_compute_vector,
    multi_resource_spec_vector,
)
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
)
from sa_hmarl.env.request_demand_mean_field import RequestDemandMeanField
from sa_hmarl.evaluation.diagnose_candidate_resource_impact import (
    _compute_spectrum_field,
    _select_r_action,
)
from sa_hmarl.evaluation.eval_c_closed_loop import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action,
)
from sa_hmarl.training.utils import generate_requests, make_env


@dataclass
class HorizonOutcome:
    current_success: bool
    future_blocked: int
    future_raw_empty: int
    future_server_overload: int
    future_no_suitable_block: int
    future_steps: int
    phi_mean: float
    phi_min: float
    phi_end: float

    @property
    def blocked_rate(self) -> float:
        return self.future_blocked / max(self.future_steps, 1)

    @property
    def collapse_rate(self) -> float:
        return self.future_raw_empty / max(self.future_steps, 1)

    def oracle_key(self) -> Tuple[float, ...]:
        """Lexicographic objective used only for diagnostic oracle ranking."""
        return (
            float(not self.current_success),
            float(self.future_blocked),
            float(self.future_raw_empty),
            float(self.future_server_overload),
            -self.phi_mean,
        )


def _rollout_candidate(
    env: Any,
    requests: Sequence[Any],
    request_index: int,
    split_id: int,
    server_id: int,
    horizon: int,
    agent_c: Any,
    agent_r: Any,
    alpha: float,
    util_threshold: float,
) -> HorizonOutcome:
    """Execute one candidate, then follow the fixed reference policy for H steps."""
    current_req = requests[request_index]
    obs_r = build_agent_r_observation(env, current_req, split_id, server_id)
    action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
    _, _, _, current_info = env.step((split_id, server_id), action_r)

    blocked = raw_empty = overload = no_block = 0
    phi_values: List[float] = []
    stop = min(len(requests), request_index + 1 + horizon)

    for future_index in range(request_index + 1, stop):
        req = requests[future_index]
        obs_c = build_agent_c_observation(env, req)
        action_idx, raw_mask = _select_c_action(agent_c, obs_c, env.net.num_slots)
        if not np.any(raw_mask):
            raw_empty += 1

        k_c, k_r = _compute_spectrum_field(obs_c, util_threshold)
        phi_values.append(float(np.log1p(k_c) + alpha * np.log1p(k_r)))

        future_split, future_server = decode_agent_c_action(
            action_idx, len(obs_c["server_utilizations"])
        )
        obs_r = build_agent_r_observation(
            env, req, future_split, future_server
        )
        action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
        _, _, _, info = env.step((future_split, future_server), action_r)
        if not info.get("success", False):
            blocked += 1
        reason = str(info.get("reason", "unknown"))
        overload += int(reason == "server_overload")
        no_block += int(reason == "no_suitable_block")

    if phi_values:
        phi_mean = float(np.mean(phi_values))
        phi_min = float(np.min(phi_values))
        phi_end = float(phi_values[-1])
    else:
        phi_mean = phi_min = phi_end = 0.0

    return HorizonOutcome(
        current_success=bool(current_info.get("success", False)),
        future_blocked=blocked,
        future_raw_empty=raw_empty,
        future_server_overload=overload,
        future_no_suitable_block=no_block,
        future_steps=len(phi_values),
        phi_mean=phi_mean,
        phi_min=phi_min,
        phi_end=phi_end,
    )


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge: float = 1.0,
) -> np.ndarray:
    """Small dependency-free standardized ridge regressor."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    penalty = np.eye(train.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ y_train)
    return test @ weights


def _average_group_rank_correlation(
    y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray
) -> float:
    """Average within-request Pearson correlation of ordinal ranks."""
    correlations = []
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        if mask.sum() < 2:
            continue
        truth = y_true[mask]
        pred = y_pred[mask]
        if np.ptp(truth) < 1e-12 or np.ptp(pred) < 1e-12:
            continue
        truth_rank = np.argsort(np.argsort(truth)).astype(np.float64)
        pred_rank = np.argsort(np.argsort(pred)).astype(np.float64)
        corr = np.corrcoef(truth_rank, pred_rank)[0, 1]
        if np.isfinite(corr):
            correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else 0.0


def _prediction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray
) -> Dict[str, float]:
    error = y_pred - y_true
    baseline = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / baseline) if baseline > 0 else 0.0,
        "mean_within_request_rank_correlation": _average_group_rank_correlation(
            y_true, y_pred, group_ids
        ),
    }


def _cross_features(base: np.ndarray, mean_field: np.ndarray) -> np.ndarray:
    interactions = np.einsum("bi,bj->bij", base, mean_field).reshape(len(base), -1)
    return np.concatenate([base, mean_field, interactions], axis=1)


def evaluate_out_of_seed_prediction(dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Leave-one-seed-out test of incremental request mean-field information."""
    seeds = np.unique(dataset["seed"])
    if len(seeds) < 2:
        return {"status": "skipped", "reason": "at least two seeds are required"}

    base = np.concatenate(
        [dataset["candidate_features"], dataset["resource_context"]], axis=1
    ).astype(np.float64)
    mean_field = dataset["request_mean_field"].astype(np.float64)
    y = dataset["future_blocked_rate"].astype(np.float64)
    groups = dataset["decision_id"]
    rows = []

    for test_seed in seeds:
        train_mask = dataset["seed"] != test_seed
        test_mask = ~train_mask
        shuffled = mean_field.copy()
        rng = np.random.RandomState(int(test_seed) + 991)
        shuffled[train_mask] = shuffled[train_mask][rng.permutation(train_mask.sum())]
        shuffled[test_mask] = shuffled[test_mask][rng.permutation(test_mask.sum())]

        feature_sets = {
            "candidate_resource": base,
            "candidate_resource_plus_mf": np.concatenate([base, mean_field], axis=1),
            "candidate_conditioned_mf": _cross_features(base, mean_field),
            "candidate_conditioned_shuffled_mf": _cross_features(base, shuffled),
        }
        fold = {"test_seed": int(test_seed), "models": {}}
        for name, features in feature_sets.items():
            prediction = _ridge_predict(
                features[train_mask], y[train_mask], features[test_mask]
            )
            fold["models"][name] = _prediction_metrics(
                y[test_mask], prediction, groups[test_mask]
            )
        rows.append(fold)

    model_names = list(rows[0]["models"])
    aggregate = {
        name: {
            metric: float(np.mean([row["models"][name][metric] for row in rows]))
            for metric in rows[0]["models"][name]
        }
        for name in model_names
    }
    return {"status": "ok", "folds": rows, "aggregate": aggregate}


def _summarize_decisions(decisions: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    decisions = list(decisions)
    if not decisions:
        return {}
    return {
        "decisions": len(decisions),
        "mean_valid_candidates": float(np.mean([d["n_candidates"] for d in decisions])),
        "mean_future_blocked_range": float(np.mean([d["blocked_range"] for d in decisions])),
        "fraction_with_blocked_difference": float(np.mean([d["blocked_range"] > 0 for d in decisions])),
        "mean_future_collapse_range": float(np.mean([d["collapse_range"] for d in decisions])),
        "fraction_with_collapse_difference": float(np.mean([d["collapse_range"] > 0 for d in decisions])),
        "reference_future_blocked_rate": float(np.mean([d["reference_blocked_rate"] for d in decisions])),
        "oracle_future_blocked_rate": float(np.mean([d["oracle_blocked_rate"] for d in decisions])),
        "oracle_blocking_headroom_pp": 100.0 * float(np.mean([
            d["reference_blocked_rate"] - d["oracle_blocked_rate"] for d in decisions
        ])),
        "reference_future_collapse_rate": float(np.mean([d["reference_collapse_rate"] for d in decisions])),
        "oracle_future_collapse_rate": float(np.mean([d["oracle_collapse_rate"] for d in decisions])),
    }


def _make_recommendation(
    decision_summary: Dict[str, float], prediction: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply conservative gates before investing in predictor training."""
    headroom = decision_summary.get("oracle_blocking_headroom_pp", 0.0)
    differing = decision_summary.get("fraction_with_blocked_difference", 0.0)
    aggregate = prediction.get("aggregate", {})
    base_rank = aggregate.get("candidate_resource", {}).get(
        "mean_within_request_rank_correlation", 0.0
    )
    mf_rank = aggregate.get("candidate_conditioned_mf", {}).get(
        "mean_within_request_rank_correlation", 0.0
    )
    shuffled_rank = aggregate.get("candidate_conditioned_shuffled_mf", {}).get(
        "mean_within_request_rank_correlation", 0.0
    )
    gates = {
        "oracle_headroom_at_least_1pp": headroom >= 1.0,
        "candidate_difference_in_at_least_10pct_states": differing >= 0.10,
        "mean_field_rank_gain_at_least_0_03": (
            mf_rank >= base_rank + 0.03 and mf_rank >= shuffled_rank + 0.03
        ),
    }
    passed = all(gates.values())
    return {
        "verdict": "PROCEED_TO_PREDICTOR" if passed else "STOP_BEFORE_PREDICTOR",
        "gates": gates,
        "reason": (
            "Candidate-conditioned mean field shows sufficient controllable headroom."
            if passed else
            "The H-step controllable effect or out-of-seed mean-field increment is too small."
        ),
    }


def run(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42, k=args.k_paths,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
    )
    agent_c = _load_ppo_c(args.rollout_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, env_proto.mod_reg, args.device)

    arrays: Dict[str, List[Any]] = {
        "seed": [], "decision_id": [], "candidate_index": [],
        "candidate_features": [], "request_mean_field": [], "resource_context": [],
        "future_blocked_rate": [], "future_collapse_rate": [],
        "future_server_overload_rate": [], "phi_mean": [], "phi_end": [],
        "is_reference_choice": [], "is_oracle_choice": [],
    }
    decisions: List[Dict[str, Any]] = []
    decision_id = 0
    start = time.time()

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            requests = generate_requests(
                env_proto, rng, src, num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
                traffic_mode=args.traffic_mode,
                regime_stay_prob=args.regime_stay_prob,
            )
            env = make_env(
                topology=args.topology, num_slots=args.num_slots,
                num_servers=args.num_servers, seed=42, k=args.k_paths,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
            )
            env.reset(requests)
            demand_mf = RequestDemandMeanField.from_config(
                args.request_mean_field_window,
                args.size_min_mb, args.size_max_mb,
                args.deadline_min, args.deadline_max,
                args.split_profile,
            )

            for request_index, req in enumerate(requests):
                obs_c = build_agent_c_observation(env, req)
                candidate_features, _ = agent_c.build_action_features(obs_c)
                raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                action_idx, _ = _select_c_action(agent_c, obs_c, env.net.num_slots)
                reference_split, reference_server = decode_agent_c_action(
                    action_idx, args.num_servers
                )
                reference_index = reference_split * args.num_servers + reference_server

                valid_indices = np.flatnonzero(raw_mask)
                snapshot = copy.deepcopy(env)
                mf_vector = demand_mf.vector()
                resource_vector = np.concatenate([
                    multi_resource_spec_vector(env),
                    multi_resource_compute_vector(env),
                ]).astype(np.float32)

                outcomes: Dict[int, HorizonOutcome] = {}
                for candidate_index in valid_indices:
                    split_id, server_id = divmod(int(candidate_index), args.num_servers)
                    outcomes[int(candidate_index)] = _rollout_candidate(
                        copy.deepcopy(snapshot), requests, request_index,
                        split_id, server_id, args.horizon,
                        agent_c, agent_r, args.alpha, args.util_threshold,
                    )

                if outcomes:
                    oracle_index = min(outcomes, key=lambda idx: outcomes[idx].oracle_key())
                    reference_outcome = outcomes.get(reference_index)
                    if reference_outcome is not None:
                        blocked_values = [o.future_blocked for o in outcomes.values()]
                        collapse_values = [o.future_raw_empty for o in outcomes.values()]
                        oracle_outcome = outcomes[oracle_index]
                        decisions.append({
                            "n_candidates": len(outcomes),
                            "blocked_range": max(blocked_values) - min(blocked_values),
                            "collapse_range": max(collapse_values) - min(collapse_values),
                            "reference_blocked_rate": reference_outcome.blocked_rate,
                            "oracle_blocked_rate": oracle_outcome.blocked_rate,
                            "reference_collapse_rate": reference_outcome.collapse_rate,
                            "oracle_collapse_rate": oracle_outcome.collapse_rate,
                        })

                    for candidate_index, outcome in outcomes.items():
                        arrays["seed"].append(seed)
                        arrays["decision_id"].append(decision_id)
                        arrays["candidate_index"].append(candidate_index)
                        arrays["candidate_features"].append(candidate_features[candidate_index])
                        arrays["request_mean_field"].append(mf_vector)
                        arrays["resource_context"].append(resource_vector)
                        arrays["future_blocked_rate"].append(outcome.blocked_rate)
                        arrays["future_collapse_rate"].append(outcome.collapse_rate)
                        arrays["future_server_overload_rate"].append(
                            outcome.future_server_overload / max(outcome.future_steps, 1)
                        )
                        arrays["phi_mean"].append(outcome.phi_mean)
                        arrays["phi_end"].append(outcome.phi_end)
                        arrays["is_reference_choice"].append(candidate_index == reference_index)
                        arrays["is_oracle_choice"].append(candidate_index == oracle_index)
                    decision_id += 1

                # Advance the shared reference trajectory exactly once.
                obs_r = build_agent_r_observation(
                    env, req, reference_split, reference_server
                )
                action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
                env.step((reference_split, reference_server), action_r)
                demand_mf.observe(req)

    dataset = {key: np.asarray(value) for key, value in arrays.items()}
    decision_summary = _summarize_decisions(decisions)
    prediction = evaluate_out_of_seed_prediction(dataset)
    report = {
        "config": {
            **vars(args),
            "seeds": seeds,
            "candidate_feature_dim": int(dataset["candidate_features"].shape[1])
            if len(dataset["candidate_features"]) else 0,
            "request_mean_field_dim": RequestDemandMeanField.DIM,
            "resource_context_dim": 11,
        },
        "samples": int(len(dataset["seed"])),
        "decision_summary": decision_summary,
        "out_of_seed_prediction": prediction,
        "recommendation": _make_recommendation(decision_summary, prediction),
        "elapsed_seconds": time.time() - start,
        "interpretation_guardrail": (
            "The oracle is a one-decision H-step counterfactual upper bound on "
            "reference-trajectory states, not a recursively closed-loop oracle policy."
        ),
    }
    return report, dataset


def _markdown(report: Dict[str, Any]) -> str:
    cfg = report["config"]
    summary = report["decision_summary"]
    lines = [
        "# Candidate-Conditioned H-Step Resource Impact Diagnostic",
        "",
        f"- Traffic: `{cfg['traffic_mode']}` (stay={cfg['regime_stay_prob']})",
        f"- Horizon: **{cfg['horizon']}** requests",
        f"- Seeds: `{cfg['seeds']}`, episodes/seed: {cfg['episodes']}",
        f"- Candidate samples: **{report['samples']}**",
        "",
        "## Counterfactual Candidate Separation",
        "",
        f"- Decisions: **{summary.get('decisions', 0)}**",
        f"- Mean valid candidates: {summary.get('mean_valid_candidates', 0):.2f}",
        f"- Requests with candidate-dependent future blocking: {summary.get('fraction_with_blocked_difference', 0):.2%}",
        f"- Mean future-blocked range: {summary.get('mean_future_blocked_range', 0):.3f}",
        f"- Requests with candidate-dependent future collapse: {summary.get('fraction_with_collapse_difference', 0):.2%}",
        f"- Mean future-collapse range: {summary.get('mean_future_collapse_range', 0):.3f}",
        "",
        "## One-Decision Oracle Headroom",
        "",
        f"- Reference future blocking rate: {summary.get('reference_future_blocked_rate', 0):.4f}",
        f"- Oracle future blocking rate: {summary.get('oracle_future_blocked_rate', 0):.4f}",
        f"- Diagnostic headroom: **{summary.get('oracle_blocking_headroom_pp', 0):.2f} pp**",
        "",
        "> This is not a recursively closed-loop oracle; it isolates one current C action.",
        "",
        "## Out-of-Seed Prediction",
        "",
    ]
    prediction = report["out_of_seed_prediction"]
    if prediction.get("status") != "ok":
        lines.append(f"Skipped: {prediction.get('reason')}")
    else:
        lines.extend(["| Features | MAE | RMSE | R2 | Within-request rank r |", "|---|---:|---:|---:|---:|"])
        for name, metrics in prediction["aggregate"].items():
            lines.append(
                f"| {name} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
                f"{metrics['r2']:.4f} | {metrics['mean_within_request_rank_correlation']:.4f} |"
            )
    lines.extend(["", f"Elapsed: {report['elapsed_seconds']:.1f}s", ""])
    recommendation = report["recommendation"]
    lines.extend([
        "## Decision",
        "",
        f"**{recommendation['verdict']}**",
        "",
        f"{recommendation['reason']}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--request_mean_field_window", type=int, default=20)
    parser.add_argument("--traffic_mode", choices=["iid", "markov_regime"], default="markov_regime")
    parser.add_argument("--regime_stay_prob", type=float, default=0.95)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_prefix", default="sa_hmarl/experiments/candidate_horizon_impact_h5")
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error("--horizon must be positive")

    report, dataset = run(args)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    prefix.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    np.savez_compressed(prefix.with_suffix(".npz"), **dataset)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
