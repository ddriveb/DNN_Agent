"""Closed-loop evaluation of PPO-C with learned post-decision reranking."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.agents.post_decision_value import PostDecisionValueNetwork
from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    decode_agent_c_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
    _get_ppo_c_logits,
    _run_episode_potential_method,
    _run_episode_ppo_c,
    _select_ppo_c_action,
    _select_r_action,
    _zscore,
)
from sa_hmarl.evaluation.generate_c_post_decision_dataset import (
    FEATURE_NAMES,
    _candidate_feature_vector,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.env.observation_builder import build_agent_r_observation
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def load_post_decision_checkpoint(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Post-decision checkpoint feature schema mismatch")
    model = PostDecisionValueNetwork(
        checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    return model, mean, std, checkpoint


def select_post_decision_action(
    model: PostDecisionValueNetwork,
    mean: np.ndarray,
    std: np.ndarray,
    candidate_features: np.ndarray,
    legal_indices: List[int],
    legal_logits: np.ndarray,
    mode: str,
    lambda_value: float,
    device: str,
) -> Tuple[int, np.ndarray]:
    normalized = (candidate_features - mean) / std
    with torch.no_grad():
        predictions = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy()
    if mode == "post_only":
        scores = predictions
    elif mode == "post_delay":
        delay_index = FEATURE_NAMES.index("min_delay_est")
        delay_estimates = candidate_features[:, delay_index]
        scores = _zscore(predictions) - lambda_value * _zscore(delay_estimates)
    elif mode == "ppo_post_rerank":
        scores = _zscore(legal_logits) + lambda_value * _zscore(predictions)
    else:
        raise ValueError(f"Unknown post-decision mode: {mode}")
    # Policy logits and action index provide deterministic tie-breaking.
    best_local = min(
        range(len(legal_indices)),
        key=lambda index: (-float(scores[index]), -float(legal_logits[index]), legal_indices[index]),
    )
    return int(legal_indices[best_local]), predictions


def _features_for_post_decision(agent_c, obs_c: Dict[str, Any]) -> np.ndarray:
    """Build the 27-dim r-feasibility prefix expected by the post-value model.

    The deployable C policy may use a different feature mode, e.g. an older
    delay-aware checkpoint with the default 17-dim feature vector.  The learned
    post-decision model, however, was trained with label-free r-feasibility
    features.  Keep those two feature streams independent.
    """
    features, _ = agent_c.build_action_features(obs_c)
    if features.shape[1] == 27:
        return features
    if features.shape[1] != 17:
        raise ValueError(
            "Post-decision model expects either default 17-dim C features "
            f"or r_feasibility 27-dim features, got {features.shape[1]}"
        )

    class _RFeasFeatureBuilder:
        feature_mode = "r_feasibility"
        ablation = False
        zero_spectrum = False

    r_features, _ = AgentC.build_action_features(_RFeasFeatureBuilder(), obs_c)
    return r_features


def _record_outcome(
    metrics: PerMethodMetrics, info: Dict[str, Any], raw_empty: bool,
    obs_c: Dict[str, Any], split_id: int, server_id: int, decision_ms: float,
    same_as_ppo: bool,
) -> None:
    success = bool(info.get("success", False))
    reason = info.get("reason", "")
    metrics.total += 1
    metrics.blocked += int(not success)
    metrics.raw_empty += int(raw_empty)
    metrics.no_suitable_block += int(reason == "no_suitable_block")
    metrics.server_overload += int(reason == "server_overload")
    metrics.deadline_failure += int(reason == "deadline_infeasible")
    if success:
        metrics.delays.append(float(info.get("delay_ms", 0.0)))
        metrics.fses.append(float(info.get("num_slots", 0.0)))
        metrics.wastes.append(float(info.get("block_waste", 0.0) or 0.0))
        metrics.path_kms.append(float(info.get("path_dist_km", 0.0) or 0.0))
    metrics.active_connections.append(0)  # overwritten by caller after env.step
    metrics.valid_c_actions.append(int(np.asarray(obs_c["agent_c_mask"], dtype=bool).sum()))
    metrics.total_valid_r_actions.append(int(sum(sum(row) for row in obs_c["feasible_counts"])))
    metrics.selected_valid_r_actions.append(int(obs_c["feasible_counts"][split_id][server_id]))
    metrics.decision_times_ms.append(decision_ms)
    metrics.actions_same.append(same_as_ppo)
    if not same_as_ppo and not success:
        metrics.changed_blocked_count += 1


def run_episode_post_decision(
    env, requests: List[Any], agent_c, agent_r, model, mean, std,
    args: argparse.Namespace, metrics: PerMethodMetrics, mode: str,
    lambda_value: float = 1.0, ppo_actions: List[int] | None = None,
) -> List[int]:
    selected_actions = []
    for request_index, req in enumerate(requests):
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        legal = np.flatnonzero(raw_mask).tolist()
        ppo_action, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
        if len(legal) <= 1:
            selected = legal[0] if legal else ppo_action
        else:
            policy_features = _features_for_post_decision(agent_c, obs_c)
            online_features = []
            for candidate_idx in legal:
                split_id, server_id = divmod(candidate_idx, len(env.mec.servers))
                online_features.append(_candidate_feature_vector(
                    env, req, obs_c, policy_features, candidate_idx, split_id, server_id
                ))
            logits = _get_ppo_c_logits(agent_c, obs_c, raw_mask)
            selected, _ = select_post_decision_action(
                model, mean, std, np.stack(online_features), legal, logits,
                mode, lambda_value, args.device,
            )
        decision_ms = (time.perf_counter() - started) * 1000.0
        selected_actions.append(int(selected))
        split_id, server_id = decode_agent_c_action(int(selected), args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_action = _select_r_action(agent_r, obs_r, env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)
        baseline_action = (
            ppo_actions[request_index]
            if ppo_actions is not None and request_index < len(ppo_actions)
            else ppo_action
        )
        _record_outcome(
            metrics, info, len(legal) == 0, obs_c, split_id, server_id,
            decision_ms, int(selected) == int(baseline_action),
        )
        metrics.active_connections[-1] = len(env.active_connections)
    return selected_actions


def run_episode_heuristic(
    env, requests: List[Any], agent_r, args: argparse.Namespace,
    metrics: PerMethodMetrics, method: str, seed: int, episode_index: int,
    ppo_actions: List[int] | None = None,
) -> List[int]:
    """Run one C-side heuristic with the same frozen PPO-R backend."""
    selected_actions = []
    server_selected_count = np.zeros(args.num_servers, dtype=int)
    rng = np.random.RandomState(seed * 1000 + episode_index)
    for request_index, req in enumerate(requests):
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        selected = select_offloading_action(
            method, env, req, obs_c, raw_mask,
            rng=rng, server_selected_count=server_selected_count,
        )
        if selected is None:
            selected = 0
        decision_ms = (time.perf_counter() - started) * 1000.0
        selected_actions.append(int(selected))
        split_id, server_id = decode_agent_c_action(int(selected), args.num_servers)
        server_selected_count[server_id] += 1
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_action = _select_r_action(agent_r, obs_r, env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)
        baseline_action = (
            ppo_actions[request_index]
            if ppo_actions is not None and request_index < len(ppo_actions)
            else selected
        )
        _record_outcome(
            metrics, info, not raw_mask.any(), obs_c, split_id, server_id,
            decision_ms, int(selected) == int(baseline_action),
        )
        metrics.active_connections[-1] = len(env.active_connections)
    return selected_actions


def _new_env(args: argparse.Namespace):
    return make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        args.slot_bw_hz, args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )


def _combine(episodes: List[PerMethodMetrics]) -> Dict[str, Any]:
    combined = PerMethodMetrics()
    for metrics in episodes:
        for field in (
            "total", "blocked", "raw_empty", "no_suitable_block",
            "server_overload", "deadline_failure", "changed_blocked_count",
        ):
            setattr(combined, field, getattr(combined, field) + getattr(metrics, field))
        for field in (
            "delays", "fses", "wastes", "path_kms", "active_connections",
            "valid_c_actions", "total_valid_r_actions", "selected_valid_r_actions",
            "actions_same", "decision_times_ms",
        ):
            getattr(combined, field).extend(getattr(metrics, field))
    return _aggregate_metrics(combined)


def _apply_learned_method_verdict(report: Dict[str, Any]) -> None:
    """Judge deployable learned methods; explicit potential is an upper bound only."""
    methods = report.get("methods", {})
    baseline = methods.get("ppo_c", {}).get("aggregate", {})
    explicit = methods.get("explicit_spectrum", {}).get("aggregate")
    learned_names = [
        name
        for name in methods
        if name == "post_only"
        or name.startswith("post_delay")
        or name.startswith("ppo_post_rerank")
    ]
    best_name = min(
        learned_names,
        key=lambda name: methods[name]["aggregate"]["blocking_rate"],
        default=None,
    )
    if not best_name or not baseline:
        report["verdict"] = "INCOMPLETE"
        return

    best = methods[best_name]["aggregate"]
    gain = baseline["blocking_rate"] - best["blocking_rate"]
    explicit_gain = (
        baseline["blocking_rate"] - explicit["blocking_rate"] if explicit else None
    )
    retained = gain / explicit_gain if explicit_gain and explicit_gain > 0 else None
    report["best_learned_method"] = best_name
    report["learned_blocking_gain"] = gain
    report["explicit_blocking_gain"] = explicit_gain
    report["explicit_gain_retained"] = retained
    report["verdict"] = (
        "PASS_POST_DECISION_CLOSED_LOOP"
        if gain >= 0.01
        and best["mean_decision_time_ms"] < 30.0
        and best["p95_decision_time_ms"] < 100.0
        and (retained is None or retained >= 0.70)
        else "FAIL_POST_DECISION_CLOSED_LOOP"
    )


def _paired_bootstrap(method_rates: List[float], baseline_rates: List[float]) -> Dict[str, float]:
    differences = np.asarray(method_rates) - np.asarray(baseline_rates)
    rng = np.random.RandomState(42)
    samples = np.asarray([
        np.mean(differences[rng.randint(0, len(differences), len(differences))])
        for _ in range(10000)
    ])
    return {
        "delta": float(np.mean(differences)),
        "ci_95_lo": float(np.percentile(samples, 2.5)),
        "ci_95_hi": float(np.percentile(samples, 97.5)),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(value) for value in args.seeds.split(",")]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    lambdas = [float(value) for value in args.lambda_values.split(",") if value.strip()]
    model, mean, std, checkpoint = load_post_decision_checkpoint(args.post_checkpoint, args.device)
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    r_before = {key: value.clone() for key, value in agent_r.policy_net.state_dict().items()}
    proto = _new_env(args)

    episodes_by_seed = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes_by_seed[seed] = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, proto.net.NUM_NODES))
            episodes_by_seed[seed].append(generate_requests(
                proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max, args.size_min_mb,
                args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile, args.traffic_mode,
                args.regime_stay_prob,
            ))

    method_specs = []
    if "ppo_c" in methods:
        method_specs.append(("ppo_c", None))
    if "post_only" in methods:
        method_specs.append(("post_only", None))
    if "post_delay" in methods:
        method_specs.extend((f"post_delay_{value:g}", value) for value in lambdas)
    if "ppo_post_rerank" in methods:
        method_specs.extend((f"ppo_post_rerank_{value:g}", value) for value in lambdas)
    if "explicit_spectrum" in methods:
        method_specs.append(("explicit_spectrum", None))
    for heuristic in ("greedy", "wo", "df", "rf", "iwd"):
        if heuristic in methods:
            method_specs.append((heuristic, None))
    per_episode: Dict[str, List[PerMethodMetrics]] = {name: [] for name, _ in method_specs}
    baseline_actions = {}
    started = time.time()

    # Baseline is always generated for paired action diagnostics.
    for seed in seeds:
        for episode_index, requests in enumerate(episodes_by_seed[seed]):
            env = _new_env(args)
            env.reset(requests)
            metrics = PerMethodMetrics()
            actions = _run_episode_ppo_c(env, requests, agent_c, agent_r, args, metrics)
            baseline_actions[(seed, episode_index)] = actions
            if "ppo_c" in per_episode:
                per_episode["ppo_c"].append(metrics)

    for method_name, lambda_value in method_specs:
        if method_name == "ppo_c":
            continue
        for seed in seeds:
            for episode_index, requests in enumerate(episodes_by_seed[seed]):
                env = _new_env(args)
                env.reset(requests)
                metrics = PerMethodMetrics()
                if method_name == "explicit_spectrum":
                    _run_episode_potential_method(
                        env, requests, agent_c, agent_r, args, metrics,
                        "spectrum_only", ppo_c_actions=baseline_actions[(seed, episode_index)],
                    )
                elif method_name in ("greedy", "wo", "df", "rf", "iwd"):
                    run_episode_heuristic(
                        env, requests, agent_r, args, metrics, method_name,
                        seed, episode_index,
                        baseline_actions[(seed, episode_index)],
                    )
                else:
                    if method_name == "post_only":
                        mode = "post_only"
                    elif method_name.startswith("post_delay"):
                        mode = "post_delay"
                    else:
                        mode = "ppo_post_rerank"
                    run_episode_post_decision(
                        env, requests, agent_c, agent_r, model, mean, std, args,
                        metrics, mode, lambda_value or 1.0,
                        baseline_actions[(seed, episode_index)],
                    )
                per_episode[method_name].append(metrics)
            print(f"[{method_name}] completed seed {seed}", flush=True)

    baseline_rates = [metrics.blocked / max(metrics.total, 1) for metrics in per_episode.get("ppo_c", [])]
    output_methods = {}
    for method_name, metrics_list in per_episode.items():
        entry = {"aggregate": _combine(metrics_list)}
        if method_name != "ppo_c" and baseline_rates:
            rates = [metrics.blocked / max(metrics.total, 1) for metrics in metrics_list]
            entry["blocking_delta_ci"] = _paired_bootstrap(rates, baseline_rates)
        output_methods[method_name] = entry
    r_unchanged = all(
        torch.equal(value, r_before[key])
        for key, value in agent_r.policy_net.state_dict().items()
    )
    report = {
        "config": vars(args), "post_checkpoint_mode": checkpoint.get("mode"),
        "methods": output_methods, "agent_r_unchanged": r_unchanged,
        "elapsed_seconds": time.time() - started,
    }
    _apply_learned_method_verdict(report)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Learned Post-Decision Closed-Loop Evaluation", "",
        "| Method | Blocking | Success delay mean/P95 | Raw empty | NSB | Overload | Decision mean/P95 | Delta 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in output_methods.items():
        aggregate = entry["aggregate"]
        ci = entry.get("blocking_delta_ci")
        ci_text = "-" if not ci else f"[{ci['ci_95_lo']*100:+.2f}, {ci['ci_95_hi']*100:+.2f}]pp"
        lines.append(
            f"| {name} | {aggregate['blocking_rate']:.2%} | "
            f"{aggregate['mean_delay_ms']:.3f}/{aggregate['p95_delay_ms']:.3f} ms | "
            f"{aggregate['raw_mask_empty_rate']:.2%} | "
            f"{aggregate['no_suitable_block_rate']:.2%} | {aggregate['server_overload_rate']:.2%} | "
            f"{aggregate['mean_decision_time_ms']:.3f}/{aggregate['p95_decision_time_ms']:.3f} | {ci_text} |"
        )
    lines += ["", f"**Verdict: {report.get('verdict', 'INCOMPLETE')}**"]
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post_checkpoint", default="sa_hmarl/checkpoints/post_decision/c_post_decision_joint.pt")
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument(
        "--methods",
        default="ppo_c,post_only,greedy,wo,df,rf,iwd",
        help="Comma-separated: ppo_c, post_only, post_delay, ppo_post_rerank, explicit_spectrum, greedy, wo, df, rf, iwd",
    )
    parser.add_argument("--lambda_values", default="0.25,0.5,1.0")
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
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
    parser.add_argument("--traffic_mode", default="iid")
    parser.add_argument("--regime_stay_prob", type=float, default=0.9)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--potential_history_window", type=int, default=12)
    parser.add_argument("--potential_probe_limit", type=int, default=6)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/c_post_decision_closed_loop.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/c_post_decision_closed_loop.md")
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(result.get("verdict", "INCOMPLETE"))
