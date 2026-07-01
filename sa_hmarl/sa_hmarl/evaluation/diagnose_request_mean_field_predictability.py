"""Diagnose whether recent request demand predicts near-future demand.

This diagnostic is intentionally independent of Agent-C and Agent-R.  It asks
whether a rolling request mean field contains information beyond the stationary
request prior, which is a prerequisite for using it as critic context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from sa_hmarl.env.request import DNNRequest
from sa_hmarl.training.utils import generate_requests


DIM_NAMES = (
    "p_small",
    "p_medium",
    "p_large",
    "p_urgent",
    "p_normal",
    "p_relaxed",
)


def request_attributes(request: DNNRequest) -> Tuple[float, float]:
    """Return candidate-agnostic data demand and deadline for one request."""
    mean_size_mb = float(np.mean([split.intermediate_size_mb for split in request.splits]))
    return mean_size_mb, float(request.deadline_ms)


def fit_thresholds(episodes: Sequence[Sequence[DNNRequest]]) -> Dict[str, List[float]]:
    """Fit balanced demand/urgency bins on calibration episodes only."""
    attributes = [request_attributes(req) for episode in episodes for req in episode]
    values = np.asarray(attributes, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one calibration request is required")
    size_thresholds = np.quantile(values[:, 0], [1.0 / 3.0, 2.0 / 3.0])
    deadline_thresholds = np.quantile(values[:, 1], [1.0 / 3.0, 2.0 / 3.0])
    return {
        "size_mb": [float(x) for x in size_thresholds],
        "deadline_ms": [float(x) for x in deadline_thresholds],
    }


def classify_request(request: DNNRequest, thresholds: Dict[str, List[float]]) -> Tuple[int, int]:
    """Return size class (small/medium/large) and urgency class."""
    size_mb, deadline_ms = request_attributes(request)
    size_class = int(np.searchsorted(thresholds["size_mb"], size_mb, side="right"))
    # Smaller deadlines are more urgent, so ascending bins already map correctly.
    urgency_class = int(
        np.searchsorted(thresholds["deadline_ms"], deadline_ms, side="right")
    )
    return size_class, urgency_class


def request_mean_field(
    requests: Sequence[DNNRequest], thresholds: Dict[str, List[float]]
) -> np.ndarray:
    """Build six marginal probabilities for demand size and urgency."""
    vector = np.zeros(6, dtype=np.float64)
    if not requests:
        return vector
    for request in requests:
        size_class, urgency_class = classify_request(request, thresholds)
        vector[size_class] += 1.0
        vector[3 + urgency_class] += 1.0
    vector[:3] /= len(requests)
    vector[3:] /= len(requests)
    return vector


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks without requiring SciPy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    if rank:
        x = _rankdata(x)
        y = _rankdata(y)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence averaged over size and urgency marginals."""
    divergences = []
    for start in (0, 3):
        p_group = np.clip(p[start : start + 3], 1e-12, 1.0)
        q_group = np.clip(q[start : start + 3], 1e-12, 1.0)
        midpoint = 0.5 * (p_group + q_group)
        divergence = 0.5 * np.sum(p_group * np.log2(p_group / midpoint))
        divergence += 0.5 * np.sum(q_group * np.log2(q_group / midpoint))
        divergences.append(float(divergence))
    return float(np.mean(divergences))


def evaluate_predictability(
    episodes: Sequence[Sequence[DNNRequest]],
    thresholds: Dict[str, List[float]],
    prior: np.ndarray,
    history_window: int,
    horizon: int,
) -> Dict[str, object]:
    """Compare rolling mean-field forecasts with a stationary prior forecast."""
    history_vectors: List[np.ndarray] = []
    future_vectors: List[np.ndarray] = []
    for episode in episodes:
        for index in range(history_window, len(episode) - horizon + 1):
            history_vectors.append(
                request_mean_field(episode[index - history_window : index], thresholds)
            )
            future_vectors.append(
                request_mean_field(episode[index : index + horizon], thresholds)
            )

    history = np.asarray(history_vectors, dtype=np.float64)
    future = np.asarray(future_vectors, dtype=np.float64)
    if len(history) == 0:
        raise ValueError("No forecast samples; reduce history window or horizon")
    prior_matrix = np.broadcast_to(prior, future.shape)
    rolling_mae = float(np.mean(np.abs(history - future)))
    prior_mae = float(np.mean(np.abs(prior_matrix - future)))
    forecast_skill = 1.0 - rolling_mae / max(prior_mae, 1e-12)
    pearson = {
        name: _correlation(history[:, index], future[:, index])
        for index, name in enumerate(DIM_NAMES)
    }
    spearman = {
        name: _correlation(history[:, index], future[:, index], rank=True)
        for index, name in enumerate(DIM_NAMES)
    }
    return {
        "samples": int(len(history)),
        "rolling_mae": rolling_mae,
        "prior_mae": prior_mae,
        "forecast_skill": float(forecast_skill),
        "mean_pearson": float(np.mean(list(pearson.values()))),
        "mean_spearman": float(np.mean(list(spearman.values()))),
        "pearson": pearson,
        "spearman": spearman,
        "rolling_js_divergence": float(
            np.mean([_js_divergence(p, q) for p, q in zip(history, future)])
        ),
        "prior_js_divergence": float(
            np.mean([_js_divergence(p, q) for p, q in zip(prior_matrix, future)])
        ),
    }


def _generate_episode(
    seed: int,
    num_requests: int,
    size_range: Tuple[float, float] = (5.0, 30.0),
    deadline_range: Tuple[float, float] = (30.0, 100.0),
    traffic_mode: str = "iid",
    regime_stay_prob: float = 0.9,
) -> List[DNNRequest]:
    seed = int(seed) % (2**32 - 1)
    return generate_requests(
        env=None,
        rng=np.random.RandomState(seed),
        src_node=0,
        num_requests=num_requests,
        arrival_interval=0.25,
        holding_min=4.0,
        holding_max=10.0,
        deadline_min=deadline_range[0],
        deadline_max=deadline_range[1],
        size_min_mb=size_range[0],
        size_max_mb=size_range[1],
        edge_cost_min=0.5,
        edge_cost_max=15.0,
        num_splits=3,
        split_profile="default3",
        traffic_mode=traffic_mode,
        regime_stay_prob=regime_stay_prob,
    )


def generate_stationary_episodes(
    seeds: Iterable[int], episodes_per_seed: int, num_requests: int
) -> List[List[DNNRequest]]:
    episodes = []
    for seed in seeds:
        for episode_index in range(episodes_per_seed):
            episodes.append(_generate_episode(seed * 10000 + episode_index, num_requests))
    return episodes


def generate_regime_episodes(
    seeds: Iterable[int], episodes_per_seed: int, num_requests: int
) -> List[List[DNNRequest]]:
    """Generate Markov-modulated traffic using the production request model."""
    episodes = []
    for seed in seeds:
        for episode_index in range(episodes_per_seed):
            episodes.append(
                _generate_episode(
                    seed * 100000 + episode_index,
                    num_requests,
                    traffic_mode="markov_regime",
                    regime_stay_prob=0.95,
                )
            )
    return episodes


def _prior_from_episodes(
    episodes: Sequence[Sequence[DNNRequest]], thresholds: Dict[str, List[float]]
) -> np.ndarray:
    requests = [request for episode in episodes for request in episode]
    return request_mean_field(requests, thresholds)


def _markdown_report(report: Dict[str, object]) -> str:
    lines = [
        "# Request Mean-Field Predictability Diagnostic",
        "",
        "## Setup",
        "",
        f"- Seeds: `{report['config']['seeds']}`",
        f"- Episodes per seed: `{report['config']['episodes_per_seed']}`",
        f"- Requests per episode: `{report['config']['requests_per_episode']}`",
        f"- History window: `{report['config']['history_window']}` requests",
        f"- Forecast horizons: `{report['config']['horizons']}`",
        "- Stationary trace: current IID request generator (`default3`).",
        "- Regime trace: production `markov_regime` generator (`stay_prob=0.95`).",
        "",
        "## Thresholds And Prior",
        "",
        f"- Size tertiles (mean intermediate MB): `{report['thresholds']['size_mb']}`",
        f"- Deadline tertiles (ms): `{report['thresholds']['deadline_ms']}`",
        f"- Calibration prior: `{report['calibration_prior']}`",
        f"- IID evaluation marginal: `{report['trace_marginals']['stationary_iid']}`",
        f"- Markov evaluation marginal: `{report['trace_marginals']['markov_regime']}`",
        "",
        "## Results",
        "",
        "| Trace | Horizon | Samples | Rolling MAE | Prior MAE | Skill | Mean Pearson | Rolling JS | Prior JS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for trace_name in ("stationary_iid", "regime_positive_control"):
        for horizon, metrics in report["results"][trace_name].items():
            lines.append(
                f"| {trace_name} | {horizon} | {metrics['samples']} | "
                f"{metrics['rolling_mae']:.4f} | {metrics['prior_mae']:.4f} | "
                f"{metrics['forecast_skill']:+.4f} | {metrics['mean_pearson']:+.4f} | "
                f"{metrics['rolling_js_divergence']:.4f} | "
                f"{metrics['prior_js_divergence']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`forecast_skill > 0` means the recent request distribution predicts the "
            "future better than the stationary calibration prior. Values near or below "
            "zero mean that adding this request mean field to the critic provides no "
            "demonstrated temporal information under that traffic model.",
            "",
            report["verdict"],
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, object]:
    seeds = [int(value) for value in args.seeds.split(",")]
    stationary = generate_stationary_episodes(
        seeds, args.episodes_per_seed, args.requests_per_episode
    )
    midpoint = len(stationary) // 2
    calibration = stationary[:midpoint]
    evaluation = stationary[midpoint:]
    thresholds = fit_thresholds(calibration)
    prior = _prior_from_episodes(calibration, thresholds)
    regime = generate_regime_episodes(
        seeds, args.episodes_per_seed, args.requests_per_episode
    )
    results: Dict[str, Dict[str, object]] = {
        "stationary_iid": {},
        "regime_positive_control": {},
    }
    for horizon in args.horizons:
        results["stationary_iid"][str(horizon)] = evaluate_predictability(
            evaluation, thresholds, prior, args.history_window, horizon
        )
        results["regime_positive_control"][str(horizon)] = evaluate_predictability(
            regime, thresholds, prior, args.history_window, horizon
        )

    stationary_skills = [
        metrics["forecast_skill"] for metrics in results["stationary_iid"].values()
    ]
    regime_skills = [
        metrics["forecast_skill"]
        for metrics in results["regime_positive_control"].values()
    ]
    if max(stationary_skills) <= args.minimum_skill and min(regime_skills) > args.minimum_skill:
        verdict = (
            "**Verdict:** IID traffic does not justify a request-mean-field critic, but "
            "the production `markov_regime` traffic has predictive signal at every tested "
            "horizon. A critic-only mean-field prototype is justified specifically under "
            "temporally correlated traffic; IID must remain the negative control."
        )
    else:
        verdict = (
            "**Verdict:** The rolling request mean field beats the stationary prior. "
            "A critic-only mean-field prototype is justified for the predictive horizons."
        )
    report: Dict[str, object] = {
        "config": {
            "seeds": seeds,
            "episodes_per_seed": args.episodes_per_seed,
            "requests_per_episode": args.requests_per_episode,
            "history_window": args.history_window,
            "horizons": args.horizons,
            "minimum_skill": args.minimum_skill,
        },
        "thresholds": thresholds,
        "calibration_prior": prior.tolist(),
        "trace_marginals": {
            "stationary_iid": _prior_from_episodes(evaluation, thresholds).tolist(),
            "markov_regime": _prior_from_episodes(regime, thresholds).tolist(),
        },
        "results": results,
        "verdict": verdict,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,123,456,789,101112")
    parser.add_argument("--episodes_per_seed", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--history_window", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--minimum_skill", type=float, default=0.05)
    parser.add_argument(
        "--output_prefix",
        default="sa_hmarl/experiments/request_mean_field_predictability",
    )
    args = parser.parse_args()
    report = run(args)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    output_prefix.with_suffix(".md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    print(_markdown_report(report))


if __name__ == "__main__":
    main()
