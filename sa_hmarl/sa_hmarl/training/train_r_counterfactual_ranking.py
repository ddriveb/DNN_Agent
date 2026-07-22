"""Train a planner-distilled listwise ranking network from R-side groups.

Each training example is a group of candidate R actions at a single state.
The offline teacher is an H-step counterfactual planner: every candidate action
is evaluated by branching the simulator, executing that action, and rolling out
future requests.  The target is a softmax over those planner returns (higher =
better).  The network learns a cheap score f_theta(phi(s, a_i)) and is trained
with KL(q || p) plus a small regression loss on normalized returns.

This is planner imitation / planner distillation, not TD bootstrapping and not
behavioral cloning of PPO-R.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from sa_hmarl.agents.counterfactual_r_ranker import (
    CounterfactualActionValueRanker,
    build_counterfactual_r_ranker,
)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _feasibility_mask(features: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Return a mask that is True only for path-mod-feasible candidates.

    Must be computed on *raw* (unnormalized) features because normalization
    collapses the binary feasibility indicator to near-zero.
    """
    try:
        idx = feature_names.index("path_mod_feasible")
    except ValueError:
        return np.ones(features.shape[:2], dtype=bool)
    return features[..., idx] > 0.5


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax over valid entries with temperature.  Invalid entries get zero weight."""
    if not mask.any(dim=-1).all():
        # Keep behavior defined for defensive use; valid datasets should have
        # at least two feasible candidates per group.
        mask = mask.clone()
        empty = ~mask.any(dim=-1)
        if empty.any():
            mask[empty, 0] = True
    logits = values.masked_fill(~mask, float("-inf")) / temperature
    probs = F.softmax(logits, dim=-1)
    probs = probs.masked_fill(~mask, 0.0)
    return probs


def _ranking_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    tau_label: float,
    tau_model: float,
) -> torch.Tensor:
    q = _masked_softmax(returns, mask, tau_label)
    p = _masked_softmax(scores, mask, tau_model)
    eps = 1e-10
    return (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1).mean()


def _regression_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    return_mean: float,
    return_std: float,
) -> torch.Tensor:
    target = (returns - return_mean) / (return_std + 1e-8)
    diff = F.smooth_l1_loss(scores, target, reduction="none")
    return (diff * mask.float()).sum() / mask.sum().clamp_min(1.0)


def _pairwise_ranking_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.1,
    delta: float = 0.01,
) -> torch.Tensor:
    """Pairwise margin loss over pairs with teacher return gap >= delta."""
    B, C = scores.shape
    valid = mask.float()  # [B, C]

    # Broadcast score/return differences: [B, C, C]
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    return_diff = returns.unsqueeze(2) - returns.unsqueeze(1)
    valid_pair = valid.unsqueeze(2) * valid.unsqueeze(1)

    positive = (return_diff > delta).float() * valid_pair
    loss = positive * F.relu(margin - score_diff)
    count = positive.sum().clamp_min(1.0)
    return loss.sum() / count


def _hard_negative_weights(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    alpha_hard: float = 2.0,
) -> torch.Tensor:
    """Return per-group hard-negative weights.

    A group is hard when the model's top-1 disagrees with the teacher's top-1.
    Weight = 1 + alpha_hard * max(0, teacher_best_return - model_top1_return).
    """
    B = scores.size(0)
    weights = torch.ones(B, device=scores.device, dtype=scores.dtype)
    for i in range(B):
        valid = mask[i]
        if valid.sum() < 2:
            continue
        s = scores[i, valid]
        r = returns[i, valid]
        teacher_best = r.max()
        pred_idx = int(s.argmax())
        pred_return = r[pred_idx]
        if pred_idx != int(r.argmax()):
            gap = torch.clamp(teacher_best - pred_return, min=0.0)
            weights[i] = 1.0 + alpha_hard * gap
    return weights


def _compute_group_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    ppo_action_index: np.ndarray,
) -> Dict[str, Any]:
    """Compute ranking and regret metrics for a set of candidate groups."""
    top1_correct = []
    top3_correct = []
    spearmans = []
    kendalls = []
    model_regrets = []
    ppo_regrets = []
    ppo_agreements = []
    rel_regrets_cond = []
    regret_deltas = []
    oracle_tie_hits = []
    ppo_oracle_tie_hits = []
    score_tie_fractional = []
    ndcgs = []

    n_groups = len(mask)
    for i in range(n_groups):
        valid = mask[i]
        if valid.sum() < 2:
            continue
        s = scores[i, valid]
        r = returns[i, valid]
        best_idx = int(r.argmax())
        best_return = float(r[best_idx])
        tie_tol = max(1e-4, 1e-3 * abs(best_return))

        order = np.argsort(-s, kind="stable")
        top1_correct.append(int(order[0] == best_idx))
        top3_correct.append(int(best_idx in order[: min(3, len(order))]))

        # Spearman and Kendall tau-b correlations.
        rank_score = stats.rankdata(-s)
        rank_return = stats.rankdata(-r)
        if len(s) >= 2:
            rho, _ = stats.spearmanr(rank_score, rank_return)
            if not np.isnan(rho):
                spearmans.append(float(rho))
            tau, _ = stats.kendalltau(rank_score, rank_return)
            if not np.isnan(tau):
                kendalls.append(float(tau))

        model_regret = float(r[best_idx] - r[order[0]])
        model_regrets.append(model_regret)
        oracle_tie_hits.append(int(abs(r[order[0]] - best_return) <= tie_tol))

        # Score-tie fractional top-1: legacy metric, kept for backward compatibility.
        max_score = s.max()
        score_ties = np.flatnonzero(np.isclose(s, max_score))
        best_tied = np.isclose(r[score_ties], best_return).any()
        if best_tied:
            score_tie_fractional.append(1.0 / len(score_ties))
        else:
            score_tie_fractional.append(0.0)

        ppo_idx = int(ppo_action_index[i])
        if 0 <= ppo_idx < len(r):
            ppo_regret = float(r[best_idx] - r[ppo_idx])
            ppo_regrets.append(ppo_regret)
            ppo_agreements.append(int(order[0] == ppo_idx))
            ppo_oracle_tie_hits.append(int(abs(r[ppo_idx] - best_return) <= tie_tol))
            regret_delta = ppo_regret - model_regret
            regret_deltas.append(regret_delta)
            if ppo_regret > 1e-12:
                rel_regrets_cond.append(regret_delta / ppo_regret)

        # NDCG@3 with shifted returns as relevance.
        rel = r - r.min()
        dcg = 0.0
        for rank, idx in enumerate(order[:3], start=1):
            dcg += rel[idx] / np.log2(rank + 1)
        ideal_order = np.argsort(-rel, kind="stable")
        idcg = 0.0
        for rank, idx in enumerate(ideal_order[:3], start=1):
            idcg += rel[idx] / np.log2(rank + 1)
        ndcgs.append(float(dcg / idcg) if idcg > 1e-12 else 1.0)

    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    # Pairwise accuracy over pairs with return gap >= 1% of return std.
    pairwise_correct = []
    pairwise_total = 0
    return_std = float(np.std(returns[mask])) if mask.any() else 1.0
    delta = max(0.01, 0.01 * return_std)
    for i in range(n_groups):
        valid = mask[i]
        n = int(valid.sum())
        if n < 2:
            continue
        s = scores[i, valid]
        r = returns[i, valid]
        for a in range(n):
            for b in range(a + 1, n):
                if abs(r[a] - r[b]) < delta:
                    continue
                pairwise_total += 1
                if (r[a] > r[b] and s[a] > s[b]) or (r[a] < r[b] and s[a] < s[b]):
                    pairwise_correct.append(1)
                else:
                    pairwise_correct.append(0)

    # Aggregate regret metrics (the stable ones).
    mean_model_regret = _mean(model_regrets)
    mean_ppo_regret = _mean(ppo_regrets)
    absolute_regret_improvement = mean_ppo_regret - mean_model_regret
    aggregate_regret_reduction = absolute_regret_improvement / max(mean_ppo_regret, 1e-12)

    # Per-group regret delta statistics.
    regret_deltas_arr = np.asarray(regret_deltas, dtype=np.float64)
    tol = 1e-6
    n_delta = max(len(regret_deltas_arr), 1)
    better_mask = regret_deltas_arr > tol
    worse_mask = regret_deltas_arr < -tol
    equal_mask = ~(better_mask | worse_mask)

    # PPO-regret bucket analysis.
    ppo_regrets_arr = np.asarray(ppo_regrets, dtype=np.float64)
    model_regrets_arr = np.asarray(model_regrets, dtype=np.float64)
    bucket_edges = [0.0, 1e-12, 0.01, 0.05, 0.1, float("inf")]
    bucket_labels = [
        "exactly_zero",
        "(0,0.01]",
        "(0.01,0.05]",
        "(0.05,0.1]",
        ">0.1",
    ]
    bucket_report = {}
    aligned_model_regrets = []
    aligned_ppo_regrets = []
    aligned_regret_deltas = []
    ppo_idx_cursor = 0
    for i in range(n_groups):
        if mask[i].sum() < 2:
            continue
        if ppo_idx_cursor < len(ppo_regrets):
            aligned_model_regrets.append(model_regrets[ppo_idx_cursor])
            aligned_ppo_regrets.append(ppo_regrets[ppo_idx_cursor])
            aligned_regret_deltas.append(regret_deltas[ppo_idx_cursor])
            ppo_idx_cursor += 1
    aligned_model_regrets = np.asarray(aligned_model_regrets, dtype=np.float64)
    aligned_ppo_regrets = np.asarray(aligned_ppo_regrets, dtype=np.float64)
    aligned_regret_deltas = np.asarray(aligned_regret_deltas, dtype=np.float64)

    for lo, hi, label in zip(bucket_edges[:-1], bucket_edges[1:], bucket_labels):
        if label == "exactly_zero":
            bmask = (aligned_ppo_regrets >= lo) & (aligned_ppo_regrets <= hi)
        else:
            bmask = (aligned_ppo_regrets > lo) & (aligned_ppo_regrets <= hi)
        if bmask.sum() == 0:
            bucket_report[label] = {"count": 0}
            continue
        bd = aligned_regret_deltas[bmask]
        bucket_report[label] = {
            "count": int(bmask.sum()),
            "mean_model_regret": float(aligned_model_regrets[bmask].mean()),
            "mean_ppo_regret": float(aligned_ppo_regrets[bmask].mean()),
            "mean_regret_delta": float(bd.mean()),
            "model_better_rate": float((bd > tol).sum() / len(bd)),
            "model_worse_rate": float((bd < -tol).sum() / len(bd)),
            "model_equal_rate": float(((bd >= -tol) & (bd <= tol)).sum() / len(bd)),
        }

    return {
        "top1_accuracy": _mean(top1_correct),
        "top3_accuracy": _mean(top3_correct),
        "score_tie_fractional_top1": _mean(score_tie_fractional),
        "oracle_tie_hit_rate": _mean(oracle_tie_hits),
        "ppo_oracle_tie_hit_rate": _mean(ppo_oracle_tie_hits),
        "ndcg_at_3": _mean(ndcgs),
        "spearman_mean": _mean(spearmans),
        "kendall_tau_b": _mean(kendalls),
        "model_regret_mean": mean_model_regret,
        "ppo_regret_mean": mean_ppo_regret,
        "absolute_regret_improvement": absolute_regret_improvement,
        "aggregate_regret_reduction": aggregate_regret_reduction,
        "conditional_mean_relative_reduction_on_ppo_error_groups": _mean(rel_regrets_cond),
        "model_better_than_ppo_rate": float(better_mask.sum() / n_delta),
        "model_worse_than_ppo_rate": float(worse_mask.sum() / n_delta),
        "model_equal_to_ppo_rate": float(equal_mask.sum() / n_delta),
        "mean_improvement_on_rescued_groups": float(regret_deltas_arr[better_mask].mean()) if better_mask.any() else 0.0,
        "mean_harm_on_harmed_groups": float(regret_deltas_arr[worse_mask].mean()) if worse_mask.any() else 0.0,
        "median_regret_delta": float(np.median(regret_deltas_arr)) if len(regret_deltas_arr) else 0.0,
        "regret_delta_p10": float(np.percentile(regret_deltas_arr, 10)) if len(regret_deltas_arr) else 0.0,
        "regret_delta_p50": float(np.percentile(regret_deltas_arr, 50)) if len(regret_deltas_arr) else 0.0,
        "regret_delta_p90": float(np.percentile(regret_deltas_arr, 90)) if len(regret_deltas_arr) else 0.0,
        "ppo_regret_buckets": bucket_report,
        "ppo_agreement": _mean(ppo_agreements),
        "pairwise_accuracy": float(np.mean(pairwise_correct)) if pairwise_correct else 0.0,
        "pairwise_pairs": pairwise_total,
        "score_std": float(np.std(scores[mask])) if mask.any() else 0.0,
    }


def _selection_score(metrics: Dict[str, Any], metric_name: str) -> float:
    """Return a higher-is-better score for checkpoint selection."""
    if metric_name == "top1":
        return float(metrics["top1_accuracy"])
    if metric_name == "top3":
        return float(metrics["top3_accuracy"])
    if metric_name == "spearman":
        return float(metrics["spearman_mean"])
    if metric_name == "kl":
        return -float(metrics["kl"])
    if metric_name == "regret":
        return -float(metrics["model_regret_mean"])
    raise ValueError(f"Unknown selection metric: {metric_name}")


def _evaluate(
    features: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    ppo_action_index: torch.Tensor,
    feas_mask: torch.Tensor,
    model: CounterfactualActionValueRanker,
    return_mean: float,
    return_std: float,
    tau_label: float,
    tau_model: float,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    model.eval()
    eff_mask = mask & feas_mask
    all_scores = []
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            end = min(start + batch_size, features.size(0))
            x = features[start:end].to(device)
            scores = model(x, eff_mask[start:end].to(device))  # [B, C]
            all_scores.append(scores.cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)
    metrics = _compute_group_metrics(scores, returns.numpy(), eff_mask.numpy(), ppo_action_index.numpy())

    # KL on the split.
    total_kl = 0.0
    total_groups = 0
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            end = min(start + batch_size, features.size(0))
            x = features[start:end].to(device)
            mask_t = eff_mask[start:end].to(device)
            scores_t = model(x, mask_t)
            returns_t = returns[start:end].to(device)
            kl = _ranking_loss(scores_t, returns_t, mask_t, tau_label, tau_model)
            total_kl += float(kl.item()) * (end - start)
            total_groups += (end - start)
    metrics["kl"] = total_kl / max(total_groups, 1)
    return metrics


def _compute_group_weights(
    train_data: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray:
    """Return normalized sampling weights for each training group (balanced mode)."""
    returns = train_data["returns"]
    mask = train_data["mask"]
    n = returns.shape[0]
    if n == 0:
        return np.ones(1, dtype=float) / 1.0

    # Return contrast per group.
    ranges = np.zeros(n, dtype=float)
    for i in range(n):
        valid = returns[i, mask[i]]
        if valid.size > 0:
            ranges[i] = float(valid.max() - valid.min())
    global_std = float(np.std(ranges)) + 1e-8
    norm_ranges = ranges / global_std

    # Pressure proxy: 1 - raw_r_valid_ratio (higher = fewer legal R actions).
    feature_names = list(metadata.get("feature_names", []))
    try:
        rvr_idx = feature_names.index("raw_r_valid_ratio")
        pressures = np.zeros(n, dtype=float)
        for i in range(n):
            valid_vals = train_data["features"][i, mask[i], rvr_idx]
            if valid_vals.size > 0:
                pressures[i] = 1.0 - float(np.mean(valid_vals))
    except ValueError:
        pressures = np.zeros(n, dtype=float)

    # Future NSB positive indicator.
    nsb = train_data.get("future_nsb_counts", np.zeros_like(mask, dtype=np.int64))
    nsb_positive = np.array([
        int(np.any(nsb[i, mask[i]] > 0)) for i in range(n)
    ], dtype=float)

    weights = (
        1.0
        + float(args.balance_alpha) * norm_ranges
        + float(args.balance_beta) * pressures
        + float(args.balance_gamma) * nsb_positive
    )
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    return weights


def _compute_stratified_depth_weights(
    train_data: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    """Return uniform-over-depth-stratum sampling weights.

    The four strata (0/1/2/3) are weighted equally in expectation, so rare deep
    path groups are not drowned out by the naturally more frequent shallow groups.
    """
    n = train_data["features"].shape[0]
    if n == 0 or "depth_stratum" not in train_data:
        return np.ones(n, dtype=float) / max(n, 1)

    strata = np.asarray(train_data["depth_stratum"], dtype=np.int32)
    weights = np.zeros(n, dtype=float)
    for stratum in range(4):
        mask = strata == stratum
        count = int(mask.sum())
        if count > 0:
            # Within a stratum, optionally upweight high-contrast groups.
            stratum_weight = 1.0 / max(count, 1)
            if getattr(args, "stratified_depth_contrast", False):
                returns = train_data["returns"]
                cand_mask = train_data["mask"]
                ranges = np.array([
                    float(returns[i, cand_mask[i]].max() - returns[i, cand_mask[i]].min())
                    if cand_mask[i].any() else 0.0
                    for i in np.flatnonzero(mask)
                ], dtype=float)
                std = float(ranges.std()) + 1e-8
                local_weights = 1.0 + np.clip(ranges / std, 0.0, 5.0)
                local_weights /= local_weights.sum()
                weights[mask] = stratum_weight * local_weights * count
            else:
                weights[mask] = stratum_weight
    weights = np.maximum(weights, 1e-12)
    weights /= weights.sum()
    return weights


def train(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    root = Path(args.dataset_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    train_data = dict(np.load(root / "train.npz", allow_pickle=False))
    val_data = dict(np.load(root / "val.npz", allow_pickle=False))
    test_data = dict(np.load(root / "test.npz", allow_pickle=False))

    mean = np.asarray(metadata["train_feature_mean"], dtype=np.float32)
    std = np.maximum(
        np.asarray(metadata["train_feature_std"], dtype=np.float32),
        float(args.feature_std_floor),
    )
    input_dim = train_data["features"].shape[2]

    # Compute feasibility masks on raw features before normalization.
    train_feas_np = _feasibility_mask(train_data["features"], metadata["feature_names"])
    val_feas_np = _feasibility_mask(val_data["features"], metadata["feature_names"])
    test_feas_np = _feasibility_mask(test_data["features"], metadata["feature_names"])

    # Normalize features (valid entries only).
    def _norm(data, feas_np):
        f = (data["features"].astype(np.float32) - mean) / std
        return {
            "features": torch.as_tensor(f, dtype=torch.float32),
            "returns": torch.as_tensor(data["returns"], dtype=torch.float32),
            "mask": torch.as_tensor(data["mask"], dtype=torch.bool),
            "ppo_action_index": torch.as_tensor(data["ppo_action_index"], dtype=torch.long),
            "feas": torch.as_tensor(feas_np, dtype=torch.bool),
        }

    train_t = _norm(train_data, train_feas_np)
    val_t = _norm(val_data, val_feas_np)
    test_t = _norm(test_data, test_feas_np)

    # Global return statistics for regression target (feasible candidates only).
    train_eff = train_t["mask"] & train_t["feas"]
    valid_returns = train_t["returns"][train_eff].numpy()
    return_mean = float(np.mean(valid_returns))
    return_std = float(np.std(valid_returns))

    model = build_counterfactual_r_ranker(
        args.model_type, input_dim, args.hidden_dims, args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n = train_t["features"].size(0)
    best_val_top1 = -1.0
    best_selection_score = float("-inf")
    best_state = None
    patience_left = args.patience
    history: List[Dict[str, Any]] = []

    # Optional group-level sampling weights.
    group_weights = None
    if args.sampling_mode == "balanced":
        group_weights = _compute_group_weights(train_data, metadata, args)
    elif args.sampling_mode == "stratified_depth":
        group_weights = _compute_stratified_depth_weights(train_data, args)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if group_weights is not None:
            order = np.random.choice(n, size=n, replace=True, p=group_weights)
        else:
            order = np.random.permutation(n)
        train_losses = []
        for start in range(0, n, args.batch_size):
            indices = order[start:start + args.batch_size]
            x = train_t["features"][indices].to(args.device)
            r = train_t["returns"][indices].to(args.device)
            m = train_t["mask"][indices].to(args.device)
            feas_m = train_t["feas"][indices].to(args.device)
            eff_m = m & feas_m

            scores = model(x, eff_m)  # [B, C]
            loss_rank = _ranking_loss(scores, r, eff_m, args.tau_label, args.tau_model)
            loss_reg = _regression_loss(scores, r, eff_m, return_mean, return_std)
            loss = loss_rank + args.reg_weight * loss_reg

            if args.lambda_pair > 0.0:
                loss_pair = _pairwise_ranking_loss(scores, r, eff_m, args.pair_margin, args.pair_delta)
                loss = loss + args.lambda_pair * loss_pair
            else:
                loss_pair = torch.tensor(0.0, device=loss.device)

            # Hard-negative upweighting on groups where model top-1 is wrong.
            if args.lambda_hard > 0.0:
                hard_weights = _hard_negative_weights(scores.detach(), r, eff_m, args.alpha_hard)
                loss_hard_rank = _ranking_loss(scores, r, eff_m, args.tau_label, args.tau_model)
                loss = loss + args.lambda_hard * (hard_weights * loss_hard_rank).mean()
            else:
                loss_hard_rank = torch.tensor(0.0, device=loss.device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Evaluate.
        train_metrics = _evaluate(
            train_t["features"], train_t["returns"], train_t["mask"], train_t["ppo_action_index"],
            train_t["feas"], model, return_mean, return_std, args.tau_label, args.tau_model,
            args.eval_batch_size, args.device,
        )
        val_metrics = _evaluate(
            val_t["features"], val_t["returns"], val_t["mask"], val_t["ppo_action_index"],
            val_t["feas"], model, return_mean, return_std, args.tau_label, args.tau_model,
            args.eval_batch_size, args.device,
        )
        train_metrics["loss"] = float(np.mean(train_losses))

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "loss_rank": float(loss_rank.item()),
            "loss_reg": float(loss_reg.item()),
            "loss_pair": float(loss_pair.item()),
            "loss_hard": float(loss_hard_rank.item()),
        })

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[epoch={epoch}] train_loss={train_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1_accuracy']:.2%} "
                f"val_top3={val_metrics['top3_accuracy']:.2%} "
                f"val_spearman={val_metrics['spearman_mean']:.3f} "
                f"val_ppo_agreement={val_metrics['ppo_agreement']:.2%} "
                f"score_std={val_metrics['score_std']:.4f}",
                flush=True,
            )

        current_selection_score = _selection_score(val_metrics, args.selection_metric)
        if current_selection_score > best_selection_score + args.min_delta:
            best_val_top1 = val_metrics["top1_accuracy"]
            best_selection_score = current_selection_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    test_metrics = _evaluate(
        test_t["features"], test_t["returns"], test_t["mask"], test_t["ppo_action_index"],
        test_t["feas"], model, return_mean, return_std, args.tau_label, args.tau_model,
        args.eval_batch_size, args.device,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "ranking_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "model_type": args.model_type,
        "feature_names": metadata["feature_names"],
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "return_mean": return_mean,
        "return_std": return_std,
        "tau_label": args.tau_label,
        "tau_model": args.tau_model,
        "best_val_top1": best_val_top1,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_selection_score,
        "test_metrics": test_metrics,
        "candidate_mode": metadata.get("candidate_mode", "v1"),
        "max_candidates": metadata.get("max_candidates", args.batch_size),
        "ppo_top_k": metadata.get("ppo_top_k", 8),
        "num_random_candidates": metadata.get("num_random_candidates", 5),
        "min_candidates": metadata.get("min_candidates", 15),
        "candidate_seed": metadata.get("candidate_seed", 12345),
    }, checkpoint_path)

    report = {
        "dataset": str(root),
        "checkpoint": str(checkpoint_path),
        "best_val_top1": best_val_top1,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_selection_score,
        "test_metrics": test_metrics,
        "history": history,
        "config": vars(args),
        "elapsed_seconds": time.time() - getattr(args, "_start_time", time.time()),
    }
    _atomic_write_json(output / "training_report.json", report)
    _write_markdown(output / "training_report.md", report, metadata)
    return report


def _write_markdown(path: Path, report: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    test = report["test_metrics"]
    lines = [
        "# Counterfactual R-Side Ranking Training", "",
        f"- Dataset: `{report['dataset']}`",
        f"- Return coefs: current_block={metadata['return_coefs']['current_block']}, "
        f"future_block={metadata['return_coefs']['future_block']}, "
        f"future_nsb={metadata['return_coefs']['future_nsb']}, "
        f"delay={metadata['return_coefs']['delay']}, fs={metadata['return_coefs']['fs']}"
        + (
            f", path_penalty={metadata['return_coefs'].get('path_penalty', 0)}, "
            f"fs_penalty={metadata['return_coefs'].get('fs_penalty', 0)}"
            if metadata['return_coefs'].get('path_penalty') or metadata['return_coefs'].get('fs_penalty')
            else ""
        ),
        f"- Horizon: {metadata['horizon']}",
        f"- Checkpoint: `{report['checkpoint']}`", "",
        f"- Model type: `{report['config'].get('model_type', 'mlp')}`", "",
        f"- Selection metric: `{report.get('selection_metric', 'top1')}`", "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Best checkpoint val top-1 | {report['best_val_top1']:.2%} |",
        f"| Best `{report.get('selection_metric', 'top1')}` selection score | "
        f"{report.get('best_selection_score', 0):.4f} |",
        f"| Test top-1 | {test['top1_accuracy']:.2%} |",
        f"| Test top-3 | {test['top3_accuracy']:.2%} |",
        f"| Test score-tie fractional top-1 | {test.get('score_tie_fractional_top1', 0):.2%} |",
        f"| Test oracle tie hit rate | {test.get('oracle_tie_hit_rate', 0):.2%} |",
        f"| Test PPO oracle tie hit rate | {test.get('ppo_oracle_tie_hit_rate', 0):.2%} |",
        f"| Test NDCG@3 | {test.get('ndcg_at_3', 0):.3f} |",
        f"| Test Spearman mean | {test['spearman_mean']:.3f} |",
        f"| Test Kendall tau-b | {test.get('kendall_tau_b', 0):.3f} |",
        f"| Test pairwise accuracy | {test.get('pairwise_accuracy', 0):.3f} |",
        f"| Test model regret | {test['model_regret_mean']:.4f} |",
        f"| Test PPO regret | {test['ppo_regret_mean']:.4f} |",
        f"| Test absolute regret improvement | {test.get('absolute_regret_improvement', 0):.4f} |",
        f"| Test aggregate regret reduction | {test.get('aggregate_regret_reduction', 0):.2%} |",
        f"| Test conditional mean relative reduction | {test.get('conditional_mean_relative_reduction_on_ppo_error_groups', 0):.4f} |",
        f"| Test model better / equal / worse rate | {test.get('model_better_than_ppo_rate', 0):.2%} / {test.get('model_equal_to_ppo_rate', 0):.2%} / {test.get('model_worse_than_ppo_rate', 0):.2%} |",
        f"| Test median regret delta | {test.get('median_regret_delta', 0):.4f} |",
        f"| Test regret delta P10/P50/P90 | {test.get('regret_delta_p10', 0):.4f} / {test.get('regret_delta_p50', 0):.4f} / {test.get('regret_delta_p90', 0):.4f} |",
        f"| Test PPO agreement | {test['ppo_agreement']:.2%} |",
        f"| Test score std | {test['score_std']:.4f} |",
        f"| Test KL | {test['kl']:.4f} |",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="sa_hmarl/datasets/r_counterfactual_ranking_v1_2_mixed_low")
    parser.add_argument("--output_dir", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low")
    parser.add_argument(
        "--model_type",
        default="mlp",
        choices=["mlp", "deepset", "set_transformer"],
    )
    parser.add_argument("--selection_metric", default="top1",
                        choices=["top1", "top3", "spearman", "kl", "regret"])
    parser.add_argument("--hidden_dims", type=lambda value: tuple(int(x) for x in value.split(",")), default=(128, 64))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--tau_label", type=float, default=0.5)
    parser.add_argument("--tau_model", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.1)
    parser.add_argument("--feature_std_floor", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument(
        "--sampling_mode",
        choices=["uniform", "balanced", "stratified_depth"],
        default="uniform",
        help=(
            "Training sampling mode. 'uniform' is the unweighted full-state natural "
            "distribution (v1.3 fix main version). 'balanced' upweights high-contrast / "
            "high-pressure / NSB-positive groups. 'stratified_depth' balances the four "
            "depth strata."
        ),
    )
    parser.add_argument("--balanced_sampler", action="store_true",
                        help="Deprecated: use --sampling_mode=balanced.")
    parser.add_argument("--balance_alpha", type=float, default=1.0,
                        help="Weight on normalized return range.")
    parser.add_argument("--balance_beta", type=float, default=1.0,
                        help="Weight on pressure (1 - raw_r_valid_ratio).")
    parser.add_argument("--balance_gamma", type=float, default=1.0,
                        help="Weight on future-NSB-positive indicator.")
    parser.add_argument(
        "--stratified_depth_contrast",
        action="store_true",
        help="Within each depth stratum, additionally upweight high return-contrast groups.",
    )
    parser.add_argument("--lambda_pair", type=float, default=0.5,
                        help="Weight for pairwise ranking loss.")
    parser.add_argument("--pair_margin", type=float, default=0.1)
    parser.add_argument("--pair_delta", type=float, default=0.01)
    parser.add_argument("--lambda_hard", type=float, default=1.0,
                        help="Weight for hard-negative listwise re-loss.")
    parser.add_argument("--alpha_hard", type=float, default=2.0,
                        help="Hard-negative weight multiplier.")
    parser.add_argument("--hard_replay_ratio", type=float, default=0.3,
                        help="Reserved; current hard-negative weighting is applied every batch.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    # Backward compatibility: legacy --balanced_sampler flag maps to sampling_mode=balanced.
    if args.balanced_sampler and args.sampling_mode == "uniform":
        args.sampling_mode = "balanced"
    args._start_time = time.time()
    report = train(args)
    print(
        f"Training complete. Best checkpoint val top-1={report['best_val_top1']:.2%} "
        f"selection_metric={report.get('selection_metric', 'top1')} "
        f"selection_score={report.get('best_selection_score', 0):.4f} "
        f"Test top-1={report['test_metrics']['top1_accuracy']:.2%}"
    )
