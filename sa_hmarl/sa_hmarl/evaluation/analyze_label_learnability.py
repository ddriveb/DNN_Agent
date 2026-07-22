"""Label learnability analysis for the SA-HMARL v1.3 medium dataset.

Reads a merged counterfactual ranking dataset and produces:
  - per-component variance decomposition
  - future-blocking vs delay/FS headroom split
  - PPO regret and oracle headroom diagnostics
  - feature-vs-return correlations
  - train/val/test distribution shift
  - multi-trace ranking stability (future-trace swap proxy)

Outputs:
    sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_LABEL_LEARNABILITY.md
    sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_LABEL_LEARNABILITY.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


def _load_split(dataset_dir: Path, split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(str(dataset_dir / f"{split}.npz"), allow_pickle=False))


def _aggregated_future_components(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Aggregate per-step future components to per-candidate scalar components."""
    mask = np.asarray(data["mask"], dtype=bool)
    # Future blocked / nsb / overload counts (NaN padding ignored).
    future_block = np.nansum(data["future_blocked_per_step"], axis=2)
    future_nsb = np.nansum(data["future_nsb_per_step"], axis=2)
    future_overload = np.nansum(data["future_overload_per_step"], axis=2)

    # Success count per candidate from non-NaN steps.
    bp = data["future_blocked_per_step"]
    nan_mask = np.isnan(bp)
    success_count = np.nansum((bp == 0.0) & (~nan_mask), axis=2).astype(np.float32)

    future_delay_sum = np.nansum(data["future_delay_per_step"], axis=2)
    future_fs_sum = np.nansum(data["future_fs_per_step"], axis=2)
    future_delay = future_delay_sum / np.maximum(success_count, 1.0)
    future_fs = future_fs_sum / np.maximum(success_count, 1.0)

    # Zero-out padded candidates so they do not enter statistics.
    for arr in (future_block, future_nsb, future_overload, future_delay, future_fs, success_count):
        arr[~mask] = 0.0
    return {
        "future_block": future_block,
        "future_nsb": future_nsb,
        "future_overload": future_overload,
        "future_delay": future_delay,
        "future_fs": future_fs,
        "success_count": success_count,
    }


def _reconstruct_returns(data: Dict[str, np.ndarray], coefs: Dict[str, float]) -> np.ndarray:
    """Reconstruct returns from saved components as a sanity check."""
    fut = _aggregated_future_components(data)
    return (
        -coefs["current_block"] * data["current_block"]
        - coefs["future_block"] * fut["future_block"]
        - coefs["future_nsb"] * fut["future_nsb"]
        - coefs["delay"] * fut["future_delay"]
        - coefs["fs"] * fut["future_fs"]
    )


_COMPONENT_TO_COEF = {
    "current_block": "current_block",
    "future_block": "future_block",
    "future_nsb": "future_nsb",
    "future_delay": "delay",
    "future_fs": "fs",
}


def _split_statistics(
    data: Dict[str, np.ndarray],
    feature_names: List[str],
    coefs: Dict[str, float],
) -> Dict[str, Any]:
    mask = np.asarray(data["mask"], dtype=bool)
    returns = np.asarray(data["returns"], dtype=np.float32)
    fut = _aggregated_future_components(data)

    n_groups = mask.shape[0]
    valid_groups = int(mask.any(axis=1).sum())
    cand_counts = mask.sum(axis=1)

    # Per-group variances.
    return_var_per_group = np.zeros(n_groups, dtype=np.float64)
    comp_vars = {
        "current_block": np.zeros(n_groups, dtype=np.float64),
        "future_block": np.zeros(n_groups, dtype=np.float64),
        "future_nsb": np.zeros(n_groups, dtype=np.float64),
        "future_delay": np.zeros(n_groups, dtype=np.float64),
        "future_fs": np.zeros(n_groups, dtype=np.float64),
    }
    groups_with_variance = 0
    for i in range(n_groups):
        idx = mask[i]
        if idx.sum() < 2:
            continue
        groups_with_variance += 1
        r = returns[i, idx]
        return_var_per_group[i] = float(np.var(r, ddof=0))
        comp_vars["current_block"][i] = float(np.var(data["current_block"][i, idx], ddof=0))
        comp_vars["future_block"][i] = float(np.var(fut["future_block"][i, idx], ddof=0))
        comp_vars["future_nsb"][i] = float(np.var(fut["future_nsb"][i, idx], ddof=0))
        comp_vars["future_delay"][i] = float(np.var(fut["future_delay"][i, idx], ddof=0))
        comp_vars["future_fs"][i] = float(np.var(fut["future_fs"][i, idx], ddof=0))

    # Fraction of return variance attributable to each component (linear model).
    explained = {k: 0.0 for k in comp_vars}
    denom = 0.0
    for i in range(n_groups):
        if return_var_per_group[i] > 1e-12:
            denom += return_var_per_group[i]
            for k in comp_vars:
                ck = _COMPONENT_TO_COEF[k]
                explained[k] += coefs[ck] * coefs[ck] * comp_vars[k][i]
    explained_frac = {k: float(v / max(denom, 1e-12)) for k, v in explained.items()}

    # Group-level headroom taxonomy.
    tol = 1e-4
    identical_future_block = 0
    only_delay_fs = 0
    future_block_differs = 0
    ppo_not_in_tie = 0
    ppo_regrets = []
    model_regrets = []
    for i in range(n_groups):
        idx = mask[i]
        if idx.sum() < 2:
            continue
        r = returns[i, idx]
        fb = fut["future_block"][i, idx]
        cb = data["current_block"][i, idx]
        r_range = float(r.max() - r.min())
        fb_range = float(fb.max() - fb.min())
        cb_range = float(cb.max() - cb.min())

        if fb_range <= tol:
            identical_future_block += 1
            if cb_range <= tol and r_range > tol:
                only_delay_fs += 1
        else:
            future_block_differs += 1

        best_val = r.max()
        tie_tol = max(1e-4, 1e-3 * abs(best_val))
        best_tie = np.abs(r - best_val) <= tie_tol
        ppo_idx = int(data["ppo_action_index"][i])
        if 0 <= ppo_idx < len(r):
            ppo_regrets.append(float(best_val - r[ppo_idx]))
            if not best_tie[ppo_idx]:
                ppo_not_in_tie += 1
            model_idx = int(np.argmax(r))
            model_regrets.append(float(best_val - r[model_idx]))  # oracle zero regret by definition

    n_eval = max(groups_with_variance, 1)
    headroom = {
        "groups_with_variance": groups_with_variance,
        "identical_future_block_pp": identical_future_block / n_eval,
        "only_delay_fs_pp": only_delay_fs / n_eval,
        "future_block_differs_pp": future_block_differs / n_eval,
        "ppo_not_in_oracle_tie_pp": ppo_not_in_tie / max(len(ppo_regrets), 1),
        "ppo_regret_mean": float(np.mean(ppo_regrets)) if ppo_regrets else 0.0,
        "ppo_regret_p50": float(np.median(ppo_regrets)) if ppo_regrets else 0.0,
        "ppo_regret_p90": float(np.percentile(ppo_regrets, 90)) if ppo_regrets else 0.0,
    }

    # Label distribution.
    label_vals = returns[mask]
    label_dist = {
        "n": int(label_vals.size),
        "mean": float(label_vals.mean()),
        "std": float(label_vals.std()),
        "min": float(label_vals.min()),
        "max": float(label_vals.max()),
    }

    # Blocking / load proxies.
    current_block_rate = float(data["current_block"][mask].mean())
    future_block_rate = float(fut["future_block"][mask].mean()) / max(coefs.get("H", 5), 1)
    future_nsb_rate = float(fut["future_nsb"][mask].mean()) / max(coefs.get("H", 5), 1)

    return {
        "groups": n_groups,
        "valid_groups": valid_groups,
        "avg_candidates": float(cand_counts.mean()),
        "return_var_mean": float(np.mean(return_var_per_group[return_var_per_group > 0]))
        if np.any(return_var_per_group > 0)
        else 0.0,
        "component_variance_mean": {k: float(np.mean(v[v > 0])) if np.any(v > 0) else 0.0 for k, v in comp_vars.items()},
        "explained_variance_frac": explained_frac,
        "headroom": headroom,
        "label_distribution": label_dist,
        "current_block_rate": current_block_rate,
        "future_block_rate": future_block_rate,
        "future_nsb_rate": future_nsb_rate,
    }


def _feature_return_correlations(
    data: Dict[str, np.ndarray], feature_names: List[str]
) -> Dict[str, Any]:
    mask = np.asarray(data["mask"], dtype=bool)
    returns = np.asarray(data["returns"], dtype=np.float32)
    features = np.asarray(data["features"], dtype=np.float32)

    # Global Spearman correlation between each feature and raw return.
    global_corr = {}
    for f_idx, f_name in enumerate(feature_names):
        x = features[:, :, f_idx][mask]
        y = returns[mask]
        if len(x) > 1 and np.std(x) > 0:
            rho, _ = stats.spearmanr(x, y)
            global_corr[f_name] = float(rho)
        else:
            global_corr[f_name] = 0.0

    # Per-group average absolute Spearman.
    per_group_abs = {f_name: [] for f_name in feature_names}
    for i in range(mask.shape[0]):
        idx = mask[i]
        if idx.sum() < 3:
            continue
        r = returns[i, idx]
        for f_idx, f_name in enumerate(feature_names):
            x = features[i, idx, f_idx]
            if np.std(x) > 0:
                rho, _ = stats.spearmanr(x, r)
                if not np.isnan(rho):
                    per_group_abs[f_name].append(abs(float(rho)))
    avg_abs = {
        f_name: float(np.mean(v)) if v else 0.0
        for f_name, v in per_group_abs.items()
    }
    return {"global_spearman": global_corr, "avg_abs_spearman_per_group": avg_abs}


def _distribution_shift(
    train: Dict[str, np.ndarray],
    val: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, data in [("train", train), ("val", val), ("test", test)]:
        mask = data["mask"].astype(bool)
        returns = data["returns"][mask]
        out[name] = {
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "current_block_rate": float(data["current_block"][mask].mean()),
        }
    out["val_train_return_diff"] = out["val"]["return_mean"] - out["train"]["return_mean"]
    out["test_train_return_diff"] = out["test"]["return_mean"] - out["train"]["return_mean"]
    return out


def _future_trace_swap_stability(
    data: Dict[str, np.ndarray], n_samples: int = 200, seed: int = 12345
) -> Dict[str, Any]:
    """Replace each sampled group's future trace with a donor group's future trace.

    This proxies the stability of candidate rankings under different common-future
    request sequences while keeping the immediate action (current_block, immediate
    delay/FS) fixed.
    """
    rng = np.random.RandomState(seed)
    mask = data["mask"].astype(bool)
    returns = data["returns"].astype(np.float32)
    n_groups = mask.shape[0]
    cand_counts = mask.sum(axis=1)
    valid = np.flatnonzero(mask.any(axis=1))
    if len(valid) == 0:
        return {"kendall_tau_b_mean": 0.0, "top1_agreement": 0.0, "n_samples": 0}

    samples = rng.choice(valid, size=min(n_samples, len(valid)), replace=False)
    taus = []
    top1_agree = []

    for i in samples:
        n_cand = int(cand_counts[i])
        if n_cand < 2:
            continue
        # Pick a donor with the same number of candidates.
        donors = np.flatnonzero(cand_counts == n_cand)
        donors = donors[donors != i]
        if len(donors) == 0:
            continue
        j = int(rng.choice(donors))

        r_orig = returns[i, :n_cand]
        # Build alternative return: keep current block/immediate delay/fs, swap future.
        cb = data["current_block"][i, :n_cand]
        imm_delay = data["immediate_delay"][i, :n_cand]
        imm_fs = data["immediate_fs"][i, :n_cand]

        fb = np.nansum(data["future_blocked_per_step"][j, :n_cand], axis=1)
        fn = np.nansum(data["future_nsb_per_step"][j, :n_cand], axis=1)
        fd_sum = np.nansum(data["future_delay_per_step"][j, :n_cand], axis=1)
        ffs_sum = np.nansum(data["future_fs_per_step"][j, :n_cand], axis=1)
        bp = data["future_blocked_per_step"][j, :n_cand]
        success = np.nansum((bp == 0.0) & (~np.isnan(bp)), axis=1).astype(np.float32)
        fd = fd_sum / np.maximum(success, 1.0)
        ffs = ffs_sum / np.maximum(success, 1.0)

        r_alt = (
            -3.0 * cb
            - 4.0 * fb
            - 3.0 * fn
            - 0.03 * fd
            - 0.05 * ffs
        )

        rank_orig = stats.rankdata(-r_orig)
        rank_alt = stats.rankdata(-r_alt)
        tau, _ = stats.kendalltau(rank_orig, rank_alt)
        if not np.isnan(tau):
            taus.append(float(tau))
        top1_agree.append(int(np.argmax(r_orig) == np.argmax(r_alt)))

    return {
        "kendall_tau_b_mean": float(np.mean(taus)) if taus else 0.0,
        "kendall_tau_b_std": float(np.std(taus)) if taus else 0.0,
        "top1_agreement": float(np.mean(top1_agree)) if top1_agree else 0.0,
        "n_samples": len(taus),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    dataset_dir = root / "sa_hmarl" / "datasets" / "r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium"
    if not (dataset_dir / "metadata.json").exists():
        print(f"[learnability] Dataset metadata not found: {dataset_dir / 'metadata.json'}")
        return 1

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_names = list(metadata.get("feature_names", []))
    coefs = dict(metadata.get("label_coefs", metadata.get("return_coefs", {})))
    coefs["H"] = int(metadata.get("H", 5))

    train = _load_split(dataset_dir, "train")
    val = _load_split(dataset_dir, "val")
    test = _load_split(dataset_dir, "test")

    # Sanity-check reconstruction.
    recon = _reconstruct_returns(train, coefs)
    diff = np.abs(recon - train["returns"])[train["mask"].astype(bool)].max()
    if diff > 1e-5:
        print(f"[learnability] WARNING: return reconstruction max diff = {diff:.3e}")

    split_stats = {}
    for name, data in [("train", train), ("val", val), ("test", test)]:
        split_stats[name] = _split_statistics(data, feature_names, coefs)
        split_stats[name]["feature_correlations"] = _feature_return_correlations(data, feature_names)

    shift = _distribution_shift(train, val, test)
    donor_proxy = _future_trace_swap_stability(train, n_samples=200)

    payload = {
        "dataset_dir": str(dataset_dir),
        "metadata": {
            "K_C": metadata.get("K_C"),
            "K_path": metadata.get("K_path"),
            "K_prop": metadata.get("K_prop"),
            "H": metadata.get("H"),
            "label_coefs": coefs,
        },
        "return_reconstruction_max_diff": float(diff),
        "split_statistics": split_stats,
        "distribution_shift": shift,
        "donor_group_index_aligned_proxy": donor_proxy,
    }

    exp_dir = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking"
    exp_dir.mkdir(parents=True, exist_ok=True)
    json_path = exp_dir / "MEDIUM_LABEL_LEARNABILITY.json"
    md_path = exp_dir / "MEDIUM_LABEL_LEARNABILITY.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# SA-HMARL v1.3 Medium Dataset Label Learnability Analysis",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- K_C={metadata.get('K_C')}, K_path={metadata.get('K_path')}, K_prop={metadata.get('K_prop')}, H={metadata.get('H')}",
        f"- Return reconstruction max diff: {diff:.3e}",
        "",
        "## Split overview",
        "",
        "| Split | Groups | Valid groups | Avg cand | Return mean | Return std | Current block % | Future block % | Future NSB % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["train", "val", "test"]:
        s = split_stats[name]
        lines.append(
            f"| {name} | {s['groups']} | {s['valid_groups']} | {s['avg_candidates']:.2f} | "
            f"{s['label_distribution']['mean']:.4f} | {s['label_distribution']['std']:.4f} | "
            f"{s['current_block_rate']:.2%} | {s['future_block_rate']:.2%} | {s['future_nsb_rate']:.2%} |"
        )

    lines.extend(["", "## Component variance (mean over groups with variance)", ""])
    lines.append("| Component | Mean within-group variance | Explained frac of return var |")
    lines.append("|---|---:|---:|")
    for k in ["current_block", "future_block", "future_nsb", "future_delay", "future_fs"]:
        mv = split_stats["train"]["component_variance_mean"][k]
        ef = split_stats["train"]["explained_variance_frac"][k]
        lines.append(f"| {k} | {mv:.6f} | {ef:.2%} |")

    h = split_stats["train"]["headroom"]
    lines.extend(["", "## Headroom taxonomy (train)", ""])
    lines.append(f"- Groups with candidate variance: {h['groups_with_variance']}")
    lines.append(f"- Future blocking identical across candidates: {h['identical_future_block_pp']:.2%}")
    lines.append(f"- Only delay/FS differ (blocking identical): {h['only_delay_fs_pp']:.2%}")
    lines.append(f"- Future blocking differs across candidates: {h['future_block_differs_pp']:.2%}")
    lines.append(f"- PPO Top-1 outside oracle tie set: {h['ppo_not_in_oracle_tie_pp']:.2%}")
    lines.append(f"- PPO regret mean / P50 / P90: {h['ppo_regret_mean']:.4f} / {h['ppo_regret_p50']:.4f} / {h['ppo_regret_p90']:.4f}")

    lines.extend(["", "## Distribution shift", ""])
    lines.append(f"- val return mean - train: {shift['val_train_return_diff']:.4f}")
    lines.append(f"- test return mean - train: {shift['test_train_return_diff']:.4f}")

    lines.extend(["", "## Feature vs return correlations (train)", "", "| Feature | Global Spearman | Avg |Abs| per group |", "|---|---:|---:|"])
    gc = split_stats["train"]["feature_correlations"]["global_spearman"]
    ag = split_stats["train"]["feature_correlations"]["avg_abs_spearman_per_group"]
    for f_name in feature_names:
        lines.append(f"| {f_name} | {gc.get(f_name, 0):.3f} | {ag.get(f_name, 0):.3f} |")

    lines.extend(["", "## Donor-group future-trace proxy (NOT a true multi-trace stability test)", ""])
    lines.append("- This proxy replaces a group's future components with those of another group that happens to have the same candidate count, then realigns by candidate index.")
    lines.append("- Different groups usually have different candidate action IDs, so this does **not** fix the state or the action; it is only a coarse sanity check.")
    lines.append(f"- Samples: {donor_proxy['n_samples']}")
    lines.append(f"- Mean Kendall tau-b: {donor_proxy['kendall_tau_b_mean']:.3f} ± {donor_proxy['kendall_tau_b_std']:.3f}")
    lines.append(f"- Top-1 agreement: {donor_proxy['top1_agreement']:.2%}")

    lines.extend(["", "## Interpretation cues", ""])
    if h["future_block_differs_pp"] < 0.1:
        lines.append("- Future blocking rarely differs across candidates; most candidate-return variance comes from delay/FS rather than future blocking.")
    lines.append("- The donor-group proxy above cannot be used to claim that the H=5 label is 'unstable under multiple future traces'. A true multi-trace experiment must fix the state and the action IDs while varying only the future request sequence.")
    if split_stats["train"]["feature_correlations"]["avg_abs_spearman_per_group"].get("raw_r_valid_ratio", 0) < 0.05:
        lines.append("- Per-group correlations between features and returns are weak; the 25-d pre-decision features may not strongly encode the ranking.")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[learnability] Wrote {md_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
