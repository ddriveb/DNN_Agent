"""Part 6: top-1 vs closed-loop gap analysis."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import stats


DATASET_DIR = Path("sa_hmarl/datasets/v135_paired_feature_ablation_medium")
CHECKPOINT_DIR = Path("sa_hmarl/experiments/v135_parallel_paired")
CLOSED_LOOP_JSON = Path("sa_hmarl/experiments/v135_parallel_paired/FEATURE_ONLY_CLOSED_LOOP.json")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_ranker(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"),
        int(ckpt.get("input_dim", ckpt.get("feature_dim"))),
        tuple(ckpt.get("hidden_dims", (128, 64))),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    return model, mean, std


def _score_groups(features: np.ndarray, mask: np.ndarray, model, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = (features - mean) / std
    X_t = torch.tensor(X[mask], dtype=torch.float32)
    with torch.no_grad():
        scores_valid = model(X_t).cpu().numpy().ravel()
    scores = np.full(features.shape[:2], -1e9, dtype=np.float32)
    scores[mask] = scores_valid
    return scores


def _per_group_analysis(features: np.ndarray, returns: np.ndarray, mask: np.ndarray, model, mean, std) -> Dict[str, Any]:
    scores = _score_groups(features, mask, model, mean, std)
    groups = []
    gap_buckets = {"[0,1e-4)": [], "[1e-4,1e-3)": [], "[1e-3,1e-2)": [], ">=1e-2": []}
    for i in range(features.shape[0]):
        m = mask[i]
        if m.sum() < 2:
            continue
        r = returns[i, m]
        s = scores[i, m]
        sorted_r = np.sort(r)[::-1]
        gap = float(sorted_r[0] - sorted_r[1])
        best_return = float(r.max())
        pred_idx = int(np.argmax(s))
        pred_return = float(r[pred_idx])
        regret = best_return - pred_return
        top1 = int(pred_idx in np.where(r == best_return)[0])
        g = {"best_return": best_return, "pred_return": pred_return, "regret": regret, "top1": top1, "gap": gap}
        groups.append(g)
        if gap < 1e-4:
            gap_buckets["[0,1e-4)"].append(g)
        elif gap < 1e-3:
            gap_buckets["[1e-4,1e-3)"].append(g)
        elif gap < 1e-2:
            gap_buckets["[1e-3,1e-2)"].append(g)
        else:
            gap_buckets[">=1e-2"].append(g)

    bucket_summary = {}
    for bname, bgroups in gap_buckets.items():
        if not bgroups:
            continue
        bucket_summary[bname] = {
            "n": len(bgroups),
            "top1_rate": float(np.mean([g["top1"] for g in bgroups])),
            "mean_regret": float(np.mean([g["regret"] for g in bgroups])),
            "mean_gap": float(np.mean([g["gap"] for g in bgroups])),
        }
    return {"groups": groups, "bucket_summary": bucket_summary}


def _action_decode(action_idx: int, num_mods: int, max_blocks: int) -> Tuple[int, int, int]:
    path_idx = action_idx // (num_mods * max_blocks)
    rem = action_idx % (num_mods * max_blocks)
    mod_idx = rem // max_blocks
    block_idx = rem % max_blocks
    return path_idx, mod_idx, block_idx


def _disagreement_analysis(v1_features: np.ndarray, ps_features: np.ndarray, returns: np.ndarray, mask: np.ndarray,
                           v1_model, v1_mean, v1_std, ps_model, ps_mean, ps_std,
                           num_mods: int, max_blocks: int) -> Dict[str, Any]:
    v1_scores = _score_groups(v1_features, mask, v1_model, v1_mean, v1_std)
    ps_scores = _score_groups(ps_features, mask, ps_model, ps_mean, ps_std)
    disagreements = 0
    path_diff = []
    mod_diff = []
    block_diff = []
    regret_diff = []
    for i in range(v1_features.shape[0]):
        m = mask[i]
        if m.sum() < 2:
            continue
        r = returns[i, m]
        v1_idx = int(np.argmax(v1_scores[i, m]))
        ps_idx = int(np.argmax(ps_scores[i, m]))
        if v1_idx != ps_idx:
            disagreements += 1
            v1_p, v1_m, v1_b = _action_decode(int(np.flatnonzero(m)[v1_idx]), num_mods, max_blocks)
            ps_p, ps_m_, ps_b = _action_decode(int(np.flatnonzero(m)[ps_idx]), num_mods, max_blocks)
            path_diff.append(abs(v1_p - ps_p))
            mod_diff.append(abs(v1_m - ps_m_))
            block_diff.append(abs(v1_b - ps_b))
            regret_diff.append(abs((r.max() - r[v1_idx]) - (r.max() - r[ps_idx])))
    total = int(mask.sum(axis=1).shape[0])
    return {
        "disagreement_rate": float(disagreements / total) if total else 0.0,
        "mean_path_diff": float(np.mean(path_diff)) if path_diff else 0.0,
        "mean_mod_diff": float(np.mean(mod_diff)) if mod_diff else 0.0,
        "mean_block_diff": float(np.mean(block_diff)) if block_diff else 0.0,
        "mean_regret_diff": float(np.mean(regret_diff)) if regret_diff else 0.0,
    }


def _bootstrap_ci(diffs: List[float], n_bootstrap: int = 10000) -> Tuple[float, float, float]:
    diffs = np.array(diffs)
    rng = np.random.RandomState(0)
    boot = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot.append(sample.mean())
    boot = np.sort(boot)
    return float(diffs.mean()), float(boot[int(0.025 * n_bootstrap)]), float(boot[int(0.975 * n_bootstrap)])


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    meta = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    test = np.load(str(DATASET_DIR / "test.npz"), allow_pickle=True)
    mask = test["mask"].astype(bool)
    returns = test["returns"].astype(np.float32)
    v1_features = test["features_v1"].astype(np.float32)
    ps_features = test["features_poststate_v1"].astype(np.float32)

    v1_model, v1_mean, v1_std = _load_ranker(CHECKPOINT_DIR / "ranker_v1_s456" / "ranking_model.pt")
    ps_model, ps_mean, ps_std = _load_ranker(CHECKPOINT_DIR / "ranker_poststate_v1_s456" / "ranking_model.pt")

    v1_analysis = _per_group_analysis(v1_features, returns, mask, v1_model, v1_mean, v1_std)
    ps_analysis = _per_group_analysis(ps_features, returns, mask, ps_model, ps_mean, ps_std)

    disag = _disagreement_analysis(
        v1_features, ps_features, returns, mask,
        v1_model, v1_mean, v1_std, ps_model, ps_mean, ps_std,
        num_mods=5, max_blocks=10,
    )

    # Closed-loop bootstrap
    cl = json.loads(CLOSED_LOOP_JSON.read_text(encoding="utf-8"))
    seeds = cl["summary"]["ranker_v1"]["seeds"]
    v1_rates = [row["blocking_rate"] for row in cl["summary"]["ranker_v1"]["per_seed"]]
    ps_rates = [row["blocking_rate"] for row in cl["summary"]["ranker_poststate_v1"]["per_seed"]]
    diffs = [v - p for v, p in zip(v1_rates, ps_rates)]
    mean_diff, ci_lo, ci_hi = _bootstrap_ci(diffs)

    out = {
        "v1_test_analysis": v1_analysis["bucket_summary"],
        "poststate_test_analysis": ps_analysis["bucket_summary"],
        "v1_overall": {
            "top1_rate": float(np.mean([g["top1"] for g in v1_analysis["groups"]])),
            "mean_regret": float(np.mean([g["regret"] for g in v1_analysis["groups"]])),
        },
        "poststate_overall": {
            "top1_rate": float(np.mean([g["top1"] for g in ps_analysis["groups"]])),
            "mean_regret": float(np.mean([g["regret"] for g in ps_analysis["groups"]])),
        },
        "disagreement": disag,
        "closed_loop_bootstrap": {
            "seeds": seeds,
            "ranker_v1_blocking_rates": v1_rates,
            "ranker_poststate_blocking_rates": ps_rates,
            "paired_diff_mean": float(mean_diff),
            "paired_diff_95ci_lo": ci_lo,
            "paired_diff_95ci_hi": ci_hi,
        },
    }
    (OUTPUT_DIR / "TOP1_CLOSED_LOOP_GAP_ANALYSIS.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Top-1 vs Closed-Loop Gap Analysis",
        "",
        f"Checkpoints: ranker_v1_s456, ranker_poststate_v1_s456",
        "",
        "## Overall test-set ranking performance",
        "",
        "| Model | Top-1 rate | Mean regret |",
        "|---|---:|---:|",
        f"| v1 | {out['v1_overall']['top1_rate']:.2%} | {out['v1_overall']['mean_regret']:.4f} |",
        f"| poststate_v1 | {out['poststate_overall']['top1_rate']:.2%} | {out['poststate_overall']['mean_regret']:.4f} |",
        "",
        "## Regret by top1-top2 return gap bucket (poststate_v1)",
        "",
        "| Gap bucket | Groups | Top-1 rate | Mean regret | Mean gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for bname, s in out["poststate_test_analysis"].items():
        lines.append(f"| {bname} | {s['n']} | {s['top1_rate']:.2%} | {s['mean_regret']:.4f} | {s['mean_gap']:.4f} |")
    lines += ["", "## v1 vs poststate action disagreement", "", "| Metric | Value |", "|---|---:|"]
    for k, v in out["disagreement"].items():
        lines.append(f"| {k} | {v:.4f} |")
    lines += ["", "## Closed-loop paired bootstrap (v1 - poststate blocking rate)", "", f"Mean diff: {mean_diff:.4%}", f"95% CI: [{ci_lo:.4%}, {ci_hi:.4%}]"]
    if ci_lo <= 0 <= ci_hi:
        lines.append("**Result:** CI includes 0 → difference is **not statistically significant** at α=0.05.")
    else:
        lines.append("**Result:** CI excludes 0 → difference is statistically significant.")
    lines.append("")

    (OUTPUT_DIR / "TOP1_CLOSED_LOOP_GAP_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Part 6 complete. Report written to", OUTPUT_DIR / "TOP1_CLOSED_LOOP_GAP_ANALYSIS.md")


if __name__ == "__main__":
    main()
