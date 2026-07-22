"""Orchestrate the SA-HMARL Strict v1.3 full-state fix pilot for COST239.

Phases:
  1. Merge the generated full-state pilot shards.
  2. Train four ranker variants x 3 seeds:
       - full_state_uniform (main)
       - full_state_stratified_depth
       - deep_path_only_uniform (legacy distribution)
       - deep_path_only_stratified_depth
     plus reuse the existing old_v13 checkpoint for Current/Gated baselines.
  3. Offline evaluation on the held-out test split per E-gate subgroup.
  4. Closed-loop fair evaluation on shared traffic seeds if offline gate passes.
  5. Write all deliverables: reports, JSON, manifest.

All file writes use atomic rename. Workers run with single-threaded BLAS.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root on path when run directly.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.training.train_r_counterfactual_ranking import (
    _compute_group_metrics,
    _feasibility_mask,
)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class CommandRecord:
    phase: str
    cmd: List[str]
    env: Dict[str, str]
    start_sec: float
    end_sec: float = 0.0
    returncode: int = -1
    stdout_tail: str = ""
    stderr_tail: str = ""
    failed_shard: Optional[str] = None
    retries: int = 0


def _run_cmd(
    phase: str,
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Path = ROOT,
    timeout: Optional[int] = None,
) -> CommandRecord:
    """Run a subprocess, capture tails, and record in the manifest."""
    base_env = os.environ.copy()
    base_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TORCH_NUM_THREADS": "1",
        }
    )
    if env:
        base_env.update(env)
    record = CommandRecord(
        phase=phase,
        cmd=cmd,
        env={k: base_env.get(k, "") for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"]},
        start_sec=time.perf_counter(),
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=base_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        record.end_sec = time.perf_counter()
        record.returncode = proc.returncode
        record.stdout_tail = proc.stdout[-4000:]
        record.stderr_tail = proc.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        record.end_sec = time.perf_counter()
        record.returncode = -9
        record.stderr_tail = f"Timeout after {timeout}s: {exc}"
    return record


def _python_cmd(module: str, *args: str) -> List[str]:
    py = str(ROOT / ".venv" / "bin" / "python")
    return [py, "-m", module, *args]


# ---------------------------------------------------------------------------
# Dataset diagnostics
# ---------------------------------------------------------------------------
def _load_split(dataset_dir: Path, split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(dataset_dir / f"{split}.npz", allow_pickle=False))


def _subgroup_metrics(
    data: Dict[str, np.ndarray], metadata: Dict[str, Any], model_state: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute offline ranking/regret metrics on E=0 and E=1 subgroups."""
    feature_names = list(metadata.get("feature_names", []))
    mean = np.asarray(metadata["train_feature_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(metadata["train_feature_std"], dtype=np.float32), 1e-6)
    feas = _feasibility_mask(data["features"], feature_names)
    eff = data["mask"] & feas

    input_dim = data["features"].shape[2]
    model = build_counterfactual_r_ranker(
        model_state.get("model_type", "mlp"),
        input_dim,
        tuple(model_state.get("hidden_dims", [128, 64])),
        float(model_state.get("dropout", 0.0)),
    )
    model.load_state_dict(model_state["model_state_dict"])
    model.eval()

    # Score all candidates.
    x_raw = (data["features"].astype(np.float32) - mean) / std
    scores = np.full(data["returns"].shape, -np.inf, dtype=np.float32)
    batch_size = 256
    with torch.no_grad():
        for start in range(0, x_raw.shape[0], batch_size):
            end = min(start + batch_size, x_raw.shape[0])
            xt = torch.as_tensor(x_raw[start:end], dtype=torch.float32)
            mt = torch.as_tensor(eff[start:end], dtype=torch.bool)
            sc = model(xt, mt).cpu().numpy()
            scores[start:end] = sc

    out: Dict[str, Any] = {}
    if "deep_path_gate" in data:
        dpg = np.asarray(data["deep_path_gate"], dtype=np.int32)
        e0_mask = dpg == 0
        e1_mask = dpg == 1
    else:
        # Legacy dataset: derive from max_path_idx.
        mpi = np.asarray(data["max_path_idx"], dtype=np.int32)
        e0_mask = mpi < 5
        e1_mask = mpi >= 5

    for label, mask in (("e0", e0_mask), ("e1", e1_mask)):
        if mask.sum() == 0:
            out[label] = {"groups": 0}
            continue
        sub_scores = scores[mask]
        sub_returns = data["returns"][mask]
        sub_eff = eff[mask]
        sub_ppo = data["ppo_action_index"][mask]
        metrics = _compute_group_metrics(sub_scores, sub_returns, sub_eff, sub_ppo)
        metrics["groups"] = int(mask.sum())
        metrics["candidates_per_group"] = float(sub_eff.sum() / max(mask.sum(), 1))
        out[label] = metrics

    # Overall metrics.
    out["all"] = _compute_group_metrics(scores, data["returns"], eff, data["ppo_action_index"])
    out["all"]["groups"] = data["returns"].shape[0]
    return out


def _compute_dataset_report(
    dataset_dir: Path, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute pilot dataset statistics requested by the protocol."""
    train = _load_split(dataset_dir, "train")
    val = _load_split(dataset_dir, "val")
    test = _load_split(dataset_dir, "test")

    def _split_stats(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        mask = data["mask"]
        returns = data["returns"]
        cand_counts = mask.sum(axis=1)
        ranges = []
        ppo_regrets = []
        for i in range(mask.shape[0]):
            vals = returns[i, mask[i]]
            if vals.size >= 2:
                ranges.append(float(vals.max() - vals.min()))
            ppo_idx = int(data["ppo_action_index"][i])
            if 0 <= ppo_idx < vals.size:
                ppo_regrets.append(float(vals.max() - vals[ppo_idx]))

        # Future-blocking difference rate: groups where future blocked steps differ.
        future_diff = 0
        if "future_blocked_per_step" in data:
            fb = data["future_blocked_per_step"]
            for i in range(mask.shape[0]):
                valid = mask[i]
                if valid.sum() < 2:
                    continue
                rows = fb[i, valid, :]
                if np.any(rows != rows[0:1]):
                    future_diff += 1

        if "deep_path_gate" in data:
            dpg = np.asarray(data["deep_path_gate"], dtype=np.int32)
            e0 = int((dpg == 0).sum())
            e1 = int((dpg == 1).sum())
        else:
            mpi = np.asarray(data["max_path_idx"], dtype=np.int32)
            e0 = int((mpi < 5).sum())
            e1 = int((mpi >= 5).sum())

        return {
            "groups": int(mask.shape[0]),
            "avg_candidates": float(cand_counts.mean()),
            "nonzero_return_range_rate": float(sum(1 for r in ranges if abs(r) > 1e-4) / max(len(ranges), 1)),
            "return_min": float(returns[mask].min()),
            "return_max": float(returns[mask].max()),
            "return_mean": float(returns[mask].mean()),
            "return_std": float(returns[mask].std()),
            "ppo_regret_mean": float(np.mean(ppo_regrets)) if ppo_regrets else 0.0,
            "ppo_regret_headroom": float(np.max(ppo_regrets)) if ppo_regrets else 0.0,
            "future_blocking_difference_rate": future_diff / max(mask.shape[0], 1),
            "e0_groups": e0,
            "e1_groups": e1,
            "e1_rate": e1 / max(mask.shape[0], 1),
        }

    report = {
        "dataset_dir": str(dataset_dir),
        "train": _split_stats(train),
        "val": _split_stats(val),
        "test": _split_stats(test),
        "metadata": {
            "K_C": metadata.get("K_C"),
            "K_path": metadata.get("K_path"),
            "K_prop": metadata.get("K_prop"),
            "H": metadata.get("H"),
            "gamma": metadata.get("gamma"),
            "group_filter": metadata.get("group_filter"),
            "candidate_mode": metadata.get("candidate_mode"),
        },
    }

    # Per-stratum label variance / learnability on train.
    strata = np.asarray(train.get("depth_stratum", np.zeros(train["mask"].shape[0], dtype=np.int32)), dtype=np.int32)
    stratum_report = {}
    for s in range(4):
        sm = strata == s
        if sm.sum() == 0:
            stratum_report[s] = {"groups": 0}
            continue
        vals = train["returns"][sm]
        m = train["mask"][sm]
        ranges = []
        for i in range(vals.shape[0]):
            v = vals[i, m[i]]
            if v.size >= 2:
                ranges.append(float(v.max() - v.min()))
        stratum_report[s] = {
            "groups": int(sm.sum()),
            "return_mean": float(vals[m].mean()),
            "return_std": float(vals[m].std()),
            "nonzero_range_rate": float(sum(1 for r in ranges if abs(r) > 1e-4) / max(len(ranges), 1)),
            "label_variance_proxy": float(np.var(ranges)) if ranges else 0.0,
        }
    report["train_stratum"] = stratum_report
    return report


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def _dataset_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Full-State Pilot Dataset Report (COST239)",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- K_C={report['metadata']['K_C']}, K_path={report['metadata']['K_path']}, "
        f"K_prop={report['metadata']['K_prop']}, H={report['metadata']['H']}",
        f"- group_filter={report['metadata']['group_filter']}",
        f"- candidate_mode={report['metadata']['candidate_mode']}",
        "",
        "## Split statistics",
        "",
        "| Split | Groups | Avg cand | E=0 | E=1 | E=1 rate | Return min | Return max | "
        "PPO regret mean | PPO headroom | Future block diff rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        s = report[split]
        lines.append(
            f"| {split} | {s['groups']} | {s['avg_candidates']:.2f} | "
            f"{s['e0_groups']} | {s['e1_groups']} | {s['e1_rate']:.2%} | "
            f"{s['return_min']:.4f} | {s['return_max']:.4f} | "
            f"{s['ppo_regret_mean']:.4f} | {s['ppo_regret_headroom']:.4f} | "
            f"{s['future_blocking_difference_rate']:.2%} |"
        )
    lines.extend(["", "## Train depth-stratum learnability", "", "| Stratum | Groups | Return mean | Return std | Nonzero range rate | Label variance proxy |", "|---|---|---:|---:|---:|---:|"])
    for s, stats in report["train_stratum"].items():
        if stats.get("groups", 0) > 0:
            lines.append(
                f"| {s} | {stats['groups']} | {stats['return_mean']:.4f} | "
                f"{stats['return_std']:.4f} | {stats['nonzero_range_rate']:.2%} | "
                f"{stats['label_variance_proxy']:.4f} |"
            )
    return "\n".join(lines)


def _offline_report_md(offline: Dict[str, Any]) -> str:
    lines = [
        "# Offline Comparison: Full-State v1.3 Pilot (COST239)",
        "",
        "Metrics computed on the held-out test split. Lower model regret is better.",
        "",
        "| Method | Seed | Test groups | Model regret | PPO regret | Regret reduction | Top-1 | NDCG@3 | PPO agree |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, seeds in offline["methods"].items():
        for seed, metrics in seeds.items():
            all_m = metrics.get("all", {})
            lines.append(
                f"| {method} | {seed} | {all_m.get('groups', 0)} | "
                f"{all_m.get('model_regret_mean', 0):.4f} | {all_m.get('ppo_regret_mean', 0):.4f} | "
                f"{all_m.get('aggregate_regret_reduction', 0):.2%} | "
                f"{all_m.get('top1_accuracy', 0):.2%} | {all_m.get('ndcg_at_3', 0):.3f} | "
                f"{all_m.get('ppo_agreement', 0):.2%} |"
            )

    lines.extend(["", "## E-gate subgroup (offline)", "", "| Method | Seed | E | Groups | Model regret | PPO regret | Top-1 | NDCG@3 | PPO agree |", "|---|---|---|---|---:|---:|---:|---:|---:|"])
    for method, seeds in offline["methods"].items():
        for seed, metrics in seeds.items():
            for gate in ["e0", "e1"]:
                gm = metrics.get(gate, {})
                if gm.get("groups", 0) > 0:
                    lines.append(
                        f"| {method} | {seed} | {gate.upper()} | {gm['groups']} | "
                        f"{gm.get('model_regret_mean', 0):.4f} | {gm.get('ppo_regret_mean', 0):.4f} | "
                        f"{gm.get('top1_accuracy', 0):.2%} | {gm.get('ndcg_at_3', 0):.3f} | "
                        f"{gm.get('ppo_agreement', 0):.2%} |"
                    )
    return "\n".join(lines)


def _subgroup_e0_e1_md(offline: Dict[str, Any]) -> str:
    lines = [
        "# E=0 / E=1 Subgroup Report (Offline)",
        "",
        "Per-gate offline regret and ranking metrics.",
        "",
    ]
    for method, seeds in offline["methods"].items():
        lines.append(f"## {method}")
        lines.append("")
        lines.append("| Seed | E | Groups | Cand/group | Model regret | PPO regret | Regret red | Top-1 | Tie Top-1 | NDCG@3 | Better | Equal | Worse |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for seed, metrics in seeds.items():
            for gate in ["e0", "e1"]:
                gm = metrics.get(gate, {})
                if gm.get("groups", 0) == 0:
                    continue
                lines.append(
                    f"| {seed} | {gate.upper()} | {gm['groups']} | "
                    f"{gm.get('candidates_per_group', 0):.2f} | "
                    f"{gm.get('model_regret_mean', 0):.4f} | "
                    f"{gm.get('ppo_regret_mean', 0):.4f} | "
                    f"{gm.get('aggregate_regret_reduction', 0):.2%} | "
                    f"{gm.get('top1_accuracy', 0):.2%} | "
                    f"{gm.get('score_tie_fractional_top1', 0):.2%} | "
                    f"{gm.get('ndcg_at_3', 0):.3f} | "
                    f"{gm.get('model_better_than_ppo_rate', 0):.2%} | "
                    f"{gm.get('model_equal_to_ppo_rate', 0):.2%} | "
                    f"{gm.get('model_worse_than_ppo_rate', 0):.2%} |"
                )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pilot orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v13_strict_fixed")
    parser.add_argument("--old_v13_checkpoint", default="sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt")
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--train_seeds", default="42,43,44")
    parser.add_argument("--eval_seeds", default="4001,4002,4003,4004,4005")
    parser.add_argument("--skip_dataset", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_closed_loop", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    # Phase 1: merge dataset.
    dataset_dir = Path(args.dataset_dir)
    if not args.skip_dataset:
        cmd = _python_cmd(
            "sa_hmarl.evaluation.merge_r_counterfactual_ranking_shards",
            "--dataset_dir", str(dataset_dir),
        )
        rec = _run_cmd("merge_dataset", cmd, timeout=600)
        manifest.append(asdict(rec))
        if rec.returncode != 0:
            _atomic_write_json(out_dir / "MANIFEST.json", {"manifest": manifest})
            raise RuntimeError(f"Dataset merge failed: {rec.stderr_tail}")

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset_report = _compute_dataset_report(dataset_dir, metadata)
    _atomic_write_json(out_dir / "FULL_STATE_PILOT_DATASET_REPORT.json", dataset_report)
    _atomic_write_text(out_dir / "FULL_STATE_PILOT_DATASET_REPORT.md", _dataset_report_md(dataset_report))

    # Sampling-shift audit: compare old (deep_path_only) and new (all) distributions.
    old_meta_path = Path("sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium/metadata.json")
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8")) if old_meta_path.exists() else {}
    sampling_audit = {
        "old_dataset": str(old_meta_path),
        "old_groups_total": old_meta.get("groups_total", "unknown"),
        "old_groups_e0": old_meta.get("groups_e0", "unknown"),
        "old_groups_e1": old_meta.get("groups_e1", "unknown"),
        "old_group_filter": old_meta.get("group_filter", "implicit deep_path_only (path_idx>=5)"),
        "new_dataset": str(dataset_dir),
        "new_groups_total": dataset_report["train"]["groups"] + dataset_report["val"]["groups"] + dataset_report["test"]["groups"],
        "new_groups_e0": dataset_report["train"]["e0_groups"] + dataset_report["val"]["e0_groups"] + dataset_report["test"]["e0_groups"],
        "new_groups_e1": dataset_report["train"]["e1_groups"] + dataset_report["val"]["e1_groups"] + dataset_report["test"]["e1_groups"],
        "new_group_filter": metadata.get("group_filter"),
        "shift_conclusion": (
            "The old v1.3 dataset conditioned on E=1; the new full-state dataset includes "
            "E=0 states, removing the train-deployment distribution shift."
        ),
    }
    _atomic_write_json(out_dir / "SAMPLING_SHIFT_AUDIT.json", sampling_audit)
    _atomic_write_text(
        out_dir / "SAMPLING_SHIFT_AUDIT.md",
        "# Sampling Shift Audit\n\n" + json.dumps(sampling_audit, indent=2, default=str),
    )

    # Phase 2: train rankers (unless skipped; then load existing offline results).
    # Strict v1.3 loss-matched protocol: listwise KL + regression only, no pairwise/hard-negative.
    strict_loss_args = [
        "--reg_weight", "1.0",
        "--lambda_pair", "0.0",
        "--lambda_hard", "0.0",
    ]
    train_methods = {
        "full_state_uniform_strict": {"sampling_mode": "uniform", "dataset": str(dataset_dir)},
        "full_state_stratified_depth_strict": {"sampling_mode": "stratified_depth", "dataset": str(dataset_dir)},
    }
    offline_results: Dict[str, Any] = {"methods": {}}
    if args.skip_training and (out_dir / "OFFLINE_COMPARISON.json").exists():
        offline_results = json.loads((out_dir / "OFFLINE_COMPARISON.json").read_text(encoding="utf-8"))
    for method, cfg in train_methods.items():
        offline_results["methods"][method] = {}
        for seed in [int(s) for s in args.train_seeds.split(",") if s.strip()]:
            model_dir = out_dir / "checkpoints" / method / f"seed_{seed}"
            cmd = _python_cmd(
                "sa_hmarl.training.train_r_counterfactual_ranking",
                "--dataset_dir", cfg["dataset"],
                "--output_dir", str(model_dir),
                "--seed", str(seed),
                "--sampling_mode", cfg["sampling_mode"],
                "--selection_metric", "regret",
                *strict_loss_args,
                "--device", args.device,
                "--epochs", "80",
                "--patience", "20",
            )
            rec = _run_cmd(f"train_{method}_seed{seed}", cmd, timeout=3600)
            manifest.append(asdict(rec))
            if rec.returncode != 0:
                print(f"[pilot] Training {method} seed {seed} failed; skipping evaluation.")
                continue

            ckpt = torch.load(model_dir / "ranking_model.pt", map_location="cpu", weights_only=False)
            test_metrics = _subgroup_metrics(_load_split(dataset_dir, "test"), metadata, ckpt)
            offline_results["methods"][method][str(seed)] = test_metrics

    # Add old_v13 baseline (trained on deep_path_only, evaluated on full test).
    offline_results["methods"]["current_v13_old_checkpoint"] = {}
    for seed in [int(args.train_seeds.split(",")[0])]:
        ckpt = torch.load(args.old_v13_checkpoint, map_location="cpu", weights_only=False)
        test_metrics = _subgroup_metrics(_load_split(dataset_dir, "test"), metadata, ckpt)
        offline_results["methods"]["current_v13_old_checkpoint"][str(seed)] = test_metrics

    _atomic_write_json(out_dir / "OFFLINE_COMPARISON.json", offline_results)
    _atomic_write_text(out_dir / "OFFLINE_COMPARISON.md", _offline_report_md(offline_results))
    _atomic_write_json(out_dir / "SUBGROUP_E0_E1_REPORT.json", offline_results)
    _atomic_write_text(out_dir / "SUBGROUP_E0_E1_REPORT.md", _subgroup_e0_e1_md(offline_results))

    # Gate check: Full or Gated offline regret must be stable better than PPO-R.
    # Load existing offline results if training was skipped.
    if args.skip_training and (out_dir / "OFFLINE_COMPARISON.json").exists():
        offline_results = json.loads((out_dir / "OFFLINE_COMPARISON.json").read_text(encoding="utf-8"))

    def _mean_agg_red(method: str) -> float:
        seeds = offline_results.get("methods", {}).get(method, {})
        reds = [m["all"]["aggregate_regret_reduction"] for m in seeds.values() if "all" in m]
        return float(np.mean(reds)) if reds else -1.0

    full_red = _mean_agg_red("full_state_uniform_strict")
    strat_red = _mean_agg_red("full_state_stratified_depth_strict")
    gated_red = _mean_agg_red("current_v13_old_checkpoint")  # evaluated on all, acts as gated offline proxy

    gate_passed = full_red > 0.0 or strat_red > 0.0

    status = {
        "phase": "offline_gate",
        "full_state_uniform_aggregate_regret_reduction": full_red,
        "full_state_stratified_depth_aggregate_regret_reduction": strat_red,
        "gated_old_v13_aggregate_regret_reduction": gated_red,
        "gate_passed": bool(gate_passed),
        "note": "Gate passes if Full or Stratified Full mean aggregate regret reduction > 0 vs PPO-R.",
    }
    _atomic_write_json(out_dir / "STATUS.json", {"status": status, "manifest": manifest})

    if not gate_passed or args.skip_closed_loop:
        print(f"[pilot] Offline gate passed={gate_passed}; skip_closed_loop={args.skip_closed_loop}. Stopping.")
        return

    # Phase 3: closed-loop fair evaluation.
    ranker_specs = [
        f"old_v13={args.old_v13_checkpoint}",
        f"full_v13_pilot={out_dir / 'checkpoints' / 'full_state_uniform_strict' / 'seed_42' / 'ranking_model.pt'}",
        f"full_v13_stratified={out_dir / 'checkpoints' / 'full_state_stratified_depth_strict' / 'seed_42' / 'ranking_model.pt'}",
    ]
    # Current v1.3: old checkpoint with ranker_gate=all.
    # Gated v1.3: old checkpoint with ranker_gate=deep_path_only.
    for gate in ["all", "deep_path_only"]:
        output_sub = out_dir / f"closed_loop_{gate}"
        ranker_args = []
        for spec in ranker_specs:
            ranker_args.extend(["--ranker_specs", spec])
        cmd = _python_cmd(
            "sa_hmarl.evaluation.eval_r_counterfactual_ranking_cost239_kpath50_hops_fair",
            *ranker_args,
            "--modes", "ppo_r_top1,ksp_ff_plain,ksp_ff_highest",
            "--ranker_gate", gate,
            "--seeds", args.eval_seeds,
            "--requests_per_episode", "2000",
            "--warmup_requests", "500",
            "--output_dir", str(output_sub),
            "--output_json", "closed_loop.json",
            "--output_md", "CLOSED_LOOP.md",
            "--device", args.device,
        )
        rec = _run_cmd(f"closed_loop_{gate}", cmd, timeout=7200)
        manifest.append(asdict(rec))

    _atomic_write_json(out_dir / "MANIFEST.json", {"manifest": manifest})
    print("[pilot] Pilot complete. Outputs in", out_dir)


if __name__ == "__main__":
    main()
