"""Merge per-shard .npz files from the v1.3 counterfactual generator into
per-split train/val/test .npz files and a training-ready metadata.json.

Reads the shards listed in ``metadata.json`` under ``dataset_dir/shards/``,
concatenates them by split, computes feature normalization statistics on the
training split, writes ``train.npz``/``val.npz``/``test.npz``, and updates the
top-level ``metadata.json`` with diagnostics and statistics.

Usage:
    PYTHONPATH=sa_hmarl python -m sa_hmarl.evaluation.merge_r_counterfactual_ranking_shards \
        --dataset_dir sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _concat_split(shards: List[Dict[str, Any]], dataset_dir: Path) -> Dict[str, np.ndarray]:
    """Load shard npz paths and concatenate arrays along the group axis.

    Object arrays (e.g. state_hash, trace_hash) are dropped because the
    training loader uses ``allow_pickle=False``.
    """
    pieces = []
    for shard in shards:
        npz_path = dataset_dir / str(shard["npz_path"])
        # Must allow pickle temporarily because shards contain object arrays.
        loaded = dict(np.load(str(npz_path), allow_pickle=True))
        # Drop object arrays before stacking.
        loaded = {k: v for k, v in loaded.items() if v.dtype != object}
        pieces.append(loaded)

    if not pieces:
        raise ValueError(f"No shard data for split")

    keys = list(pieces[0].keys())
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        arrays = [p[key] for p in pieces]
        out[key] = np.concatenate(arrays, axis=0)
    return out


def _split_diagnostics(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute per-split dataset diagnostics."""
    mask = np.asarray(data["mask"], dtype=bool)
    returns = np.asarray(data["returns"], dtype=np.float32)
    n_groups = mask.shape[0]
    cand_counts = mask.sum(axis=1)
    valid_groups = int((cand_counts >= 1).sum())

    # Nonzero return range rate among groups with >=2 candidates.
    ranges = []
    for i in range(n_groups):
        vals = returns[i, mask[i]]
        if vals.size >= 2:
            ranges.append(float(vals.max() - vals.min()))
    nonzero_range_rate = (
        float(sum(1 for r in ranges if abs(r) > 1e-4)) / max(len(ranges), 1)
    )

    # PPO top-1 matches the best-return candidate.
    ppo_top1_count = 0
    oracle_headroom_sum = 0.0
    for i in range(n_groups):
        vals = returns[i, mask[i]]
        if vals.size == 0:
            continue
        best_idx = int(vals.argmax())
        ppo_idx = int(data["ppo_action_index"][i])
        if 0 <= ppo_idx < vals.size and ppo_idx == best_idx:
            ppo_top1_count += 1
        oracle_headroom_sum += float(vals.max() - vals[ppo_idx]) if 0 <= ppo_idx < vals.size else 0.0

    # Depth-stratum and E-gate statistics (v1.3 fix).
    e0_groups = 0
    e1_groups = 0
    stratum_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    if "deep_path_gate" in data:
        dpg = np.asarray(data["deep_path_gate"], dtype=np.int32)
        e0_groups = int((dpg == 0).sum())
        e1_groups = int((dpg == 1).sum())
    if "depth_stratum" in data:
        ds = np.asarray(data["depth_stratum"], dtype=np.int32)
        for stratum in range(4):
            stratum_counts[stratum] = int((ds == stratum).sum())

    return {
        "groups": n_groups,
        "valid_groups": valid_groups,
        "avg_candidates": float(cand_counts.mean()),
        "p50_candidates": float(np.median(cand_counts)),
        "p90_candidates": float(np.percentile(cand_counts, 90)),
        "min_candidates": int(cand_counts.min()),
        "max_candidates": int(cand_counts.max()),
        "nonzero_return_range_rate": nonzero_range_rate,
        "ppo_top1_rate": ppo_top1_count / max(valid_groups, 1),
        "oracle_headroom_pp": oracle_headroom_sum / max(valid_groups, 1),
        "return_mean": float(returns[mask].mean()),
        "return_std": float(returns[mask].std()),
        "groups_e0": e0_groups,
        "groups_e1": e1_groups,
        "e1_rate": e1_groups / max(n_groups, 1),
        "depth_stratum_counts": stratum_counts,
    }


def _feature_stats(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    features = np.asarray(data["features"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=bool)
    valid = features[mask]
    return {
        "feature_mean": valid.mean(axis=0).tolist(),
        "feature_std": np.maximum(valid.std(axis=0), 1e-6).tolist(),
        "feature_min": valid.min(axis=0).tolist(),
        "feature_max": valid.max(axis=0).tolist(),
    }


def _label_distribution(returns: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    vals = returns[mask]
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p5": float(np.percentile(vals, 5)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
    }


def merge(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    shards = metadata.get("shards", [])
    # Also discover any shard metadata files on disk; this protects against
    # generator runs where completed shards were marked "skipped" and omitted.
    for shard_meta_path in sorted(dataset_dir.glob("shards/*.metadata.json")):
        try:
            sm = json.loads(shard_meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if sm.get("status") != "ok":
            continue
        if sm.get("shard_id") in {s.get("shard_id") for s in shards}:
            continue
        npz_path = dataset_dir / "shards" / f"{sm['shard_id']}.npz"
        sm["npz_path"] = str(npz_path.relative_to(dataset_dir))
        shards.append(sm)
    if not shards:
        raise ValueError("No shards found")

    split_shards: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for shard in shards:
        split = shard.get("split") or shard.get("shard_id", "").split("_")[0]
        if split in split_shards:
            split_shards[split].append(shard)

    split_data: Dict[str, Dict[str, np.ndarray]] = {}
    diagnostics: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for split in ["train", "val", "test"]:
        if not split_shards[split]:
            continue
        data = _concat_split(split_shards[split], dataset_dir)
        diag = _split_diagnostics(data)
        diag.update(_feature_stats(data))
        diag["label_distribution"] = _label_distribution(data["returns"], data["mask"])
        diagnostics[split] = diag
        split_data[split] = data
        _atomic_save_npz(dataset_dir / f"{split}.npz", **data)
        print(f"[merge] {split}: {diag['groups']} groups, avg_cand={diag['avg_candidates']:.2f}")

    # Normalization stats from training split.
    train_data = split_data["train"]
    train_features = np.asarray(train_data["features"], dtype=np.float32)
    train_mask = np.asarray(train_data["mask"], dtype=bool)
    valid_features = train_features[train_mask].astype(np.float64)
    feature_mean = valid_features.mean(axis=0).astype(np.float32).tolist()
    feature_std = np.maximum(valid_features.std(axis=0), 1e-6).astype(np.float32).tolist()

    # Preserve original metadata fields and add training-ready fields.
    merged = copy.deepcopy(metadata)
    merged["diagnostics"] = diagnostics
    merged["train_feature_mean"] = feature_mean
    merged["train_feature_std"] = feature_std
    merged["feature_dim"] = len(feature_mean)
    merged["max_candidates"] = metadata.get("max_candidates", metadata.get("K_prop"))
    merged["ppo_top_k"] = metadata.get("ppo_top_k", metadata.get("K_prop"))
    merged["candidate_mode"] = metadata.get("candidate_mode", "ppo_r_topk_only")
    merged["num_random_candidates"] = metadata.get("num_random_candidates", 0)
    merged["min_candidates"] = metadata.get("min_candidates", 1)
    merged["candidate_seed"] = metadata.get("candidate_seed", 12345)
    merged["horizon"] = metadata.get("H", metadata.get("horizon", 5))
    merged["gamma"] = metadata.get("gamma", 1.0)
    return_coefs = dict(metadata.get("label_coefs", metadata.get("return_coefs", {})))
    for key, default in [
        ("current_block", 0.0),
        ("future_block", 0.0),
        ("future_nsb", 0.0),
        ("future_server_overload", 0.0),
        ("future_optical", 0.0),
        ("future_overload", 0.0),
        ("future_other", 0.0),
        ("delay", 0.0),
        ("fs", 0.0),
        ("path_penalty", 0.0),
        ("fs_penalty", 0.0),
        ("return_mode", "legacy_total_return"),
    ]:
        return_coefs.setdefault(key, default)
    merged["return_coefs"] = return_coefs
    merged["merge_seconds"] = time.perf_counter() - t0
    merged["merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Backup original metadata and write merged version.
    (dataset_dir / "metadata_raw.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    _atomic_write_json(meta_path, merged)

    _write_report(dataset_dir / "DATASET_REPORT.md", merged)
    print(f"[merge] Updated {meta_path}")
    return merged


def _write_report(path: Path, metadata: Dict[str, Any]) -> None:
    label_coefs = metadata.get("return_coefs", metadata.get("label_coefs", {}))
    lines = [
        "# SA-HMARL v1.3 Counterfactual Ranking Dataset Report",
        "",
        f"- Dataset dir: `{metadata.get('config', {}).get('output_dir', '')}`",
        f"- Topology: `{metadata.get('topology')}`",
        f"- K_C={metadata.get('K_C')}, K_path={metadata.get('K_path')}, K_prop={metadata.get('K_prop')}, H={metadata.get('H')}",
        f"- Candidate mode: `{metadata.get('candidate_mode')}`",
        f"- Path sort (C/R): `{metadata.get('path_sort_strategy_c')}` / `{metadata.get('path_sort_strategy_r')}`",
        f"- Block sort (C/R): `{metadata.get('block_sort_strategy_c')}` / `{metadata.get('block_sort_strategy_r')}`",
        f"- Gamma: {metadata.get('gamma')}",
        "",
        "## Label coefficients",
        "",
        "```json",
        json.dumps(label_coefs, indent=2),
        "```",
        "",
        "## Diagnostics",
        "",
        "| Split | Groups | Avg cand | P50 cand | PPO top-1 | Oracle headroom | Nonzero range | Return mean | Return std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, diag in metadata.get("diagnostics", {}).items():
        lines.append(
            f"| {split} | {diag['groups']} | {diag['avg_candidates']:.2f} | "
            f"{diag['p50_candidates']:.0f} | {diag['ppo_top1_rate']:.2%} | "
            f"{diag['oracle_headroom_pp']:.4f} | {diag['nonzero_return_range_rate']:.2%} | "
            f"{diag['return_mean']:.4f} | {diag['return_std']:.4f} |"
        )

    lines.extend(["", "## Label distribution (train)", ""])
    train_label = metadata.get("diagnostics", {}).get("train", {}).get("label_distribution", {})
    if train_label:
        lines.append("```json")
        lines.append(json.dumps(train_label, indent=2))
        lines.append("```")

    lines.extend(["", "## Feature statistics (train valid candidates)", ""])
    lines.append(f"- feature_dim={metadata.get('feature_dim')}")
    lines.append(f"- mean[0:5]={metadata.get('train_feature_mean', [])[:5]}")
    lines.append(f"- std[0:5]={metadata.get('train_feature_std', [])[:5]}")

    lines.extend(["", "## E-gate and depth stratum (v1.3 fix)", ""])
    lines.append(f"- group_filter={metadata.get('group_filter', 'unknown')}")
    lines.append(f"- total E=0 groups: {metadata.get('groups_e0', 'N/A')}")
    lines.append(f"- total E=1 groups: {metadata.get('groups_e1', 'N/A')}")
    lines.append(f"- total groups: {metadata.get('groups_total', 'N/A')}")
    lines.append(f"- depth stratum counts (total): {metadata.get('depth_stratum_counts', {})}")
    lines.append("")
    lines.append("| Split | E=0 | E=1 | E=1 rate | Stratum 0 | Stratum 1 | Stratum 2 | Stratum 3 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for split, diag in metadata.get("diagnostics", {}).items():
        sc = diag.get("depth_stratum_counts", {0: 0, 1: 0, 2: 0, 3: 0})
        lines.append(
            f"| {split} | {diag.get('groups_e0', 0)} | {diag.get('groups_e1', 0)} | "
            f"{diag.get('e1_rate', 0):.2%} | {sc.get(0, 0)} | {sc.get(1, 0)} | "
            f"{sc.get(2, 0)} | {sc.get(3, 0)} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
