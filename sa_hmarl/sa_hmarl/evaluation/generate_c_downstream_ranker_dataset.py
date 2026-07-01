"""Generate a C-side downstream-RMSA-aware ranking dataset.

The heavy counterfactual work is delegated to
``diagnose_c_rankability_with_v12``.  This script converts its per-candidate
records into group-wise listwise ranking arrays suitable for training a small
C-ranker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sa_hmarl.evaluation.diagnose_c_rankability_with_v12 import (
    build_parser as build_diagnostic_parser,
    run as run_diagnostic,
)


FEATURE_NAMES = [
    "split_id_norm",
    "server_id_norm",
    "valid_c_count_norm",
    "legal_r_count_norm",
    "log_legal_r_count_norm",
    "best_r_score",
    "top3_r_score_mean",
    "r_score_range",
    "current_success",
    "current_delay_norm",
    "current_fs_norm",
]


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _feature_vector(cand: Dict[str, Any], state: Dict[str, Any], args: argparse.Namespace) -> List[float]:
    legal_r = _to_float(cand.get("legal_r_count"))
    max_r = max(args.k_paths * len(args.mod_names) * args.max_blocks, 1)
    best_r = _to_float(cand.get("best_r_score"))
    top3_r = _to_float(cand.get("top3_r_score_mean"))
    score_range = _to_float(cand.get("r_score_range"))
    return [
        _to_float(cand.get("split_id")) / max(args.num_splits - 1, 1),
        _to_float(cand.get("server_id")) / max(args.num_servers - 1, 1),
        _to_float(state.get("valid_c_count")) / max(args.num_splits * args.num_servers, 1),
        legal_r / max_r,
        np.log1p(legal_r) / np.log1p(max_r),
        best_r,
        top3_r,
        score_range,
        1.0 if cand.get("current_success") else 0.0,
        _to_float(cand.get("current_delay_ms")) / max(args.deadline_max, 1.0),
        _to_float(cand.get("current_fs")) / max(args.num_slots, 1),
    ]


def _split_states(states: List[Dict[str, Any]], train_frac: float, val_frac: float):
    states = [s for s in states if s.get("valid_c_count", 0) > 0]
    states.sort(key=lambda s: (s["seed"], s["episode"], s["request"]))
    n = len(states)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return {
        "train": states[:n_train],
        "val": states[n_train:n_train + n_val],
        "test": states[n_train + n_val:],
    }


def _build_arrays(
    split_states: List[Dict[str, Any]],
    candidates_by_state: Dict[tuple, List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    features: List[List[float]] = []
    returns: List[float] = []
    group_ids: List[int] = []
    c_indices: List[int] = []
    selected_by_ppo: List[int] = []

    for group_id, state in enumerate(split_states):
        key = (state["seed"], state["episode"], state["request"])
        cands = candidates_by_state.get(key, [])
        if len(cands) < 2:
            continue
        for cand in cands:
            features.append(_feature_vector(cand, state, args))
            returns.append(_to_float(cand.get("return")))
            group_ids.append(group_id)
            c_indices.append(int(cand["c_idx"]))
            selected_by_ppo.append(1 if cand.get("selected_by_ppo") else 0)

    return {
        "features": np.asarray(features, dtype=np.float32),
        "returns": np.asarray(returns, dtype=np.float32),
        "group_ids": np.asarray(group_ids, dtype=np.int64),
        "c_indices": np.asarray(c_indices, dtype=np.int64),
        "selected_by_ppo": np.asarray(selected_by_ppo, dtype=np.int64),
    }


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    diag_args = build_diagnostic_parser().parse_args([])
    for key, value in vars(args).items():
        if hasattr(diag_args, key):
            setattr(diag_args, key, value)
    diag_args.output_json = str(Path(args.output_dir) / "raw_diagnostic.json")
    diag_args.output_md = str(Path(args.output_dir) / "raw_diagnostic.md")
    report = run_diagnostic(diag_args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    args.mod_names = ["BPSK", "QPSK", "8QAM", "16QAM"]
    candidates_by_state: Dict[tuple, List[Dict[str, Any]]] = {}
    for cand in report["candidates"]:
        key = (cand["seed"], cand["episode"], cand["request"])
        candidates_by_state.setdefault(key, []).append(cand)

    splits = _split_states(report["states"], args.train_frac, args.val_frac)
    metadata: Dict[str, Any] = {
        "feature_names": FEATURE_NAMES,
        "config": vars(args),
        "diagnostic_summary": report["summary"],
        "splits": {},
    }

    train_arrays = None
    for split_name, states in splits.items():
        arrays = _build_arrays(states, candidates_by_state, args)
        if split_name == "train":
            train_arrays = arrays
        _atomic_save_npz(out_dir / f"{split_name}.npz", **arrays)
        group_count = len(set(arrays["group_ids"].tolist())) if len(arrays["group_ids"]) else 0
        metadata["splits"][split_name] = {
            "states": len(states),
            "groups": int(group_count),
            "candidates": int(len(arrays["features"])),
        }

    if train_arrays is None or len(train_arrays["features"]) == 0:
        raise RuntimeError("No training features generated")
    mean = train_arrays["features"].mean(axis=0)
    std = train_arrays["features"].std(axis=0)
    std[std < 1e-6] = 1.0
    np.save(out_dir / "feature_mean.npy", mean.astype(np.float32))
    np.save(out_dir / "feature_std.npy", std.astype(np.float32))

    metadata["feature_mean"] = mean.astype(float).tolist()
    metadata["feature_std"] = std.astype(float).tolist()
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report(out_dir / "generation_report.md", metadata)
    return metadata


def _write_report(path: Path, metadata: Dict[str, Any]) -> None:
    lines = [
        "# C-side Downstream-RMSA Ranker Dataset",
        "",
        f"Diagnostic verdict: `{metadata['diagnostic_summary']['verdict']}`",
        "",
        "| Split | States | Groups | Candidates |",
        "|---|---:|---:|---:|",
    ]
    for name, row in metadata["splits"].items():
        lines.append(f"| {name} | {row['states']} | {row['groups']} | {row['candidates']} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = build_diagnostic_parser()
    parser.description = __doc__
    parser.add_argument("--output_dir", default="sa_hmarl/datasets/c_downstream_ranker")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    metadata = generate(parsed)
    print(json.dumps(metadata["splits"], indent=2))
