"""Create an explicit-only afterstate feature view from a v1.35_poststate_v1 dataset.

Keeps only the 16 afterstate+delta dimensions (indices 25-40 of poststate_v1).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from sa_hmarl.evaluation.merge_r_counterfactual_ranking_shards import (
    _atomic_save_npz,
    _atomic_write_json,
    _label_distribution,
    _split_diagnostics,
)
from sa_hmarl.evaluation.r_poststate_features import POSTSTATE_V1_FEATURE_NAMES


EXPLICIT_NAMES = POSTSTATE_V1_FEATURE_NAMES[25:]


def _feature_stats(features: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    valid = features[mask]
    return {
        "feature_mean": valid.mean(axis=0).tolist(),
        "feature_std": np.maximum(valid.std(axis=0), 1e-6).tolist(),
        "feature_min": valid.min(axis=0).tolist(),
        "feature_max": valid.max(axis=0).tolist(),
    }


def slice_dataset(v135_dir: Path, explicit_dir: Path) -> Dict[str, Any]:
    v135_dir = Path(v135_dir)
    explicit_dir = Path(explicit_dir)
    explicit_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((v135_dir / "metadata.json").read_text(encoding="utf-8"))
    new_meta = copy.deepcopy(metadata)
    new_meta["feature_mode"] = "v1.35_explicit_afterstate"
    new_meta["feature_names"] = list(EXPLICIT_NAMES)
    new_meta["feature_dim"] = len(EXPLICIT_NAMES)
    new_meta["derived_from"] = str(v135_dir)

    diagnostics: Dict[str, Any] = {}
    split_data: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        src = v135_dir / f"{split}.npz"
        if not src.exists():
            continue
        data = dict(np.load(str(src), allow_pickle=True))
        data = {k: v for k, v in data.items() if v.dtype != object}
        data["features"] = np.asarray(data["features"], dtype=np.float32)[:, :, 25:]
        diag = _split_diagnostics(data)
        diag.update(_feature_stats(data["features"], np.asarray(data["mask"], dtype=bool)))
        diag["label_distribution"] = _label_distribution(data["returns"], data["mask"])
        diagnostics[split] = diag
        split_data[split] = data
        _atomic_save_npz(explicit_dir / f"{split}.npz", **data)
        print(f"[slice-explicit] {split}: {diag['groups']} groups, avg_cand={diag['avg_candidates']:.2f}")

    train_features = np.asarray(split_data["train"]["features"], dtype=np.float32)
    train_mask = np.asarray(split_data["train"]["mask"], dtype=bool)
    valid_features = train_features[train_mask].astype(np.float64)
    new_meta["train_feature_mean"] = valid_features.mean(axis=0).astype(np.float32).tolist()
    new_meta["train_feature_std"] = np.maximum(valid_features.std(axis=0), 1e-6).astype(np.float32).tolist()
    new_meta["diagnostics"] = diagnostics
    new_meta["feature_dim"] = len(EXPLICIT_NAMES)
    new_meta["merged_at"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")

    _atomic_write_json(explicit_dir / "metadata.json", new_meta)
    print(f"[slice-explicit] Wrote explicit-only view to {explicit_dir}")
    return new_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v135_dir", default="sa_hmarl/datasets/v135_pilot_cost239_v135")
    parser.add_argument("--explicit_dir", default="sa_hmarl/datasets/v135_pilot_cost239_v135_explicit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    slice_dataset(Path(args.v135_dir), Path(args.explicit_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
