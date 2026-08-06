#!/usr/bin/env python3
"""N1-100 training with XLRON-consistent settings.

Key differences from standard N1-100:
- holding_truncation=2.0 (XLRON rejection sampling)
- warmup=3000 (Doherty 2025)
- 10 seeds for training
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.dataset import (
    collect_teacher_dataset,
)
from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.protocol import (
    EVAL_REQUESTS,
    TOPOLOGY_CONFIG,
)
from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.training import (
    load_dense_dataset,
    train_model,
)

# XLRON-consistent settings
HOLDING_TRUNCATION = 2.0
WARMUP = 3000
# Seed segments (2026-08-06, fair-format rerun): 62011-15 train,
# 62016-17 valid, 62091 smoke — disjoint from 618xx (cost239/abilene),
# 619xx (NS1), 62001-05 (calibration) and all legacy 694xx-697xx.
TOPO_SPECS = {
    "xlron_nsfnet_deeprmsa": {
        "train": (62011, 62012, 62013, 62014, 62015),
        "valid": (62016, 62017),
        "smoke": 62091,
        "dir": "nsfnet",
    },
    "xlron_jpn48": {
        "train": (62011, 62012, 62013, 62014, 62015),
        "valid": (62016, 62017),
        "smoke": 62091,
        "dir": "jpn48",
    },
    "xlron_german17": {
        "train": (69601, 69602, 69603, 69604, 69605),
        "valid": (69611, 69612),
        "smoke": 69691,
        "dir": "german17",
    },
    "xlron_usnet_gcnrmsa": {
        "train": (62011, 62012, 62013, 62014, 62015),
        "valid": (62016, 62017),
        "smoke": 62091,
        "dir": "usnet",
    },
    "cost239_deeprmsa": {
        "train": (61811, 61812, 61813, 61814, 61815),
        "valid": (61816, 61817),
        "smoke": 61891,
        "dir": "cost239",
    },
    "abilene": {
        "train": (61811, 61812, 61813, 61814, 61815),
        "valid": (61816, 61817),
        "smoke": 61891,
        "dir": "abilene",
    },
}
BASE_ROOT = Path(
    "sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train"), default="train")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--topology", choices=list(TOPO_SPECS),
                        default="xlron_nsfnet_deeprmsa")
    args = parser.parse_args()
    topology = args.topology
    spec = TOPO_SPECS[topology]
    config = TOPOLOGY_CONFIG[topology]
    BASE = BASE_ROOT / spec["dir"]
    dataset_dir = BASE / "datasets"

    if args.mode == "smoke":
        seeds = (spec["smoke"],)
        requests = 800
        warmup = 100
    else:
        seeds = spec["train"] + spec["valid"]
        requests = EVAL_REQUESTS
        warmup = WARMUP

    paths = {}
    for seed in seeds:
        path = dataset_dir / f"seed_{seed}.npz"
        paths[seed] = path
        if not path.exists():
            print(f"Collecting seed {seed} with truncation={HOLDING_TRUNCATION}...", flush=True)
            print(collect_teacher_dataset(
                topology=topology, config=config, seed=seed,
                warmup=warmup, record_requests=requests, output_path=path,
                holding_truncation=HOLDING_TRUNCATION,
                path_sort_strategy="xlron",
            ), flush=True)

    if args.mode == "smoke":
        complete = load_dense_dataset([paths[spec["smoke"]]])
        split = max(1, int(0.8 * len(complete.features)))
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.training import DenseDataset
        train = DenseDataset(
            features=complete.features[:split],
            valid=complete.valid[:split],
            targets=complete.targets[:split],
        )
        validation = DenseDataset(
            features=complete.features[split:],
            valid=complete.valid[split:],
            targets=complete.targets[split:],
        )
    else:
        train = load_dense_dataset([paths[s] for s in spec["train"]])
        validation = load_dense_dataset([paths[s] for s in spec["valid"]])

    print(f"Training on {len(train.features)} samples, validating on {len(validation.features)}", flush=True)
    result = train_model(
        train, validation, output_dir=BASE / "training",
        epochs=args.epochs, device_name=args.device,
    )
    print("Training complete:", result, flush=True)


if __name__ == "__main__":
    main()
