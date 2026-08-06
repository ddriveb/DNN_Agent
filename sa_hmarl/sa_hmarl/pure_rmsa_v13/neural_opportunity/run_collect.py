"""CLI for exact Opportunity-only Teacher collection."""
from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import collect_teacher_dataset
from .protocol import (
    SMOKE_SEED,
    TOPOLOGY_CONFIG,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "validation"), required=True)
    parser.add_argument("--topologies", nargs="+", choices=tuple(TOPOLOGY_CONFIG))
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--record-requests", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/datasets"
        ),
    )
    args = parser.parse_args()
    topologies = args.topologies or list(TOPOLOGY_CONFIG)
    if args.mode == "smoke":
        seeds = (SMOKE_SEED,)
        warmup = args.warmup if args.warmup is not None else 100
        requests = args.record_requests if args.record_requests is not None else 500
    elif args.mode == "train":
        seeds = TRAIN_SEEDS
        warmup = args.warmup if args.warmup is not None else 1000
        requests = args.record_requests if args.record_requests is not None else 10000
    else:
        seeds = VALIDATION_SEEDS
        warmup = args.warmup if args.warmup is not None else 1000
        requests = args.record_requests if args.record_requests is not None else 10000
    for topology in topologies:
        for seed in seeds:
            path = args.output_dir / topology / f"seed_{seed}.npz"
            result = collect_teacher_dataset(
                topology=topology,
                config=TOPOLOGY_CONFIG[topology],
                seed=seed,
                warmup=warmup,
                record_requests=requests,
                output_path=path,
            )
            print(result)


if __name__ == "__main__":
    main()
