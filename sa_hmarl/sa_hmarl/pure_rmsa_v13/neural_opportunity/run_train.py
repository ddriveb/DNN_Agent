"""Train one topology-specific Neural Opportunity price map."""
from __future__ import annotations

import argparse
from pathlib import Path

from .training import DenseDataset, load_dense_dataset, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--validation", nargs="+", type=Path)
    parser.add_argument("--smoke-split", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train = load_dense_dataset(args.train)
    if args.smoke_split:
        split = max(1, int(0.8 * len(train.features)))
        validation = DenseDataset(
            features=train.features[split:],
            valid=train.valid[split:],
            targets=train.targets[split:],
        )
        train = DenseDataset(
            features=train.features[:split],
            valid=train.valid[:split],
            targets=train.targets[:split],
        )
    else:
        if not args.validation:
            raise ValueError("--validation is required without --smoke-split")
        validation = load_dense_dataset(args.validation)
    result = train_model(
        train,
        validation,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(result["best_validation"])


if __name__ == "__main__":
    main()
