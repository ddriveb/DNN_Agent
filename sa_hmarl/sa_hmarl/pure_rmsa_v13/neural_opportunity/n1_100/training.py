"""Price regression plus decision-focused ranking for Neural Opportunity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import TinyOpportunityMLP, TinyOpportunityWeights
from .protocol import FEATURE_DIM, NUM_SLOTS, PROTOCOL_ID


@dataclass(frozen=True)
class DenseDataset:
    features: np.ndarray
    valid: np.ndarray
    targets: np.ndarray


def load_dense_dataset(paths: list[Path]) -> DenseDataset:
    features = []
    valid = []
    targets = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            features.append(np.asarray(data["features"], dtype=np.float32))
            valid.append(np.asarray(data["valid"], dtype=np.bool_))
            targets.append(np.asarray(data["targets"], dtype=np.float32))
    return DenseDataset(
        features=np.concatenate(features),
        valid=np.concatenate(valid),
        targets=np.concatenate(targets),
    )


def _forward_grid(model, features):
    batch, paths, dim = features.shape
    return model(features.reshape(batch * paths, dim)).reshape(
        batch, paths, NUM_SLOTS
    )


def composite_loss(model, features, valid, targets, target_scale):
    predicted = _forward_grid(model, features)
    normalized_target = targets / target_scale
    price = F.smooth_l1_loss(
        predicted[valid], normalized_target[valid], beta=0.05
    )
    ranks = features[:, :, :1] * (49.0 / target_scale)
    target_total = normalized_target + ranks
    predicted_total = predicted + ranks
    masked_target = target_total.masked_fill(~valid, 1e6)
    masked_predicted = predicted_total.masked_fill(~valid, 1e6)
    target_logits = (-masked_target / 0.10).masked_fill(~valid, -1e6)
    predicted_logits = (-masked_predicted / 0.10).masked_fill(~valid, -1e6)
    target_distribution = torch.softmax(
        target_logits.reshape(features.shape[0], -1), dim=1
    ).reshape_as(target_logits)
    predicted_log_distribution = torch.log_softmax(
        predicted_logits.reshape(features.shape[0], -1), dim=1
    ).reshape_as(predicted_logits)
    listwise = -torch.sum(
        target_distribution * predicted_log_distribution, dim=(1, 2)
    ).mean()
    best_flat = torch.argmin(
        masked_target.reshape(features.shape[0], -1), dim=1
    )
    pred_flat = predicted_total.reshape(features.shape[0], -1)
    target_flat = target_total.reshape(features.shape[0], -1)
    valid_flat = valid.reshape(features.shape[0], -1)
    rows = torch.arange(features.shape[0], device=features.device)
    pred_best = pred_flat[rows, best_flat][:, None]
    target_best = target_flat[rows, best_flat][:, None]
    desired_margin = torch.clamp(target_flat - target_best, 0.0, 0.25)
    pair_values = F.relu(desired_margin - (pred_flat - pred_best))
    pair_mask = valid_flat.clone()
    pair_mask[rows, best_flat] = False
    pairwise = (
        pair_values[pair_mask].mean()
        if torch.any(pair_mask)
        else predicted.sum() * 0.0
    )
    total = price + 0.5 * listwise + 0.5 * pairwise
    return total, price, listwise, pairwise


def evaluate_model(model, dataset: DenseDataset, target_scale: float, device):
    features = torch.from_numpy(dataset.features).to(device)
    valid = torch.from_numpy(dataset.valid).to(device)
    targets = torch.from_numpy(dataset.targets).to(device)
    with torch.inference_mode():
        predicted = _forward_grid(model, features) * target_scale
        ranks = features[:, :, :1] * 49.0
        teacher_total = (targets + ranks).masked_fill(~valid, 1e9)
        predicted_total = (predicted + ranks).masked_fill(~valid, 1e9)
        teacher_best = torch.argmin(teacher_total.reshape(len(features), -1), dim=1)
        predicted_best = torch.argmin(
            predicted_total.reshape(len(features), -1), dim=1
        )
        recall = torch.mean((teacher_best == predicted_best).float()).item()
        flat_teacher = teacher_total.reshape(len(features), -1)
        rows = torch.arange(len(features), device=device)
        minimum = flat_teacher[rows, teacher_best]
        chosen = flat_teacher[rows, predicted_best]
        finite_teacher = (targets + ranks).masked_fill(~valid, -1e9).reshape(
            len(features), -1
        )
        maximum = torch.max(finite_teacher, dim=1).values
        denominator = torch.clamp(maximum - minimum, min=1e-6)
        normalized_regret = torch.mean((chosen - minimum) / denominator).item()
        mae = torch.mean(torch.abs(predicted[valid] - targets[valid])).item()
    return {
        "top1_recall": recall,
        "mean_normalized_regret": normalized_regret,
        "valid_price_mae": mae,
    }


def train_model(
    train: DenseDataset,
    validation: DenseDataset,
    *,
    output_dir: Path,
    epochs: int = 60,
    batch_size: int = 128,
    seed: int = 20260802,
    device_name: str = "cpu",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    model = TinyOpportunityMLP().to(device)
    valid_targets = train.targets[train.valid]
    target_scale = max(1.0, float(np.percentile(valid_targets, 95)))
    tensors = TensorDataset(
        torch.from_numpy(train.features),
        torch.from_numpy(train.valid),
        torch.from_numpy(train.targets),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_regret = float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for features, valid, targets in loader:
            features = features.to(device)
            valid = valid.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = composite_loss(
                model, features, valid, targets, target_scale
            )
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sums += np.asarray([float(loss.detach().cpu()) for loss in losses])
            batches += 1
        metrics = evaluate_model(model, validation, target_scale, device)
        row = {
            "epoch": epoch + 1,
            "loss": float(sums[0] / batches),
            "price_loss": float(sums[1] / batches),
            "listwise_loss": float(sums[2] / batches),
            "pairwise_loss": float(sums[3] / batches),
            **metrics,
        }
        history.append(row)
        if metrics["mean_normalized_regret"] < best_regret:
            best_regret = metrics["mean_normalized_regret"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, validation, target_scale, device)
    deployment = TinyOpportunityWeights.from_torch(model)
    deployment = TinyOpportunityWeights(
        w1=deployment.w1,
        b1=deployment.b1,
        w2=deployment.w2 * target_scale,
        b2=deployment.b2 * target_scale,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment.save(output_dir / "deployment_weights.npz")
    torch.save(
        {
            "protocol_id": PROTOCOL_ID,
            "feature_dim": FEATURE_DIM,
            "target_scale": target_scale,
            "state_dict": best_state,
        },
        output_dir / "training_checkpoint.pt",
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "target_scale": target_scale,
        "train_samples": len(train.features),
        "validation_samples": len(validation.features),
        "best_validation": final_metrics,
        "history": history,
    }
    (output_dir / "TRAINING_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
