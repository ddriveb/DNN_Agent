"""Tiny shared candidate scorer used for training and deployment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .protocol import (
    FEATURE_DIM,
    FEATURE_SCHEMA_ID,
    HIDDEN_DIM,
    OUTPUT_DIM,
    PROTOCOL_ID,
)


class TinyOpportunityMLP(nn.Module):
    """One shared MLP evaluated over every legal candidate in one call."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, OUTPUT_DIM)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(features)))


@dataclass(frozen=True)
class TinyOpportunityWeights:
    """NumPy deployment envelope for the same two-layer MLP."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def zeros(
        cls,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> "TinyOpportunityWeights":
        return cls(
            w1=np.zeros((feature_dim, hidden_dim), dtype=np.float32),
            b1=np.zeros(hidden_dim, dtype=np.float32),
            w2=np.zeros((hidden_dim, OUTPUT_DIM), dtype=np.float32),
            b2=np.zeros(OUTPUT_DIM, dtype=np.float32),
        )

    @classmethod
    def from_torch(cls, model: TinyOpportunityMLP) -> "TinyOpportunityWeights":
        return cls(
            w1=model.fc1.weight.detach().cpu().numpy().T.astype(
                np.float32, copy=True
            ),
            b1=model.fc1.bias.detach().cpu().numpy().astype(
                np.float32, copy=True
            ),
            w2=model.fc2.weight.detach().cpu().numpy().T.astype(
                np.float32, copy=True
            ),
            b2=model.fc2.bias.detach().cpu().numpy().astype(
                np.float32, copy=True
            ),
        )

    def forward(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(features @ self.w1 + self.b1, 0.0)
        return hidden @ self.w2 + self.b2

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            protocol_id=np.asarray(PROTOCOL_ID),
            feature_schema_id=np.asarray(FEATURE_SCHEMA_ID),
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyOpportunityWeights":
        with np.load(Path(path), allow_pickle=False) as data:
            protocol_id = str(data["protocol_id"])
            feature_schema_id = str(data["feature_schema_id"])
            if protocol_id != PROTOCOL_ID:
                raise ValueError(f"checkpoint protocol mismatch: {protocol_id}")
            if feature_schema_id != FEATURE_SCHEMA_ID:
                raise ValueError(
                    f"checkpoint feature schema mismatch: {feature_schema_id}"
                )
            return cls(
                w1=np.asarray(data["w1"], dtype=np.float32),
                b1=np.asarray(data["b1"], dtype=np.float32),
                w2=np.asarray(data["w2"], dtype=np.float32),
                b2=np.asarray(data["b2"], dtype=np.float32),
            )
