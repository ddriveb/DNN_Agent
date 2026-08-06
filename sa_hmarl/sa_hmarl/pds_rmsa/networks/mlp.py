"""Shared MLP architecture for PDS and PreD value networks."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    """ input -> 256 -> 128 -> 1 with SiLU activations and no dropout. """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PDSMLP(SharedMLP):
    """Value network V_pds(Ω+) for the post-decision-state approach."""

    def __init__(self, input_dim: int):
        super().__init__(input_dim)


class PreDMLP(SharedMLP):
    """Action-value network Q_pred(S,a) for the PreD-DQN approach."""

    def __init__(self, input_dim: int):
        super().__init__(input_dim)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def build_network(input_dim: int, mode: str) -> SharedMLP:
    """Factory to build either a PDS or PreD value network."""
    mode = str(mode).lower()
    if mode == "pds":
        return PDSMLP(input_dim)
    if mode == "pred":
        return PreDMLP(input_dim)
    raise ValueError(f"Unknown mode: {mode}. Use 'pds' or 'pred'.")
