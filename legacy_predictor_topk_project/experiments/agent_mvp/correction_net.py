"""CorrectionNet v1/v2: Predicts long-term value correction for each action.

Architecture:
  Input: action_features + global_trend_features
  Output: correction_score (scalar)

Training:
  Supervised regression on episode returns.
  Target = discounted return from current step to episode end.

Usage:
  final_score = predictor_score + lambda_corr * correction_score
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import torch.nn as nn


GLOBAL_STATE_DIM = 30
ACTION_FEATURE_DIM_V1 = 7
ACTION_FEATURE_DIM_V2 = 13

ACTION_FEATURE_NAMES_V2 = [
    "bw_norm",
    "compute_cost",
    "interm_norm",
    "p_success",
    "delay_norm",
    "server_load",
    "deadline_slack_norm",
    "delta_frag",
    "delta_lfb_ratio",
    "future_blocking_risk",
    "path_lfb_ratio",
    "path_utilization",
    "mapper_feasible",
]


def extract_global_trend_features(global_state):
    """Extract compact global trend features from the 30D global state."""
    frag = float(global_state[13]) if len(global_state) > 13 else 0.0
    max_free = float(global_state[14]) if len(global_state) > 14 else 0.0
    server_utils = global_state[15:20] if len(global_state) >= 20 else global_state[-6:-1]
    avg_load = float(np.mean(server_utils)) if len(server_utils) > 0 else 0.5
    return np.array([frag, max_free, avg_load], dtype=np.float32)


def compose_correction_input(action_feat, global_state):
    """Compose CorrectionNet input from per-action features + global trends."""
    action_feat = np.asarray(action_feat, dtype=np.float32)
    action_feat = np.clip(action_feat, -100.0, 100.0)
    global_feat = extract_global_trend_features(global_state)
    return np.concatenate([action_feat, global_feat]).astype(np.float32)


class CorrectionNet(nn.Module):
    """Small MLP that predicts long-term value correction for an action."""

    def __init__(self, input_dim: int = 10, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, input_dim) -> (batch,) correction scores."""
        return self.net(x).squeeze(-1)


def compute_episode_returns(transitions, gamma=0.95):
    """Compute discounted returns for each transition in an episode.

    transitions: list of dicts from one episode (ordered)
    Returns: list of returns (same length as transitions)
    """
    returns = []
    G = 0.0
    for t in reversed(transitions):
        G = t["reward"] + gamma * G
        returns.append(G)
    returns = list(reversed(returns))
    return returns


def extract_correction_features(transition, state_dict):
    """Extract input features for CorrectionNet.

    Features:
      - action_features (7D): bw_norm, compute_cost, interm_norm, P_success,
                               delay_norm, server_load, deadline_slack
      - global_trend (3D): recent blocking rate (window=20),
                           current frag index,
                           current avg server load
    """
    action_id = transition["action"]
    action_feat = state_dict["action_features"][action_id].copy()
    global_state = state_dict["global_state"]
    return compose_correction_input(action_feat, global_state)


class CorrectionDataset:
    """Dataset for training CorrectionNet."""

    def __init__(self, transitions, state_dicts, gamma=0.95):
        self.inputs = []
        self.targets = []

        # Group transitions by episode
        episodes = []
        current_episode = []
        for t, s in zip(transitions, state_dicts):
            current_episode.append((t, s))
            if t["done"]:
                episodes.append(current_episode)
                current_episode = []
        if current_episode:
            episodes.append(current_episode)

        # Compute returns and extract features
        for ep in episodes:
            ep_trans = [t for t, s in ep]
            ep_states = [s for t, s in ep]
            returns = compute_episode_returns(ep_trans, gamma)

            for (t, s), G in zip(ep, returns):
                feat = extract_correction_features(t, s)
                self.inputs.append(feat)
                self.targets.append(G)

        self.inputs = np.stack(self.inputs)
        self.targets = np.array(self.targets, dtype=np.float32)

        # Normalize targets
        self.target_mean = float(np.mean(self.targets))
        self.target_std = float(np.std(self.targets)) + 1e-6
        self.targets_norm = (self.targets - self.target_mean) / self.target_std

        print(f"CorrectionDataset: {len(self.inputs)} samples")
        print(f"  Input dim: {self.inputs.shape[1]}")
        print(f"  Target range: [{self.targets.min():.2f}, {self.targets.max():.2f}]")
        print(f"  Target mean: {self.target_mean:.2f}, std: {self.target_std:.2f}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.inputs[idx]).float(),
            torch.tensor(self.targets_norm[idx], dtype=torch.float32),
        )
