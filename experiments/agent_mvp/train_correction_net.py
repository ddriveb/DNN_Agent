"""Train CorrectionNet v1.

Supervised regression on episode returns.
Input: action local features + global trends
Output: normalized long-term return
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
import argparse

from correction_net import (
    ACTION_FEATURE_NAMES_V2,
    CorrectionNet,
    GLOBAL_STATE_DIM,
    ACTION_FEATURE_DIM_V1,
    compose_correction_input,
    compute_episode_returns,
)


def extract_features_from_flat(flat_state, action_id, num_actions=15, global_state_dim=GLOBAL_STATE_DIM, action_feat_dim=ACTION_FEATURE_DIM_V1):
    """Extract CorrectionNet input from flat_state.

    flat_state = [global_state, action_features (num_actions * action_feat_dim)]
    """
    global_state = flat_state[:global_state_dim]
    action_feats = flat_state[
        global_state_dim:global_state_dim + num_actions * action_feat_dim
    ].reshape(num_actions, action_feat_dim)
    action_feat = action_feats[action_id]
    return compose_correction_input(action_feat, global_state)


def build_dataset(replay_path, gamma=0.95):
    with open(replay_path, "rb") as f:
        data = pickle.load(f)

    transitions = data["transitions"]
    num_actions = data.get("num_actions", 15)
    global_state_dim = data.get("global_state_dim", GLOBAL_STATE_DIM)
    action_feat_dim = data.get("action_feat_dim", ACTION_FEATURE_DIM_V1)

    # Group by episode
    episodes = []
    current = []
    for t in transitions:
        current.append(t)
        if t["done"]:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)

    all_inputs = []
    all_targets = []

    for ep in episodes:
        returns = compute_episode_returns(ep, gamma)
        for t, G in zip(ep, returns):
            feat = extract_features_from_flat(
                t["state"],
                t["action"],
                num_actions=num_actions,
                global_state_dim=global_state_dim,
                action_feat_dim=action_feat_dim,
            )
            all_inputs.append(feat)
            all_targets.append(G)

    inputs = np.stack(all_inputs)
    targets = np.array(all_targets, dtype=np.float32)

    # Normalize targets
    target_mean = float(np.mean(targets))
    target_std = float(np.std(targets)) + 1e-6
    targets_norm = (targets - target_mean) / target_std

    print(f"Dataset: {len(inputs)} samples, input_dim={inputs.shape[1]}")
    print(f"Target: mean={target_mean:.3f}, std={target_std:.3f}, range=[{targets.min():.2f}, {targets.max():.2f}]")

    return inputs, targets_norm, target_mean, target_std


def train(
    replay_path="data/replay_buffer_topk2_fragfeat_v1_30k.pkl",
    epochs=100,
    batch_size=256,
    lr=1e-3,
    val_split=0.2,
    device="cpu",
    checkpoint_name="correction_net_fragfeat_v1.pt",
    history_name="correction_net_fragfeat_v1_history.json",
):
    inputs, targets, target_mean, target_std = build_dataset(replay_path)
    input_dim = inputs.shape[1]

    # Split
    n = len(inputs)
    indices = np.random.permutation(n)
    split = int((1 - val_split) * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = TensorDataset(
        torch.from_numpy(inputs[train_idx]).float(),
        torch.from_numpy(targets[train_idx]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(inputs[val_idx]).float(),
        torch.from_numpy(targets[val_idx]).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Normalize inputs (z-score from training set)
    train_inputs = inputs[train_idx]
    input_mean = np.mean(train_inputs, axis=0)
    input_std = np.std(train_inputs, axis=0) + 1e-6
    train_inputs_norm = (train_inputs - input_mean) / input_std
    val_inputs_norm = (inputs[val_idx] - input_mean) / input_std

    train_ds = TensorDataset(
        torch.from_numpy(train_inputs_norm).float(),
        torch.from_numpy(targets[train_idx]).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_inputs_norm).float(),
        torch.from_numpy(targets[val_idx]).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CorrectionNet(input_dim, hidden_dims=(64, 64)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    print(f"\nTraining CorrectionNet:")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Input dim: {input_dim}")
    print(f"  Input mean range: [{input_mean.min():.3f}, {input_mean.max():.3f}]")
    print(f"  Input std range:  [{input_std.min():.3f}, {input_std.max():.3f}]")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest val loss: {best_val_loss:.4f}")

    # Save
    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    replay_meta = {}
    with open(replay_path, "rb") as f:
        replay_meta = pickle.load(f)
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
        "history": history,
        "dataset_size": int(len(inputs)),
        "replay_path": str(replay_path),
        "action_feat_dim": int(replay_meta.get("action_feat_dim", ACTION_FEATURE_DIM_V1)),
        "action_feature_names": replay_meta.get("action_feature_names", list(ACTION_FEATURE_NAMES_V2)),
    }, out_dir / checkpoint_name)

    with open(out_dir / history_name, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved to {out_dir / checkpoint_name}")
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_path", default="data/replay_buffer_topk2_fragfeat_v1_30k.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--checkpoint_name", default="correction_net_fragfeat_v1.pt")
    parser.add_argument("--history_name", default="correction_net_fragfeat_v1_history.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    train(
        replay_path=args.replay_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=device,
        checkpoint_name=args.checkpoint_name,
        history_name=args.history_name,
    )


if __name__ == "__main__":
    main()
