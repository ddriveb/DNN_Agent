"""Train Imitation Agent via behavior cloning.

Small MLP: state → hidden → action logits
Loss: CrossEntropy on teacher's chosen action
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from typing import Dict


class ImitationDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).long()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class ImitationAgent(nn.Module):
    """Lightweight MLP for behavior cloning."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(128, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

    def predict_action(self, state):
        """Return best action_id for a given state."""
        self.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).float().unsqueeze(0)
            logits = self(state)
            return int(torch.argmax(logits, dim=-1).item())


def train(
    dataset_path: str = "data/imitation_dataset.pkl",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    device: str = "cpu",
) -> Dict:
    # Load data
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    states = data["states"]
    actions = data["actions"]
    state_dim = data["state_dim"]
    num_actions = data["num_actions"]
    print(f"Loaded dataset: {len(states)} samples, state_dim={state_dim}, num_actions={num_actions}")

    # Split train/val
    n = len(states)
    indices = np.random.permutation(n)
    split = int((1 - val_split) * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = ImitationDataset(states[train_idx], actions[train_idx])
    val_dataset = ImitationDataset(states[val_idx], actions[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    model = ImitationAgent(state_dim, num_actions, hidden_dims=(128, 128), dropout=0.1)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_states)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_actions).sum().item()
            train_total += len(batch_actions)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                logits = model(batch_states)
                loss = criterion(logits, batch_actions)

                val_loss += loss.item() * len(batch_states)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == batch_actions).sum().item()
                val_total += len(batch_actions)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # Save
    out_dir = Path(__file__).parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "state_dim": state_dim,
        "num_actions": num_actions,
        "hidden_dims": (128, 128),
        "best_val_acc": best_val_acc,
        "history": history,
    }
    torch.save(checkpoint, out_dir / "imitation_agent.pt")
    print(f"Saved checkpoint to {out_dir / 'imitation_agent.pt'}")

    # Also save as json summary
    with open(out_dir / "imitation_training.json", "w") as f:
        json.dump({
            "state_dim": state_dim,
            "num_actions": num_actions,
            "best_val_acc": float(best_val_acc),
            "final_train_acc": float(history["train_acc"][-1]),
            "final_val_acc": float(history["val_acc"][-1]),
        }, f, indent=2)

    return checkpoint


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/imitation_dataset.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
