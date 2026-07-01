"""Training loop + evaluation + calibration plot."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class OpticalDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "partition": torch.tensor(s["partition"], dtype=torch.long),
            "src": torch.tensor(s["src"], dtype=torch.long),
            "dst": torch.tensor(s["dst"], dtype=torch.long),
            "z": torch.from_numpy(s["z"].copy()).float(),
            "success": torch.tensor(s["success"], dtype=torch.float32),
            "delay": torch.tensor(s["delay"], dtype=torch.float32),
        }


def train_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3):
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history = {"train_loss": [], "val_acc": [], "val_auc": [], "val_mae": [], "val_ece": []}

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        for batch in train_loader:
            partition = batch["partition"]
            src = batch["src"]
            dst = batch["dst"]
            z = batch["z"]
            success = batch["success"]
            delay = batch["delay"]

            optimizer.zero_grad()
            success_logit, delay_pred = predictor(partition, src, dst, z)
            loss = bce(success_logit, success)
            mask = success > 0.5
            if mask.sum() > 0:
                loss = loss + mse(delay_pred[mask], delay[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate_predictor(predictor, val_loader)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_acc"].append(metrics["accuracy"])
        history["val_auc"].append(metrics["auc"])
        history["val_mae"].append(metrics["delay_mae"])
        history["val_ece"].append(metrics["ece"])

        print(
            f"  Epoch {epoch+1:02d}: loss={history['train_loss'][-1]:.4f} "
            f"acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
            f"mae={metrics['delay_mae']:.6f} ece={metrics['ece']:.4f}"
        )
    return history


def evaluate_predictor(predictor, loader):
    predictor.eval()
    all_success, all_prob, all_delay, all_delay_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logit, delay_pred = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            prob = torch.sigmoid(logit).numpy()
            success = batch["success"].numpy()
            delay = batch["delay"].numpy()
            all_success.extend(success.tolist())
            all_prob.extend(prob.tolist())
            mask = success > 0.5
            if mask.sum() > 0:
                all_delay.extend(delay[mask].tolist())
                all_delay_pred.extend(delay_pred.numpy()[mask].tolist())

    y_true = np.array(all_success)
    y_prob = np.array(all_prob)
    acc = accuracy_score(y_true, y_prob > 0.5)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    mae = float(np.mean(np.abs(np.array(all_delay) - np.array(all_delay_pred)))) if all_delay else 0.0
    ece = compute_ece(y_true, y_prob, n_bins=10)
    return {"accuracy": acc, "auc": auc, "delay_mae": mae, "ece": ece}


def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * np.abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece / len(y_true))


def plot_calibration(predictor, loader, save_path="calibration.png"):
    predictor.eval()
    probs, truths = [], []
    with torch.no_grad():
        for batch in loader:
            logit, _ = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            probs.extend(torch.sigmoid(logit).numpy().tolist())
            truths.extend(batch["success"].numpy().tolist())

    probs = np.array(probs)
    truths = np.array(truths)
    n_bins = 10
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    accs, confs = [], []
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i+1]) if i < n_bins - 1 else (probs >= edges[i]) & (probs <= edges[i+1])
        if mask.sum() == 0:
            accs.append(0.0)
            confs.append(0.0)
        else:
            accs.append(float(truths[mask].mean()))
            confs.append(float(probs[mask].mean()))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(confs, accs, "o-", label="Predictor")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Success Rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Calibration plot saved: {save_path}")


def measure_latency(predictor, encoder, mapper, num_runs=1000):
    partition = torch.tensor([0], dtype=torch.long)
    src = torch.tensor([0], dtype=torch.long)
    dst = torch.tensor([5], dtype=torch.long)

    t0 = time.time()
    for _ in range(num_runs):
        _ = encoder.encode(0, 5)
    enc_ms = (time.time() - t0) / num_runs * 1000

    z = torch.from_numpy(encoder.encode(0, 5)).unsqueeze(0).float()
    _ = predictor(partition, src, dst, z)
    t0 = time.time()
    for _ in range(num_runs):
        _ = predictor(partition, src, dst, z)
    pred_ms = (time.time() - t0) / num_runs * 1000

    mapper.net.reset()
    for link in mapper.net.link_states:
        mask = np.random.rand(mapper.net.num_slots) < 0.3
        mapper.net.link_states[link][mask] = True
    t0 = time.time()
    for _ in range(num_runs):
        _ = mapper.map(0, 5, 4)
    map_ms = (time.time() - t0) / num_runs * 1000
    mapper.net.reset()
    return {"encoder_ms": enc_ms, "predictor_ms": pred_ms, "mapper_ms": map_ms}
