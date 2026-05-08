"""Training loop for PINN-Predictor with optional GradNorm."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from pinn_predictor import PINNPredictor, GradNormPINNPredictor


class PINNDataset(Dataset):
    """Dataset for PINN predictor."""
    def __init__(self, samples):
        self.samples = samples
        self.has_pinn = "S_current" in samples[0]
        if self.has_pinn:
            self.S_current = torch.from_numpy(samples[0]["S_current"]).float()
            self.num_links = samples[0]["S_current"].shape[0]
            self.num_slots = samples[0]["S_current"].shape[1]
        else:
            self.S_current = None
            self.num_links = None
            self.num_slots = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "path_id": torch.tensor(s["path_id"], dtype=torch.long),
            "src": torch.tensor(s["src"], dtype=torch.long),
            "dst": torch.tensor(s["dst"], dtype=torch.long),
            "bw": torch.tensor(s["bw"], dtype=torch.float32),
            "z": torch.from_numpy(s["z"].copy()).float(),
            "success": torch.tensor(s["success"], dtype=torch.float32),
            "delay": torch.tensor(s["delay"], dtype=torch.float32),
        }
        if self.has_pinn:
            item["S_current"] = self.S_current
            item["path_link_mask"] = torch.from_numpy(s["path_link_mask"].copy()).float()
            item["num_links"] = self.num_links
            item["num_slots"] = self.num_slots
            item["start_slot"] = torch.tensor(s.get("start_slot", -1), dtype=torch.long)
        return item


def pinn_collate_fn(batch):
    """Custom collate for PINN: stack tensors, keep S_current as scalar."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ("S_current", "num_links", "num_slots"):
            result[key] = batch[0][key]
        else:
            result[key] = torch.stack([b[key] for b in batch])
    return result


def train_pinn_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3,
                         alpha_physics=0.01, use_gradnorm=False):
    """
    Train PINN predictor.
    If use_gradnorm=True, predictor must be GradNormPINNPredictor.
    """
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    history = {
        "train_loss": [], "val_acc": [], "val_auc": [], "val_mae": [], "val_ece": [],
        "physics_loss": [], "w_data": [], "w_physics": [],
    }

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        total_physics = 0.0
        total_data = 0.0
        num_batches = 0

        for batch in train_loader:
            path_id = batch["path_id"]
            src = batch["src"]
            dst = batch["dst"]
            bw = batch["bw"]
            z = batch["z"]
            success = batch["success"]
            delay = batch["delay"]
            S_current = batch.get("S_current")
            path_link_mask = batch.get("path_link_mask")

            optimizer.zero_grad()

            start_slot = batch.get("start_slot")
            if use_gradnorm:
                # GradNorm training
                out = predictor.forward_train_gradnorm(
                    path_id, src, dst, bw, z, S_current, path_link_mask,
                    success, delay, start_slot=start_slot
                )
                loss = out["total_loss"]
                total_physics += out["loss_physics"].item()
                total_data += out["loss_data"].item()
                history["w_data"].append(out["w_data"])
                history["w_physics"].append(out["w_physics"])
            else:
                # Standard PINN training
                out = predictor.forward_train(path_id, src, dst, bw, z,
                                              S_current, path_link_mask,
                                              start_slot=start_slot)
                loss_data = bce(out["success_logit"], success)
                mask = success > 0.5
                if mask.sum() > 0:
                    loss_data = loss_data + mse(out["delay_pred"][mask], delay[mask])
                loss = loss_data + out["loss_xhat"] + alpha_physics * out["physics_loss"]
                total_physics += out["physics_loss"].item()
                total_data += loss_data.item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # Validation
        metrics = evaluate_pinn_predictor(predictor, val_loader)
        history["train_loss"].append(total_loss / num_batches)
        history["val_acc"].append(metrics["accuracy"])
        history["val_auc"].append(metrics["auc"])
        history["val_mae"].append(metrics["delay_mae"])
        history["val_ece"].append(metrics["ece"])
        history["physics_loss"].append(total_physics / num_batches)

        print(
            f"  Epoch {epoch+1:02d}: loss={history['train_loss'][-1]:.4f} "
            f"acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
            f"mae={metrics['delay_mae']:.6f} ece={metrics['ece']:.4f} "
            f"phys={history['physics_loss'][-1]:.2f}"
        )
    return history


def evaluate_pinn_predictor(predictor, loader):
    """Evaluate using standard forward (no physics head)."""
    predictor.eval()
    all_success, all_prob, all_delay, all_delay_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logit, delay_pred = predictor(
                batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"]
            )
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
