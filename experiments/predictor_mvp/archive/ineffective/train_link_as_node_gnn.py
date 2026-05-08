"""Training loop + evaluation for Link-As-Node GAT Predictor."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from link_as_node_gnn_predictor import LinkAsNodeGATPredictor


class LinkAsNodeDataset(Dataset):
    """Dataset for Link-As-Node GAT predictor."""
    def __init__(self, samples):
        self.samples = samples
        # Extract fixed converted_edge_index and num_links from first sample
        self.has_graph = "link_node_features" in samples[0]
        if self.has_graph:
            self.converted_edge_index = torch.from_numpy(samples[0]["converted_edge_index"]).long()
            self.num_links = samples[0]["link_node_features"].shape[0]
        else:
            self.converted_edge_index = None
            self.num_links = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "path_id": torch.tensor(s["path_id"], dtype=torch.long),
            "src": torch.tensor(s["src"], dtype=torch.long),
            "dst": torch.tensor(s["dst"], dtype=torch.long),
            "bw": torch.tensor(s["bw"], dtype=torch.float32),
            "success": torch.tensor(s["success"], dtype=torch.float32),
            "delay": torch.tensor(s["delay"], dtype=torch.float32),
        }
        if self.has_graph:
            item["link_node_features"] = torch.from_numpy(s["link_node_features"].copy()).float()
            item["path_mask"] = torch.from_numpy(s["path_mask"].copy()).float()
            item["converted_edge_index"] = self.converted_edge_index
            item["num_links"] = self.num_links
        return item


def link_as_node_collate_fn(batch):
    """Custom collate: stack tensors, keep converted_edge_index/num_links as scalars."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ("converted_edge_index", "num_links"):
            result[key] = batch[0][key]
        else:
            result[key] = torch.stack([b[key] for b in batch])
    return result


def train_link_as_node_gnn(predictor, train_loader, val_loader, epochs=30, lr=1e-3):
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history = {"train_loss": [], "val_acc": [], "val_auc": [], "val_mae": [], "val_ece": []}

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        for batch in train_loader:
            path_id = batch["path_id"]
            src = batch["src"]
            dst = batch["dst"]
            bw = batch["bw"]
            success = batch["success"]
            delay = batch["delay"]
            link_node_features = batch.get("link_node_features")
            path_masks = batch.get("path_mask")
            converted_edge_index = batch.get("converted_edge_index")

            optimizer.zero_grad()
            logits, delay_pred = predictor.forward_batch(
                path_id, src, dst, bw,
                link_node_features, converted_edge_index, path_masks
            )
            loss = bce(logits, success)
            mask = success > 0.5
            if mask.sum() > 0:
                loss = loss + mse(delay_pred[mask], delay[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate_link_as_node_gnn(predictor, val_loader)
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


def evaluate_link_as_node_gnn(predictor, loader):
    predictor.eval()
    all_success, all_prob, all_delay, all_delay_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            path_id = batch["path_id"]
            src = batch["src"]
            dst = batch["dst"]
            bw = batch["bw"]
            success = batch["success"]
            delay = batch["delay"]
            link_node_features = batch.get("link_node_features")
            path_masks = batch.get("path_mask")
            converted_edge_index = batch.get("converted_edge_index")

            logits, delay_pred = predictor.forward_batch(
                path_id, src, dst, bw,
                link_node_features, converted_edge_index, path_masks
            )
            prob = torch.sigmoid(logits).numpy()
            all_success.extend(success.numpy().tolist())
            all_prob.extend(prob.tolist())
            mask = success.numpy() > 0.5
            if mask.sum() > 0:
                all_delay.extend(delay.numpy()[mask].tolist())
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
