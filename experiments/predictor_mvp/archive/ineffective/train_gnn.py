"""Training loop for GNN predictor."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from gnn_predictor import GNNPredictor


class GNNDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        # Extract fixed edge_index and num_nodes from first sample (if available)
        self.has_graph = "edge_features" in samples[0]
        if self.has_graph:
            self.edge_index = torch.from_numpy(samples[0]["edge_index"]).long()
            self.num_nodes = samples[0]["num_nodes"]
        else:
            self.edge_index = None
            self.num_nodes = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "partition": torch.tensor(s["partition"], dtype=torch.long),
            "src": torch.tensor(s["src"], dtype=torch.long),
            "dst": torch.tensor(s["dst"], dtype=torch.long),
            "success": torch.tensor(s["success"], dtype=torch.float32),
            "delay": torch.tensor(s["delay"], dtype=torch.float32),
        }
        if self.has_graph:
            item["edge_features"] = torch.from_numpy(s["edge_features"].copy()).float()
            item["edge_index"] = self.edge_index
            item["num_nodes"] = self.num_nodes
        return item


def gnn_collate_fn(batch):
    """Custom collate for GNN: stack edge_features, keep edge_index/num_nodes as scalars."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ("edge_index", "num_nodes"):
            result[key] = batch[0][key]  # assume all same
        else:
            result[key] = torch.stack([b[key] for b in batch])
    return result


def train_gnn_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3):
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        for batch in train_loader:
            partition = batch["partition"]
            src = batch["src"]
            dst = batch["dst"]
            success = batch["success"]
            delay = batch["delay"]
            edge_features = batch.get("edge_features")
            edge_index = batch.get("edge_index")
            num_nodes = batch.get("num_nodes")

            optimizer.zero_grad()

            # Forward batch
            logits, delay_pred = predictor.forward_batch(
                partition, src, dst, edge_features, edge_index, num_nodes
            )

            loss = bce(logits, success)
            mask = success > 0.5
            if mask.sum() > 0:
                loss = loss + mse(delay_pred[mask], delay[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        metrics = evaluate_gnn_predictor(predictor, val_loader)
        print(
            f"  Epoch {epoch+1:02d}: loss={total_loss/len(train_loader):.4f} "
            f"acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
            f"mae={metrics['delay_mae']:.6f} ece={metrics['ece']:.4f}"
        )
    return metrics


def evaluate_gnn_predictor(predictor, loader):
    predictor.eval()
    all_success, all_prob, all_delay, all_delay_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            partition = batch["partition"]
            src = batch["src"]
            dst = batch["dst"]
            success = batch["success"]
            delay = batch["delay"]
            edge_features = batch.get("edge_features")
            edge_index = batch.get("edge_index")
            num_nodes = batch.get("num_nodes")

            logits, delay_pred = predictor.forward_batch(
                partition, src, dst, edge_features, edge_index, num_nodes
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
