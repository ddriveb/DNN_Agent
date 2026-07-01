"""
Post-hoc calibration methods for cross-topology transfer.
"""
import numpy as np
from scipy.optimize import minimize_scalar


class TemperatureScaling:
    """Learn a single temperature T to scale logits."""
    def __init__(self):
        self.T = 1.0

    def fit(self, logits, labels):
        """
        logits: (N,) array of model logits (before sigmoid)
        labels: (N,) array of 0/1 ground truth
        """
        def nll(T):
            if T <= 0:
                return 1e10
            probs = 1.0 / (1.0 + np.exp(-logits / T))
            probs = np.clip(probs, 1e-8, 1 - 1e-8)
            return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.T = result.x
        return self

    def calibrate(self, logits):
        probs = 1.0 / (1.0 + np.exp(-logits / self.T))
        return probs


class PlattScaling:
    """Learn a and b for sigmoid(a * logit + b)."""
    def __init__(self):
        self.a = 1.0
        self.b = 0.0

    def fit(self, logits, labels, max_iter=1000):
        """
        Fit via gradient descent on NLL.
        """
        a, b = 1.0, 0.0
        lr = 0.1
        N = len(labels)
        
        for _ in range(max_iter):
            z = a * logits + b
            probs = 1.0 / (1.0 + np.exp(-z))
            probs = np.clip(probs, 1e-8, 1 - 1e-8)
            
            # Gradients
            dz = probs - labels
            grad_a = np.mean(dz * logits)
            grad_b = np.mean(dz)
            
            a -= lr * grad_a
            b -= lr * grad_b
        
        self.a = a
        self.b = b
        return self

    def calibrate(self, logits):
        z = self.a * logits + self.b
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs


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


def evaluate_with_calibration(predictor, loader, calibrator=None):
    """Evaluate predictor. If calibrator is provided, apply post-hoc calibration."""
    import torch
    predictor.eval()
    all_success = []
    all_logit = []
    all_delay = []
    all_delay_pred = []
    
    with torch.no_grad():
        for batch in loader:
            logit, delay_pred = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            all_logit.extend(logit.numpy().tolist())
            all_success.extend(batch["success"].numpy().tolist())
            delay = batch["delay"].numpy()
            mask = delay > 0
            if mask.sum() > 0:
                all_delay.extend(delay[mask].tolist())
                all_delay_pred.extend(delay_pred.numpy()[mask].tolist())
    
    logits = np.array(all_logit)
    y_true = np.array(all_success)
    
    if calibrator is not None:
        y_prob = calibrator.calibrate(logits)
    else:
        y_prob = 1.0 / (1.0 + np.exp(-logits))
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    acc = accuracy_score(y_true, y_prob > 0.5)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    mae = float(np.mean(np.abs(np.array(all_delay) - np.array(all_delay_pred)))) if all_delay else 0.0
    ece = compute_ece(y_true, y_prob)
    
    return {"accuracy": acc, "auc": auc, "delay_mae": mae, "ece": ece}
