"""
Pure-NumPy MLP fallback (no PyTorch required).
Use this if torch installation is slow / unavailable.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyPredictor:
    """Tiny MLP: (partition_onehot, src_onehot, dst_onehot, z) -> (success_prob, delay)."""

    def __init__(self, state_dim, num_partitions=3, num_servers=14, hidden_dim=64, seed=42):
        rng = np.random.RandomState(seed)
        self.input_dim = num_partitions + num_servers + num_servers + state_dim
        self.hidden_dim = hidden_dim

        # Xavier init
        def init(m, n):
            return rng.randn(m, n) * np.sqrt(2.0 / m)

        self.W1 = init(self.input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = init(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = init(hidden_dim, 2)
        self.b3 = np.zeros(2)

    def forward(self, X):
        h1 = relu(X @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        return out[:, 0], out[:, 1], h2  # h2 for dropout grad if needed

    def predict_proba(self, X):
        logit, delay, _ = self.forward(X)
        return sigmoid(logit), delay

    def _build_X(self, partition, src, dst, z):
        """Convert discrete indices + state into flat feature vector."""
        B = len(partition)
        X = np.zeros((B, self.input_dim), dtype=np.float32)
        off = 0
        # partition one-hot
        for i, p in enumerate(partition):
            X[i, off + int(p)] = 1.0
        off += 3
        # src one-hot
        for i, s in enumerate(src):
            X[i, off + int(s)] = 1.0
        off += 14
        # dst one-hot
        for i, d in enumerate(dst):
            X[i, off + int(d)] = 1.0
        off += 14
        # state
        X[:, off:] = z
        return X

    def train(self, samples, epochs=30, lr=1e-3, batch_size=256):
        """
        samples: list of dicts from DatasetGenerator
        Returns history dict.
        """
        # Pre-convert to numpy arrays
        partitions = np.array([s["partition"] for s in samples], dtype=np.int32)
        srcs = np.array([s["src"] for s in samples], dtype=np.int32)
        dsts = np.array([s["dst"] for s in samples], dtype=np.int32)
        zs = np.array([s["z"] for s in samples], dtype=np.float32)
        successes = np.array([s["success"] for s in samples], dtype=np.float32)
        delays = np.array([s["delay"] for s in samples], dtype=np.float32)

        N = len(samples)
        history = {"train_loss": [], "val_acc": [], "val_auc": [], "val_mae": [], "val_ece": []}

        # Simple train/val split (last 20%)
        split = int(0.8 * N)
        idx = np.random.permutation(N)
        tr_idx, val_idx = idx[:split], idx[split:]

        for epoch in range(epochs):
            np.random.shuffle(tr_idx)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(tr_idx), batch_size):
                bid = tr_idx[i:i + batch_size]
                X = self._build_X(partitions[bid], srcs[bid], dsts[bid], zs[bid])
                y = successes[bid]
                d = delays[bid]

                # Forward
                h1 = relu(X @ self.W1 + self.b1)
                # Training dropout 10%
                mask = (np.random.rand(*h1.shape) > 0.1).astype(np.float32) / 0.9
                h1 = h1 * mask
                h2 = relu(h1 @ self.W2 + self.b2)
                mask2 = (np.random.rand(*h2.shape) > 0.1).astype(np.float32) / 0.9
                h2 = h2 * mask2
                out = h2 @ self.W3 + self.b3
                logit = out[:, 0]
                delay_pred = out[:, 1]

                # Loss
                prob = sigmoid(logit)
                bce = -np.mean(y * np.log(prob + 1e-8) + (1 - y) * np.log(1 - prob + 1e-8))
                mask_succ = y > 0.5
                mse = np.mean((delay_pred[mask_succ] - d[mask_succ]) ** 2) if mask_succ.sum() > 0 else 0.0
                loss = bce + mse
                total_loss += loss
                n_batches += 1

                # Backprop (manual)
                # dL/dlogit
                dlogit = prob - y
                ddelay = np.zeros_like(delay_pred)
                ddelay[mask_succ] = 2 * (delay_pred[mask_succ] - d[mask_succ]) / mask_succ.sum() if mask_succ.sum() > 0 else 0
                dout = np.stack([dlogit, ddelay], axis=1)  # (B,2)

                dW3 = h2.T @ dout / len(bid)
                db3 = np.mean(dout, axis=0)
                dh2 = dout @ self.W3.T
                dh2[h2 <= 0] = 0  # ReLU backprop
                dh2 = dh2 * mask2

                dW2 = h1.T @ dh2 / len(bid)
                db2 = np.mean(dh2, axis=0)
                dh1 = dh2 @ self.W2.T
                dh1[h1 <= 0] = 0
                dh1 = dh1 * mask

                dW1 = X.T @ dh1 / len(bid)
                db1 = np.mean(dh1, axis=0)

                # Update
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W3 -= lr * dW3
                self.b3 -= lr * db3

            # Validation
            Xv = self._build_X(partitions[val_idx], srcs[val_idx], dsts[val_idx], zs[val_idx])
            yv = successes[val_idx]
            dv = delays[val_idx]
            logit_v, delay_v = self.predict_proba(Xv)
            prob_v = logit_v
            acc = np.mean((prob_v > 0.5) == yv)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(yv, prob_v) if len(set(yv)) > 1 else 0.5
            mae = float(np.mean(np.abs(delay_v[yv > 0.5] - dv[yv > 0.5]))) if (yv > 0.5).any() else 0.0
            ece = self._compute_ece(yv, prob_v)

            history["train_loss"].append(total_loss / n_batches)
            history["val_acc"].append(acc)
            history["val_auc"].append(auc)
            history["val_mae"].append(mae)
            history["val_ece"].append(ece)
            print(
                f"  Epoch {epoch+1:02d}: loss={history['train_loss'][-1]:.4f} "
                f"acc={acc:.4f} auc={auc:.4f} mae={mae:.6f} ece={ece:.4f}"
            )
        return history

    @staticmethod
    def _compute_ece(y_true, y_prob, n_bins=10):
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
            if mask.sum() == 0:
                continue
            ece += mask.sum() * np.abs(y_prob[mask].mean() - y_true[mask].mean())
        return float(ece / len(y_true))
