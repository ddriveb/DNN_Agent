"""Scikit-learn MLP fallback — much faster than NumPy manual backprop."""
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor


class SklearnPredictor:
    def __init__(self, state_dim, hidden_dim=64):
        self.input_dim = 3 + 14 + 14 + state_dim
        self.cls = MLPClassifier(
            hidden_layer_sizes=(hidden_dim, hidden_dim),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=1,
            warm_start=True,
            early_stopping=False,
        )
        self.reg = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, hidden_dim),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=1,
            warm_start=True,
            early_stopping=False,
        )
        self._cls_fitted = False
        self._reg_fitted = False

    def _build_X(self, samples):
        B = len(samples)
        X = np.zeros((B, self.input_dim), dtype=np.float32)
        for i, s in enumerate(samples):
            off = 0
            X[i, off + int(s["partition"])] = 1.0
            off += 3
            X[i, off + int(s["src"])] = 1.0
            off += 14
            X[i, off + int(s["dst"])] = 1.0
            off += 14
            X[i, off:] = s["z"]
        return X

    def train(self, train_samples, val_samples, epochs=30):
        X_train = self._build_X(train_samples)
        y_train = np.array([s["success"] for s in train_samples])
        d_train = np.array([s["delay"] for s in train_samples])

        X_val = self._build_X(val_samples)
        y_val = np.array([s["success"] for s in val_samples])
        d_val = np.array([s["delay"] for s in val_samples])

        history = {"train_loss": [], "val_acc": [], "val_auc": [], "val_mae": [], "val_ece": []}

        for epoch in range(epochs):
            self.cls.fit(X_train, y_train)
            self._cls_fitted = True

            succ_mask = y_train > 0.5
            if succ_mask.sum() > 0:
                self.reg.fit(X_train[succ_mask], d_train[succ_mask])
                self._reg_fitted = True

            prob_val = self.cls.predict_proba(X_val)[:, 1]
            pred_delay = self.reg.predict(X_val) if self._reg_fitted else np.zeros(len(val_samples))

            from sklearn.metrics import roc_auc_score, accuracy_score
            acc = accuracy_score(y_val, prob_val > 0.5)
            auc = roc_auc_score(y_val, prob_val) if len(set(y_val)) > 1 else 0.5
            mae = float(np.mean(np.abs(pred_delay[y_val > 0.5] - d_val[y_val > 0.5]))) if (y_val > 0.5).any() else 0.0
            ece = self._compute_ece(y_val, prob_val)

            # BCE as proxy for training loss
            prob_train = self.cls.predict_proba(X_train)[:, 1]
            train_loss = float(-np.mean(y_train * np.log(prob_train + 1e-8) + (1 - y_train) * np.log(1 - prob_train + 1e-8)))

            history["train_loss"].append(train_loss)
            history["val_acc"].append(acc)
            history["val_auc"].append(auc)
            history["val_mae"].append(mae)
            history["val_ece"].append(ece)

            print(
                f"  Epoch {epoch+1:02d}: loss={train_loss:.4f} "
                f"acc={acc:.4f} auc={auc:.4f} mae={mae:.6f} ece={ece:.4f}"
            )
        return history

    def predict(self, samples):
        X = self._build_X(samples)
        prob = self.cls.predict_proba(X)[:, 1] if self._cls_fitted else np.ones(len(samples)) * 0.5
        delay = self.reg.predict(X) if self._reg_fitted else np.zeros(len(samples))
        return prob, delay

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
