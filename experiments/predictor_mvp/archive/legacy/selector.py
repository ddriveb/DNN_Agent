"""
Selector: Choose the best path_id given (src, dst, bw, z).
Supports multiple strategies: max_prob, min_delay, weighted_score.
"""
import numpy as np
import torch


class Selector:
    """
    Selector evaluates all candidate paths and picks the best one.

    Usage:
        selector = Selector(predictor, encoder, strategy="max_prob")
        best, all_candidates = selector.select(src=0, dst=5, bw=8)
    """

    def __init__(self, predictor, encoder, strategy="max_prob", delay_weight=0.01):
        self.predictor = predictor
        self.encoder = encoder
        self.strategy = strategy
        self.delay_weight = delay_weight
        self.num_paths = 3  # KSP paths

    def select(self, src, dst, bw):
        """
        Evaluate all candidate paths and return the best one + full list.

        Args:
            src: source node id
            dst: destination node id
            bw: bandwidth requirement in slots (float or int)

        Returns:
            best: dict with keys {path_id, success_prob, delay}
            candidates: list of dicts for all paths
        """
        z = self.encoder.encode(src, dst)
        z_tensor = torch.from_numpy(z).unsqueeze(0).float()
        bw_tensor = torch.tensor([float(bw)], dtype=torch.float32)
        src_tensor = torch.tensor([src], dtype=torch.long)
        dst_tensor = torch.tensor([dst], dtype=torch.long)

        self.predictor.eval()
        candidates = []
        with torch.no_grad():
            for path_id in range(self.num_paths):
                path_tensor = torch.tensor([path_id], dtype=torch.long)
                logit, delay_pred = self.predictor(
                    path_tensor, src_tensor, dst_tensor, bw_tensor, z_tensor
                )
                prob = torch.sigmoid(logit).item()
                delay = delay_pred.item()
                candidates.append({
                    "path_id": path_id,
                    "success_prob": prob,
                    "delay": delay,
                })

        if self.strategy == "max_prob":
            best = max(candidates, key=lambda x: x["success_prob"])
        elif self.strategy == "min_delay":
            # Only consider paths with reasonable success probability
            viable = [c for c in candidates if c["success_prob"] > 0.5]
            if not viable:
                viable = candidates
            best = min(viable, key=lambda x: x["delay"])
        elif self.strategy == "weighted":
            best = max(
                candidates,
                key=lambda x: x["success_prob"] - self.delay_weight * x["delay"]
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return best, candidates

    def batch_select(self, requests):
        """
        Batch evaluation for multiple (src, dst, bw) requests.

        Args:
            requests: list of dicts [{"src": 0, "dst": 5, "bw": 8}, ...]

        Returns:
            list of (best, candidates) tuples
        """
        return [self.select(r["src"], r["dst"], r["bw"]) for r in requests]
