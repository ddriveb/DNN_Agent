"""Analyze predictor behavior: P_success distribution, calibration, discrimination.

Why does nsfnet-specific predictor outperform mixed on USNET?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
# No sklearn — implement metrics in pure numpy

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator


def load_predictor(path: str, max_servers: int = 128):
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location="cpu", weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    return predictor


def generate_test_data(topology: str, num_slots: int, num_samples: int, seed: int):
    """Generate labeled test data for a given topology."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)

    # Use moderate load parameters
    if topology == "nsfnet":
        arr, ht, preload = 5.0, 10.0, 500
    elif topology == "usnet":
        arr, ht, preload = 8.0, 12.0, 800
    else:
        arr, ht, preload = 15.0, 15.0, 1500

    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)
    return samples


def evaluate_predictor_on_data(predictor, samples, name: str):
    """Evaluate predictor on a dataset and print metrics."""
    partitions = torch.tensor([s["partition"] for s in samples], dtype=torch.long)
    srcs = torch.tensor([s["src"] for s in samples], dtype=torch.long)
    dsts = torch.tensor([s["dst"] for s in samples], dtype=torch.long)
    zs = torch.stack([torch.from_numpy(s["z"].copy()) for s in samples])
    labels = np.array([s["success"] for s in samples])

    with torch.no_grad():
        logits, delays = predictor(partitions, srcs, dsts, zs)
        probs = torch.sigmoid(logits).numpy()

    # Metrics
    auc = compute_auc(labels, probs)
    brier = compute_brier(labels, probs)
    acc = np.mean((probs > 0.5) == labels)

    # Calibration (ECE)
    ece = compute_ece(labels, probs, n_bins=10)

    # Discrimination: mean prob for success vs failure
    success_probs = probs[labels == 1]
    failure_probs = probs[labels == 0]
    prob_gap = np.mean(success_probs) - np.mean(failure_probs)

    print(f"\n{name}:")
    print(f"  AUC={auc:.4f}  Brier={brier:.4f}  Acc={acc:.4f}  ECE={ece:.4f}")
    print(f"  P(success|label=1)={np.mean(success_probs):.4f}  P(success|label=0)={np.mean(failure_probs):.4f}  Gap={prob_gap:.4f}")
    print(f"  P(success) distribution: min={probs.min():.3f}  max={probs.max():.3f}  mean={probs.mean():.3f}  std={probs.std():.3f}")

    return {
        "auc": auc, "brier": brier, "acc": acc, "ece": ece,
        "prob_gap": prob_gap,
        "success_mean_prob": float(np.mean(success_probs)),
        "failure_mean_prob": float(np.mean(failure_probs)),
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
    }


def compute_auc(y_true, y_prob):
    """Compute AUC using Mann-Whitney U statistic."""
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    # Rank all scores
    sorted_idx = np.argsort(y_prob)
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[sorted_idx] = np.arange(1, len(y_prob) + 1)
    # Handle ties
    unique_vals, inverse, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    for i, val in enumerate(unique_vals):
        mask = inverse == i
        if counts[i] > 1:
            ranks[mask] = ranks[mask].mean()
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    rank_sum_pos = np.sum(ranks[y_true == 1])
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_brier(y_true, y_prob):
    """Compute Brier score: mean squared error of probability predictions."""
    return float(np.mean((y_prob - y_true) ** 2))


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


def main():
    base = Path(__file__).parent.parent / "predictor_mvp"
    predictors = {
        "mixed-v2b": load_predictor(str(base / "pretrained_mixed_v2b.pt"), 128),
        "nsfnet-v2b": load_predictor(str(base / "pretrained_nsfnet_v2b.pt"), 128),
    }

    print("=" * 70)
    print("PREDICTOR ANALYSIS: Why does nsfnet-specific beat mixed on USNET?")
    print("=" * 70)

    # Test on NSFNET
    print("\n--- Test data: NSFNET (in-domain for nsfnet-specific) ---")
    nsfnet_samples = generate_test_data("nsfnet", 32, 5000, seed=100)
    nsfnet_results = {}
    for name, predictor in predictors.items():
        nsfnet_results[name] = evaluate_predictor_on_data(predictor, nsfnet_samples, name)

    # Test on USNET
    print("\n--- Test data: USNET (cross-domain) ---")
    usnet_samples = generate_test_data("usnet", 64, 5000, seed=100)
    usnet_results = {}
    for name, predictor in predictors.items():
        usnet_results[name] = evaluate_predictor_on_data(predictor, usnet_samples, name)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-topology gap")
    print("=" * 70)

    for metric in ["auc", "brier", "ece", "prob_gap"]:
        print(f"\n{metric.upper()}:")
        for name in predictors.keys():
            nsf_val = nsfnet_results[name][metric]
            us_val = usnet_results[name][metric]
            gap = us_val - nsf_val
            print(f"  {name:<15}: NSFNET={nsf_val:.4f}  USNET={us_val:.4f}  Gap={gap:+.4f}")

    # Save
    import json
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "predictor_analysis.json", "w") as f:
        json.dump({
            "nsfnet": {k: {m: float(v) for m, v in d.items()} for k, d in nsfnet_results.items()},
            "usnet": {k: {m: float(v) for m, v in d.items()} for k, d in usnet_results.items()},
        }, f, indent=2)
    print(f"\nSaved to {out_dir / 'predictor_analysis.json'}")


if __name__ == "__main__":
    main()
