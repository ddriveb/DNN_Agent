"""
PINN-Predictor experiment: compare Baseline vs PINN under different sample sizes.
Train on NSFNET, test on NSFNET (same-topology generalization).
Sample sizes: 500, 2000, 8000
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from pinn_predictor import PINNPredictor, GradNormPINNPredictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor
from train_pinn import PINNDataset, pinn_collate_fn, train_pinn_predictor, evaluate_pinn_predictor
from torch.utils.data import DataLoader

MAX_SERVERS = 128
SAMPLE_SIZES = [500, 2000, 8000]
TEST_SIZE = 2000
EPOCHS = 20


def generate_samples(topology, num_samples, seed=42, include_pinn=False):
    """Generate dataset for a given sample size."""
    if topology == "nsfnet":
        n_nodes, num_slots, arr, ht, preload = 14, 32, 5.0, 10.0, 500
    else:
        raise ValueError(f"Unknown topology: {topology}")

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht,
                        preload=preload, include_pinn=include_pinn)


def get_num_links_from_samples(samples):
    """Extract num_links from PINN samples."""
    if "S_current" in samples[0]:
        return samples[0]["S_current"].shape[0]
    return None


def train_baseline(samples, epochs=20):
    """Train baseline MLP predictor."""
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    state_dim = len(samples[0]["z"])
    predictor = Predictor(state_dim, num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor


def train_pinn(samples, epochs=20, use_gradnorm=False):
    """Train PINN predictor."""
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(PINNDataset(samples[:split]), batch_size=256,
                              shuffle=True, collate_fn=pinn_collate_fn)
    val_loader = DataLoader(PINNDataset(samples[split:]), batch_size=256,
                            shuffle=False, collate_fn=pinn_collate_fn)

    state_dim = len(samples[0]["z"])
    num_links = get_num_links_from_samples(samples)
    num_slots = samples[0]["S_current"].shape[1]

    if use_gradnorm:
        predictor = GradNormPINNPredictor(
            state_dim, num_links, num_slots,
            num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64
        )
    else:
        predictor = PINNPredictor(
            state_dim, num_links, num_slots,
            num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64
        )

    train_pinn_predictor(predictor, train_loader, val_loader, epochs=epochs,
                         lr=1e-3, alpha_physics=0.01, use_gradnorm=use_gradnorm)
    return predictor


def evaluate_on_test(predictor, test_samples, is_pinn=False):
    """Evaluate predictor on test set."""
    if is_pinn:
        test_loader = DataLoader(PINNDataset(test_samples), batch_size=256,
                                 shuffle=False, collate_fn=pinn_collate_fn)
        return evaluate_pinn_predictor(predictor, test_loader)
    else:
        test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
        return evaluate_predictor(predictor, test_loader)


if __name__ == "__main__":
    print("=" * 70)
    print("PINN-PREDICTOR EXPERIMENT: Sample Efficiency Comparison")
    print("=" * 70)

    # Generate test set (shared across all experiments)
    print("\nGenerating test set (2000 samples) ...")
    test_samples = generate_samples("nsfnet", TEST_SIZE, seed=999, include_pinn=True)
    test_samples_baseline = generate_samples("nsfnet", TEST_SIZE, seed=999, include_pinn=False)

    results = []

    for n_samples in SAMPLE_SIZES:
        print(f"\n{'='*70}")
        print(f"SAMPLE SIZE = {n_samples}")
        print(f"{'='*70}")

        # Generate training data
        print(f"Generating {n_samples} training samples ...")
        train_samples_pinn = generate_samples("nsfnet", n_samples, seed=42, include_pinn=True)
        train_samples_base = generate_samples("nsfnet", n_samples, seed=42, include_pinn=False)

        # --- Baseline ---
        print(f"\n--- Baseline MLP ({n_samples} samples) ---")
        baseline_model = train_baseline(train_samples_base, epochs=EPOCHS)
        base_metrics = evaluate_on_test(baseline_model, test_samples_baseline, is_pinn=False)
        print(f"  Test AUC={base_metrics['auc']:.4f}  ECE={base_metrics['ece']:.4f}  "
              f"ACC={base_metrics['accuracy']:.4f}")

        # --- PINN Standard ---
        print(f"\n--- PINN Standard ({n_samples} samples) ---")
        pinn_model = train_pinn(train_samples_pinn, epochs=EPOCHS, use_gradnorm=False)
        pinn_metrics = evaluate_on_test(pinn_model, test_samples, is_pinn=True)
        print(f"  Test AUC={pinn_metrics['auc']:.4f}  ECE={pinn_metrics['ece']:.4f}  "
              f"ACC={pinn_metrics['accuracy']:.4f}")

        # --- PINN + GradNorm ---
        print(f"\n--- PINN + GradNorm ({n_samples} samples) ---")
        gn_model = train_pinn(train_samples_pinn, epochs=EPOCHS, use_gradnorm=True)
        gn_metrics = evaluate_on_test(gn_model, test_samples, is_pinn=True)
        print(f"  Test AUC={gn_metrics['auc']:.4f}  ECE={gn_metrics['ece']:.4f}  "
              f"ACC={gn_metrics['accuracy']:.4f}")

        results.append({
            "n_samples": n_samples,
            "baseline_auc": base_metrics["auc"],
            "baseline_ece": base_metrics["ece"],
            "baseline_acc": base_metrics["accuracy"],
            "pinn_auc": pinn_metrics["auc"],
            "pinn_ece": pinn_metrics["ece"],
            "pinn_acc": pinn_metrics["accuracy"],
            "gradnorm_auc": gn_metrics["auc"],
            "gradnorm_ece": gn_metrics["ece"],
            "gradnorm_acc": gn_metrics["accuracy"],
        })

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'N':>6} | {'Base AUC':>9} {'PINN AUC':>9} {'GN AUC':>9} | "
          f"{'Base ECE':>9} {'PINN ECE':>9} {'GN ECE':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_samples']:>6} | "
              f"{r['baseline_auc']:>9.4f} {r['pinn_auc']:>9.4f} {r['gradnorm_auc']:>9.4f} | "
              f"{r['baseline_ece']:>9.4f} {r['pinn_ece']:>9.4f} {r['gradnorm_ece']:>9.4f}")
    print("=" * 70)
