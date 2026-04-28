"""
Cross-topology transfer learning.
Train on one topology, test on others (zero-shot generalization).
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor
from torch.utils.data import DataLoader


MAX_SERVERS = 64

# Topology configs: (name, num_slots, arrival_rate, avg_ht, preload)
TOPOLOGIES = {
    "nsfnet":    (14, 32,  5.0, 10.0, 500),
    "usnet":     (28, 64,  8.0, 12.0, 800),
    "cost266":   (28, 64,  8.0, 12.0, 800),
    "germany50": (50, 128, 15.0, 15.0, 1500),
    "random50":  (50, 128, 15.0, 15.0, 1500),
}


def generate_data(topology, version, num_samples, seed=42):
    """Generate dataset for a given topology and encoder version."""
    n_nodes, num_slots, arr, ht, preload = TOPOLOGIES[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)

    if version == "v0":
        encoder.encode = encoder.encode_v0
    elif version == "v2b":
        encoder.encode = encoder.encode_v2b
    else:
        raise ValueError(version)

    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)
    return samples, net.NUM_NODES


def train_predictor_on_data(samples, state_dim, epochs=30):
    """Train a predictor on given samples."""
    np.random.seed(42)
    torch.manual_seed(42)

    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(state_dim, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    history = train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor, history


def evaluate_on_data(predictor, samples):
    """Evaluate a trained predictor on new samples (different topology)."""
    loader = DataLoader(OpticalDataset(samples), batch_size=256, shuffle=False)
    return evaluate_predictor(predictor, loader)


def run_cross_topology(train_topo, test_topo, version, train_samples=15000, test_samples=5000):
    """Train on train_topo, test on test_topo."""
    print(f"\n{'='*70}")
    print(f"Train: {train_topo} → Test: {test_topo} | Encoder: {version}")
    print(f"{'='*70}")

    # Generate training data
    print("  Generating training data ...")
    train_samples_data, _ = generate_data(train_topo, version, train_samples, seed=42)
    print(f"    Train: {len(train_samples_data)} | SR: {np.mean([s['success'] for s in train_samples_data]):.3f}")

    # Train
    state_dim = len(train_samples_data[0]["z"])
    print(f"    State dim: {state_dim}")
    predictor, _ = train_predictor_on_data(train_samples_data, state_dim, epochs=30)

    # Generate test data
    print("  Generating test data ...")
    test_samples_data, _ = generate_data(test_topo, version, test_samples, seed=123)
    print(f"    Test: {len(test_samples_data)} | SR: {np.mean([s['success'] for s in test_samples_data]):.3f}")

    # Evaluate
    metrics = evaluate_on_data(predictor, test_samples_data)
    print(f"\n  Test Results — acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
          f"mae={metrics['delay_mae']:.6f} ece={metrics['ece']:.4f}")

    return metrics


def run_intra_topology(topo, version, num_samples=15000):
    """Train and test on the same topology (baseline)."""
    print(f"\n{'='*70}")
    print(f"Intra-topology: {topo} | Encoder: {version}")
    print(f"{'='*70}")

    print("  Generating data ...")
    samples, _ = generate_data(topo, version, num_samples, seed=42)
    print(f"    Samples: {len(samples)} | SR: {np.mean([s['success'] for s in samples]):.3f}")

    state_dim = len(samples[0]["z"])
    print(f"    State dim: {state_dim}")

    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(state_dim, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    history = train_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3)

    metrics = evaluate_predictor(predictor, val_loader)
    print(f"\n  Val Results — acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
          f"mae={metrics['delay_mae']:.6f} ece={metrics['ece']:.4f}")
    return metrics


if __name__ == "__main__":
    results = {}

    # Experiment 1: Intra-topology baselines
    print("\n" + "="*70)
    print("PART 1: INTRA-TOPOLOGY BASELINES")
    print("="*70)
    for topo in ["nsfnet", "usnet", "cost266", "germany50", "random50"]:
        for ver in ["v0", "v2b"]:
            results[(topo, topo, ver)] = run_intra_topology(topo, ver, num_samples=15000)

    # Experiment 2: Cross-topology transfer (NSFNET -> others)
    print("\n" + "="*70)
    print("PART 2: CROSS-TOPOLOGY TRANSFER (NSFNET -> X)")
    print("="*70)
    for test_topo in ["usnet", "cost266", "germany50", "random50"]:
        for ver in ["v0", "v2b"]:
            results[("nsfnet", test_topo, ver)] = run_cross_topology(
                "nsfnet", test_topo, ver, train_samples=15000, test_samples=5000
            )

    # Summary tables
    print("\n" + "="*70)
    print("SUMMARY TABLE 1: INTRA-TOPOLOGY BASELINES")
    print("="*70)
    print(f"{'Topology':<12} {'Nodes':>5} {'Version':<6} {'Acc':>6} {'AUC':>6} {'ECE':>6}")
    print("-"*70)
    for topo in ["nsfnet", "usnet", "cost266", "germany50", "random50"]:
        n_nodes = TOPOLOGIES[topo][0]
        for ver in ["v0", "v2b"]:
            r = results[(topo, topo, ver)]
            print(f"{topo:<12} {n_nodes:>5} {ver:<6} {r['accuracy']:6.4f} {r['auc']:6.4f} {r['ece']:6.4f}")

    print("\n" + "="*70)
    print("SUMMARY TABLE 2: CROSS-TOPOLOGY TRANSFER (NSFNET -> X)")
    print("="*70)
    print(f"{'Train':<10} {'Test':<12} {'Ver':<5} {'Acc':>6} {'AUC':>6} {'ECE':>6} {'ΔAUC(v2b-v0)':>12}")
    print("-"*70)
    for test_topo in ["usnet", "cost266", "germany50", "random50"]:
        n_nodes = TOPOLOGIES[test_topo][0]
        r0 = results[("nsfnet", test_topo, "v0")]
        rb = results[("nsfnet", test_topo, "v2b")]
        delta = rb['auc'] - r0['auc']
        print(f"{'nsfnet':<10} {test_topo:<12} {'v0':<5} {r0['accuracy']:6.4f} {r0['auc']:6.4f} {r0['ece']:6.4f}")
        print(f"{'nsfnet':<10} {test_topo:<12} {'v2b':<5} {rb['accuracy']:6.4f} {rb['auc']:6.4f} {rb['ece']:6.4f} {delta:>+11.4f}")

    print("\n" + "="*70)
    print("SUMMARY TABLE 3: TRANSFER GAP (Intra - Cross)")
    print("="*70)
    print(f"{'Test Topo':<12} {'Ver':<5} {'Intra AUC':>10} {'Cross AUC':>10} {'Gap':>8}")
    print("-"*70)
    for test_topo in ["usnet", "cost266", "germany50", "random50"]:
        for ver in ["v0", "v2b"]:
            intra = results[(test_topo, test_topo, ver)]['auc']
            cross = results[("nsfnet", test_topo, ver)]['auc']
            gap = intra - cross
            print(f"{test_topo:<12} {ver:<5} {intra:10.4f} {cross:10.4f} {gap:8.4f}")
