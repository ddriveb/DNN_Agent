"""
Large-scale generalization test across 4 topologies.
Intra-topology training + evaluation.
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor, plot_calibration
from torch.utils.data import DataLoader


# Topology configs: (name, num_slots, arrival_rate, avg_ht, preload)
TOPOLOGIES = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "usnet":     (28, 64,   8.0, 12.0, 800),
    "cost266":   (28, 64,   8.0, 12.0, 800),
    "random50":  (50, 128,  15.0, 15.0, 1500),
    "random80":  (80, 256,  25.0, 20.0, 2500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def run_single(topology, version, num_samples=10000, epochs=30):
    print(f"\n{'='*70}")
    print(f"Topology: {topology} | Encoder: {version} | Samples: {num_samples}")
    print(f"{'='*70}")

    np.random.seed(42)
    torch.manual_seed(42)

    n_nodes, num_slots, arr, ht, preload = TOPOLOGIES[topology]

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=42)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)

    if version == "v0":
        encoder.encode = encoder.encode_v0
    elif version == "v2b":
        encoder.encode = encoder.encode_v2b
    else:
        raise ValueError(version)

    print(f"  Nodes: {net.NUM_NODES}, Slots: {num_slots}")
    print("  Generating dataset ...")
    gen = DatasetGenerator(net, mapper, encoder, seed=42)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)

    succ_rate = np.mean([s["success"] for s in samples])
    print(f"  Dataset: {len(samples)} | Success rate: {succ_rate:.3f} | State dim: {len(samples[0]['z'])}")

    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(len(samples[0]["z"]), num_partitions=3, max_servers=128, hidden_dim=64)
    history = train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)

    final = evaluate_predictor(predictor, val_loader)
    print(f"\n  Final — acc={final['accuracy']:.4f} auc={final['auc']:.4f} mae={final['delay_mae']:.6f} ece={final['ece']:.4f}")

    plot_calibration(predictor, val_loader, save_path=f"calibration_{topology}_{version}_large.png")
    return final


if __name__ == "__main__":
    results = {}

    test_topos = ["nsfnet", "usnet", "cost266", "random50", "random80", "random100"]
    for topo in test_topos:
        for ver in ["v0", "v2b"]:
            results[(topo, ver)] = run_single(topo, ver, num_samples=10000, epochs=30)

    print("\n" + "="*70)
    print("LARGE-SCALE GENERALIZATION SUMMARY (Intra-Topology)")
    print("="*70)
    print(f"{'Topology':<12} {'Nodes':>5} {'Version':<6} {'Acc':>6} {'AUC':>6} {'ECE':>6}")
    print("-"*70)
    for topo in test_topos:
        n_nodes = TOPOLOGIES[topo][0]
        for ver in ["v0", "v2b"]:
            r = results[(topo, ver)]
            print(f"{topo:<12} {n_nodes:>5} {ver:<6} {r['accuracy']:6.4f} {r['auc']:6.4f} {r['ece']:6.4f}")

    print("\n" + "="*70)
    print("AUC IMPROVEMENT (v2b - v0)")
    print("="*70)
    for topo in test_topos:
        n_nodes = TOPOLOGIES[topo][0]
        delta = results[(topo, "v2b")]["auc"] - results[(topo, "v0")]["auc"]
        print(f"{topo:<12} ({n_nodes:>3} nodes):  +{delta:.4f}")
