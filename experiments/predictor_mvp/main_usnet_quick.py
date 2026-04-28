"""Quick USNET generalization test (lightweight)."""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor, plot_calibration
from torch.utils.data import DataLoader


def run(topology, version, num_samples=5000, epochs=20):
    print(f"\n{'='*60}")
    print(f"Topology: {topology} | Encoder: {version}")
    print(f"{'='*60}")

    np.random.seed(42)
    torch.manual_seed(42)

    slots = 64 if topology == "usnet" else 32
    arr = 8.0 if topology == "usnet" else 5.0
    ht = 12.0 if topology == "usnet" else 10.0

    net = OpticalNetwork(topology=topology, num_slots=slots, seed=42)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)

    if version == "v0":
        encoder.encode = encoder.encode_v0
    elif version == "v2b":
        encoder.encode = encoder.encode_v2b

    gen = DatasetGenerator(net, mapper, encoder, seed=42)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=300)

    succ_rate = np.mean([s["success"] for s in samples])
    print(f"  Dataset: {len(samples)} | Success rate: {succ_rate:.3f} | State dim: {len(samples[0]['z'])}")

    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(len(samples[0]["z"]), num_partitions=3, num_servers=net.NUM_NODES, hidden_dim=64)
    history = train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)

    final = evaluate_predictor(predictor, val_loader)
    print(f"\n  Final — acc={final['accuracy']:.4f} auc={final['auc']:.4f} ece={final['ece']:.4f}")
    plot_calibration(predictor, val_loader, save_path=f"calibration_{topology}_{version}.png")
    return final


if __name__ == "__main__":
    print("NSFNET quick baseline:")
    r_nsf_v0 = run("nsfnet", "v0", num_samples=5000, epochs=20)
    r_nsf_v2b = run("nsfnet", "v2b", num_samples=5000, epochs=20)

    print("\nUSNET generalization:")
    r_us_v0 = run("usnet", "v0", num_samples=5000, epochs=20)
    r_us_v2b = run("usnet", "v2b", num_samples=5000, epochs=20)

    print("\n" + "="*60)
    print("GENERALIZATION SUMMARY")
    print("="*60)
    print(f"{'Topology':<10} {'Version':<6} {'Acc':>6} {'AUC':>6} {'ECE':>6}")
    for topo, ver, res in [("nsfnet","v0",r_nsf_v0), ("nsfnet","v2b",r_nsf_v2b),
                           ("usnet","v0",r_us_v0), ("usnet","v2b",r_us_v2b)]:
        print(f"{topo:<10} {ver:<6} {res['accuracy']:6.4f} {res['auc']:6.4f} {res['ece']:6.4f}")
