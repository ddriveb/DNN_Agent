"""
Robustness test: v2b predictor stability across random seeds.
Tests both intra-topology (NSFNET) and large-scale (Random100).
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


TOPOLOGIES = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def run_single_seed(topology, seed, num_samples=8000, epochs=25):
    """Run one experiment with a given random seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_nodes, num_slots, arr, ht, preload = TOPOLOGIES[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b

    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)

    succ_rate = np.mean([s["success"] for s in samples])

    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(len(samples[0]["z"]), num_partitions=3, max_servers=128, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)

    metrics = evaluate_predictor(predictor, val_loader)
    metrics["seed"] = seed
    metrics["sr"] = succ_rate
    return metrics


if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 2024]

    for topo in ["nsfnet", "random100"]:
        print(f"\n{'='*70}")
        print(f"ROBUSTNESS TEST: {topo.upper()} | v2b | {len(seeds)} seeds")
        print(f"{'='*70}")

        results = []
        for seed in seeds:
            r = run_single_seed(topo, seed, num_samples=8000, epochs=25)
            results.append(r)
            print(f"  Seed {seed:4d}: acc={r['accuracy']:.4f} auc={r['auc']:.4f} "
                  f"mae={r['delay_mae']:.6f} ece={r['ece']:.4f} | SR={r['sr']:.3f}")

        # Statistics
        aucs = [r["auc"] for r in results]
        accs = [r["accuracy"] for r in results]
        eces = [r["ece"] for r in results]

        print(f"\n  {'Metric':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*50}")
        for name, vals in [("AUC", aucs), ("Acc", accs), ("ECE", eces)]:
            print(f"  {name:<10} {np.mean(vals):8.4f} {np.std(vals):8.4f} "
                  f"{np.min(vals):8.4f} {np.max(vals):8.4f}")
