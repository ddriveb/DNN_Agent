"""
Topology-conditioned predictor experiment.
Compare v2b vs v2t (v2b + topology statistics) on cross-topology transfer.
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor
from calibration_methods import PlattScaling, evaluate_with_calibration
from torch.utils.data import DataLoader

MAX_SERVERS = 128

TOPO_CONFIGS = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def generate_samples(topology, encoder_version, num_samples, seed=42):
    n_nodes, num_slots, arr, ht, preload = TOPO_CONFIGS[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    if encoder_version == "v2b":
        encoder.encode = encoder.encode_v2b
    elif encoder_version == "v2t":
        encoder.encode = encoder.encode_v2t
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)


def train_and_eval(train_topo, test_topo, encoder_version, train_n=15000, test_n=3000, calib_n=2000, epochs=30):
    print(f"\n{'='*70}")
    print(f"Train: {train_topo} → Test: {test_topo} | Encoder: {encoder_version}")
    print(f"{'='*70}")

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate training data
    print(f"  Generating {train_n} training samples on {train_topo} ...")
    train_samples = generate_samples(train_topo, encoder_version, train_n, seed=42)
    print(f"    Train: {len(train_samples)} | SR: {np.mean([s['success'] for s in train_samples]):.3f} | z_dim: {len(train_samples[0]['z'])}")

    split = int(0.8 * len(train_samples))
    train_loader = DataLoader(OpticalDataset(train_samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(train_samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(len(train_samples[0]["z"]), num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)

    # Generate test data
    print(f"  Generating {test_n} test samples on {test_topo} ...")
    test_samples = generate_samples(test_topo, encoder_version, test_n, seed=200)
    print(f"    Test: {len(test_samples)} | SR: {np.mean([s['success'] for s in test_samples]):.3f}")

    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw = evaluate_predictor(predictor, test_loader)
    print(f"  Raw    — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

    # Platt Scaling
    calib_samples = generate_samples(test_topo, encoder_version, calib_n, seed=100)
    calib_loader = DataLoader(OpticalDataset(calib_samples), batch_size=256, shuffle=False)
    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in calib_loader:
            logit, _ = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            all_logits.extend(logit.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    platt = PlattScaling().fit(np.array(all_logits), np.array(all_labels))
    platt_result = evaluate_with_calibration(predictor, test_loader, calibrator=platt)
    print(f"  +Platt — acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:.4f}")

    return {"raw": raw, "platt": platt_result}


if __name__ == "__main__":
    results = {}

    # Baseline: v2b
    results[("v2b", "nsfnet", "random100")] = train_and_eval(
        "nsfnet", "random100", "v2b", train_n=15000, test_n=3000, calib_n=2000, epochs=30
    )

    # Topology-conditioned: v2t
    results[("v2t", "nsfnet", "random100")] = train_and_eval(
        "nsfnet", "random100", "v2t", train_n=15000, test_n=3000, calib_n=2000, epochs=30
    )

    # Also test intra-topology for v2t on Random100
    print(f"\n{'='*70}")
    print("INTRA-TOPOLOGY: Random100 | Encoder: v2t")
    print(f"{'='*70}")
    np.random.seed(42)
    torch.manual_seed(42)
    intra_samples = generate_samples("random100", "v2t", 10000, seed=42)
    print(f"  Train: {len(intra_samples)} | SR: {np.mean([s['success'] for s in intra_samples]):.3f} | z_dim: {len(intra_samples[0]['z'])}")
    split = int(0.8 * len(intra_samples))
    train_loader = DataLoader(OpticalDataset(intra_samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(intra_samples[split:]), batch_size=256, shuffle=False)
    predictor = Predictor(len(intra_samples[0]["z"]), num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3)
    intra = evaluate_predictor(predictor, val_loader)
    print(f"  Intra  — acc={intra['accuracy']:.4f} auc={intra['auc']:.4f} ece={intra['ece']:.4f}")

    # Summary
    print("\n" + "="*70)
    print("TOPOLOGY-CONDITIONED PREDICTOR SUMMARY")
    print("="*70)
    print(f"{'Encoder':<8} {'Train→Test':<20} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12}")
    print("-"*70)
    for key, r in results.items():
        enc, train, test = key
        print(f"{enc:<8} {train+'→'+test:<20} {r['raw']['auc']:10.4f} {r['platt']['auc']:12.4f} "
              f"{r['raw']['ece']:10.4f} {r['platt']['ece']:12.4f}")
    print(f"{'v2t':<8} {'Random100(intra)':<20} {intra['auc']:10.4f} {'—':>12} {intra['ece']:10.4f} {'—':>12}")
