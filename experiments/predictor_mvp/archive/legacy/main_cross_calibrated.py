"""
Cross-topology transfer with post-hoc calibration.
Train on NSFNET, test on diverse topologies with Temperature/Platt scaling.
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor
from calibration_methods import TemperatureScaling, PlattScaling, evaluate_with_calibration
from torch.utils.data import DataLoader


MAX_SERVERS = 128

# Topology configs: (name, num_slots, arrival_rate, avg_ht, preload)
TOPOLOGIES = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "usnet":     (28, 64,   8.0, 12.0, 800),
    "cost266":   (28, 64,   8.0, 12.0, 800),
    "germany50": (50, 128, 15.0, 15.0, 1500),
    "random50":  (50, 128, 15.0, 15.0, 1500),
    "random80":  (80, 256, 25.0, 20.0, 2500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def generate_data(topology, version, num_samples, seed=42):
    n_nodes, num_slots, arr, ht, preload = TOPOLOGIES[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    if version == "v0":
        encoder.encode = encoder.encode_v0
    elif version == "v2b":
        encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)
    return samples


def collect_logits_labels(predictor, loader):
    """Collect raw logits and labels from a dataloader."""
    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logit, _ = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            all_logits.extend(logit.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    return np.array(all_logits), np.array(all_labels)


def run_experiment(version, train_topo="nsfnet", test_topo="usnet"):
    print(f"\n{'='*75}")
    print(f"Train: {train_topo} → Test: {test_topo} | Encoder: {version}")
    print(f"{'='*75}")

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Train on source topology
    print("  [1/4] Training on source topology ...")
    train_samples = generate_data(train_topo, version, 15000, seed=42)
    print(f"    Train: {len(train_samples)} | SR: {np.mean([s['success'] for s in train_samples]):.3f}")
    state_dim = len(train_samples[0]["z"])

    split = int(0.8 * len(train_samples))
    train_loader = DataLoader(OpticalDataset(train_samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(train_samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(state_dim, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3)

    # 2. Generate calibration + test data on target topology
    print("  [2/4] Generating calibration data on target ...")
    calib_samples = generate_data(test_topo, version, 2000, seed=100)
    print(f"    Calib: {len(calib_samples)} | SR: {np.mean([s['success'] for s in calib_samples]):.3f}")

    print("  [3/4] Generating test data on target ...")
    test_samples = generate_data(test_topo, version, 3000, seed=200)
    print(f"    Test:  {len(test_samples)} | SR: {np.mean([s['success'] for s in test_samples]):.3f}")

    # 3. Fit calibrators
    calib_loader = DataLoader(OpticalDataset(calib_samples), batch_size=256, shuffle=False)
    calib_logits, calib_labels = collect_logits_labels(predictor, calib_loader)

    ts = TemperatureScaling().fit(calib_logits, calib_labels)
    platt = PlattScaling().fit(calib_logits, calib_labels)
    print(f"    Temperature Scaling: T={ts.T:.4f}")
    print(f"    Platt Scaling: a={platt.a:.4f}, b={platt.b:.4f}")

    # 4. Evaluate on test set (3 modes)
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)

    metrics_raw = evaluate_with_calibration(predictor, test_loader, calibrator=None)
    metrics_ts = evaluate_with_calibration(predictor, test_loader, calibrator=ts)
    metrics_platt = evaluate_with_calibration(predictor, test_loader, calibrator=platt)

    print(f"\n  [4/4] Test Results on {test_topo}:")
    print(f"    {'Method':<15} {'Acc':>6} {'AUC':>6} {'ECE':>6}")
    print(f"    {'-'*40}")
    for name, m in [("Raw", metrics_raw), ("TempScale", metrics_ts), ("Platt", metrics_platt)]:
        print(f"    {name:<15} {m['accuracy']:6.4f} {m['auc']:6.4f} {m['ece']:6.4f}")

    return {
        "raw": metrics_raw, "ts": metrics_ts, "platt": metrics_platt,
        "T": ts.T, "a": platt.a, "b": platt.b
    }


if __name__ == "__main__":
    results = {}
    test_topologies = ["usnet", "cost266", "germany50", "random50", "random80", "random100"]

    for ver in ["v0", "v2b"]:
        for test_topo in test_topologies:
            results[(ver, test_topo)] = run_experiment(ver, "nsfnet", test_topo)

    # Summary tables
    print("\n" + "="*75)
    print("SUMMARY: Cross-Topology AUC (Raw / TempScale / Platt)")
    print("="*75)
    print(f"{'Test Topo':<12} {'Nodes':>5} {'Ver':<5} {'Raw':>6} {'+T.S.':>6} {'+Platt':>7}")
    print("-"*55)
    for test_topo in test_topologies:
        n_nodes = TOPOLOGIES[test_topo][0]
        for ver in ["v0", "v2b"]:
            r = results[(ver, test_topo)]
            print(f"{test_topo:<12} {n_nodes:>5} {ver:<5} "
                  f"{r['raw']['auc']:6.4f} {r['ts']['auc']:6.4f} {r['platt']['auc']:7.4f}")

    print("\n" + "="*75)
    print("SUMMARY: Cross-Topology ECE (Raw / TempScale / Platt)")
    print("="*75)
    print(f"{'Test Topo':<12} {'Nodes':>5} {'Ver':<5} {'Raw':>6} {'+T.S.':>6} {'+Platt':>7}")
    print("-"*55)
    for test_topo in test_topologies:
        n_nodes = TOPOLOGIES[test_topo][0]
        for ver in ["v0", "v2b"]:
            r = results[(ver, test_topo)]
            print(f"{test_topo:<12} {n_nodes:>5} {ver:<5} "
                  f"{r['raw']['ece']:6.4f} {r['ts']['ece']:6.4f} {r['platt']['ece']:7.4f}")

    print("\n" + "="*75)
    print("CALIBRATION PARAMETER SUMMARY")
    print("="*75)
    print(f"{'Test Topo':<12} {'Ver':<5} {'T':>8} {'a':>8} {'b':>8}")
    print("-"*45)
    for test_topo in test_topologies:
        for ver in ["v0", "v2b"]:
            r = results[(ver, test_topo)]
            print(f"{test_topo:<12} {ver:<5} {r['T']:8.4f} {r['a']:8.4f} {r['b']:8.4f}")
