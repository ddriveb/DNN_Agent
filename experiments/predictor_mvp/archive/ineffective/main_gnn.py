"""
GNN predictor experiment.
Train on NSFNET, zero-shot test on Random100.
Compare GNN vs v2b baseline.
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from gnn_predictor import GNNPredictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor
from train_gnn import GNNDataset, gnn_collate_fn, train_gnn_predictor, evaluate_gnn_predictor
from calibration_methods import PlattScaling, evaluate_with_calibration
from torch.utils.data import DataLoader

MAX_SERVERS = 128

TOPO_CONFIGS = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def generate_samples(topology, encoder_version, num_samples, seed=42, include_graph=False):
    n_nodes, num_slots, arr, ht, preload = TOPO_CONFIGS[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    if encoder_version == "v0":
        encoder.encode = encoder.encode_v0
    elif encoder_version == "v2b":
        encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload,
                        include_edge_features=include_graph)


def train_v2b_baseline(samples, epochs=30):
    print(f"\n{'='*70}")
    print(f"TRAIN v2b BASELINE on NSFNET")
    print(f"{'='*70}")
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)
    predictor = Predictor(len(samples[0]["z"]), num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor


def train_gnn(samples, epochs=30):
    print(f"\n{'='*70}")
    print(f"TRAIN GNN on NSFNET")
    print(f"{'='*70}")
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_dataset = GNNDataset(samples[:split])
    val_dataset = GNNDataset(samples[split:])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=gnn_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=gnn_collate_fn)

    num_edges = train_dataset.edge_index.size(0)
    predictor = GNNPredictor(max_nodes=MAX_SERVERS, edge_feat_dim=3, node_hidden=16,
                             num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_gnn_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor


def evaluate_v2b_cross(predictor, test_samples, calib_samples):
    print(f"\n{'='*70}")
    print(f"EVALUATE v2b: NSFNET → Random100")
    print(f"{'='*70}")
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw = evaluate_predictor(predictor, test_loader)
    print(f"  Raw    — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

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


def evaluate_gnn_cross(predictor, test_samples, calib_samples):
    print(f"\n{'='*70}")
    print(f"EVALUATE GNN: NSFNET → Random100")
    print(f"{'='*70}")
    test_dataset = GNNDataset(test_samples)
    calib_dataset = GNNDataset(calib_samples)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=gnn_collate_fn)
    calib_loader = DataLoader(calib_dataset, batch_size=128, shuffle=False, collate_fn=gnn_collate_fn)

    raw = evaluate_gnn_predictor(predictor, test_loader)
    print(f"  Raw    — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in calib_loader:
            logits, _ = predictor.forward_batch(
                batch["partition"], batch["src"], batch["dst"],
                batch["edge_features"], batch["edge_index"], batch["num_nodes"]
            )
            all_logits.extend(logits.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    platt = PlattScaling().fit(np.array(all_logits), np.array(all_labels))
    # Custom Platt evaluation for GNN (evaluate_with_calibration expects 'z' key)
    predictor.eval()
    all_success, all_prob = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, _ = predictor.forward_batch(
                batch["partition"], batch["src"], batch["dst"],
                batch["edge_features"], batch["edge_index"], batch["num_nodes"]
            )
            probs = platt.calibrate(logits.numpy())
            all_success.extend(batch["success"].numpy().tolist())
            all_prob.extend(probs.tolist())
    from sklearn.metrics import accuracy_score, roc_auc_score
    from train import compute_ece as _compute_ece
    y_true = np.array(all_success)
    y_prob = np.array(all_prob)
    acc = accuracy_score(y_true, y_prob > 0.5)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    ece = _compute_ece(y_true, y_prob)
    platt_result = {"accuracy": acc, "auc": auc, "delay_mae": 0.0, "ece": ece}
    print(f"  +Platt — acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:.4f}")
    return {"raw": raw, "platt": platt_result}


if __name__ == "__main__":
    # Generate datasets
    print("Generating NSFNET v2b samples ...")
    nsf_v2b = generate_samples("nsfnet", "v2b", 15000, seed=42, include_graph=False)
    print("Generating NSFNET GNN samples ...")
    nsf_gnn = generate_samples("nsfnet", "v2b", 15000, seed=42, include_graph=True)
    print("Generating Random100 v2b test samples ...")
    rand100_v2b_test = generate_samples("random100", "v2b", 3000, seed=200, include_graph=False)
    print("Generating Random100 GNN test samples ...")
    rand100_gnn_test = generate_samples("random100", "v2b", 3000, seed=200, include_graph=True)
    print("Generating Random100 GNN calib samples ...")
    rand100_gnn_calib = generate_samples("random100", "v2b", 2000, seed=100, include_graph=True)

    # Train v2b baseline
    v2b_predictor = train_v2b_baseline(nsf_v2b, epochs=30)
    torch.save(v2b_predictor.state_dict(), "gnn_exp_v2b.pt")

    # Train GNN
    gnn_predictor = train_gnn(nsf_gnn, epochs=30)
    torch.save(gnn_predictor.state_dict(), "gnn_exp_gnn.pt")

    # Evaluate cross
    v2b_results = evaluate_v2b_cross(v2b_predictor, rand100_v2b_test, rand100_gnn_calib)
    gnn_results = evaluate_gnn_cross(gnn_predictor, rand100_gnn_test, rand100_gnn_calib)

    # Summary
    print("\n" + "="*70)
    print("GNN EXPERIMENT SUMMARY: NSFNET → Random100")
    print("="*70)
    print(f"{'Model':<12} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12}")
    print("-"*70)
    print(f"{'v2b':<12} {v2b_results['raw']['auc']:10.4f} {v2b_results['platt']['auc']:12.4f} "
          f"{v2b_results['raw']['ece']:10.4f} {v2b_results['platt']['ece']:12.4f}")
    print(f"{'GNN':<12} {gnn_results['raw']['auc']:10.4f} {gnn_results['platt']['auc']:12.4f} "
          f"{gnn_results['raw']['ece']:10.4f} {gnn_results['platt']['ece']:12.4f}")
