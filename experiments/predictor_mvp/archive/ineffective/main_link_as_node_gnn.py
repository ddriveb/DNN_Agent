"""
Link-As-Node GAT experiment.
Train on NSFNET, zero-shot test on Random100.
Compare: v2b baseline vs Old GNN vs Link-As-Node GAT.
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from gnn_predictor import GNNPredictor
from link_as_node_gnn_predictor import LinkAsNodeGATPredictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor
from train_gnn import GNNDataset, gnn_collate_fn, train_gnn_predictor, evaluate_gnn_predictor
from train_link_as_node_gnn import LinkAsNodeDataset, link_as_node_collate_fn, train_link_as_node_gnn, evaluate_link_as_node_gnn
from calibration_methods import PlattScaling
from torch.utils.data import DataLoader

MAX_SERVERS = 128

TOPO_CONFIGS = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def generate_samples(topology, encoder_version, num_samples, seed=42,
                     include_graph=False, include_link_as_node=False):
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
                        include_edge_features=include_graph,
                        include_link_as_node=include_link_as_node)


def train_v2b_baseline(samples, epochs=30):
    print(f"\n{'='*70}")
    print(f"TRAIN v2b BASELINE on NSFNET")
    print(f"{'='*70}")
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)
    predictor = Predictor(len(samples[0]["z"]), num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor


def train_old_gnn(samples, epochs=30):
    print(f"\n{'='*70}")
    print(f"TRAIN Old GNN on NSFNET")
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


def train_link_as_node_gat(samples, epochs=30):
    print(f"\n{'='*70}")
    print(f"TRAIN Link-As-Node GAT on NSFNET")
    print(f"{'='*70}")
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_dataset = LinkAsNodeDataset(samples[:split])
    val_dataset = LinkAsNodeDataset(samples[split:])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=link_as_node_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=link_as_node_collate_fn)

    num_links = train_dataset.num_links
    predictor = LinkAsNodeGATPredictor(
        num_links=num_links,
        link_feat_dim=6,
        link_hidden=64,
        num_gat_layers=3,
        num_heads=8,
        dropout=0.1,
        num_partitions=3,
        max_servers=MAX_SERVERS,
        hidden_dim=64,
    )
    train_link_as_node_gnn(predictor, train_loader, val_loader, epochs=25, lr=1e-3)
    return predictor


def evaluate_v2b_cross(predictor, test_samples, calib_samples):
    print(f"\n{'='*70}")
    print(f"EVALUATE v2b: NSFNET -> Random100")
    print(f"{'='*70}")
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw = evaluate_predictor(predictor, test_loader)
    print(f"  Raw    -- acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

    calib_loader = DataLoader(OpticalDataset(calib_samples), batch_size=256, shuffle=False)
    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in calib_loader:
            logit, _ = predictor(batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"])
            all_logits.extend(logit.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    platt = PlattScaling().fit(np.array(all_logits), np.array(all_labels))

    # Evaluate with calibration on test set
    predictor.eval()
    all_success, all_prob = [], []
    with torch.no_grad():
        for batch in test_loader:
            logit, _ = predictor(batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"])
            probs = platt.calibrate(logit.numpy())
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
    print(f"  +Platt -- acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:.4f}")
    return {"raw": raw, "platt": platt_result}


def evaluate_old_gnn_cross(predictor, test_samples, calib_samples):
    print(f"\n{'='*70}")
    print(f"EVALUATE Old GNN: NSFNET -> Random100")
    print(f"{'='*70}")
    test_dataset = GNNDataset(test_samples)
    calib_dataset = GNNDataset(calib_samples)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=gnn_collate_fn)
    calib_loader = DataLoader(calib_dataset, batch_size=128, shuffle=False, collate_fn=gnn_collate_fn)

    raw = evaluate_gnn_predictor(predictor, test_loader)
    print(f"  Raw    -- acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

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
    print(f"  +Platt -- acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:.4f}")
    return {"raw": raw, "platt": platt_result}


def evaluate_link_as_node_gat_cross(predictor, test_samples, calib_samples):
    print(f"\n{'='*70}")
    print(f"EVALUATE Link-As-Node GAT: NSFNET -> Random100")
    print(f"{'='*70}")
    test_dataset = LinkAsNodeDataset(test_samples)
    calib_dataset = LinkAsNodeDataset(calib_samples)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=link_as_node_collate_fn)
    calib_loader = DataLoader(calib_dataset, batch_size=128, shuffle=False, collate_fn=link_as_node_collate_fn)

    raw = evaluate_link_as_node_gnn(predictor, test_loader)
    print(f"  Raw    -- acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in calib_loader:
            logits, _ = predictor.forward_batch(
                batch["path_id"], batch["src"], batch["dst"], batch["bw"],
                batch["link_node_features"], batch["converted_edge_index"], batch["path_mask"]
            )
            all_logits.extend(logits.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    platt = PlattScaling().fit(np.array(all_logits), np.array(all_labels))

    predictor.eval()
    all_success, all_prob = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, _ = predictor.forward_batch(
                batch["path_id"], batch["src"], batch["dst"], batch["bw"],
                batch["link_node_features"], batch["converted_edge_index"], batch["path_mask"]
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
    print(f"  +Platt -- acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:.4f}")
    return {"raw": raw, "platt": platt_result}


if __name__ == "__main__":
    # Generate datasets
    print("Generating NSFNET v2b samples ...")
    nsf_v2b = generate_samples("nsfnet", "v2b", 8000, seed=42, include_graph=False, include_link_as_node=False)
    print("Generating NSFNET old GNN samples ...")
    nsf_old_gnn = generate_samples("nsfnet", "v2b", 8000, seed=42, include_graph=True, include_link_as_node=False)
    # Old GNN dataset expects 'partition' key instead of 'path_id'
    for s in nsf_old_gnn:
        s["partition"] = s.pop("path_id")
    print("Generating NSFNET Link-As-Node GAT samples ...")
    nsf_lan_gat = generate_samples("nsfnet", "v2b", 8000, seed=42, include_graph=False, include_link_as_node=True)

    print("Generating Random100 v2b test samples ...")
    rand100_v2b_test = generate_samples("random100", "v2b", 1500, seed=200, include_graph=False, include_link_as_node=False)
    print("Generating Random100 old GNN test samples ...")
    rand100_old_gnn_test = generate_samples("random100", "v2b", 1500, seed=200, include_graph=True, include_link_as_node=False)
    for s in rand100_old_gnn_test:
        s["partition"] = s.pop("path_id")
    print("Generating Random100 Link-As-Node GAT test samples ...")
    rand100_lan_gat_test = generate_samples("random100", "v2b", 1500, seed=200, include_graph=False, include_link_as_node=True)

    print("Generating Random100 calibration samples (shared for all) ...")
    rand100_calib_v2b = generate_samples("random100", "v2b", 1000, seed=100, include_graph=False, include_link_as_node=False)
    rand100_calib_old_gnn = generate_samples("random100", "v2b", 1000, seed=100, include_graph=True, include_link_as_node=False)
    for s in rand100_calib_old_gnn:
        s["partition"] = s.pop("path_id")
    rand100_calib_lan_gat = generate_samples("random100", "v2b", 1000, seed=100, include_graph=False, include_link_as_node=True)

    # Train models
    v2b_predictor = train_v2b_baseline(nsf_v2b, epochs=20)
    torch.save(v2b_predictor.state_dict(), "lan_exp_v2b.pt")

    old_gnn_predictor = train_old_gnn(nsf_old_gnn, epochs=20)
    torch.save(old_gnn_predictor.state_dict(), "lan_exp_old_gnn.pt")

    lan_gat_predictor = train_link_as_node_gat(nsf_lan_gat, epochs=20)
    torch.save(lan_gat_predictor.state_dict(), "lan_exp_lan_gat.pt")

    # Evaluate cross-topology
    v2b_results = evaluate_v2b_cross(v2b_predictor, rand100_v2b_test, rand100_calib_v2b)
    old_gnn_results = evaluate_old_gnn_cross(old_gnn_predictor, rand100_old_gnn_test, rand100_calib_old_gnn)
    lan_gat_results = evaluate_link_as_node_gat_cross(lan_gat_predictor, rand100_lan_gat_test, rand100_calib_lan_gat)

    # Summary
    print("\n" + "="*70)
    print("LINK-AS-NODE GAT EXPERIMENT SUMMARY: NSFNET -> Random100")
    print("="*70)
    print(f"{'Model':<20} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12}")
    print("-"*70)
    print(f"{'v2b baseline':<20} {v2b_results['raw']['auc']:10.4f} {v2b_results['platt']['auc']:12.4f} "
          f"{v2b_results['raw']['ece']:10.4f} {v2b_results['platt']['ece']:12.4f}")
    print(f"{'Old GNN':<20} {old_gnn_results['raw']['auc']:10.4f} {old_gnn_results['platt']['auc']:12.4f} "
          f"{old_gnn_results['raw']['ece']:10.4f} {old_gnn_results['platt']['ece']:12.4f}")
    print(f"{'Link-As-Node GAT':<20} {lan_gat_results['raw']['auc']:10.4f} {lan_gat_results['platt']['auc']:12.4f} "
          f"{lan_gat_results['raw']['ece']:10.4f} {lan_gat_results['platt']['ece']:12.4f}")
    print("="*70)
