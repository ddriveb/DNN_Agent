"""
Hybrid Predictor experiment.
Compare: Baseline MLP vs Link-As-Node GAT vs Hybrid (v2b + GAT)
Train on NSFNET, test on NSFNET (intra) and Random100 (cross).
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from link_as_node_gnn_predictor import LinkAsNodeGATPredictor
from hybrid_predictor import HybridPredictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor
from train_link_as_node_gnn import LinkAsNodeDataset, link_as_node_collate_fn, train_link_as_node_gnn, evaluate_link_as_node_gnn
from train_hybrid import HybridDataset, hybrid_collate_fn, train_hybrid_predictor, evaluate_hybrid_predictor
from calibration_methods import PlattScaling
from torch.utils.data import DataLoader

MAX_SERVERS = 128
EPOCHS = 20


def generate_samples(topology, num_samples, seed=42, include_graph=False):
    """Generate dataset."""
    if topology == "nsfnet":
        n_nodes, num_slots, arr, ht, preload = 14, 32, 5.0, 10.0, 500
    elif topology == "random100":
        n_nodes, num_slots, arr, ht, preload = 100, 320, 30.0, 25.0, 3000
    else:
        raise ValueError(f"Unknown topology: {topology}")

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht,
                        preload=preload, include_edge_features=False,
                        include_link_as_node=include_graph)


def train_baseline(samples, epochs=20):
    print(f"\n{'='*60}")
    print(f"TRAIN Baseline MLP")
    print(f"{'='*60}")
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)
    predictor = Predictor(len(samples[0]["z"]), num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor


def train_gat(samples, epochs=20):
    print(f"\n{'='*60}")
    print(f"TRAIN Link-As-Node GAT")
    print(f"{'='*60}")
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(LinkAsNodeDataset(samples[:split]), batch_size=128, shuffle=True, collate_fn=link_as_node_collate_fn)
    val_loader = DataLoader(LinkAsNodeDataset(samples[split:]), batch_size=128, shuffle=False, collate_fn=link_as_node_collate_fn)

    num_links = train_loader.dataset.num_links
    predictor = LinkAsNodeGATPredictor(
        num_links=num_links, link_feat_dim=6, link_hidden=32,
        num_gat_layers=2, num_heads=4, dropout=0.1,
        num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64,
    )
    train_link_as_node_gnn(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor


def train_hybrid(samples, epochs=20):
    print(f"\n{'='*60}")
    print(f"TRAIN Hybrid (v2b + GAT)")
    print(f"{'='*60}")
    np.random.seed(42)
    torch.manual_seed(42)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(HybridDataset(samples[:split]), batch_size=128, shuffle=True, collate_fn=hybrid_collate_fn)
    val_loader = DataLoader(HybridDataset(samples[split:]), batch_size=128, shuffle=False, collate_fn=hybrid_collate_fn)

    state_dim = len(samples[0]["z"])
    num_links = train_loader.dataset.num_links
    predictor = HybridPredictor(
        state_dim=state_dim, num_links=num_links, link_feat_dim=6, link_hidden=32,
        num_gat_layers=2, num_heads=4, dropout=0.1,
        num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64,
    )
    train_hybrid_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor


def evaluate_baseline(predictor, test_samples):
    loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    return evaluate_predictor(predictor, loader)


def evaluate_gat(predictor, test_samples):
    loader = DataLoader(LinkAsNodeDataset(test_samples), batch_size=128, shuffle=False, collate_fn=link_as_node_collate_fn)
    return evaluate_link_as_node_gnn(predictor, loader)


def evaluate_hybrid(predictor, test_samples):
    loader = DataLoader(HybridDataset(test_samples), batch_size=128, shuffle=False, collate_fn=hybrid_collate_fn)
    return evaluate_hybrid_predictor(predictor, loader)


def apply_platt(predictor, calib_samples, test_samples, model_type="baseline"):
    """Apply Platt scaling and evaluate."""
    from train import compute_ece as _compute_ece
    from sklearn.metrics import accuracy_score, roc_auc_score

    if model_type == "baseline":
        calib_loader = DataLoader(OpticalDataset(calib_samples), batch_size=256, shuffle=False)
        test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    elif model_type == "gat":
        calib_loader = DataLoader(LinkAsNodeDataset(calib_samples), batch_size=128, shuffle=False, collate_fn=link_as_node_collate_fn)
        test_loader = DataLoader(LinkAsNodeDataset(test_samples), batch_size=128, shuffle=False, collate_fn=link_as_node_collate_fn)
    else:  # hybrid
        calib_loader = DataLoader(HybridDataset(calib_samples), batch_size=128, shuffle=False, collate_fn=hybrid_collate_fn)
        test_loader = DataLoader(HybridDataset(test_samples), batch_size=128, shuffle=False, collate_fn=hybrid_collate_fn)

    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in calib_loader:
            if model_type == "baseline":
                logit, _ = predictor(batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"])
            elif model_type == "gat":
                logit, _ = predictor.forward_batch(
                    batch["path_id"], batch["src"], batch["dst"], batch["bw"],
                    batch["link_node_features"], batch["converted_edge_index"], batch["path_mask"]
                )
            else:
                logit, _ = predictor.forward_batch(
                    batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"],
                    batch["link_node_features"], batch["converted_edge_index"], batch["path_mask"]
                )
            all_logits.extend(logit.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    platt = PlattScaling().fit(np.array(all_logits), np.array(all_labels))

    predictor.eval()
    all_success, all_prob = [], []
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "baseline":
                logit, _ = predictor(batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"])
            elif model_type == "gat":
                logit, _ = predictor.forward_batch(
                    batch["path_id"], batch["src"], batch["dst"], batch["bw"],
                    batch["link_node_features"], batch["converted_edge_index"], batch["path_mask"]
                )
            else:
                logit, _ = predictor.forward_batch(
                    batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"],
                    batch["link_node_features"], batch["converted_edge_index"], batch["path_mask"]
                )
            probs = platt.calibrate(logit.numpy())
            all_success.extend(batch["success"].numpy().tolist())
            all_prob.extend(probs.tolist())

    y_true = np.array(all_success)
    y_prob = np.array(all_prob)
    acc = accuracy_score(y_true, y_prob > 0.5)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    ece = _compute_ece(y_true, y_prob)
    return {"accuracy": acc, "auc": auc, "delay_mae": 0.0, "ece": ece}


if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID PREDICTOR EXPERIMENT")
    print("=" * 70)

    # Generate datasets
    print("\nGenerating NSFNET training samples (8000) ...")
    nsf_train = generate_samples("nsfnet", 8000, seed=42, include_graph=True)

    print("Generating NSFNET test samples (2000) ...")
    nsf_test = generate_samples("nsfnet", 2000, seed=200, include_graph=True)

    print("Generating Random100 test samples (2000) ...")
    rand100_test = generate_samples("random100", 2000, seed=300, include_graph=True)

    print("Generating Random100 calibration samples (1000) ...")
    rand100_calib = generate_samples("random100", 1000, seed=400, include_graph=True)

    # Train models
    baseline_model = train_baseline(nsf_train, epochs=EPOCHS)
    gat_model = train_gat(nsf_train, epochs=EPOCHS)
    hybrid_model = train_hybrid(nsf_train, epochs=EPOCHS)

    # Evaluate on NSFNET (intra)
    print(f"\n{'='*70}")
    print("EVALUATE: NSFNET (intra-topology)")
    print(f"{'='*70}")
    base_nsf = evaluate_baseline(baseline_model, nsf_test)
    gat_nsf = evaluate_gat(gat_model, nsf_test)
    hybrid_nsf = evaluate_hybrid(hybrid_model, nsf_test)

    print(f"  Baseline  -- AUC={base_nsf['auc']:.4f} ECE={base_nsf['ece']:.4f} ACC={base_nsf['accuracy']:.4f}")
    print(f"  GAT       -- AUC={gat_nsf['auc']:.4f} ECE={gat_nsf['ece']:.4f} ACC={gat_nsf['accuracy']:.4f}")
    print(f"  Hybrid    -- AUC={hybrid_nsf['auc']:.4f} ECE={hybrid_nsf['ece']:.4f} ACC={hybrid_nsf['accuracy']:.4f}")

    # Evaluate on Random100 (cross) raw
    print(f"\n{'='*70}")
    print("EVALUATE: Random100 (cross-topology, RAW)")
    print(f"{'='*70}")
    base_rand = evaluate_baseline(baseline_model, rand100_test)
    gat_rand = evaluate_gat(gat_model, rand100_test)
    hybrid_rand = evaluate_hybrid(hybrid_model, rand100_test)

    print(f"  Baseline  -- AUC={base_rand['auc']:.4f} ECE={base_rand['ece']:.4f} ACC={base_rand['accuracy']:.4f}")
    print(f"  GAT       -- AUC={gat_rand['auc']:.4f} ECE={gat_rand['ece']:.4f} ACC={gat_rand['accuracy']:.4f}")
    print(f"  Hybrid    -- AUC={hybrid_rand['auc']:.4f} ECE={hybrid_rand['ece']:.4f} ACC={hybrid_rand['accuracy']:.4f}")

    # Evaluate on Random100 with Platt
    print(f"\n{'='*70}")
    print("EVALUATE: Random100 (cross-topology, +PLATT)")
    print(f"{'='*70}")
    base_rand_platt = apply_platt(baseline_model, rand100_calib, rand100_test, "baseline")
    gat_rand_platt = apply_platt(gat_model, rand100_calib, rand100_test, "gat")
    hybrid_rand_platt = apply_platt(hybrid_model, rand100_calib, rand100_test, "hybrid")

    print(f"  Baseline  -- AUC={base_rand_platt['auc']:.4f} ECE={base_rand_platt['ece']:.4f} ACC={base_rand_platt['accuracy']:.4f}")
    print(f"  GAT       -- AUC={gat_rand_platt['auc']:.4f} ECE={gat_rand_platt['ece']:.4f} ACC={gat_rand_platt['accuracy']:.4f}")
    print(f"  Hybrid    -- AUC={hybrid_rand_platt['auc']:.4f} ECE={hybrid_rand_platt['ece']:.4f} ACC={hybrid_rand_platt['accuracy']:.4f}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'NSF AUC':>9} {'Rand AUC':>10} {'+Platt':>10} {'Raw ECE':>9} {'+Platt ECE':>11}")
    print("-" * 70)
    print(f"{'Baseline':<12} {base_nsf['auc']:9.4f} {base_rand['auc']:10.4f} {base_rand_platt['auc']:10.4f} "
          f"{base_rand['ece']:9.4f} {base_rand_platt['ece']:11.4f}")
    print(f"{'GAT':<12} {gat_nsf['auc']:9.4f} {gat_rand['auc']:10.4f} {gat_rand_platt['auc']:10.4f} "
          f"{gat_rand['ece']:9.4f} {gat_rand_platt['ece']:11.4f}")
    print(f"{'Hybrid':<12} {hybrid_nsf['auc']:9.4f} {hybrid_rand['auc']:10.4f} {hybrid_rand_platt['auc']:10.4f} "
          f"{hybrid_rand['ece']:9.4f} {hybrid_rand_platt['ece']:11.4f}")
    print("=" * 70)
