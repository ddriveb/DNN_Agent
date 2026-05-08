"""Quick eval: load saved GNN/v2b models and evaluate on Random100."""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from gnn_predictor import GNNPredictor
from dataset import DatasetGenerator
from train import OpticalDataset, evaluate_predictor
from train_gnn import GNNDataset, gnn_collate_fn, evaluate_gnn_predictor
from calibration_methods import PlattScaling
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


# Generate test data
print("Generating Random100 v2b test ...")
rand100_v2b_test = generate_samples("random100", "v2b", 3000, seed=200, include_graph=False)
print("Generating Random100 GNN test ...")
rand100_gnn_test = generate_samples("random100", "v2b", 3000, seed=200, include_graph=True)
print("Generating Random100 GNN calib ...")
rand100_gnn_calib = generate_samples("random100", "v2b", 2000, seed=100, include_graph=True)

# Load v2b model
print("\nLoading v2b model ...")
v2b_predictor = Predictor(len(rand100_v2b_test[0]["z"]), num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
v2b_predictor.load_state_dict(torch.load("gnn_exp_v2b.pt"))

# Load GNN model
print("Loading GNN model ...")
gnn_predictor = GNNPredictor(max_nodes=MAX_SERVERS, edge_feat_dim=3, node_hidden=16,
                             num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
gnn_predictor.load_state_dict(torch.load("gnn_exp_gnn.pt"))

# Evaluate v2b
test_loader = DataLoader(OpticalDataset(rand100_v2b_test), batch_size=256, shuffle=False)
v2b_raw = evaluate_predictor(v2b_predictor, test_loader)

calib_loader = DataLoader(OpticalDataset(rand100_gnn_calib), batch_size=256, shuffle=False)
v2b_predictor.eval()
all_logits, all_labels = [], []
with torch.no_grad():
    for batch in calib_loader:
        logit, _ = v2b_predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
        all_logits.extend(logit.numpy().tolist())
        all_labels.extend(batch["success"].numpy().tolist())
platt_v2b = PlattScaling().fit(np.array(all_logits), np.array(all_labels))

v2b_predictor.eval()
all_success, all_prob = [], []
with torch.no_grad():
    for batch in test_loader:
        logit, _ = v2b_predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
        probs = platt_v2b.calibrate(logit.numpy())
        all_success.extend(batch["success"].numpy().tolist())
        all_prob.extend(probs.tolist())
from sklearn.metrics import accuracy_score, roc_auc_score
from train import compute_ece as _compute_ece
y_true = np.array(all_success)
y_prob = np.array(all_prob)
v2b_platt = {
    "accuracy": accuracy_score(y_true, y_prob > 0.5),
    "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
    "ece": _compute_ece(y_true, y_prob),
    "delay_mae": 0.0,
}

# Evaluate GNN
test_gnn_loader = DataLoader(GNNDataset(rand100_gnn_test), batch_size=128, shuffle=False, collate_fn=gnn_collate_fn)
gnn_raw = evaluate_gnn_predictor(gnn_predictor, test_gnn_loader)

calib_gnn_loader = DataLoader(GNNDataset(rand100_gnn_calib), batch_size=128, shuffle=False, collate_fn=gnn_collate_fn)
gnn_predictor.eval()
all_logits, all_labels = [], []
with torch.no_grad():
    for batch in calib_gnn_loader:
        logits, _ = gnn_predictor.forward_batch(
            batch["partition"], batch["src"], batch["dst"],
            batch["edge_features"], batch["edge_index"], batch["num_nodes"]
        )
        all_logits.extend(logits.numpy().tolist())
        all_labels.extend(batch["success"].numpy().tolist())
platt_gnn = PlattScaling().fit(np.array(all_logits), np.array(all_labels))

gnn_predictor.eval()
all_success, all_prob = [], []
with torch.no_grad():
    for batch in test_gnn_loader:
        logits, _ = gnn_predictor.forward_batch(
            batch["partition"], batch["src"], batch["dst"],
            batch["edge_features"], batch["edge_index"], batch["num_nodes"]
        )
        probs = platt_gnn.calibrate(logits.numpy())
        all_success.extend(batch["success"].numpy().tolist())
        all_prob.extend(probs.tolist())
y_true = np.array(all_success)
y_prob = np.array(all_prob)
gnn_platt = {
    "accuracy": accuracy_score(y_true, y_prob > 0.5),
    "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
    "ece": _compute_ece(y_true, y_prob),
    "delay_mae": 0.0,
}

# Summary
print("\n" + "="*70)
print("GNN EXPERIMENT SUMMARY: NSFNET → Random100")
print("="*70)
print(f"{'Model':<12} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12}")
print("-"*70)
print(f"{'v2b':<12} {v2b_raw['auc']:10.4f} {v2b_platt['auc']:12.4f} "
      f"{v2b_raw['ece']:10.4f} {v2b_platt['ece']:12.4f}")
print(f"{'GNN':<12} {gnn_raw['auc']:10.4f} {gnn_platt['auc']:12.4f} "
      f"{gnn_raw['ece']:10.4f} {gnn_platt['ece']:12.4f}")
