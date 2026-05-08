"""Topology-conditioned: v2t cross-topology NSFNET -> Random100."""
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

def generate_samples(topology, encoder_version, num_samples, seed=42):
    configs = {
        "nsfnet":    (14, 32,   5.0, 10.0, 500),
        "random100": (100, 320, 30.0, 25.0, 3000),
    }
    n_nodes, num_slots, arr, ht, preload = configs[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2t if encoder_version == "v2t" else encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)

np.random.seed(42)
torch.manual_seed(42)

# Train on NSFNET
print("="*70)
print("v2t: Train NSFNET → Test Random100")
print("="*70)
train_samples = generate_samples("nsfnet", "v2t", 15000, seed=42)
print(f"Train: {len(train_samples)} | SR: {np.mean([s['success'] for s in train_samples]):.3f} | z_dim: {len(train_samples[0]['z'])}")

split = int(0.8 * len(train_samples))
train_loader = DataLoader(OpticalDataset(train_samples[:split]), batch_size=256, shuffle=True)
val_loader = DataLoader(OpticalDataset(train_samples[split:]), batch_size=256, shuffle=False)
predictor = Predictor(len(train_samples[0]["z"]), num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
train_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3)

# Test on Random100
test_samples = generate_samples("random100", "v2t", 3000, seed=200)
print(f"Test: {len(test_samples)} | SR: {np.mean([s['success'] for s in test_samples]):.3f}")
test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
raw = evaluate_predictor(predictor, test_loader)
print(f"Raw    — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

calib_samples = generate_samples("random100", "v2t", 2000, seed=100)
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
print(f"+Platt — acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:.4f}")

# Save
with open("result_v2t_cross.txt", "w") as f:
    f.write(f"v2t NSFNET->Random100: AUC={raw['auc']:.4f} ECE={raw['ece']:.4f} "
            f"Platt_AUC={platt_result['auc']:.4f} Platt_ECE={platt_result['ece']:.4f}\n")
print("Saved to result_v2t_cross.txt")
