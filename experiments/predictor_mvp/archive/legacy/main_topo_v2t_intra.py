"""Topology-conditioned: v2t intra-topology Random100 -> Random100."""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor
from torch.utils.data import DataLoader

MAX_SERVERS = 128

np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("v2t: Intra-topology Random100")
print("="*70)

n_nodes, num_slots, arr, ht, preload = (100, 320, 30.0, 25.0, 3000)
net = OpticalNetwork(topology="random100", num_slots=num_slots, seed=42)
mapper = KSPMapper(net, k=3)
encoder = Encoder(net, k=3)
encoder.encode = encoder.encode_v2t
gen = DatasetGenerator(net, mapper, encoder, seed=42)
samples = gen.generate(10000, arrival_rate=arr, avg_holding_time=ht, preload=preload)

print(f"Train: {len(samples)} | SR: {np.mean([s['success'] for s in samples]):.3f} | z_dim: {len(samples[0]['z'])}")
split = int(0.8 * len(samples))
train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)
predictor = Predictor(len(samples[0]["z"]), num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
train_predictor(predictor, train_loader, val_loader, epochs=30, lr=1e-3)

metrics = evaluate_predictor(predictor, val_loader)
print(f"Intra  — acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} ece={metrics['ece']:.4f}")

with open("result_v2t_intra.txt", "w") as f:
    f.write(f"v2t Random100(intra): AUC={metrics['auc']:.4f} ECE={metrics['ece']:.4f}\n")
print("Saved to result_v2t_intra.txt")
