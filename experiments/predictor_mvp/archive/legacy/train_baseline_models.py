"""
Train baseline models with the bw-aware interface.
- Small model: NSFNET 15K, 30 epochs
- Large model: Random100 10K, 5 epochs (fine-tuned from small)
Both use 8-bin v2b encoder (state_dim=18).
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

NUM_BINS = 8
STATE_DIM = 2 + NUM_BINS * 2  # 18
MAX_SERVERS = 128


def log(msg):
    print(msg, flush=True)


def generate_data(topology, num_samples, seed=42):
    if topology == "nsfnet":
        n, slots, arr, ht, preload = 14, 32, 5.0, 10.0, 500
    elif topology == "random100":
        n, slots, arr, ht, preload = 100, 320, 30.0, 25.0, 3000
    else:
        raise ValueError(topology)

    net = OpticalNetwork(topology=topology, num_slots=slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = lambda src, dst: encoder.encode_v2b(src, dst, num_bins=NUM_BINS)
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)


def main():
    log("=" * 70)
    log("TRAINING BASELINE MODELS (8-bin + bw-aware)")
    log("=" * 70)

    # ========================================================================
    # 1. Train Small Model (NSFNET, 15K, 30ep)
    # ========================================================================
    log("\n[1/3] Generating NSFNET training data (15K)...")
    nsfnet_train = generate_data("nsfnet", 15000, seed=100)
    log(f"  Generated {len(nsfnet_train)} samples, SR={np.mean([s['success'] for s in nsfnet_train]):.3f}")

    log("\n[2/3] Training Small Model (NSFNET, 30 epochs)...")
    np.random.seed(100)
    torch.manual_seed(100)
    small_model = Predictor(STATE_DIM, num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)

    split = int(0.8 * len(nsfnet_train))
    train_ld = DataLoader(OpticalDataset(nsfnet_train[:split]), batch_size=256, shuffle=True)
    val_ld = DataLoader(OpticalDataset(nsfnet_train[split:]), batch_size=256, shuffle=False)
    history = train_predictor(small_model, train_ld, val_ld, epochs=30, lr=5e-4)
    torch.save(small_model.state_dict(), "pretrained_nsfnet_v2b_8bin_bw.pt")
    log("  Saved: pretrained_nsfnet_v2b_8bin_bw.pt")

    # Quick intra evaluation
    test_ld = DataLoader(OpticalDataset(nsfnet_train[split:]), batch_size=256, shuffle=False)
    metrics = evaluate_predictor(small_model, test_ld)
    log(f"  Intra NSFNET — AUC={metrics['auc']:.4f} ECE={metrics['ece']:.4f} MAE={metrics['delay_mae']:.4f}")

    # ========================================================================
    # 2. Train Large Model (Random100, 10K, 5ep from small)
    # ========================================================================
    log("\n[3/3] Generating Random100 training data (10K)...")
    r100_train = generate_data("random100", 10000, seed=100)
    log(f"  Generated {len(r100_train)} samples, SR={np.mean([s['success'] for s in r100_train]):.3f}")

    log("\nTraining Large Model (Random100, 5 epochs, from small model)...")
    np.random.seed(100)
    torch.manual_seed(100)
    large_model = Predictor(STATE_DIM, num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)
    # Load small model weights (same architecture)
    large_model.load_state_dict(torch.load("pretrained_nsfnet_v2b_8bin_bw.pt"))

    split = int(0.8 * len(r100_train))
    train_ld = DataLoader(OpticalDataset(r100_train[:split]), batch_size=256, shuffle=True)
    val_ld = DataLoader(OpticalDataset(r100_train[split:]), batch_size=256, shuffle=False)
    history = train_predictor(large_model, train_ld, val_ld, epochs=5, lr=5e-4)
    torch.save(large_model.state_dict(), "large_model_random100_8bin_bw.pt")
    log("  Saved: large_model_random100_8bin_bw.pt")

    # Quick intra evaluation
    test_ld = DataLoader(OpticalDataset(r100_train[split:]), batch_size=256, shuffle=False)
    metrics = evaluate_predictor(large_model, test_ld)
    log(f"  Intra Random100 — AUC={metrics['auc']:.4f} ECE={metrics['ece']:.4f} MAE={metrics['delay_mae']:.4f}")

    log("\n" + "=" * 70)
    log("BASELINE MODELS TRAINING COMPLETE")
    log("=" * 70)
    log("Model files:")
    log("  pretrained_nsfnet_v2b_8bin_bw.pt  (small model)")
    log("  large_model_random100_8bin_bw.pt  (large model)")


if __name__ == "__main__":
    main()
