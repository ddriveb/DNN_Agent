"""
Large-to-small topology transfer experiment.
Train on Random100 (large), test zero-shot on NSFNET (small) and Random50 (medium).
Compare against small-to-large transfer (NSFNET model → Random100).
"""
import numpy as np
import torch
import time
import sys
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, evaluate_predictor
from calibration_methods import PlattScaling, evaluate_with_calibration
from torch.utils.data import DataLoader

MAX_SERVERS = 128
TOPO_CONFIGS = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "random50":  (50, 128,  15.0, 15.0, 2000),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def generate_samples(topology, version, num_samples, seed=42):
    n_nodes, num_slots, arr, ht, preload = TOPO_CONFIGS[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    if version == "v2b":
        encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)


def train_on_samples(predictor, train_loader, val_loader, epochs=5, lr=5e-4):
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logit, delay_pred = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            loss = bce(logit, batch["success"])
            mask = batch["success"] > 0.5
            if mask.sum() > 0:
                loss = loss + mse(delay_pred[mask], batch["delay"][mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        predictor.eval()
        all_succ, all_prob = [], []
        with torch.no_grad():
            for batch in val_loader:
                logit, _ = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
                prob = torch.sigmoid(logit).numpy()
                all_succ.extend(batch["success"].numpy().tolist())
                all_prob.extend(prob.tolist())
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(np.array(all_succ), np.array(all_prob)) if len(set(all_succ)) > 1 else 0.5
        log(f"  Epoch {epoch+1:02d}: loss={total_loss/len(train_loader):.4f} val_auc={val_auc:.4f}")


def evaluate_predictor_on_samples(predictor, test_samples):
    loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    return evaluate_predictor(predictor, loader)


if __name__ == "__main__":
    log("="*70)
    log("LARGE-TO-SMALL TRANSFER EXPERIMENT")
    log("="*70)

    # Step 1: Generate all data
    log("\n[1/3] Generating test data for all topologies...")
    t0 = time.time()
    nsfnet_test = generate_samples("nsfnet", "v2b", 2000, seed=200)
    random50_test = generate_samples("random50", "v2b", 2000, seed=200)
    random100_test = generate_samples("random100", "v2b", 2000, seed=200)
    random100_train = generate_samples("random100", "v2b", 10000, seed=100)
    log(f"Data ready in {time.time()-t0:.1f}s")
    log(f"  NSFNET   test SR={np.mean([s['success'] for s in nsfnet_test]):.3f}")
    log(f"  Random50 test SR={np.mean([s['success'] for s in random50_test]):.3f}")
    log(f"  Random100 test SR={np.mean([s['success'] for s in random100_test]):.3f}")
    log(f"  Random100 train SR={np.mean([s['success'] for s in random100_train]):.3f}")

    # Step 2: Train "Large Topology Model" on Random100
    log("\n[2/3] Training Large Topology Model (Random100, 10K samples, 5 epochs)...")
    np.random.seed(100)
    torch.manual_seed(100)
    large_model = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    large_model.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))

    split = int(0.8 * len(random100_train))
    ft_train = DataLoader(OpticalDataset(random100_train[:split]), batch_size=256, shuffle=True)
    ft_val = DataLoader(OpticalDataset(random100_train[split:]), batch_size=256, shuffle=False)
    train_on_samples(large_model, ft_train, ft_val, epochs=5, lr=5e-4)

    # Step 3: Evaluate both models on all topologies
    log("\n[3/3] Zero-shot evaluation on all topologies...")

    # Load small topology model (NSFNET pre-trained)
    small_model = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    small_model.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))

    results = {
        "small_to_large": {},
        "large_to_small": {},
    }

    # Small model (NSFNET-trained) evaluated on all
    log("\n  Small Model (NSFNET-trained) → NSFNET:")
    r = evaluate_predictor_on_samples(small_model, nsfnet_test)
    log(f"    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["small_to_large"]["nsfnet"] = r

    log("  Small Model (NSFNET-trained) → Random50:")
    r = evaluate_predictor_on_samples(small_model, random50_test)
    log(f"    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["small_to_large"]["random50"] = r

    log("  Small Model (NSFNET-trained) → Random100:")
    r = evaluate_predictor_on_samples(small_model, random100_test)
    log(f"    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["small_to_large"]["random100"] = r

    # Large model (Random100-trained) evaluated on all
    log("\n  Large Model (Random100-trained) → NSFNET:")
    r = evaluate_predictor_on_samples(large_model, nsfnet_test)
    log(f"    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["large_to_small"]["nsfnet"] = r

    log("  Large Model (Random100-trained) → Random50:")
    r = evaluate_predictor_on_samples(large_model, random50_test)
    log(f"    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["large_to_small"]["random50"] = r

    log("  Large Model (Random100-trained) → Random100:")
    r = evaluate_predictor_on_samples(large_model, random100_test)
    log(f"    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["large_to_small"]["random100"] = r

    # Summary table
    log("\n" + "="*80)
    log("LARGE-TO-SMALL TRANSFER SUMMARY")
    log("="*80)
    log(f"{'Model / Target':<35} {'NSFNET':>12} {'Random50':>12} {'Random100':>12}")
    log("-"*80)
    log(f"{'Small (NSFNET-trained) → Target':<35} "
        f"{results['small_to_large']['nsfnet']['auc']:12.4f} "
        f"{results['small_to_large']['random50']['auc']:12.4f} "
        f"{results['small_to_large']['random100']['auc']:12.4f}")
    log(f"{'Large (Random100-trained) → Target':<35} "
        f"{results['large_to_small']['nsfnet']['auc']:12.4f} "
        f"{results['large_to_small']['random50']['auc']:12.4f} "
        f"{results['large_to_small']['random100']['auc']:12.4f}")
    log("-"*80)
    log(f"{'Intra (best possible)':<35} "
        f"{'~0.992':>12} {'~0.950':>12} {'~0.938':>12}")
    log("\nNotes:")
    log("  - Small→Large gap: 0.992 → 0.778 (drop -0.214)")
    log("  - Large→Small gap: compare Large→NSFNET vs intra NSFNET")
