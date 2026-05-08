"""
Full re-training of the experiment matrix with 8-bin v2b encoder.
Trains new baseline models and re-runs core transfer experiments.
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
NUM_BINS = 8
STATE_DIM = 2 + NUM_BINS * 2  # 18


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def generate_samples(topology, num_samples, seed=42):
    if topology == "nsfnet":
        n_nodes, num_slots, arr, ht, preload = 14, 32, 5.0, 10.0, 500
    elif topology == "random50":
        n_nodes, num_slots, arr, ht, preload = 50, 128, 15.0, 15.0, 2000
    elif topology == "random100":
        n_nodes, num_slots, arr, ht, preload = 100, 320, 30.0, 25.0, 3000
    else:
        raise ValueError(topology)

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = lambda src, dst: encoder.encode_v2b(src, dst, num_bins=NUM_BINS)
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)


def train_model(predictor, train_loader, val_loader, epochs=5, lr=5e-4):
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
        if (epoch + 1) % 5 == 0 or epoch < 3:
            log(f"  Epoch {epoch+1:02d}: loss={total_loss/len(train_loader):.4f} val_auc={val_auc:.4f}")
    return predictor


def evaluate_zero_shot(predictor, test_samples):
    loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    return evaluate_predictor(predictor, loader)


def save_if_best(path, predictor, val_auc):
    torch.save(predictor.state_dict(), path)


if __name__ == "__main__":
    log("="*70)
    log("FULL 8-BIN EXPERIMENT MATRIX")
    log("="*70)

    # ========================================================================
    # STEP 1: Generate all data
    # ========================================================================
    log("\n[STEP 1] Generating all datasets with 8-bin encoder...")
    t0 = time.time()
    nsfnet_train = generate_samples("nsfnet", 15000, seed=100)
    r100_train = generate_samples("random100", 10000, seed=100)
    nsfnet_test = generate_samples("nsfnet", 2000, seed=200)
    r50_test = generate_samples("random50", 2000, seed=200)
    r100_test_same = generate_samples("random100", 2000, seed=200)
    r100_test_diff = generate_samples("random100", 2000, seed=201)
    log(f"Data ready in {time.time()-t0:.1f}s")
    log(f"  NSFNET train  SR={np.mean([s['success'] for s in nsfnet_train]):.3f}")
    log(f"  R100 train    SR={np.mean([s['success'] for s in r100_train]):.3f}")
    log(f"  NSFNET test   SR={np.mean([s['success'] for s in nsfnet_test]):.3f}")
    log(f"  R50 test      SR={np.mean([s['success'] for s in r50_test]):.3f}")
    log(f"  R100 test(s)  SR={np.mean([s['success'] for s in r100_test_same]):.3f}")
    log(f"  R100 test(d)  SR={np.mean([s['success'] for s in r100_test_diff]):.3f}")

    # ========================================================================
    # STEP 2: Train Small Model (NSFNET, 8-bin, 30 epochs)
    # ========================================================================
    log("\n[STEP 2] Training Small Model (NSFNET, 15K, 30ep, 8-bin)...")
    np.random.seed(100)
    torch.manual_seed(100)
    small_model = Predictor(STATE_DIM, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)

    split = int(0.8 * len(nsfnet_train))
    train_ld = DataLoader(OpticalDataset(nsfnet_train[:split]), batch_size=256, shuffle=True)
    val_ld = DataLoader(OpticalDataset(nsfnet_train[split:]), batch_size=256, shuffle=False)
    small_model = train_model(small_model, train_ld, val_ld, epochs=30, lr=5e-4)
    torch.save(small_model.state_dict(), "pretrained_nsfnet_v2b_8bin.pt")
    log("  Saved: pretrained_nsfnet_v2b_8bin.pt")

    # ========================================================================
    # STEP 3: Train Large Model (Random100, 8-bin, 5 epochs from small)
    # ========================================================================
    log("\n[STEP 3] Training Large Model (Random100, 10K, 5ep, 8-bin)...")
    np.random.seed(100)
    torch.manual_seed(100)
    large_model = Predictor(STATE_DIM, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    # Load compatible weights from small model
    pretrained = torch.load("pretrained_nsfnet_v2b_8bin.pt")
    model_dict = large_model.state_dict()
    matched = {k: v for k, v in pretrained.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(matched)
    large_model.load_state_dict(model_dict, strict=False)

    split = int(0.8 * len(r100_train))
    train_ld = DataLoader(OpticalDataset(r100_train[:split]), batch_size=256, shuffle=True)
    val_ld = DataLoader(OpticalDataset(r100_train[split:]), batch_size=256, shuffle=False)
    large_model = train_model(large_model, train_ld, val_ld, epochs=5, lr=5e-4)
    torch.save(large_model.state_dict(), "large_model_random100_8bin.pt")
    log("  Saved: large_model_random100_8bin.pt")

    # ========================================================================
    # STEP 4: Evaluate full transfer matrix
    # ========================================================================
    log("\n[STEP 4] Evaluating transfer matrix...")

    results = {}

    # Small model evaluations
    log("\n--- Small Model (NSFNET-trained) ---")
    r = evaluate_zero_shot(small_model, nsfnet_test)
    log(f"  → NSFNET (intra):      AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["small→nsfnet"] = r

    r = evaluate_zero_shot(small_model, r100_test_same)
    log(f"  → Random100 (same):    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["small→r100same"] = r

    r = evaluate_zero_shot(small_model, r50_test)
    log(f"  → Random50:            AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["small→r50"] = r

    # Large model evaluations
    log("\n--- Large Model (Random100-trained) ---")
    r = evaluate_zero_shot(large_model, r100_test_same)
    log(f"  → Random100 (same):    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["large→r100same"] = r

    r = evaluate_zero_shot(large_model, r100_test_diff)
    log(f"  → Random100 (diff):    AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["large→r100diff"] = r

    r = evaluate_zero_shot(large_model, r50_test)
    log(f"  → Random50:            AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["large→r50"] = r

    r = evaluate_zero_shot(large_model, nsfnet_test)
    log(f"  → NSFNET:              AUC={r['auc']:.4f} ECE={r['ece']:.4f} MAE={r['delay_mae']:.4f}")
    results["large→nsfnet"] = r

    # ========================================================================
    # Summary
    # ========================================================================
    log("\n" + "="*75)
    log("FULL 8-BIN TRANSFER MATRIX SUMMARY")
    log("="*75)
    log(f"{'Transfer':<25} {'AUC':>8} {'ECE':>8} {'MAE':>8}")
    log("-"*75)
    log(f"{'Small→NSFNET (intra)':<25} {results['small→nsfnet']['auc']:8.4f} {results['small→nsfnet']['ece']:8.4f} {results['small→nsfnet']['delay_mae']:8.4f}")
    log(f"{'Small→Random100':<25} {results['small→r100same']['auc']:8.4f} {results['small→r100same']['ece']:8.4f} {results['small→r100same']['delay_mae']:8.4f}")
    log(f"{'Small→Random50':<25} {results['small→r50']['auc']:8.4f} {results['small→r50']['ece']:8.4f} {results['small→r50']['delay_mae']:8.4f}")
    log("-"*75)
    log(f"{'Large→Random100 (intra)':<25} {results['large→r100same']['auc']:8.4f} {results['large→r100same']['ece']:8.4f} {results['large→r100same']['delay_mae']:8.4f}")
    log(f"{'Large→Random100 (diff)':<25} {results['large→r100diff']['auc']:8.4f} {results['large→r100diff']['ece']:8.4f} {results['large→r100diff']['delay_mae']:8.4f}")
    log(f"{'Large→Random50':<25} {results['large→r50']['auc']:8.4f} {results['large→r50']['ece']:8.4f} {results['large→r50']['delay_mae']:8.4f}")
    log(f"{'Large→NSFNET':<25} {results['large→nsfnet']['auc']:8.4f} {results['large→nsfnet']['ece']:8.4f} {results['large→nsfnet']['delay_mae']:8.4f}")
    log("="*75)
    log("\nComparison with 4-bin reference:")
    log("  4-bin Small→NSFNET intra:     AUC≈0.992")
    log("  4-bin Small→Random100:         AUC≈0.778")
    log("  4-bin Large→Random50:          AUC≈0.888")
    log("  4-bin Large→NSFNET:            AUC≈0.961")
    log("  4-bin Large→Random100(diff):   AUC≈0.799-0.817")
