"""
Large-to-small transfer + noise fine-tuning on Random50.
Train large model on Random100, then fine-tune on Random50 with/without noise.
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
        log(f"  Epoch {epoch+1:02d}: loss={total_loss/len(train_loader):.4f} val_auc={val_auc:.4f}")


def finetune_with_noise(predictor, train_loader, val_loader, epochs=5, lr=5e-4, noise_std=0.0):
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            z = batch["z"]
            if noise_std > 0.0:
                z = z + torch.randn_like(z) * noise_std
            logit, delay_pred = predictor(batch["partition"], batch["src"], batch["dst"], z)
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


def evaluate_predictor_on_samples(predictor, test_samples, calib_samples):
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw = evaluate_predictor(predictor, test_loader)
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
    return raw, platt_result


if __name__ == "__main__":
    log("="*70)
    log("LARGE-TO-SMALL + NOISE FINE-TUNE (Random100→Random50)")
    log("="*70)

    # Step 1: Generate all data
    log("\n[1/3] Generating data...")
    t0 = time.time()
    r100_train = generate_samples("random100", "v2b", 10000, seed=100)
    r50_ft2k = generate_samples("random50", "v2b", 2000, seed=200)
    r50_ft5k = generate_samples("random50", "v2b", 5000, seed=201)
    r50_test = generate_samples("random50", "v2b", 2000, seed=300)
    r50_calib = generate_samples("random50", "v2b", 2000, seed=400)
    log(f"Data ready in {time.time()-t0:.1f}s")
    log(f"  R100 train SR={np.mean([s['success'] for s in r100_train]):.3f}")
    log(f"  R50 ft2k  SR={np.mean([s['success'] for s in r50_ft2k]):.3f}")
    log(f"  R50 ft5k  SR={np.mean([s['success'] for s in r50_ft5k]):.3f}")
    log(f"  R50 test  SR={np.mean([s['success'] for s in r50_test]):.3f}")

    # Step 2: Train Large Model on Random100
    log("\n[2/3] Training Large Model (Random100, 10K, 5 epochs)...")
    np.random.seed(100)
    torch.manual_seed(100)
    large_model = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    large_model.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))

    split = int(0.8 * len(r100_train))
    ft_train = DataLoader(OpticalDataset(r100_train[:split]), batch_size=256, shuffle=True)
    ft_val = DataLoader(OpticalDataset(r100_train[split:]), batch_size=256, shuffle=False)
    train_model(large_model, ft_train, ft_val, epochs=5, lr=5e-4)
    torch.save(large_model.state_dict(), "large_model_random100.pt")
    log("  Saved: large_model_random100.pt")

    # Step 3: Zero-shot large model on Random50
    log("\n[3/3] Evaluating on Random50...")
    log("\n  Large Model ZERO-SHOT → Random50:")
    zs_raw, zs_platt = evaluate_predictor_on_samples(large_model, r50_test, r50_calib)
    log(f"    Raw    — auc={zs_raw['auc']:.4f} ece={zs_raw['ece']:.4f} mae={zs_raw['delay_mae']:.4f}")
    log(f"    +Platt — auc={zs_platt['auc']:.4f} ece={zs_platt['ece']:.4f} mae={zs_platt['delay_mae']:.4f}")

    # Fine-tune configs
    configs = [
        (0.00, r50_ft2k, "large_2k_no_noise"),
        (0.15, r50_ft2k, "large_2k_noise_0.15"),
        (0.00, r50_ft5k, "large_5k_no_noise"),
        (0.15, r50_ft5k, "large_5k_noise_0.15"),
    ]

    results = {}
    for noise_std, ft_samples, name in configs:
        log(f"\n  Config: {name}")
        predictor = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
        predictor.load_state_dict(torch.load("large_model_random100.pt"))

        split = int(0.8 * len(ft_samples))
        ft_train_ld = DataLoader(OpticalDataset(ft_samples[:split]), batch_size=256, shuffle=True)
        ft_val_ld = DataLoader(OpticalDataset(ft_samples[split:]), batch_size=256, shuffle=False)
        finetune_with_noise(predictor, ft_train_ld, ft_val_ld, epochs=5, lr=5e-4, noise_std=noise_std)

        raw, platt = evaluate_predictor_on_samples(predictor, r50_test, r50_calib)
        log(f"    Raw    — auc={raw['auc']:.4f} ece={raw['ece']:.4f} mae={raw['delay_mae']:.4f}")
        log(f"    +Platt — auc={platt['auc']:.4f} ece={platt['ece']:.4f} mae={platt['delay_mae']:.4f}")
        results[name] = {"raw": raw, "platt": platt}

    # Summary
    log("\n" + "="*80)
    log("LARGE→SMALL + NOISE FINE-TUNE SUMMARY (Random50 target)")
    log("="*80)
    log(f"{'Config':<25} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12} {'MAE(Raw)':>10}")
    log("-"*80)
    log(f"{'zero-shot (large)':<25} {zs_raw['auc']:10.4f} {zs_platt['auc']:12.4f} {zs_raw['ece']:10.4f} {zs_platt['ece']:12.4f} {zs_raw['delay_mae']:10.4f}")
    for noise_std, ft_samples, name in configs:
        r = results[name]
        log(f"{name:<25} {r['raw']['auc']:10.4f} {r['platt']['auc']:12.4f} "
            f"{r['raw']['ece']:10.4f} {r['platt']['ece']:12.4f} {r['raw']['delay_mae']:10.4f}")
    log("-"*80)
    log(f"{'intra Random50 (ref)':<25} {'~0.950':>10} {'~0.950':>12} {'~0.020':>10} {'~0.020':>12} {'~0.010':>10}")
    log(f"{'small→Random50 (ref)':<25} {'0.8481':>10} {'0.8481':>12} {'0.1531':>10} {'0.1531':>12} {'0.0407':>10}")
