"""
Noise augmentation with varying sample sizes.
Tests sigma=0.15 vs no noise on 2K and 5K Random100 fine-tune samples.
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


def evaluate_with_platt(predictor, test_loader, calib_samples):
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


def run_config(noise_std, ft_samples, test_samples, calib_samples, epochs=5):
    label = f"σ={noise_std:.2f}_N={len(ft_samples)}"
    log(f"\n{'='*60}")
    log(f"Config: {label} | Epochs: {epochs}")
    log(f"{'='*60}")

    np.random.seed(100)
    torch.manual_seed(100)

    predictor = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    predictor.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))

    split = int(0.8 * len(ft_samples))
    ft_train = DataLoader(OpticalDataset(ft_samples[:split]), batch_size=256, shuffle=True)
    ft_val = DataLoader(OpticalDataset(ft_samples[split:]), batch_size=256, shuffle=False)

    finetune_with_noise(predictor, ft_train, ft_val, epochs=epochs, lr=5e-4, noise_std=noise_std)

    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw, platt = evaluate_with_platt(predictor, test_loader, calib_samples)
    log(f"  Raw    — auc={raw['auc']:.4f} ece={raw['ece']:.4f} mae={raw['delay_mae']:.4f}")
    log(f"  +Platt — auc={platt['auc']:.4f} ece={platt['ece']:.4f} mae={platt['delay_mae']:.4f}")
    return {"raw": raw, "platt": platt}


if __name__ == "__main__":
    log("Pre-generating data...")
    t0 = time.time()
    ft_2k = generate_samples("random100", "v2b", 2000, seed=100)
    ft_5k = generate_samples("random100", "v2b", 5000, seed=101)
    test_samples = generate_samples("random100", "v2b", 2000, seed=200)
    calib_samples = generate_samples("random100", "v2b", 2000, seed=300)
    log(f"Data ready in {time.time()-t0:.1f}s")
    log(f"  FT_2K={len(ft_2k)} SR={np.mean([s['success'] for s in ft_2k]):.3f}")
    log(f"  FT_5K={len(ft_5k)} SR={np.mean([s['success'] for s in ft_5k]):.3f}")
    log(f"  Test={len(test_samples)} SR={np.mean([s['success'] for s in test_samples]):.3f}")

    configs = [
        (0.00, ft_2k, "no_noise_2k"),
        (0.15, ft_2k, "noise_0.15_2k"),
        (0.00, ft_5k, "no_noise_5k"),
        (0.15, ft_5k, "noise_0.15_5k"),
    ]

    results = {}
    for noise_std, ft_samples, name in configs:
        results[name] = run_config(noise_std, ft_samples, test_samples, calib_samples, epochs=5)

    # Zero-shot
    log(f"\n{'='*60}")
    log("ZERO-SHOT BASELINE")
    log(f"{'='*60}")
    predictor = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    predictor.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    zs_raw, zs_platt = evaluate_with_platt(predictor, test_loader, calib_samples)
    log(f"  Raw    — auc={zs_raw['auc']:.4f} ece={zs_raw['ece']:.4f} mae={zs_raw['delay_mae']:.4f}")
    log(f"  +Platt — auc={zs_platt['auc']:.4f} ece={zs_platt['ece']:.4f} mae={zs_platt['delay_mae']:.4f}")

    # Summary
    log("\n" + "="*75)
    log("NOISE + SAMPLE SIZE SCALING (Random100, σ=0.15, 5 epochs)")
    log("="*75)
    log(f"{'Config':<20} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12} {'MAE(Raw)':>10}")
    log("-"*75)
    log(f"{'zero-shot':<20} {zs_raw['auc']:10.4f} {zs_platt['auc']:12.4f} {zs_raw['ece']:10.4f} {zs_platt['ece']:12.4f} {zs_raw['delay_mae']:10.4f}")
    for noise_std, ft_samples, name in configs:
        r = results[name]
        log(f"{name:<20} {r['raw']['auc']:10.4f} {r['platt']['auc']:12.4f} "
            f"{r['raw']['ece']:10.4f} {r['platt']['ece']:12.4f} {r['raw']['delay_mae']:10.4f}")
