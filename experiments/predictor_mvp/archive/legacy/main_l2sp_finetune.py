"""
L2-SP Regularized Fine-Tuning + Path-Level Data Augmentation.
L2-SP penalizes deviation from pretrained weights:
    Loss = BCE + lambda * ||theta - theta_pretrain||^2
Data augmentation adds Gaussian noise to the 10D state vector z.
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


def l2sp_loss(model, pretrained_state_dict, lambda_l2sp):
    """L2-SP: penalize squared deviation from pretrained weights."""
    if lambda_l2sp == 0.0:
        return 0.0
    loss = 0.0
    for name, param in model.named_parameters():
        if name in pretrained_state_dict and param.requires_grad:
            loss += torch.sum((param - pretrained_state_dict[name].to(param.device)) ** 2)
    return lambda_l2sp * loss


def finetune(predictor, pretrained_state_dict, train_loader, val_loader,
             epochs=5, lr=5e-4, l2sp_lambda=0.0, noise_std=0.0):
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
            # Add L2-SP regularization
            loss = loss + l2sp_loss(predictor, pretrained_state_dict, l2sp_lambda)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Quick val eval
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


def run_config(l2sp_lambda, noise_std, ft_samples, test_samples, calib_samples, epochs=5):
    label = f"L2={l2sp_lambda:.0e}_N={noise_std:.2f}"
    log(f"\n{'='*60}")
    log(f"Config: {label} | Epochs: {epochs}")
    log(f"{'='*60}")

    np.random.seed(100)
    torch.manual_seed(100)

    predictor = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    pretrained_state_dict = torch.load("pretrained_nsfnet_v2b.pt")
    predictor.load_state_dict(pretrained_state_dict)

    split = int(0.8 * len(ft_samples))
    ft_train = DataLoader(OpticalDataset(ft_samples[:split]), batch_size=256, shuffle=True)
    ft_val = DataLoader(OpticalDataset(ft_samples[split:]), batch_size=256, shuffle=False)

    finetune(predictor, pretrained_state_dict, ft_train, ft_val,
             epochs=epochs, lr=5e-4, l2sp_lambda=l2sp_lambda, noise_std=noise_std)

    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw, platt = evaluate_with_platt(predictor, test_loader, calib_samples)
    log(f"  Raw    — auc={raw['auc']:.4f} ece={raw['ece']:.4f} mae={raw['delay_mae']:.4f}")
    log(f"  +Platt — auc={platt['auc']:.4f} ece={platt['ece']:.4f} mae={platt['delay_mae']:.4f}")
    return {"raw": raw, "platt": platt}


if __name__ == "__main__":
    log("Pre-generating data...")
    t0 = time.time()
    ft_samples = generate_samples("random100", "v2b", 2000, seed=100)
    test_samples = generate_samples("random100", "v2b", 2000, seed=200)
    calib_samples = generate_samples("random100", "v2b", 2000, seed=300)
    log(f"Data ready in {time.time()-t0:.1f}s")
    log(f"  FT={len(ft_samples)} SR={np.mean([s['success'] for s in ft_samples]):.3f}")
    log(f"  Test={len(test_samples)} SR={np.mean([s['success'] for s in test_samples]):.3f}")

    # Experiment grid
    configs = [
        # Baselines
        (0.0, 0.0, "full_baseline"),
        # L2-SP only
        (1e-4, 0.0, "l2sp_1e-4"),
        (1e-3, 0.0, "l2sp_1e-3"),
        (1e-2, 0.0, "l2sp_1e-2"),
        (1e-1, 0.0, "l2sp_1e-1"),
        # Noise only
        (0.0, 0.01, "noise_0.01"),
        (0.0, 0.05, "noise_0.05"),
        (0.0, 0.10, "noise_0.10"),
        # Combined best guesses
        (1e-3, 0.05, "l2sp_1e-3_noise_0.05"),
        (1e-2, 0.05, "l2sp_1e-2_noise_0.05"),
    ]

    results = {}
    for l2sp, noise, name in configs:
        results[name] = run_config(l2sp, noise, ft_samples, test_samples, calib_samples, epochs=5)

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
    log("\n" + "="*80)
    log("L2-SP + NOISE AUGMENTATION SUMMARY (Random100, 2K FT, 5 epochs)")
    log("="*80)
    log(f"{'Config':<25} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12} {'MAE(Raw)':>10}")
    log("-"*80)
    log(f"{'zero-shot':<25} {zs_raw['auc']:10.4f} {zs_platt['auc']:12.4f} {zs_raw['ece']:10.4f} {zs_platt['ece']:12.4f} {zs_raw['delay_mae']:10.4f}")
    for l2sp, noise, name in configs:
        r = results[name]
        log(f"{name:<25} {r['raw']['auc']:10.4f} {r['platt']['auc']:12.4f} "
            f"{r['raw']['ece']:10.4f} {r['platt']['ece']:12.4f} {r['raw']['delay_mae']:10.4f}")
