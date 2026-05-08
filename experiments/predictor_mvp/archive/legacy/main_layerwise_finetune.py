"""
Layer-wise fine-tuning experiment.
Freeze bottom layers, only fine-tune top layers to avoid catastrophic forgetting.
Pre-generates data once and reuses across all strategies.
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


def apply_freeze_strategy(predictor, strategy):
    for p in predictor.parameters():
        p.requires_grad = True

    if strategy == "full":
        return

    for p in predictor.partition_embed.parameters():
        p.requires_grad = False
    for p in predictor.src_embed.parameters():
        p.requires_grad = False
    for p in predictor.dst_embed.parameters():
        p.requires_grad = False

    if strategy == "embed_only":
        return

    for p in predictor.mlp[0].parameters():
        p.requires_grad = False

    if strategy == "first_two":
        for p in predictor.mlp[3].parameters():
            p.requires_grad = False
        return

    if strategy == "output_only":
        for p in predictor.mlp[3].parameters():
            p.requires_grad = False
        return


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def finetune(predictor, train_loader, val_loader, epochs=3, lr=5e-4):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, predictor.parameters()), lr=lr)
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


def run_strategy(strategy, ft_samples, test_samples, calib_samples, epochs=3):
    log(f"\n{'='*60}")
    log(f"Strategy: {strategy} | FT: {len(ft_samples)} | Epochs: {epochs}")
    log(f"{'='*60}")

    np.random.seed(100)
    torch.manual_seed(100)

    predictor = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    predictor.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))
    apply_freeze_strategy(predictor, strategy)
    log(f"  Trainable params: {count_trainable(predictor)} / {sum(p.numel() for p in predictor.parameters())}")

    split = int(0.8 * len(ft_samples))
    ft_train = DataLoader(OpticalDataset(ft_samples[:split]), batch_size=256, shuffle=True)
    ft_val = DataLoader(OpticalDataset(ft_samples[split:]), batch_size=256, shuffle=False)

    finetune(predictor, ft_train, ft_val, epochs=epochs, lr=5e-4)

    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw, platt = evaluate_with_platt(predictor, test_loader, calib_samples)
    log(f"  Raw    — auc={raw['auc']:.4f} ece={raw['ece']:.4f} mae={raw['delay_mae']:.4f}")
    log(f"  +Platt — auc={platt['auc']:.4f} ece={platt['ece']:.4f} mae={platt['delay_mae']:.4f}")
    return {"raw": raw, "platt": platt}


if __name__ == "__main__":
    # Pre-generate all data once
    log("Pre-generating data (this may take a few minutes)...")
    t0 = time.time()

    log("  [1/3] Fine-tune samples (2000)...")
    ft_samples = generate_samples("random100", "v2b", 2000, seed=100)

    log("  [2/3] Test samples (2000)...")
    test_samples = generate_samples("random100", "v2b", 2000, seed=200)

    log("  [3/3] Calibration samples (2000)...")
    calib_samples = generate_samples("random100", "v2b", 2000, seed=300)

    log(f"Data generation complete in {time.time()-t0:.1f}s")
    log(f"  FT   SR={np.mean([s['success'] for s in ft_samples]):.3f}")
    log(f"  Test SR={np.mean([s['success'] for s in test_samples]):.3f}")

    strategies = ["output_only", "first_two", "embed_only", "full"]
    results = {}

    for strat in strategies:
        results[strat] = run_strategy(strat, ft_samples, test_samples, calib_samples, epochs=3)

    # Zero-shot baseline
    log(f"\n{'='*60}")
    log("ZERO-SHOT BASELINE (no fine-tune)")
    log(f"{'='*60}")
    predictor = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    predictor.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    zs_raw, zs_platt = evaluate_with_platt(predictor, test_loader, calib_samples)
    log(f"  Raw    — auc={zs_raw['auc']:.4f} ece={zs_raw['ece']:.4f} mae={zs_raw['delay_mae']:.4f}")
    log(f"  +Platt — auc={zs_platt['auc']:.4f} ece={zs_platt['ece']:.4f} mae={zs_platt['delay_mae']:.4f}")

    # Summary
    log("\n" + "="*70)
    log("LAYER-WISE FINE-TUNE SUMMARY (Random100, 2K samples, 3 epochs)")
    log("="*70)
    log(f"{'Strategy':<18} {'AUC(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Raw)':>10} {'ECE(Platt)':>12} {'MAE(Raw)':>10}")
    log("-"*70)
    log(f"{'zero-shot':<18} {zs_raw['auc']:10.4f} {zs_platt['auc']:12.4f} {zs_raw['ece']:10.4f} {zs_platt['ece']:12.4f} {zs_raw['delay_mae']:10.4f}")
    for strat in strategies:
        r = results[strat]
        log(f"{strat:<18} {r['raw']['auc']:10.4f} {r['platt']['auc']:12.4f} "
            f"{r['raw']['ece']:10.4f} {r['platt']['ece']:12.4f} {r['raw']['delay_mae']:10.4f}")
