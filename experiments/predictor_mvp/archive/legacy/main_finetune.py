"""
Fine-tuning experiment: Pre-train on NSFNET, fine-tune on Random100.
Compare data budgets: 500, 2K, 5K samples.
"""
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

TOPOLOGIES = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def generate_data(topology, version, num_samples, seed=42):
    n_nodes, num_slots, arr, ht, preload = TOPOLOGIES[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    if version == "v0":
        encoder.encode = encoder.encode_v0
    elif version == "v2b":
        encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)
    return samples


def pretrain_on_nsfnet(version="v2b", num_samples=15000, epochs=30):
    print(f"\n{'='*70}")
    print(f"PRE-TRAIN: NSFNET | Encoder: {version}")
    print(f"{'='*70}")
    np.random.seed(42)
    torch.manual_seed(42)
    samples = generate_data("nsfnet", version, num_samples, seed=42)
    print(f"  Train: {len(samples)} | SR: {np.mean([s['success'] for s in samples]):.3f}")
    state_dim = len(samples[0]["z"])
    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)
    predictor = Predictor(state_dim, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor, state_dim


def finetune_and_evaluate(predictor, state_dim, version="v2b",
                          finetune_n=2000, finetune_epochs=5, test_n=3000):
    print(f"\n{'='*70}")
    print(f"FINE-TUNE: Random100 | N={finetune_n} | Epochs={finetune_epochs}")
    print(f"{'='*70}")

    np.random.seed(100)
    torch.manual_seed(100)

    # Generate fine-tune + test data
    ft_samples = generate_data("random100", version, finetune_n, seed=100)
    test_samples = generate_data("random100", version, test_n, seed=200)
    print(f"  Fine-tune: {len(ft_samples)} | SR: {np.mean([s['success'] for s in ft_samples]):.3f}")
    print(f"  Test:      {len(test_samples)} | SR: {np.mean([s['success'] for s in test_samples]):.3f}")

    # Fine-tune
    if finetune_n > 0:
        split = int(0.8 * len(ft_samples))
        ft_train = DataLoader(OpticalDataset(ft_samples[:split]), batch_size=256, shuffle=True)
        ft_val = DataLoader(OpticalDataset(ft_samples[split:]), batch_size=256, shuffle=False)
        # Use lower LR for fine-tuning
        train_predictor(predictor, ft_train, ft_val, epochs=finetune_epochs, lr=5e-4)

    # Evaluate on test set
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)

    # Raw
    raw = evaluate_predictor(predictor, test_loader)
    print(f"\n  Raw      — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")

    # Platt on fine-tune data
    ft_all_loader = DataLoader(OpticalDataset(ft_samples), batch_size=256, shuffle=False)
    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in ft_all_loader:
            logit, _ = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            all_logits.extend(logit.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    platt = PlattScaling().fit(np.array(all_logits), np.array(all_labels))
    platt_result = evaluate_with_calibration(predictor, test_loader, calibrator=platt)
    print(f"  +Platt   — acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:.4f}")

    return {"raw": raw, "platt": platt_result}


def zero_shot_baseline(predictor, state_dim, version="v2b", test_n=3000):
    print(f"\n{'='*70}")
    print(f"ZERO-SHOT BASELINE: NSFNET → Random100")
    print(f"{'='*70}")
    test_samples = generate_data("random100", version, test_n, seed=200)
    print(f"  Test: {len(test_samples)} | SR: {np.mean([s['success'] for s in test_samples]):.3f}")
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw = evaluate_predictor(predictor, test_loader)
    print(f"  Raw    — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:.4f}")
    return raw


if __name__ == "__main__":
    # Pre-train on NSFNET
    predictor, state_dim = pretrain_on_nsfnet(version="v2b", num_samples=15000, epochs=30)

    # Save pretrained
    torch.save(predictor.state_dict(), "pretrained_nsfnet_v2b.pt")

    # Zero-shot baseline (no fine-tune)
    zs = zero_shot_baseline(predictor, state_dim, version="v2b", test_n=3000)

    # Fine-tune experiments
    configs = [
        (0, 0),      # zero-shot already done above, but re-run for consistency
        (500, 5),
        (2000, 5),
        (5000, 10),
    ]

    results = {"zero_shot": zs}
    for ft_n, ft_ep in configs:
        if ft_n == 0:
            continue
        # Reload pretrained for each experiment
        predictor_reload = Predictor(state_dim, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
        predictor_reload.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))
        r = finetune_and_evaluate(predictor_reload, state_dim, "v2b",
                                   finetune_n=ft_n, finetune_epochs=ft_ep, test_n=3000)
        results[(ft_n, ft_ep)] = r

    # Summary
    print("\n" + "="*70)
    print("FINE-TUNE SUMMARY: Random100")
    print("="*70)
    print(f"{'Strategy':<20} {'AUC':>6} {'ECE':>6}")
    print("-"*40)
    print(f"{'Zero-shot':<20} {zs['auc']:6.4f} {zs['ece']:6.4f}")
    for key, r in results.items():
        if key == "zero_shot":
            continue
        ft_n, ft_ep = key
        label = f"FT {ft_n}@{ft_ep}ep +Platt"
        print(f"{label:<20} {r['platt']['auc']:6.4f} {r['platt']['ece']:6.4f}")
