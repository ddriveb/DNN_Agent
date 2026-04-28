"""
Mixed-topology training: train on NSFNET + USNET + Random50,
then zero-shot test on Random100.
Compare against NSFNET-only pre-training baseline.
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

TOPO_CONFIGS = {
    "nsfnet":    (14, 32,   5.0, 10.0, 500),
    "usnet":     (28, 64,   8.0, 12.0, 800),
    "random50":  (50, 128,  15.0, 15.0, 1500),
    "random100": (100, 320, 30.0, 25.0, 3000),
}


def generate_topo_samples(topology, version, num_samples, seed):
    n_nodes, num_slots, arr, ht, preload = TOPO_CONFIGS[topology]
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    if version == "v0":
        encoder.encode = encoder.encode_v0
    elif version == "v2b":
        encoder.encode = encoder.encode_v2b
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)


def train_on_mixed(version="v2b", epochs=30):
    print(f"\n{'='*70}")
    print(f"MIXED TRAINING: NSFNET(5K) + USNET(5K) + Random50(5K) | Encoder: {version}")
    print(f"{'='*70}")

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate mixed dataset
    s1 = generate_topo_samples("nsfnet", version, 5000, seed=42)
    s2 = generate_topo_samples("usnet", version, 5000, seed=43)
    s3 = generate_topo_samples("random50", version, 5000, seed=44)
    all_samples = s1 + s2 + s3
    np.random.shuffle(all_samples)

    print(f"  Total samples: {len(all_samples)}")
    print(f"  NSFNET  SR: {np.mean([s['success'] for s in s1]):.3f}")
    print(f"  USNET   SR: {np.mean([s['success'] for s in s2]):.3f}")
    print(f"  Random50 SR: {np.mean([s['success'] for s in s3]):.3f}")

    state_dim = len(all_samples[0]["z"])
    split = int(0.8 * len(all_samples))
    train_loader = DataLoader(OpticalDataset(all_samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(all_samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(state_dim, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor, state_dim


def evaluate_on_random100(predictor, version="v2b", test_n=3000, calib_n=2000):
    print(f"\n{'='*70}")
    print(f"EVALUATE: Mixed-trained → Random100")
    print(f"{'='*70}")

    test_samples = generate_topo_samples("random100", version, test_n, seed=200)
    calib_samples = generate_topo_samples("random100", version, calib_n, seed=100)
    print(f"  Test:  {len(test_samples)} | SR: {np.mean([s['success'] for s in test_samples]):.3f}")
    print(f"  Calib: {len(calib_samples)} | SR: {np.mean([s['success'] for s in calib_samples]):.3f}")

    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw = evaluate_predictor(predictor, test_loader)
    print(f"  Raw    — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:4f}")

    # Platt
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
    print(f"  +Platt — acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:4f}")

    return {"raw": raw, "platt": platt_result}


def finetune_mixed_on_random100(predictor, state_dim, version="v2b",
                                 finetune_n=2000, finetune_epochs=5, test_n=3000):
    print(f"\n{'='*70}")
    print(f"FINE-TUNE: Mixed-trained → Random100 | N={finetune_n}")
    print(f"{'='*70}")

    np.random.seed(100)
    torch.manual_seed(100)

    ft_samples = generate_topo_samples("random100", version, finetune_n, seed=100)
    test_samples = generate_topo_samples("random100", version, test_n, seed=200)
    print(f"  Fine-tune: {len(ft_samples)} | SR: {np.mean([s['success'] for s in ft_samples]):.3f}")
    print(f"  Test:      {len(test_samples)} | SR: {np.mean([s['success'] for s in test_samples]):.3f}")

    split = int(0.8 * len(ft_samples))
    ft_train = DataLoader(OpticalDataset(ft_samples[:split]), batch_size=256, shuffle=True)
    ft_val = DataLoader(OpticalDataset(ft_samples[split:]), batch_size=256, shuffle=False)

    train_predictor(predictor, ft_train, ft_val, epochs=finetune_epochs, lr=5e-4)

    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    raw = evaluate_predictor(predictor, test_loader)

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

    print(f"  Raw    — acc={raw['accuracy']:.4f} auc={raw['auc']:.4f} ece={raw['ece']:4f}")
    print(f"  +Platt — acc={platt_result['accuracy']:.4f} auc={platt_result['auc']:.4f} ece={platt_result['ece']:4f}")
    return {"raw": raw, "platt": platt_result}


if __name__ == "__main__":
    # Train mixed model
    mixed_predictor, state_dim = train_on_mixed(version="v2b", epochs=30)
    torch.save(mixed_predictor.state_dict(), "pretrained_mixed_v2b.pt")

    # Zero-shot on Random100
    mixed_zs = evaluate_on_random100(mixed_predictor, version="v2b", test_n=3000, calib_n=2000)

    # Fine-tune mixed model on Random100 (2K samples)
    mixed_ft = finetune_mixed_on_random100(mixed_predictor, state_dim, version="v2b",
                                            finetune_n=2000, finetune_epochs=5, test_n=3000)

    # For comparison, also run NSFNET-only baseline
    print("\n" + "="*70)
    print("NSFNET-ONLY BASELINE (for comparison)")
    print("="*70)
    from main_finetune import pretrain_on_nsfnet, zero_shot_baseline
    nsf_predictor, _ = pretrain_on_nsfnet(version="v2b", num_samples=15000, epochs=30)
    nsf_zs = zero_shot_baseline(nsf_predictor, state_dim, version="v2b", test_n=3000)

    # Summary
    print("\n" + "="*70)
    print("MIXED TRAINING SUMMARY: Random100")
    print("="*70)
    print(f"{'Strategy':<30} {'AUC':>6} {'ECE':>6}")
    print("-"*50)
    print(f"{'NSFNET-only → Zero-shot':<30} {nsf_zs['auc']:6.4f} {nsf_zs['ece']:6.4f}")
    print(f"{'Mixed → Zero-shot':<30} {mixed_zs['raw']['auc']:6.4f} {mixed_zs['raw']['ece']:6.4f}")
    print(f"{'Mixed → Zero-shot + Platt':<30} {mixed_zs['platt']['auc']:6.4f} {mixed_zs['platt']['ece']:6.4f}")
    print(f"{'Mixed → FT 2K + Platt':<30} {mixed_ft['platt']['auc']:6.4f} {mixed_ft['platt']['ece']:6.4f}")
