"""
Full transfer matrix for 8-bin + bw-aware models.
Compares small model (NSFNET-trained) vs large model (Random100-trained)
on zero-shot transfer to all target topologies.
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from calibration_methods import PlattScaling, evaluate_with_calibration
from torch.utils.data import DataLoader
from train import OpticalDataset

MAX_SERVERS = 128
NUM_BINS = 8
STATE_DIM = 2 + NUM_BINS * 2


def log(msg):
    print(msg, flush=True)


def generate_data(topology, num_samples, seed=42):
    if topology == "nsfnet":
        n, slots, arr, ht, preload = 14, 32, 5.0, 10.0, 500
    elif topology == "random50":
        n, slots, arr, ht, preload = 50, 128, 15.0, 15.0, 2000
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


def evaluate_with_platt(predictor, test_samples, calib_samples):
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    calib_loader = DataLoader(OpticalDataset(calib_samples), batch_size=256, shuffle=False)

    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in calib_loader:
            logit, _ = predictor(batch["path_id"], batch["src"], batch["dst"], batch["bw"], batch["z"])
            all_logits.extend(logit.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    platt = PlattScaling().fit(np.array(all_logits), np.array(all_labels))

    raw = evaluate_with_calibration(predictor, test_loader, calibrator=None)
    platt_result = evaluate_with_calibration(predictor, test_loader, calibrator=platt)
    return raw, platt_result


def main():
    log("=" * 75)
    log("8-BIN + BW-AWARE TRANSFER MATRIX")
    log("=" * 75)

    # Load models
    log("\nLoading models...")
    small = Predictor(STATE_DIM, num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)
    small.load_state_dict(torch.load("pretrained_nsfnet_v2b_8bin_bw.pt"))
    large = Predictor(STATE_DIM, num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)
    large.load_state_dict(torch.load("large_model_random100_8bin_bw.pt"))
    log("  Loaded: small (NSFNET-trained) + large (Random100-trained)")

    # Generate data
    log("\nGenerating test & calibration data...")
    tests = {
        "nsfnet": generate_data("nsfnet", 2000, seed=200),
        "random50": generate_data("random50", 2000, seed=200),
        "r100same": generate_data("random100", 2000, seed=200),
        "r100diff": generate_data("random100", 2000, seed=201),
    }
    calibs = {
        "nsfnet": generate_data("nsfnet", 2000, seed=300),
        "random50": generate_data("random50", 2000, seed=300),
        "r100same": generate_data("random100", 2000, seed=300),
        "r100diff": generate_data("random100", 2000, seed=301),
    }

    configs = [
        ("Small→NSFNET (intra)", small, "nsfnet", "nsfnet"),
        ("Small→Random50", small, "random50", "random50"),
        ("Small→R100same", small, "r100same", "r100same"),
        ("Large→R100same (intra)", large, "r100same", "r100same"),
        ("Large→R100diff", large, "r100diff", "r100diff"),
        ("Large→Random50", large, "random50", "random50"),
        ("Large→NSFNET", large, "nsfnet", "nsfnet"),
    ]

    results = {}
    for name, model, test_key, calib_key in configs:
        log(f"\nEvaluating: {name}")
        raw, platt = evaluate_with_platt(model, tests[test_key], calibs[calib_key])
        log(f"  Raw   — AUC={raw['auc']:.4f} ECE={raw['ece']:.4f}")
        log(f"  Platt — AUC={platt['auc']:.4f} ECE={platt['ece']:.4f}")
        results[name] = {"raw": raw, "platt": platt}

    # Summary
    log("\n" + "=" * 90)
    log("TRANSFER MATRIX SUMMARY (8-bin + bw-aware)")
    log("=" * 90)
    log(f"{'Transfer':<25} {'AUC(Raw)':>10} {'ECE(Raw)':>10} {'AUC(Platt)':>12} {'ECE(Platt)':>12}")
    log("-" * 90)
    for name, _, _, _ in configs:
        r = results[name]
        log(f"{name:<25} {r['raw']['auc']:10.4f} {r['raw']['ece']:10.4f} {r['platt']['auc']:12.4f} {r['platt']['ece']:12.4f}")
    log("=" * 90)

    log("\nComparison with 4-bin (no bw) reference:")
    log("  4-bin Small→NSFNET intra:     AUC≈0.992")
    log("  4-bin Small→Random100:         AUC≈0.778")
    log("  4-bin Small→Random50:          AUC≈0.848")
    log("  4-bin Large→Random100 intra:   AUC≈0.938")
    log("  4-bin Large→Random50:          AUC≈0.888")
    log("  4-bin Large→NSFNET:            AUC≈0.961")


if __name__ == "__main__":
    main()
