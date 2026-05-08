"""
8-bin calibration comparison: Raw vs Platt vs per-topology Temperature.
Tests all model-target pairs with 2K calibration samples per topology.
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
from calibration_methods import PlattScaling, TemperatureScaling, evaluate_with_calibration
from torch.utils.data import DataLoader

MAX_SERVERS = 128
NUM_BINS = 8
STATE_DIM = 2 + NUM_BINS * 2


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def generate_samples(topology, num_samples, seed=42):
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


def evaluate_all_methods(predictor, test_samples, calib_samples):
    """Returns dict with raw, platt, and temperature metrics."""
    test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
    calib_loader = DataLoader(OpticalDataset(calib_samples), batch_size=256, shuffle=False)

    # Collect logits and labels on calibration set
    predictor.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in calib_loader:
            logit, _ = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            all_logits.extend(logit.numpy().tolist())
            all_labels.extend(batch["success"].numpy().tolist())
    logits = np.array(all_logits)
    labels = np.array(all_labels)

    # Raw
    raw = evaluate_with_calibration(predictor, test_loader, calibrator=None)

    # Platt
    platt = PlattScaling().fit(logits, labels)
    platt_result = evaluate_with_calibration(predictor, test_loader, calibrator=platt)

    # Temperature (per-topology)
    temp = TemperatureScaling().fit(logits, labels)
    temp_result = evaluate_with_calibration(predictor, test_loader, calibrator=temp)

    return {
        "raw": raw,
        "platt": platt_result,
        "temp": temp_result,
        "temp_T": temp.T,
        "platt_a": platt.a,
        "platt_b": platt.b,
    }


if __name__ == "__main__":
    log("="*70)
    log("8-BIN CALIBRATION: RAW vs PLATT vs TEMPERATURE")
    log("="*70)

    # Load models
    log("\nLoading 8-bin models...")
    small_model = Predictor(STATE_DIM, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    small_model.load_state_dict(torch.load("pretrained_nsfnet_v2b_8bin.pt"))
    large_model = Predictor(STATE_DIM, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    large_model.load_state_dict(torch.load("large_model_random100_8bin.pt"))

    # Generate test and calibration data
    log("\nGenerating test & calibration data...")
    t0 = time.time()
    tests = {
        "nsfnet":    generate_samples("nsfnet", 2000, seed=200),
        "random50":  generate_samples("random50", 2000, seed=200),
        "r100same":  generate_samples("random100", 2000, seed=200),
        "r100diff":  generate_samples("random100", 2000, seed=201),
    }
    calibs = {
        "nsfnet":    generate_samples("nsfnet", 2000, seed=300),
        "random50":  generate_samples("random50", 2000, seed=300),
        "r100same":  generate_samples("random100", 2000, seed=300),
        "r100diff":  generate_samples("random100", 2000, seed=301),
    }
    log(f"Data ready in {time.time()-t0:.1f}s")

    # Evaluation matrix
    configs = [
        ("Small→NSFNET",    small_model, "nsfnet",    "nsfnet"),
        ("Small→Random50",  small_model, "random50",  "random50"),
        ("Small→R100same", small_model, "r100same",  "r100same"),
        ("Large→R100same", large_model, "r100same",  "r100same"),
        ("Large→R100diff", large_model, "r100diff",  "r100diff"),
        ("Large→Random50", large_model, "random50",  "random50"),
        ("Large→NSFNET",   large_model, "nsfnet",    "nsfnet"),
    ]

    results = {}
    for name, model, test_key, calib_key in configs:
        log(f"\nEvaluating: {name}")
        r = evaluate_all_methods(model, tests[test_key], calibs[calib_key])
        log(f"  Raw  — AUC={r['raw']['auc']:.4f} ECE={r['raw']['ece']:.4f}")
        log(f"  Platt— AUC={r['platt']['auc']:.4f} ECE={r['platt']['ece']:.4f} (a={r['platt_a']:.3f}, b={r['platt_b']:.3f})")
        log(f"  Temp — AUC={r['temp']['auc']:.4f} ECE={r['temp']['ece']:.4f} (T={r['temp_T']:.3f})")
        results[name] = r

    # Summary table
    log("\n" + "="*90)
    log("8-BIN CALIBRATION SUMMARY")
    log("="*90)
    log(f"{'Transfer':<18} {'Raw AUC':>8} {'Raw ECE':>8} {'Platt ECE':>10} {'Temp ECE':>10} {'Temp T':>8}")
    log("-"*90)
    for name, _, _, _ in configs:
        r = results[name]
        log(f"{name:<18} {r['raw']['auc']:8.4f} {r['raw']['ece']:8.4f} {r['platt']['ece']:10.4f} {r['temp']['ece']:10.4f} {r['temp_T']:8.3f}")
    log("="*90)

    # Best method per transfer
    log("\nBest calibration method per transfer (lowest ECE):")
    for name, _, _, _ in configs:
        r = results[name]
        best = min([("Raw", r["raw"]["ece"]), ("Platt", r["platt"]["ece"]), ("Temp", r["temp"]["ece"])], key=lambda x: x[1])
        log(f"  {name:<18} → {best[0]} (ECE={best[1]:.4f})")
