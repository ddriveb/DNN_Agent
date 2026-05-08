"""
Same-scale topology transfer experiment.
Test whether a model trained on Random100 (seed=100) generalizes to
a different Random100 instance (seed=201) with different random edges.
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
    log("SAME-SCALE TOPOLOGY TRANSFER (Random100→Random100 variant)")
    log("="*70)

    # Generate test data for same-scale different topologies
    log("\nGenerating test data...")
    t0 = time.time()
    # Same topology (seed=200) — already tested before, but regenerate for fair comparison
    r100_same = generate_samples("random100", "v2b", 2000, seed=200)
    # Different topology (seed=201) — different random graph
    r100_diff = generate_samples("random100", "v2b", 2000, seed=201)
    # Another different topology (seed=202)
    r100_diff2 = generate_samples("random100", "v2b", 2000, seed=202)
    # Calibration data (use same seed as training for fairness, or different? Use seed=300)
    calib_same = generate_samples("random100", "v2b", 2000, seed=300)
    calib_diff = generate_samples("random100", "v2b", 2000, seed=301)
    log(f"Data ready in {time.time()-t0:.1f}s")
    log(f"  Same seed=200    SR={np.mean([s['success'] for s in r100_same]):.3f}")
    log(f"  Diff seed=201    SR={np.mean([s['success'] for s in r100_diff]):.3f}")
    log(f"  Diff seed=202    SR={np.mean([s['success'] for s in r100_diff2]):.3f}")

    # Load models
    log("\nLoading models...")
    # Large model (Random100-trained)
    large_model = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    large_model.load_state_dict(torch.load("large_model_random100.pt"))
    # Small model (NSFNET-trained)
    small_model = Predictor(10, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)
    small_model.load_state_dict(torch.load("pretrained_nsfnet_v2b.pt"))

    # Evaluate Large Model
    log("\n--- Large Model (Random100-trained) ---")
    log("  → Same Random100 (seed=200):")
    r, p = evaluate_predictor_on_samples(large_model, r100_same, calib_same)
    log(f"    Raw    — auc={r['auc']:.4f} ece={r['ece']:.4f} mae={r['delay_mae']:.4f}")
    log(f"    +Platt — auc={p['auc']:.4f} ece={p['ece']:.4f} mae={p['delay_mae']:.4f}")

    log("  → Diff Random100 (seed=201):")
    r, p = evaluate_predictor_on_samples(large_model, r100_diff, calib_diff)
    log(f"    Raw    — auc={r['auc']:.4f} ece={r['ece']:.4f} mae={r['delay_mae']:.4f}")
    log(f"    +Platt — auc={p['auc']:.4f} ece={p['ece']:.4f} mae={p['delay_mae']:.4f}")

    log("  → Diff Random100 (seed=202):")
    r2, p2 = evaluate_predictor_on_samples(large_model, r100_diff2, calib_diff)
    log(f"    Raw    — auc={r2['auc']:.4f} ece={r2['ece']:.4f} mae={r2['delay_mae']:.4f}")
    log(f"    +Platt — auc={p2['auc']:.4f} ece={p2['ece']:.4f} mae={p2['delay_mae']:.4f}")

    # Evaluate Small Model
    log("\n--- Small Model (NSFNET-trained) ---")
    log("  → Same Random100 (seed=200):")
    r, p = evaluate_predictor_on_samples(small_model, r100_same, calib_same)
    log(f"    Raw    — auc={r['auc']:.4f} ece={r['ece']:.4f} mae={r['delay_mae']:.4f}")
    log(f"    +Platt — auc={p['auc']:.4f} ece={p['ece']:.4f} mae={p['delay_mae']:.4f}")

    log("  → Diff Random100 (seed=201):")
    r, p = evaluate_predictor_on_samples(small_model, r100_diff, calib_diff)
    log(f"    Raw    — auc={r['auc']:.4f} ece={r['ece']:.4f} mae={r['delay_mae']:.4f}")
    log(f"    +Platt — auc={p['auc']:.4f} ece={p['ece']:.4f} mae={p['delay_mae']:.4f}")

    log("  → Diff Random100 (seed=202):")
    r2, p2 = evaluate_predictor_on_samples(small_model, r100_diff2, calib_diff)
    log(f"    Raw    — auc={r2['auc']:.4f} ece={r2['ece']:.4f} mae={r2['delay_mae']:.4f}")
    log(f"    +Platt — auc={p2['auc']:.4f} ece={p2['ece']:.4f} mae={p2['delay_mae']:.4f}")

    # Summary
    log("\n" + "="*75)
    log("SAME-SCALE TOPOLOGY TRANSFER SUMMARY")
    log("="*75)
    log(f"{'Model / Target':<30} {'Same(seed=200)':>14} {'Diff(seed=201)':>14} {'Diff(seed=202)':>14}")
    log("-"*75)
    log(f"{'Large (R100-trained) AUC':<30} {'0.7884*':>14} {'TBD':>14} {'TBD':>14}")
    log(f"{'Small (NSFNET-trained) AUC':<30} {'0.7783*':>14} {'TBD':>14} {'TBD':>14}")
    log("-"*75)
    log("* Values from previous experiments for reference")
