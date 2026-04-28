"""
Predictor MVP — PyTorch backend.
Ablation + Calibration + Latency + Generalization
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor, evaluate_predictor, plot_calibration, measure_latency
from torch.utils.data import DataLoader


def run_experiment(
    encoder_version="v2",
    topology="nsfnet",
    num_slots=32,
    num_samples=20000,
    epochs=30,
    arrival_rate=5.0,
    avg_holding_time=10.0,
    preload=500,
):
    print(f"\n{'='*60}")
    print(f"Topology: {topology} | Encoder: {encoder_version}")
    print(f"{'='*60}")

    np.random.seed(42)
    torch.manual_seed(42)

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=42)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)

    if encoder_version == "v0":
        encoder.encode = encoder.encode_v0
    elif encoder_version == "v1":
        encoder.encode = encoder.encode_v1
    elif encoder_version == "v2":
        encoder.encode = encoder.encode_v2
    elif encoder_version == "v2b":
        encoder.encode = encoder.encode_v2b
    elif encoder_version == "v2n":
        encoder.encode = encoder.encode_v2n
    elif encoder_version == "v2s":
        encoder.encode = encoder.encode_v2s
    else:
        raise ValueError(encoder_version)

    print("Generating dataset ...")
    gen = DatasetGenerator(net, mapper, encoder, seed=42)
    samples = gen.generate(
        num_samples,
        arrival_rate=arrival_rate,
        avg_holding_time=avg_holding_time,
        preload=preload,
    )

    succ_rate = np.mean([s["success"] for s in samples])
    print(f"  Dataset size: {len(samples)}  |  Success rate: {succ_rate:.3f}")

    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    state_dim = len(samples[0]["z"])
    print(f"  State dim: {state_dim}")

    predictor = Predictor(state_dim, num_partitions=3, num_servers=net.NUM_NODES, hidden_dim=64)
    history = train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)

    final = evaluate_predictor(predictor, val_loader)
    print(f"\n  Final — acc={final['accuracy']:.4f} auc={final['auc']:.4f} "
          f"mae={final['delay_mae']:.6f} ece={final['ece']:.4f}")

    plot_calibration(predictor, val_loader, save_path=f"calibration_{topology}_{encoder_version}.png")

    lat = measure_latency(predictor, encoder, mapper, num_runs=500)
    print(f"  Latency — encoder={lat['encoder_ms']*1000:.1f}us  "
          f"predictor={lat['predictor_ms']*1000:.1f}us  "
          f"mapper={lat['mapper_ms']*1000:.1f}us")

    return {"metrics": final, "latency": lat, "history": history}


if __name__ == "__main__":
    all_results = {}

    # Experiment 1: NSFNET ablation with coarse/noisy variants
    print("\n" + "="*60)
    print("EXPERIMENT 1: NSFNET Ablation (exact / binned / noisy / summary)")
    print("="*60)
    for version in ["v0", "v1", "v2", "v2b", "v2n", "v2s"]:
        all_results[("nsfnet", version)] = run_experiment(
            version, topology="nsfnet", num_slots=32, num_samples=20000,
            arrival_rate=5.0, avg_holding_time=10.0, preload=500
        )

    # Experiment 2: USNET generalization
    print("\n" + "="*60)
    print("EXPERIMENT 2: USNET Generalization")
    print("="*60)
    for version in ["v0", "v2", "v2b"]:
        all_results[("usnet", version)] = run_experiment(
            version, topology="usnet", num_slots=64, num_samples=20000,
            arrival_rate=8.0, avg_holding_time=12.0, preload=800
        )

    # Summary tables
    print("\n" + "="*60)
    print("SUMMARY TABLE 1: NSFNET Ablation")
    print("="*60)
    print(f"{'Version':<8} {'Acc':>6} {'AUC':>6} {'DelayMAE':>10} {'ECE':>6} {'z_dim':>5}")
    for v in ["v0", "v1", "v2", "v2b", "v2n", "v2s"]:
        m = all_results[("nsfnet", v)]["metrics"]
        dim = len(all_results[("nsfnet", v)]["history"])
        # hack: read dim from first sample
        print(f"{v:<8} {m['accuracy']:6.4f} {m['auc']:6.4f} {m['delay_mae']:10.6f} {m['ece']:6.4f}")

    print("\n" + "="*60)
    print("SUMMARY TABLE 2: USNET Generalization")
    print("="*60)
    print(f"{'Version':<8} {'Acc':>6} {'AUC':>6} {'DelayMAE':>10} {'ECE':>6}")
    for v in ["v0", "v2", "v2b"]:
        m = all_results[("usnet", v)]["metrics"]
        print(f"{v:<8} {m['accuracy']:6.4f} {m['auc']:6.4f} {m['delay_mae']:10.6f} {m['ece']:6.4f}")
