"""
Predictor MVP — Scikit-learn backend (fast, no PyTorch).
"""
import numpy as np
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from sklearn_predictor import SklearnPredictor
from dataset import DatasetGenerator
import time


def run_experiment(encoder_version="v2", num_samples=8000, epochs=20):
    print(f"\n{'='*60}")
    print(f"Encoder version: {encoder_version}  (Sklearn Predictor)")
    print(f"{'='*60}")

    np.random.seed(42)
    net = OpticalNetwork(num_slots=128, seed=42)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)

    if encoder_version == "v0":
        encoder.encode = encoder.encode_v0
    elif encoder_version == "v1":
        encoder.encode = encoder.encode_v1
    elif encoder_version == "v2":
        encoder.encode = encoder.encode_v2
    else:
        raise ValueError(encoder_version)

    print("Generating dataset ...")
    gen = DatasetGenerator(net, mapper, encoder, seed=42)
    samples = gen.generate(num_samples, arrival_rate=2.0, avg_holding_time=3.0)
    succ_rate = np.mean([s["success"] for s in samples])
    print(f"  Dataset size: {len(samples)}  |  Success rate: {succ_rate:.3f}")

    state_dim = len(samples[0]["z"])
    print(f"  State dim: {state_dim}")

    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    predictor = SklearnPredictor(state_dim, hidden_dim=64)
    history = predictor.train(train_samples, val_samples, epochs=epochs)

    print(f"\n  Final — acc={history['val_acc'][-1]:.4f} auc={history['val_auc'][-1]:.4f} "
          f"mae={history['val_mae'][-1]:.6f} ece={history['val_ece'][-1]:.4f}")

    # Latency
    t0 = time.time()
    for _ in range(500):
        _ = encoder.encode(0, 5)
    enc_us = (time.time() - t0) / 500 * 1e6

    _ = predictor.predict([samples[0]])
    t0 = time.time()
    for _ in range(500):
        _ = predictor.predict([samples[0]])
    pred_us = (time.time() - t0) / 500 * 1e6

    mapper.net.reset()
    for link in mapper.net.link_states:
        mask = np.random.rand(mapper.net.num_slots) < 0.3
        mapper.net.link_states[link][mask] = True
    t0 = time.time()
    for _ in range(500):
        _ = mapper.map(0, 5, 4)
    map_us = (time.time() - t0) / 500 * 1e6
    mapper.net.reset()

    print(f"  Latency — encoder={enc_us:.1f}us  predictor={pred_us:.1f}us  mapper={map_us:.1f}us")

    return {
        "metrics": {
            "accuracy": history["val_acc"][-1],
            "auc": history["val_auc"][-1],
            "delay_mae": history["val_mae"][-1],
            "ece": history["val_ece"][-1],
        },
        "latency": {"encoder_us": enc_us, "predictor_us": pred_us, "mapper_us": map_us},
    }


if __name__ == "__main__":
    results = {}
    for version in ["v0", "v1", "v2"]:
        results[version] = run_experiment(version, num_samples=8000, epochs=20)

    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print(f"{'Version':<8} {'Acc':>6} {'AUC':>6} {'DelayMAE':>10} {'ECE':>6} {'z_dim':>5}")
    for v in ["v0", "v1", "v2"]:
        m = results[v]["metrics"]
        dim = {0: 1, 1: 2, 2: 7}[int(v[1])]
        print(f"{v:<8} {m['accuracy']:6.4f} {m['auc']:6.4f} {m['delay_mae']:10.6f} {m['ece']:6.4f} {dim:>5}")
