"""
8-bin vs 4-bin v2b feature comparison.
Test whether increasing histogram bins from 4 to 8 improves intra-topology performance.
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
from torch.utils.data import DataLoader

MAX_SERVERS = 128


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def generate_with_bins(topology, version, num_samples, num_bins, seed=42):
    if topology == "random100":
        n_nodes, num_slots, arr, ht, preload = 100, 320, 30.0, 25.0, 3000
    else:
        raise ValueError(f"Unknown topology: {topology}")

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)

    if version == "v2b":
        def encode_wrapper(src, dst):
            return encoder.encode_v2b(src, dst, num_bins=num_bins)
        encoder.encode = encode_wrapper

    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    return gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)


def train_from_pretrained(predictor, pretrained_path, train_loader, val_loader, epochs=5, lr=5e-4):
    # Load pretrained weights where dimensions match
    pretrained = torch.load(pretrained_path)
    model_dict = predictor.state_dict()
    # Filter out layers with mismatched shapes (e.g., first MLP layer if state_dim changed)
    matched = {k: v for k, v in pretrained.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(matched)
    predictor.load_state_dict(model_dict, strict=False)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
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


def evaluate(predictor, test_loader):
    return evaluate_predictor(predictor, test_loader)


if __name__ == "__main__":
    log("="*70)
    log("4-BIN vs 8-BIN v2b COMPARISON (Random100 intra)")
    log("="*70)

    sample_size = 8000
    test_size = 2000

    # Generate data
    log(f"\nGenerating {sample_size} training + {test_size} test samples for each bin config...")
    t0 = time.time()
    train_4b = generate_with_bins("random100", "v2b", sample_size, num_bins=4, seed=100)
    train_8b = generate_with_bins("random100", "v2b", sample_size, num_bins=8, seed=100)
    test_4b = generate_with_bins("random100", "v2b", test_size, num_bins=4, seed=200)
    test_8b = generate_with_bins("random100", "v2b", test_size, num_bins=8, seed=200)
    log(f"Data ready in {time.time()-t0:.1f}s")
    log(f"  4-bin train SR={np.mean([s['success'] for s in train_4b]):.3f}")
    log(f"  8-bin train SR={np.mean([s['success'] for s in train_8b]):.3f}")

    configs = [
        (4, train_4b, test_4b, "v2b_4bin"),
        (8, train_8b, test_8b, "v2b_8bin"),
    ]

    results = {}
    for num_bins, train_samples, test_samples, name in configs:
        log(f"\n{'='*60}")
        log(f"Config: {name} | state_dim={2 + num_bins*2}")
        log(f"{'='*60}")

        np.random.seed(100)
        torch.manual_seed(100)

        state_dim = 2 + num_bins * 2
        predictor = Predictor(state_dim, num_partitions=3, max_servers=MAX_SERVERS, hidden_dim=64)

        split = int(0.8 * len(train_samples))
        ft_train = DataLoader(OpticalDataset(train_samples[:split]), batch_size=256, shuffle=True)
        ft_val = DataLoader(OpticalDataset(train_samples[split:]), batch_size=256, shuffle=False)

        # Try loading pretrained embeddings (always compatible)
        train_from_pretrained(predictor, "pretrained_nsfnet_v2b.pt", ft_train, ft_val, epochs=5, lr=5e-4)

        test_loader = DataLoader(OpticalDataset(test_samples), batch_size=256, shuffle=False)
        r = evaluate(predictor, test_loader)
        log(f"  Test — auc={r['auc']:.4f} ece={r['ece']:.4f} mae={r['delay_mae']:.4f}")
        results[name] = r

    # Summary
    log("\n" + "="*70)
    log("4-BIN vs 8-BIN SUMMARY (Random100 intra, 8K train, 2K test)")
    log("="*70)
    log(f"{'Config':<15} {'State Dim':>10} {'AUC':>10} {'ECE':>10} {'MAE':>10}")
    log("-"*70)
    for num_bins, _, _, name in configs:
        r = results[name]
        state_dim = 2 + num_bins * 2
        log(f"{name:<15} {state_dim:>10} {r['auc']:10.4f} {r['ece']:10.4f} {r['delay_mae']:10.4f}")
    log("-"*70)
    log("Reference: Previous Random100 intra (4-bin, 15K samples, 30ep) AUC≈0.938")
