"""Joint predictor training on paper topologies with held-out evaluation.

This script trains the predictor on a mixture of paper-style topologies
(e.g. Net-1 + Net-2) and evaluates zero-shot transfer on a held-out topology
(e.g. Net-3). It also compares against the existing NSFNET-only predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_mvp"))

from eval_fragmentation_mainline import load_predictor
from paper_env import create_paper_env
from predictor import Predictor


class OpticalDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "partition": torch.tensor(s["partition"], dtype=torch.long),
            "src": torch.tensor(s["src"], dtype=torch.long),
            "dst": torch.tensor(s["dst"], dtype=torch.long),
            "z": torch.from_numpy(s["z"].copy()).float(),
            "success": torch.tensor(s["success"], dtype=torch.float32),
            "delay": torch.tensor(s["delay"], dtype=torch.float32),
        }


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    pos = y_prob[y_true > 0.5]
    neg = y_prob[y_true <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return float(wins / (len(pos) * len(neg)))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= low) & (y_prob < high)
        else:
            mask = (y_prob >= low) & (y_prob <= high)
        if not np.any(mask):
            continue
        ece += float(np.sum(mask)) * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return float(ece / max(len(y_true), 1))


def evaluate_predictor(predictor, loader):
    predictor.eval()
    all_success, all_prob, all_delay, all_delay_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            logit, delay_pred = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            prob = torch.sigmoid(logit).numpy()
            success = batch["success"].numpy()
            delay = batch["delay"].numpy()
            all_success.extend(success.tolist())
            all_prob.extend(prob.tolist())
            mask = success > 0.5
            if mask.sum() > 0:
                all_delay.extend(delay[mask].tolist())
                all_delay_pred.extend(delay_pred.numpy()[mask].tolist())
    y_true = np.array(all_success, dtype=np.float32)
    y_prob = np.array(all_prob, dtype=np.float32)
    acc = float(np.mean((y_prob > 0.5) == (y_true > 0.5)))
    auc = compute_auc(y_true, y_prob)
    mae = float(np.mean(np.abs(np.array(all_delay) - np.array(all_delay_pred)))) if all_delay else 0.0
    ece = compute_ece(y_true, y_prob, n_bins=10)
    return {"accuracy": acc, "auc": auc, "delay_mae": mae, "ece": ece}


def train_predictor(predictor, train_loader, val_loader, *, epochs: int = 30, lr: float = 1e-3):
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    history = {"train_loss": [], "val_acc": [], "val_auc": [], "val_mae": [], "val_ece": []}
    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            success_logit, delay_pred = predictor(batch["partition"], batch["src"], batch["dst"], batch["z"])
            loss = bce(success_logit, batch["success"])
            mask = batch["success"] > 0.5
            if int(mask.sum()) > 0:
                loss = loss + mse(delay_pred[mask], batch["delay"][mask])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        metrics = evaluate_predictor(predictor, val_loader)
        history["train_loss"].append(total_loss / max(len(train_loader), 1))
        history["val_acc"].append(metrics["accuracy"])
        history["val_auc"].append(metrics["auc"])
        history["val_mae"].append(metrics["delay_mae"])
        history["val_ece"].append(metrics["ece"])
        print(
            f"  Epoch {epoch + 1:02d}: loss={history['train_loss'][-1]:.4f} "
            f"acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f} "
            f"mae={metrics['delay_mae']:.6f} ece={metrics['ece']:.4f}"
        )
    return history


BANDWIDTH_MAP = {0: 1, 1: 4, 2: 8}


def generate_predictor_samples_for_topology(
    topology_file: Path,
    *,
    num_samples: int,
    seed: int,
    arrival_rate: float = 5.0,
    avg_holding_time: float = 10.0,
    preload: int = 0,
):
    """Generate predictor samples under the paper-style event-driven dynamics."""
    rng = np.random.RandomState(seed)
    env, encoder, _mec = create_paper_env(
        topology_file,
        load_factor=0.6,
        fragmentation=0.2,
        seed=seed,
    )
    net = env.net
    mapper = env.mapper
    server_nodes = list(env.mec.server_node_ids)

    net.reset()
    active = []
    t = 0.0

    def random_action():
        src = int(rng.choice(server_nodes))
        dst = int(rng.choice(server_nodes))
        while dst == src:
            dst = int(rng.choice(server_nodes))
        partition = int(rng.randint(0, 3))
        return src, dst, partition, BANDWIDTH_MAP[partition]

    for _ in range(preload):
        src, dst, partition, bw = random_action()
        success, path, start_slot, delay = mapper.map(src, dst, bw)
        if success:
            ht = float(rng.exponential(avg_holding_time * 2.0))
            active.append((path, start_slot, bw, t + ht))

    samples = []
    while len(samples) < num_samples:
        t += float(rng.exponential(1.0 / max(arrival_rate, 1e-6)))
        new_active = []
        for path, start, bw, release_t in active:
            if release_t <= t:
                net.release(path, start, bw)
            else:
                new_active.append((path, start, bw, release_t))
        active = new_active

        src, dst, partition, bw = random_action()
        z = encoder.encode(src, dst)
        success, path, start_slot, delay = mapper.map(src, dst, bw)
        samples.append(
            {
                "src": src,
                "dst": dst,
                "partition": partition,
                "z": z.astype(np.float32),
                "success": 1.0 if success else 0.0,
                "delay": float(delay) if success else 0.0,
                "topology": topology_file.stem,
            }
        )
        if success:
            ht = float(rng.exponential(avg_holding_time))
            active.append((path, start_slot, bw, t + ht))

    for path, start, bw, _release_t in active:
        net.release(path, start, bw)
    return samples


def make_loader(samples, batch_size: int, shuffle: bool):
    return DataLoader(OpticalDataset(samples), batch_size=batch_size, shuffle=shuffle)


def train_joint_predictor(
    train_topologies: list[Path],
    *,
    samples_per_topology: int,
    epochs: int,
    batch_size: int,
    max_servers: int,
    seed: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    topo_samples = {}
    all_samples = []
    for topo_idx, topo in enumerate(train_topologies):
        samples = generate_predictor_samples_for_topology(
            topo,
            num_samples=samples_per_topology,
            seed=seed + topo_idx,
            preload=max(200, samples_per_topology // 10),
        )
        topo_samples[topo.stem] = samples
        all_samples.extend(samples)

    rng = np.random.RandomState(seed)
    rng.shuffle(all_samples)
    state_dim = len(all_samples[0]["z"])

    split = int(0.8 * len(all_samples))
    train_loader = make_loader(all_samples[:split], batch_size=batch_size, shuffle=True)
    val_loader = make_loader(all_samples[split:], batch_size=batch_size, shuffle=False)

    predictor = Predictor(state_dim, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    history = train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=1e-3)
    return predictor, state_dim, history, topo_samples


def evaluate_predictor_on_topology(
    predictor,
    topology_file: Path,
    *,
    num_samples: int,
    seeds: list[int],
):
    per_seed = []
    for seed in seeds:
        samples = generate_predictor_samples_for_topology(
            topology_file,
            num_samples=num_samples,
            seed=seed,
            preload=max(200, num_samples // 10),
        )
        loader = make_loader(samples, batch_size=256, shuffle=False)
        metrics = evaluate_predictor(predictor, loader)
        per_seed.append(metrics)

    out = {}
    for key in ["accuracy", "auc", "delay_mae", "ece"]:
        vals = [float(m[key]) for m in per_seed]
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_topologies", default="net1,net2")
    parser.add_argument("--holdout_topology", default="net3")
    parser.add_argument("--samples_per_topology", type=int, default=4000)
    parser.add_argument("--test_samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_servers", type=int, default=128)
    parser.add_argument("--compare_nsfnet_baseline", action="store_true")
    parser.add_argument("--baseline_ckpt", default=None)
    parser.add_argument("--output_ckpt", default="pretrained_paper_joint_v2b.pt")
    parser.add_argument("--output_json", default="paper_joint_predictor_results.json")
    args = parser.parse_args()

    device = "cpu"
    root = Path(__file__).parent
    topo_dir = root / "topologies"
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    train_topologies = [topo_dir / f"{name.strip()}.txt" for name in args.train_topologies.split(",") if name.strip()]
    holdout_topology = topo_dir / f"{args.holdout_topology}.txt"
    eval_seeds = [123, 456, 789]

    print(f"Device: {device}")
    print(f"Joint training on: {[p.stem for p in train_topologies]}")
    print(f"Held-out topology: {holdout_topology.stem}")

    predictor, state_dim, history, topo_samples = train_joint_predictor(
        train_topologies,
        samples_per_topology=args.samples_per_topology,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_servers=args.max_servers,
        seed=args.seed,
    )
    predictor.eval()

    joint_train_metrics = {
        name: {
            "num_samples": len(samples),
            "success_rate": float(np.mean([s["success"] for s in samples])),
        }
        for name, samples in topo_samples.items()
    }

    holdout_metrics = evaluate_predictor_on_topology(
        predictor,
        holdout_topology,
        num_samples=args.test_samples,
        seeds=eval_seeds,
    )

    out = {
        "config": vars(args),
        "state_dim": state_dim,
        "train_topologies": [p.stem for p in train_topologies],
        "holdout_topology": holdout_topology.stem,
        "joint_train_metrics": joint_train_metrics,
        "holdout_zero_shot": holdout_metrics,
    }

    if args.compare_nsfnet_baseline:
        baseline_path = (
            Path(args.baseline_ckpt)
            if args.baseline_ckpt
            else (root.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
        )
        baseline = load_predictor(str(baseline_path), max_servers=args.max_servers, device="cpu")
        baseline_metrics = evaluate_predictor_on_topology(
            baseline,
            holdout_topology,
            num_samples=args.test_samples,
            seeds=eval_seeds,
        )
        out["nsfnet_baseline_zero_shot"] = baseline_metrics
        out["improvement_over_nsfnet_baseline"] = {
            key: (
                float(holdout_metrics[key]["mean"] - baseline_metrics[key]["mean"])
                if key != "delay_mae" and key != "ece"
                else float(baseline_metrics[key]["mean"] - holdout_metrics[key]["mean"])
            )
            for key in ["accuracy", "auc", "delay_mae", "ece"]
        }

    ckpt_path = result_dir / args.output_ckpt
    torch.save(predictor.state_dict(), ckpt_path)
    json_path = result_dir / args.output_json
    json_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    print(f"Saved predictor to {ckpt_path}")
    print(f"Saved report to {json_path}")


if __name__ == "__main__":
    main()
