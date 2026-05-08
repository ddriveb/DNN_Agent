"""
Full DNN offloading pipeline demo.
Shows: Agent → AgentInterface → Selector → Predictor → Decision
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from agent_interface import AgentInterface
from train import OpticalDataset, train_predictor
from torch.utils.data import DataLoader

NUM_BINS = 8
STATE_DIM = 2 + NUM_BINS * 2
MAX_SERVERS = 128


def log(msg):
    print(msg, flush=True)


def quick_train(topology="nsfnet", num_samples=5000, epochs=10, seed=42):
    if topology == "nsfnet":
        n, slots, arr, ht, preload = 14, 32, 5.0, 10.0, 500
    elif topology == "random100":
        n, slots, arr, ht, preload = 100, 320, 30.0, 25.0, 3000
    else:
        raise ValueError(topology)

    net = OpticalNetwork(topology=topology, num_slots=slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = lambda src, dst: encoder.encode_v2b(src, dst, num_bins=NUM_BINS)
    gen = DatasetGenerator(net, mapper, encoder, seed=seed)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)

    split = int(0.8 * len(samples))
    train_ld = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_ld = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(STATE_DIM, num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64)
    history = train_predictor(predictor, train_ld, val_ld, epochs=epochs, lr=5e-4)
    return predictor, encoder, net, mapper


if __name__ == "__main__":
    log("=" * 70)
    log("FULL DNN OFFLOADING PIPELINE DEMO")
    log("=" * 70)

    # Step 1: Train a model with decoupled bw/path_id
    log("\n[1/4] Training predictor (NSFNET, 5K, 10ep, bw/path decoupled)...")
    predictor, encoder, net, mapper = quick_train("nsfnet", 5000, 10, seed=42)
    log("  Training complete.\n")

    # Step 2: Create AgentInterface
    log("[2/4] Creating AgentInterface...")
    iface = AgentInterface(predictor, encoder, selector_strategy="max_prob")
    log("  AgentInterface ready.\n")

    # Step 3: Agent proposes candidate actions
    log("[3/4] Agent proposes DNN offload candidates:")
    candidates = [
        {"split_point": 1, "target_server": 5,  "desc": "Light offload"},
        {"split_point": 2, "target_server": 8,  "desc": "Medium offload"},
        {"split_point": 4, "target_server": 10, "desc": "Heavy offload"},
        {"split_point": 5, "target_server": 12, "desc": "Max offload"},
    ]

    for c in candidates:
        log(f"  → {c['desc']}: split_point={c['split_point']}, target={c['target_server']}")

    # Step 4: Evaluate each candidate + select best
    log("\n[4/4] Evaluating candidates with different Selector strategies...")

    strategies = ["max_prob", "min_delay", "weighted"]

    for strategy in strategies:
        log(f"\n--- Strategy: {strategy} ---")
        iface.selector.strategy = strategy
        results = iface.batch_evaluate(candidates)

        for c, r in zip(candidates, results):
            log(f"  {c['desc']:<15} "
                f"bw={r['bw_slots']:2d}slots "
                f"path={r['best_path_id']} "
                f"P={r['success_prob']:.4f} "
                f"delay={r['delay']:.4f}")

        best = max(results, key=lambda x: x["success_prob"])
        log(f"  >> BEST ACTION: split_point={best['split_point']}, "
            f"target={best['target_server']}, path={best['best_path_id']}, "
            f"P={best['success_prob']:.4f}")

    # Step 5: Show full path breakdown for best candidate under max_prob
    log("\n" + "=" * 70)
    log("DETAILED PATH BREAKDOWN (max_prob strategy)")
    log("=" * 70)
    iface.selector.strategy = "max_prob"
    best_action = iface.select_best_action(candidates)
    log(f"Best action: split_point={best_action['split_point']}, "
        f"target={best_action['target_server']}, bw={best_action['bw_slots']} slots")
    log("Path evaluations:")
    for p in best_action["all_paths"]:
        marker = " ★ BEST" if p["path_id"] == best_action["best_path_id"] else ""
        log(f"  Path {p['path_id']}: P_success={p['success_prob']:.4f}, "
            f"delay={p['delay']:.4f}{marker}")

    # Step 6: Architecture summary
    log("\n" + "=" * 70)
    log("PIPELINE ARCHITECTURE")
    log("=" * 70)
    log("┌────────────────────────────────────────────┐")
    log("│  DNN Agent                                 │")
    log("│  Output: (split_point, target_server)      │")
    log("└────────────────────────────────────────────┘")
    log("              ↓")
    log("┌────────────────────────────────────────────┐")
    log("│  AgentInterface                            │")
    log("│  - data_size_after_split() → bw            │")
    log("│  - get_src_dst() → (src, dst)              │")
    log("└────────────────────────────────────────────┘")
    log("              ↓")
    log("┌────────────────────────────────────────────┐")
    log("│  Selector                                  │")
    log("│  - Evaluates path_id ∈ {0,1,2}             │")
    log("│  - Strategy: max_prob / min_delay / weighted│")
    log("└────────────────────────────────────────────┘")
    log("              ↓")
    log("┌────────────────────────────────────────────┐")
    log("│  Predictor + Encoder                       │")
    log("│  - Input: (path_id, src, dst, bw, z)       │")
    log("│  - Output: (P_success, delay)              │")
    log("└────────────────────────────────────────────┘")
    log("              ↓")
    log("┌────────────────────────────────────────────┐")
    log("│  Decision: (split_point, target, path_id)  │")
    log("└────────────────────────────────────────────┘")
    log("=" * 70)
