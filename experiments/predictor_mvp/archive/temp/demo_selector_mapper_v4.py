"""
Selector+Mapper closed-loop demo v4.
FIXED: Training data now correctly tests each path_id individually.
"""
import os
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from hybrid_encoder import HybridEncoder
from hybrid_predictor import HybridPredictor
from hybrid_selector import HybridSelector
from train_hybrid import HybridDataset, hybrid_collate_fn, train_hybrid_predictor
from torch.utils.data import DataLoader

MAX_SERVERS = 128
SEED = 42

# --- Tunable parameters ------------------------------------------------------
TRAIN_SAMPLES = int(os.environ.get("TRAIN_SAMPLES", "3000"))
TRAIN_EPOCHS  = int(os.environ.get("TRAIN_EPOCHS", "10"))

# Load calibration: target blocking for Shortest ~10-20%
def calibrate_load(topology, num_slots):
    """Calibrated for ~12-15% Shortest blocking rate."""
    if topology == "nsfnet":
        return 0.25, 8.0   # BR ~12%
    elif topology.startswith("random"):
        return 0.30, 20.0  # BR ~15% (estimated)
    return 0.5, 5.0


# --- CORRECTED training data generation --------------------------------------
def generate_train_data_fixed(topology, num_slots, n_samples, seed=42):
    """
    Generate training data where each sample's 'success' label
    correctly corresponds to the specified path_id.
    """
    print(f"  Generating {n_samples} training samples ({topology}) [FIXED] ...")
    arr, ht = calibrate_load(topology, num_slots)
    # Use SAME load as test to ensure realistic state distribution
    arr_train = arr
    preload = 500 if topology == "nsfnet" else 5000

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    rng = np.random.RandomState(seed)

    # Pre-load network
    active = []
    t = 0.0
    for _ in range(preload):
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, net.NUM_NODES)
        bw = [1, 2, 4, 8, 16, 32][rng.randint(0, 6)]
        bw = min(bw, num_slots)
        success, path, start_slot, delay = mapper.map(src, dst, bw)
        if success:
            active.append((path, start_slot, bw, t + rng.exponential(ht * 2)))

    # Precompute link-as-node structures
    from link_as_node_gnn_predictor import build_converted_topology, compute_link_node_features
    link_index_map, converted_edge_index = build_converted_topology(net)
    path_masks = {}
    for src in range(net.NUM_NODES):
        for dst in range(net.NUM_NODES):
            if src == dst:
                continue
            paths = mapper._path_cache.get((src, dst), [])
            for pid, path in enumerate(paths):
                mask = np.zeros(len(link_index_map), dtype=np.float32)
                for idx in range(len(path) - 1):
                    link = (min(path[idx], path[idx + 1]), max(path[idx], path[idx + 1]))
                    if link in link_index_map:
                        mask[link_index_map[link]] = 1.0
                path_masks[(src, dst, pid)] = mask

    # Generate samples with correct per-path_id labels
    samples = []
    bw_choices = [b for b in [1, 2, 4, 8, 16, 32] if b <= num_slots]
    while len(samples) < n_samples:
        t += rng.exponential(1.0 / arr_train)
        new_active = []
        for path, start, bw, rt in active:
            if rt <= t:
                net.release(path, start, bw)
            else:
                new_active.append((path, start, bw, rt))
        active = new_active

        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, net.NUM_NODES)
        bw = bw_choices[rng.randint(0, len(bw_choices))]
        path_id = rng.randint(0, 3)

        z = encoder.encode_v2b(src, dst)
        link_feats = compute_link_node_features(net, link_index_map)

        # CRITICAL FIX: test ONLY the specified path_id
        result = mapper.execute_path(src, dst, bw, path_id)
        success = result["success"]
        delay = result["delay"] if success else 0.0

        sample = {
            "src": src, "dst": dst, "path_id": path_id, "bw": float(bw),
            "z": z, "success": 1.0 if success else 0.0,
            "delay": delay,
            "link_node_features": link_feats,
            "converted_edge_index": converted_edge_index,
            "path_mask": path_masks.get((src, dst, path_id),
                                         np.zeros(len(link_index_map), dtype=np.float32)),
        }
        samples.append(sample)

        if success:
            active.append((result["path"], result["start_slot"], bw, t + rng.exponential(ht)))

    for path, start, bw, _ in active:
        net.release(path, start, bw)

    # Report class balance
    n_pos = sum(1 for s in samples if s["success"] > 0.5)
    print(f"    Class balance: {n_pos}/{len(samples)} positive ({n_pos/len(samples)*100:.1f}%)")
    return samples


def train_hybrid_model(samples, epochs=10):
    print(f"  Training Hybrid Predictor ({epochs} epochs) ...")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(HybridDataset(samples[:split]), batch_size=128,
                              shuffle=True, collate_fn=hybrid_collate_fn)
    val_loader = DataLoader(HybridDataset(samples[split:]), batch_size=128,
                            shuffle=False, collate_fn=hybrid_collate_fn)

    state_dim = len(samples[0]["z"])
    num_links = train_loader.dataset.num_links
    model = HybridPredictor(
        state_dim=state_dim, num_links=num_links, link_feat_dim=6, link_hidden=32,
        num_gat_layers=2, num_heads=4, dropout=0.3,
        num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64,
    )
    train_hybrid_predictor(model, train_loader, val_loader, epochs=epochs, lr=1e-3,
                           class_weight=True, early_stop_patience=2)
    return model


def run_strategy(name, net, mapper, selector, requests, strategy_fn,
                 holding_time, arrival_interval, seed=42):
    rng = np.random.RandomState(seed)
    net.reset()
    results = []
    active = []
    t = 0.0

    for req in requests:
        t += arrival_interval
        new_active = []
        for path, start_slot, bw_req, rt in active:
            if rt <= t:
                net.release(path, start_slot, bw_req)
            else:
                new_active.append((path, start_slot, bw_req, rt))
        active = new_active

        src, dst, bw = req["src"], req["dst"], req["bw"]
        result = strategy_fn(net, mapper, selector, src, dst, bw)
        results.append(result)
        if result["success"]:
            ht = rng.exponential(holding_time)
            active.append((result["path"], result["start_slot"], bw, t + ht))

    successes = [r for r in results if r["success"]]
    blocks = [r for r in results if not r["success"]]
    occupied_counts = [np.sum(link) for link in net.link_states.values()]
    utilization = float(np.mean(occupied_counts) / net.num_slots)

    return {
        "name": name,
        "total": len(results),
        "blocked": len(blocks),
        "blocking_rate": len(blocks) / len(results) if results else 0.0,
        "avg_delay": float(np.mean([r["delay"] for r in successes])) if successes else 0.0,
        "avg_frag_change": float(np.mean([r.get("frag_change", 0.0) for r in results])),
        "utilization": utilization,
    }


def strategy_random(net, mapper, selector, src, dst, bw):
    path_id = np.random.randint(0, 3)
    return mapper.execute_path(src, dst, bw, path_id)


def strategy_shortest(net, mapper, selector, src, dst, bw):
    return mapper.execute_path(src, dst, bw, path_id=0)


def strategy_predictor_maxprob(net, mapper, selector, src, dst, bw):
    best, _ = selector.select(src, dst, bw)
    return mapper.execute_path(src, dst, bw, path_id=best["path_id"])


def strategy_predictor_fragaware(net, mapper, selector, src, dst, bw):
    best, candidates = selector.select(src, dst, bw)
    probs = [c["success_prob"] for c in candidates]
    max_prob = max(probs) if probs else 0
    threshold = max_prob * 0.9
    qualified = [c for c in candidates if c["success_prob"] >= threshold]
    if len(qualified) <= 1:
        return mapper.execute_path(src, dst, bw, path_id=best["path_id"])

    best_frag = None
    best_path_id = best["path_id"]
    for cand in qualified:
        pid = cand["path_id"]
        result = mapper.execute_path(src, dst, bw, path_id=pid)
        if result["success"]:
            frag = result.get("frag_change", 0)
            if best_frag is None or frag < best_frag:
                best_frag = frag
                best_path_id = pid
            net.release(result["path"], result["start_slot"], bw)

    return mapper.execute_path(src, dst, bw, path_id=best_path_id)


def generate_requests(net, num_requests, bw_choices, seed=42):
    rng = np.random.RandomState(seed)
    requests = []
    for _ in range(num_requests):
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, net.NUM_NODES)
        bw = bw_choices[rng.randint(0, len(bw_choices))]
        requests.append({"src": src, "dst": dst, "bw": bw})
    return requests


def experiment(topology, num_slots, num_requests, bw_choices, label):
    arr, ht = calibrate_load(topology, num_slots)
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {label}")
    print(f"  Topology: {topology}, Slots: {num_slots}")
    print(f"  Load: arr={arr:.2f}, ht={ht:.1f}")
    print(f"  Train: {TRAIN_SAMPLES} samples, {TRAIN_EPOCHS} epochs")
    print(f"{'='*70}")

    train_samples = generate_train_data_fixed(topology, num_slots, TRAIN_SAMPLES, seed=SEED)
    hybrid_model = train_hybrid_model(train_samples, epochs=TRAIN_EPOCHS)

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=SEED)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    selector = HybridSelector(hybrid_model, encoder, strategy="max_prob")

    requests = generate_requests(net, num_requests, bw_choices, seed=SEED + 100)

    m_random   = run_strategy("Random",   net, mapper, selector, requests, strategy_random,
                              ht, arr, seed=SEED)
    m_shortest = run_strategy("Shortest", net, mapper, selector, requests, strategy_shortest,
                              ht, arr, seed=SEED)
    m_pred_mp  = run_strategy("Pred-MaxP", net, mapper, selector, requests, strategy_predictor_maxprob,
                              ht, arr, seed=SEED)
    m_pred_fa  = run_strategy("Pred-Frag", net, mapper, selector, requests, strategy_predictor_fragaware,
                              ht, arr, seed=SEED)

    base_br = m_shortest["blocking_rate"]
    pred_br = m_pred_mp["blocking_rate"]
    frag_br = m_pred_fa["blocking_rate"]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Strategy':<12} {'Blocked':>8} {'Block%':>8} {'AvgDly':>10} {'AvgFrag':>10} {'Util%':>8}")
    print("-" * 70)
    for m in [m_random, m_shortest, m_pred_mp, m_pred_fa]:
        print(f"{m['name']:<12} {m['blocked']:>8} {m['blocking_rate']*100:>7.2f}% "
              f"{m['avg_delay']*1000:>8.3f}ms {m['avg_frag_change']:>9.4f} {m['utilization']*100:>7.1f}%")
    print("-" * 70)

    def rel_imp(base, val):
        return (base - val) / base * 100 if base > 1e-9 else 0.0

    imp_mp = rel_imp(base_br, pred_br)
    imp_fa = rel_imp(base_br, frag_br)
    print(f"\nRELATIVE IMPROVEMENTS (vs Shortest={base_br*100:.2f}%):")
    print(f"  Pred-MaxP: {imp_mp:+.1f}%  ({pred_br*100:.2f}%)")
    print(f"  Pred-Frag: {imp_fa:+.1f}%  ({frag_br*100:.2f}%)")
    print(f"  Random:    {rel_imp(base_br, m_random['blocking_rate']):+.1f}%")

    grade = "EXCELLENT" if max(imp_mp, imp_fa) >= 30 else ("GOOD" if max(imp_mp, imp_fa) >= 20 else ("PASS" if max(imp_mp, imp_fa) >= 15 else "FAIL"))
    print(f"  Grade: {grade}")
    print(f"{'='*70}")

    return {
        "label": label,
        "random_br": m_random["blocking_rate"],
        "shortest_br": base_br,
        "pred_mp_br": pred_br,
        "pred_fa_br": frag_br,
        "imp_mp": imp_mp,
        "imp_fa": imp_fa,
        "utilization": m_pred_mp["utilization"],
    }


def main():
    print("=" * 70)
    print("SELECTOR+MAPPER CLOSED-LOOP DEMO v4 (FIXED + CLASS_WEIGHT + DROPOUT=0.3)")
    print(f"  Train samples: {TRAIN_SAMPLES}, Epochs: {TRAIN_EPOCHS}")
    print("=" * 70)

    results = []
    results.append(experiment(
        topology="nsfnet", num_slots=32, num_requests=2000,
        bw_choices=[1, 2, 4, 8],
        label="NSFNET-HighLoad"
    ))
    results.append(experiment(
        topology="random100", num_slots=128, num_requests=2000,
        bw_choices=[2, 4, 8, 16],
        label="Random100-HighLoad"
    ))

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<20} {'Base BR':>10} {'Pred-MP':>10} {'Pred-Frag':>10} {'Improve':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<20} {r['shortest_br']*100:>9.2f}% {r['pred_mp_br']*100:>9.2f}% "
              f"{r['pred_fa_br']*100:>9.2f}% {max(r['imp_mp'], r['imp_fa']):>+9.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
