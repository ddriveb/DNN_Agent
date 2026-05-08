"""
Multi-seed validation of the best config (v4 quick mode, no class_weight).
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
from link_as_node_gnn_predictor import build_converted_topology, compute_link_node_features

MAX_SERVERS = 128


def calibrate_load(topology, num_slots):
    if topology == "nsfnet":
        return 0.25, 8.0
    elif topology.startswith("random"):
        return 0.30, 20.0
    return 0.5, 5.0


def generate_train_data_fixed(topology, num_slots, n_samples, seed=42):
    arr, ht = calibrate_load(topology, num_slots)
    preload = 500 if topology == "nsfnet" else 5000

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    rng = np.random.RandomState(seed)

    active = []
    t = 0.0
    for _ in range(preload):
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, net.NUM_NODES)
        bw = min([1, 2, 4, 8, 16, 32][rng.randint(0, 6)], num_slots)
        success, path, start_slot, delay = mapper.map(src, dst, bw)
        if success:
            active.append((path, start_slot, bw, t + rng.exponential(ht * 2)))

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

    samples = []
    bw_choices = [b for b in [1, 2, 4, 8, 16, 32] if b <= num_slots]
    while len(samples) < n_samples:
        t += rng.exponential(1.0 / arr)
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
        result = mapper.execute_path(src, dst, bw, path_id)
        success = result["success"]
        delay = result["delay"] if success else 0.0

        samples.append({
            "src": src, "dst": dst, "path_id": path_id, "bw": float(bw),
            "z": z, "success": 1.0 if success else 0.0, "delay": delay,
            "link_node_features": link_feats,
            "converted_edge_index": converted_edge_index,
            "path_mask": path_masks.get((src, dst, path_id),
                                         np.zeros(len(link_index_map), dtype=np.float32)),
        })
        if success:
            active.append((result["path"], result["start_slot"], bw, t + rng.exponential(ht)))

    for path, start, bw, _ in active:
        net.release(path, start, bw)
    return samples


def train_model(samples, epochs=5, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    split = int(0.8 * len(samples))
    train_loader = DataLoader(HybridDataset(samples[:split]), batch_size=128,
                              shuffle=True, collate_fn=hybrid_collate_fn)
    val_loader = DataLoader(HybridDataset(samples[split:]), batch_size=128,
                            shuffle=False, collate_fn=hybrid_collate_fn)

    state_dim = len(samples[0]["z"])
    num_links = train_loader.dataset.num_links
    model = HybridPredictor(
        state_dim=state_dim, num_links=num_links, link_feat_dim=6, link_hidden=32,
        num_gat_layers=2, num_heads=4, dropout=0.1,
        num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64,
    )
    train_hybrid_predictor(model, train_loader, val_loader, epochs=epochs, lr=1e-3,
                           class_weight=False, early_stop_patience=999)
    return model


def run_strategy(name, net, mapper, selector, requests, strategy_fn, ht, arr, seed=42):
    rng = np.random.RandomState(seed)
    net.reset()
    results = []
    active = []
    t = 0.0
    for req in requests:
        t += arr
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
            ht_val = rng.exponential(ht)
            active.append((result["path"], result["start_slot"], bw, t + ht_val))
    blocks = [r for r in results if not r["success"]]
    return len(blocks) / len(results) if results else 0.0


def strategy_shortest(net, mapper, selector, src, dst, bw):
    return mapper.execute_path(src, dst, bw, path_id=0)


def strategy_fragaware(net, mapper, selector, src, dst, bw):
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


def experiment(topology, num_slots, num_requests, bw_choices, seed):
    arr, ht = calibrate_load(topology, num_slots)
    train_samples = generate_train_data_fixed(topology, num_slots, 1000, seed=seed)
    model = train_model(train_samples, epochs=5, seed=seed)

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    selector = HybridSelector(model, encoder, strategy="max_prob")

    rng = np.random.RandomState(seed + 100)
    requests = []
    for _ in range(num_requests):
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, net.NUM_NODES)
        bw = bw_choices[rng.randint(0, len(bw_choices))]
        requests.append({"src": src, "dst": dst, "bw": bw})

    base_br = run_strategy("Shortest", net, mapper, selector, requests, strategy_shortest,
                           ht, arr, seed=seed)
    frag_br = run_strategy("Frag", net, mapper, selector, requests, strategy_fragaware,
                           ht, arr, seed=seed)
    imp = (base_br - frag_br) / base_br * 100 if base_br > 1e-9 else 0.0
    return base_br, frag_br, imp


print("=" * 70)
print("MULTI-SEED VALIDATION (3 seeds, 1000 samples, 5 epochs)")
print("=" * 70)

for topology, slots, reqs, bws, label in [
    ("nsfnet", 32, 2000, [1, 2, 4, 8], "NSFNET"),
    ("random100", 128, 2000, [2, 4, 8, 16], "Random100"),
]:
    print(f"\n{label}:")
    print(f"  {'Seed':>6} {'Base BR%':>10} {'Frag BR%':>10} {'Improve%':>10}")
    improvements = []
    for seed in [42, 123, 456]:
        base_br, frag_br, imp = experiment(topology, slots, reqs, bws, seed)
        improvements.append(imp)
        print(f"  {seed:>6} {base_br*100:>9.2f}% {frag_br*100:>9.2f}% {imp:>+9.1f}%")
    print(f"  {'Mean':>6} {'':>10} {'':>10} {np.mean(improvements):>+9.1f}%")
    print(f"  {'Std':>6} {'':>10} {'':>10} {np.std(improvements):>9.1f}%")
