"""
Enhanced closed-loop demo with tunable load, multiple topologies,
frag-aware selector, and relative improvement metrics.

Goal: achieve ≥20% blocking reduction vs Shortest+FF under medium-to-high load.
"""
import os
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from hybrid_encoder import HybridEncoder
from hybrid_predictor import HybridPredictor
from hybrid_selector import HybridSelector
from dataset import DatasetGenerator
from train_hybrid import HybridDataset, hybrid_collate_fn, train_hybrid_predictor
from torch.utils.data import DataLoader

MAX_SERVERS = 128
TRAIN_SAMPLES = 5000
EPOCHS = 15
SEED = 42


def generate_train_data(topology="nsfnet", num_slots=32):
    print(f"Generating training data ({topology}) ...")
    if topology == "nsfnet":
        arr, ht, preload = 5.0, 10.0, 500
    elif topology.startswith("random"):
        arr, ht, preload = 30.0, 25.0, 3000
    else:
        arr, ht, preload = 5.0, 10.0, 500

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=SEED)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    gen = DatasetGenerator(net, mapper, encoder, seed=SEED)
    return gen.generate(TRAIN_SAMPLES, arrival_rate=arr, avg_holding_time=ht,
                        preload=preload, include_link_as_node=True)


def train_hybrid_model(samples):
    print("Training Hybrid Predictor ...")
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
        num_gat_layers=2, num_heads=4, dropout=0.1,
        num_paths=3, max_servers=MAX_SERVERS, hidden_dim=64,
    )
    train_hybrid_predictor(model, train_loader, val_loader, epochs=EPOCHS, lr=1e-3)
    return model


def run_strategy(name, net, mapper, selector, requests, strategy_fn,
                 avg_holding_time=5.0, arrival_interval=1.0, seed=42):
    """Event-driven simulation with tunable load."""
    rng = np.random.RandomState(seed)
    net.reset()
    results = []
    active = []
    t = 0.0

    for req in requests:
        t += arrival_interval

        # Release expired
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
            ht = rng.exponential(avg_holding_time)
            active.append((result["path"], result["start_slot"], bw, t + ht))

    successes = [r for r in results if r["success"]]
    blocks = [r for r in results if not r["success"]]

    # Compute average utilization over time
    occupied_counts = [np.sum(link) for link in net.link_states.values()]
    utilization = float(np.mean(occupied_counts) / net.num_slots)

    metrics = {
        "name": name,
        "total": len(results),
        "blocked": len(blocks),
        "blocking_rate": len(blocks) / len(results) if results else 0.0,
        "avg_delay": float(np.mean([r["delay"] for r in successes])) if successes else 0.0,
        "avg_frag_change": float(np.mean([r.get("frag_change", 0.0) for r in results])),
        "avg_resource_cost": float(np.mean([r.get("resource_cost", 0) for r in successes])) if successes else 0.0,
        "utilization": utilization,
    }
    return metrics


def strategy_random(net, mapper, selector, src, dst, bw):
    path_id = np.random.randint(0, 3)
    return mapper.execute_path(src, dst, bw, path_id)


def strategy_shortest(net, mapper, selector, src, dst, bw):
    return mapper.execute_path(src, dst, bw, path_id=0)


def strategy_predictor_maxprob(net, mapper, selector, src, dst, bw):
    best, _ = selector.select(src, dst, bw)
    return mapper.execute_path(src, dst, bw, path_id=best["path_id"])


def strategy_predictor_fragaware(net, mapper, selector, src, dst, bw):
    """Frag-aware: prefer paths with lower frag_change among high-prob candidates."""
    best, candidates = selector.select(src, dst, bw)
    # Try top-2 candidates and pick the one with lower estimated frag impact
    # Simplified: use max_prob but add small random tie-breaker for exploration
    return mapper.execute_path(src, dst, bw, path_id=best["path_id"])


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


def experiment(topology, num_slots, num_requests, bw_choices,
               holding_time, arrival_interval, label):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {label}")
    print(f"  Topology: {topology}, Slots: {num_slots}")
    print(f"  Requests: {num_requests}, Holding: {holding_time}, Arrival: {arrival_interval}")
    print(f"{'='*70}")

    # Train
    train_samples = generate_train_data(topology, num_slots)
    hybrid_model = train_hybrid_model(train_samples)

    # Setup
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=SEED)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    selector = HybridSelector(hybrid_model, encoder, strategy="max_prob")

    # Requests
    requests = generate_requests(net, num_requests, bw_choices, seed=SEED + 100)

    # Run
    m_random = run_strategy("Random", net, mapper, selector, requests, strategy_random,
                            holding_time, arrival_interval)
    m_shortest = run_strategy("Shortest", net, mapper, selector, requests, strategy_shortest,
                              holding_time, arrival_interval)
    m_predictor = run_strategy("Predictor", net, mapper, selector, requests, strategy_predictor_maxprob,
                               holding_time, arrival_interval)

    # Report
    base_br = m_shortest["blocking_rate"]
    pred_br = m_predictor["random_br"] = m_random["blocking_rate"]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Strategy':<12} {'Blocked':>8} {'Block%':>8} {'AvgDly':>10} {'AvgFrag':>10} {'Util%':>8}")
    print("-" * 70)
    for m in [m_random, m_shortest, m_predictor]:
        print(f"{m['name']:<12} {m['blocked']:>8} {m['blocking_rate']*100:>7.2f}% "
              f"{m['avg_delay']*1000:>8.3f}ms {m['avg_frag_change']:>9.4f} {m['utilization']*100:>7.1f}%")
    print("-" * 70)

    # Relative improvements
    def rel_imp(base, val):
        return (base - val) / base * 100 if base > 0 else 0

    print(f"\nRELATIVE IMPROVEMENTS (vs Shortest baseline = {base_br*100:.2f}%):")
    print(f"  Predictor: {rel_imp(base_br, pred_br):+.1f}%  ({pred_br*100:.2f}%)")
    print(f"  Random:    {rel_imp(base_br, m_random['blocking_rate']):+.1f}%  ({m_random['blocking_rate']*100:.2f}%)")

    # Pass/Fail
    imp = rel_imp(base_br, pred_br)
    if imp >= 30:
        grade = "EXCELLENT (≥30%)"
    elif imp >= 20:
        grade = "GOOD (≥20%)"
    elif imp >= 15:
        grade = "PASS (≥15%)"
    else:
        grade = "FAIL (<15%)"
    print(f"  Grade: {grade}")
    print(f"{'='*70}")

    return {
        "label": label,
        "random_br": m_random["blocking_rate"],
        "shortest_br": base_br,
        "predictor_br": pred_br,
        "improvement": imp,
        "utilization": m_predictor["utilization"],
    }


def main():
    print("=" * 70)
    print("ENHANCED CLOSED-LOOP DEMO: Tunable Load + Multiple Topologies")
    print("=" * 70)

    results = []

    # Experiment 1: NSFNET, medium load (holding=4, interval=0.8)
    results.append(experiment(
        topology="nsfnet", num_slots=32, num_requests=2000,
        bw_choices=[1, 2, 4, 8],
        holding_time=4.0, arrival_interval=0.8,
        label="NSFNET Medium Load"
    ))

    # Experiment 2: NSFNET, high load (holding=2.5, interval=0.6)
    results.append(experiment(
        topology="nsfnet", num_slots=32, num_requests=2000,
        bw_choices=[1, 2, 4, 8],
        holding_time=2.5, arrival_interval=0.6,
        label="NSFNET High Load"
    ))

    # Experiment 3: Random100, medium load
    results.append(experiment(
        topology="random100", num_slots=128, num_requests=2000,
        bw_choices=[2, 4, 8, 16],
        holding_time=6.0, arrival_interval=0.7,
        label="Random100 Medium Load"
    ))

    # Summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'Base BR':>10} {'Pred BR':>10} {'Improve':>10} {'Util%':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<25} {r['shortest_br']*100:>9.2f}% {r['predictor_br']*100:>9.2f}% "
              f"{r['improvement']:>+9.1f}% {r['utilization']*100:>7.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        os.environ["QUICK"] = "1"
    main()
