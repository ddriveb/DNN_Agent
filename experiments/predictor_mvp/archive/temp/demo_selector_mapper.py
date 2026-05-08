"""
Closed-loop demo: Selector + Mapper with Hybrid Predictor.
Compare three strategies on 1000 random requests:
  1. Random: randomly pick path_id
  2. Shortest: always path_id=0
  3. Predictor: HybridSelector picks best path

Metrics: blocking rate, avg delay, avg frag_change, resource utilization.
"""
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
NUM_REQUESTS = 1000
TRAIN_SAMPLES = 3000
EPOCHS = 15
SEED = 42


def generate_train_data():
    """Generate training data for Hybrid Predictor."""
    print("Generating training data ...")
    net = OpticalNetwork(topology="nsfnet", num_slots=32, seed=SEED)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    gen = DatasetGenerator(net, mapper, encoder, seed=SEED)
    return gen.generate(TRAIN_SAMPLES, arrival_rate=5.0, avg_holding_time=10.0,
                        preload=500, include_link_as_node=True)


def train_hybrid_model(samples):
    """Train Hybrid Predictor."""
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


def run_strategy(name, net, mapper, selector, requests, strategy_fn, avg_holding_time=10.0, seed=42):
    """
    Run a strategy on a sequence of requests with dynamic release.
    Successful requests hold resources for exponential holding time, then release.
    """
    rng = np.random.RandomState(seed)
    net.reset()
    results = []
    active = []  # (path, start_slot, bw, release_time)
    t = 0.0

    for i, req in enumerate(requests):
        t += 1.0  # discrete time step; could be Poisson inter-arrival

        # Release expired connections
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

    # Compute average network utilization over the run
    total_link_slots = net.num_slots * len(net.link_states)
    occupied_counts = []
    for link in net.link_states.values():
        occupied_counts.append(np.sum(link))
    utilization = float(np.mean(occupied_counts) / net.num_slots) if total_link_slots > 0 else 0.0

    metrics = {
        "name": name,
        "total": len(results),
        "blocked": len(blocks),
        "blocking_rate": len(blocks) / len(results),
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


def strategy_predictor(net, mapper, selector, src, dst, bw):
    best, _ = selector.select(src, dst, bw)
    return mapper.execute_path(src, dst, bw, path_id=best["path_id"])


def generate_requests(net, num_requests, seed=42):
    """Generate a random request sequence."""
    rng = np.random.RandomState(seed)
    requests = []
    bw_choices = [1, 2, 4, 8, 16]
    for _ in range(num_requests):
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, net.NUM_NODES)
        bw = bw_choices[rng.randint(0, len(bw_choices))]
        requests.append({"src": src, "dst": dst, "bw": bw})
    return requests


def main():
    print("=" * 70)
    print("CLOSED-LOOP DEMO: Selector + Mapper + Hybrid Predictor")
    print("=" * 70)

    # 1. Train Hybrid Predictor
    train_samples = generate_train_data()
    hybrid_model = train_hybrid_model(train_samples)

    # 2. Setup environment and components
    net = OpticalNetwork(topology="nsfnet", num_slots=32, seed=SEED)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    selector = HybridSelector(hybrid_model, encoder, strategy="max_prob")

    # 3. Generate request sequence
    requests = generate_requests(net, NUM_REQUESTS, seed=SEED + 100)

    # 4. Run three strategies
    print("\n" + "=" * 70)
    print("RUNNING STRATEGIES ...")
    print("=" * 70)

    metrics_random = run_strategy("Random", net, mapper, selector, requests, strategy_random)
    metrics_shortest = run_strategy("Shortest", net, mapper, selector, requests, strategy_shortest)
    metrics_predictor = run_strategy("Predictor", net, mapper, selector, requests, strategy_predictor)

    # 5. Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY ({} requests)".format(NUM_REQUESTS))
    print("=" * 70)
    print(f"{'Strategy':<12} {'Blocked':>8} {'Block%':>8} {'AvgDly':>10} {'AvgFrag':>10} {'AvgRes':>10}")
    print("-" * 70)
    for m in [metrics_random, metrics_shortest, metrics_predictor]:
        print(f"{m['name']:<12} {m['blocked']:>8} {m['blocking_rate']*100:>7.2f}% "
              f"{m['avg_delay']*1000:>8.3f}ms {m['avg_frag_change']:>9.4f} {m['avg_resource_cost']:>9.1f}")
    print("=" * 70)

    # 6. Key comparison
    print("\nKEY FINDINGS:")
    base_br = metrics_shortest["blocking_rate"]
    pred_br = metrics_predictor["blocking_rate"]
    improvement = (base_br - pred_br) / base_br * 100 if base_br > 0 else 0
    print(f"  Predictor vs Shortest: blocking rate {base_br*100:.2f}% → {pred_br*100:.2f}% "
          f"(improvement {improvement:+.1f}%)")

    base_br = metrics_random["blocking_rate"]
    improvement = (base_br - pred_br) / base_br * 100 if base_br > 0 else 0
    print(f"  Predictor vs Random:   blocking rate {base_br*100:.2f}% → {pred_br*100:.2f}% "
          f"(improvement {improvement:+.1f}%)")


if __name__ == "__main__":
    main()
