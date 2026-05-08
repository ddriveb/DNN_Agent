"""
DNN Offload Predictor Demo.
Demonstrates the upgraded predictor serving DNN offloading decisions.

Architecture:
    Agent proposes a_high = (split_point, target_server)
    ↓
    bw = data_size_after_split(split_point)     # bandwidth in slots
    src, dst = get_src_dst(target_server)       # source → target
    ↓
    for each candidate path_id in {0,1,2}:
        z = encoder.encode(src, dst)            # current spectrum state
        P_success, delay = predictor(path_id, src, dst, bw, z)
    ↓
    Selector picks best (split_point, target_server, path_id)
    ↓
    Mapper executes actual path-slot allocation
"""
import numpy as np
import torch
from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dataset import DatasetGenerator
from train import OpticalDataset, train_predictor
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# 1. DNN Offload → Optical Parameters
# ---------------------------------------------------------------------------

def data_size_after_split(split_point):
    """
    Map DNN split point to required bandwidth (slots).
    More layers offloaded → larger intermediate activation → more slots.
    """
    # Example: split after layer N → N feature maps need to traverse the network
    mapping = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32}
    return mapping.get(split_point, 4)


def get_src_dst(target_server, source_server=0):
    """In DNN offloading, src is the current node, dst is the target server."""
    return source_server, target_server


# ---------------------------------------------------------------------------
# 2. Quick training of a bw-aware model (for demo purposes)
# ---------------------------------------------------------------------------

def quick_train_bw_model(topology="nsfnet", num_samples=5000, epochs=10):
    """Train a small bw-aware predictor for demonstration."""
    if topology == "nsfnet":
        n_nodes, num_slots, arr, ht, preload = 14, 32, 5.0, 10.0, 500
    elif topology == "random100":
        n_nodes, num_slots, arr, ht, preload = 100, 320, 30.0, 25.0, 3000
    else:
        raise ValueError(topology)

    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=42)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = lambda src, dst: encoder.encode_v2b(src, dst, num_bins=8)

    gen = DatasetGenerator(net, mapper, encoder, seed=42)
    samples = gen.generate(num_samples, arrival_rate=arr, avg_holding_time=ht, preload=preload)

    split = int(0.8 * len(samples))
    train_loader = DataLoader(OpticalDataset(samples[:split]), batch_size=256, shuffle=True)
    val_loader = DataLoader(OpticalDataset(samples[split:]), batch_size=256, shuffle=False)

    predictor = Predictor(state_dim=18, num_paths=3, max_servers=128, hidden_dim=64)
    history = train_predictor(predictor, train_loader, val_loader, epochs=epochs, lr=5e-4)
    return predictor, encoder, net, mapper


# ---------------------------------------------------------------------------
# 3. Offload Evaluation Interface
# ---------------------------------------------------------------------------

def evaluate_offload_action(predictor, encoder, split_point, target_server, source_server=0):
    """
    Evaluate a single DNN offload candidate action.
    Returns a list of path candidates with predicted success_prob and delay.
    """
    bw_slots = data_size_after_split(split_point)
    src, dst = get_src_dst(target_server, source_server)

    # Encode current network state for (src, dst)
    z = encoder.encode(src, dst)
    z_tensor = torch.from_numpy(z).unsqueeze(0).float()

    results = []
    predictor.eval()
    with torch.no_grad():
        for path_id in range(3):  # 3 candidate KSP paths
            path_t = torch.tensor([path_id], dtype=torch.long)
            src_t = torch.tensor([src], dtype=torch.long)
            dst_t = torch.tensor([dst], dtype=torch.long)
            bw_t = torch.tensor([float(bw_slots)], dtype=torch.float32)

            logit, delay_pred = predictor(path_t, src_t, dst_t, bw_t, z_tensor)
            prob = torch.sigmoid(logit).item()
            delay = delay_pred.item()

            results.append({
                "path_id": path_id,
                "bw_slots": bw_slots,
                "success_prob": prob,
                "delay": delay,
            })

    return results


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("DNN OFFLOAD PREDICTOR DEMO")
    print("=" * 70)

    # Train a quick demo model
    print("\n[1/3] Training a bw-aware predictor (NSFNET, 5K samples, 10 epochs)...")
    predictor, encoder, net, mapper = quick_train_bw_model("nsfnet", 5000, 10)
    print("  Training complete.")

    # Simulate agent proposing several offload candidates
    print("\n[2/3] Agent proposes DNN offload candidates:")
    candidates = [
        {"split_point": 1, "target": 5,  "desc": "Light offload (2 slots)"},
        {"split_point": 3, "target": 10, "desc": "Medium offload (8 slots)"},
        {"split_point": 5, "target": 13, "desc": "Heavy offload (32 slots)"},
    ]

    for cand in candidates:
        sp = cand["split_point"]
        tgt = cand["target"]
        bw = data_size_after_split(sp)
        print(f"\n  Candidate: {cand['desc']}")
        print(f"    split_point={sp}, target_server={tgt}, bw={bw} slots")

        path_results = evaluate_offload_action(predictor, encoder, sp, tgt)
        best = max(path_results, key=lambda x: x["success_prob"])

        print(f"    Path evaluations:")
        for r in path_results:
            print(f"      Path {r['path_id']}: P_success={r['success_prob']:.4f}, delay={r['delay']:.4f}")
        print(f"    >> Best: Path {best['path_id']} (P={best['success_prob']:.4f})")

    # Show the architecture mapping
    print("\n[3/3] Architecture mapping:")
    print("  DNN Agent            →  (split_point, target_server)")
    print("  ↓")
    print("  data_size_after_split →  bw (slots)")
    print("  get_src_dst           →  (src, dst)")
    print("  Encoder.encode        →  z (18D spectrum state)")
    print("  ↓")
    print("  Predictor(path_id, src, dst, bw, z) → (P_success, delay)")
    print("  ↓")
    print("  Selector              →  best (split_point, target, path_id)")
    print("  ↓")
    print("  Mapper.map(src, dst, bw) → actual path-slot allocation")
    print("=" * 70)
