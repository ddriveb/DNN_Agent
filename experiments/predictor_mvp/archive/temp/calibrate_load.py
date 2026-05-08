"""
Search for load parameters that yield ~15% Shortest blocking rate
and ~30-40% negative samples in training data.
"""
import numpy as np
from env import OpticalNetwork
from mapper import KSPMapper

def test_load(topology, num_slots, arr, ht, num_requests=2000, seed=42):
    """Run Shortest-only simulation and return blocking rate."""
    rng = np.random.RandomState(seed)
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    
    # Generate requests
    bw_choices = [1, 2, 4, 8] if topology == "nsfnet" else [2, 4, 8, 16]
    requests = []
    for _ in range(num_requests):
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src:
            dst = rng.randint(0, net.NUM_NODES)
        bw = bw_choices[rng.randint(0, len(bw_choices))]
        requests.append((src, dst, bw))
    
    # Simulate Shortest only
    net.reset()
    active = []
    t = 0.0
    blocked = 0
    for src, dst, bw in requests:
        t += arr
        new_active = []
        for path, start, bw_req, rt in active:
            if rt <= t:
                net.release(path, start, bw_req)
            else:
                new_active.append((path, start, bw_req, rt))
        active = new_active
        
        result = mapper.execute_path(src, dst, bw, path_id=0)
        if result["success"]:
            ht_val = rng.exponential(ht)
            active.append((result["path"], result["start_slot"], bw, t + ht_val))
        else:
            blocked += 1
    
    br = blocked / num_requests
    util = np.mean([np.sum(link) for link in net.link_states.values()]) / num_slots
    return br, util


# Search NSFNET parameters
print("=" * 60)
print("NSFNET LOAD CALIBRATION")
print("=" * 60)
print(f"{'arr':>8} {'ht':>8} {'BR%':>10} {'Util%':>10}")
for arr in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]:
    for ht in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        br, util = test_load("nsfnet", 32, arr, ht, num_requests=2000)
        flag = " ***" if 0.10 <= br <= 0.25 else ""
        print(f"{arr:>8.2f} {ht:>8.1f} {br*100:>9.2f}% {util*100:>9.1f}%{flag}")

print()
print("=" * 60)
print("RANDOM100 LOAD CALIBRATION")
print("=" * 60)
print(f"{'arr':>8} {'ht':>8} {'BR%':>10} {'Util%':>10}")
for arr in [2.0, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.03]:
    for ht in [5.0, 8.0, 10.0, 15.0, 20.0]:
        br, util = test_load("random100", 128, arr, ht, num_requests=2000)
        flag = " ***" if 0.10 <= br <= 0.25 else ""
        print(f"{arr:>8.2f} {ht:>8.1f} {br*100:>9.2f}% {util*100:>9.1f}%{flag}")
