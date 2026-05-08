"""Test 3000 samples / 5 epochs config."""
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

def gen_data(topo, slots, n, seed):
    arr, ht = (0.25, 8.0) if topo == "nsfnet" else (0.30, 20.0)
    preload = 500 if topo == "nsfnet" else 5000
    net = OpticalNetwork(topology=topo, num_slots=slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    rng = np.random.RandomState(seed)
    active = []
    t = 0.0
    for _ in range(preload):
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src: dst = rng.randint(0, net.NUM_NODES)
        bw = min([1,2,4,8,16,32][rng.randint(0,6)], slots)
        s, p, ss, d = mapper.map(src, dst, bw)
        if s: active.append((p, ss, bw, t + rng.exponential(ht*2)))
    lim, cei = build_converted_topology(net)
    pms = {}
    for src in range(net.NUM_NODES):
        for dst in range(net.NUM_NODES):
            if src == dst: continue
            paths = mapper._path_cache.get((src,dst),[])
            for pid, path in enumerate(paths):
                m = np.zeros(len(lim), dtype=np.float32)
                for i in range(len(path)-1):
                    l = (min(path[i],path[i+1]), max(path[i],path[i+1]))
                    if l in lim: m[lim[l]] = 1.0
                pms[(src,dst,pid)] = m
    samples = []
    bws = [b for b in [1,2,4,8,16,32] if b <= slots]
    while len(samples) < n:
        t += rng.exponential(1.0/arr)
        na = []
        for p, s, bw, rt in active:
            if rt <= t: net.release(p, s, bw)
            else: na.append((p,s,bw,rt))
        active = na
        src = rng.randint(0, net.NUM_NODES)
        dst = rng.randint(0, net.NUM_NODES)
        while dst == src: dst = rng.randint(0, net.NUM_NODES)
        bw = bws[rng.randint(0, len(bws))]
        pid = rng.randint(0, 3)
        z = encoder.encode_v2b(src, dst)
        lf = compute_link_node_features(net, lim)
        r = mapper.execute_path(src, dst, bw, pid)
        samples.append({"src":src,"dst":dst,"path_id":pid,"bw":float(bw),
            "z":z,"success":1.0 if r["success"] else 0.0,"delay":r["delay"] if r["success"] else 0.0,
            "link_node_features":lf,"converted_edge_index":cei,
            "path_mask":pms.get((src,dst,pid), np.zeros(len(lim),dtype=np.float32))})
        if r["success"]: active.append((r["path"],r["start_slot"],bw,t+rng.exponential(ht)))
    for p,s,bw,_ in active: net.release(p,s,bw)
    return samples

def train(samples, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    split = int(0.8*len(samples))
    tl = DataLoader(HybridDataset(samples[:split]), batch_size=128, shuffle=True, collate_fn=hybrid_collate_fn)
    vl = DataLoader(HybridDataset(samples[split:]), batch_size=128, shuffle=False, collate_fn=hybrid_collate_fn)
    model = HybridPredictor(state_dim=len(samples[0]["z"]), num_links=tl.dataset.num_links,
        link_feat_dim=6, link_hidden=32, num_gat_layers=2, num_heads=4, dropout=0.1,
        num_paths=3, max_servers=128, hidden_dim=64)
    train_hybrid_predictor(model, tl, vl, epochs=5, lr=1e-3, class_weight=False, early_stop_patience=999)
    return model

def sim(net, mapper, selector, reqs, fn, ht, arr, seed):
    rng = np.random.RandomState(seed); net.reset(); active=[]; t=0.0; blocks=0
    for req in reqs:
        t += arr
        na=[]
        for p,ss,bw,rt in active:
            if rt <= t: net.release(p,ss,bw)
            else: na.append((p,ss,bw,rt))
        active = na
        r = fn(net, mapper, selector, req["src"], req["dst"], req["bw"])
        if r["success"]: active.append((r["path"],r["start_slot"],req["bw"],t+rng.exponential(ht)))
        else: blocks += 1
    return blocks / len(reqs)

def shortest(net, mapper, sel, src, dst, bw): return mapper.execute_path(src, dst, bw, 0)

def fragaware(net, mapper, sel, src, dst, bw):
    best, cands = sel.select(src, dst, bw)
    probs = [c["success_prob"] for c in cands]
    mp = max(probs) if probs else 0
    qual = [c for c in cands if c["success_prob"] >= mp*0.9]
    if len(qual) <= 1: return mapper.execute_path(src, dst, bw, best["path_id"])
    bf = None; bpid = best["path_id"]
    for cand in qual:
        pid = cand["path_id"]
        r = mapper.execute_path(src, dst, bw, pid)
        if r["success"]:
            f = r.get("frag_change", 0)
            if bf is None or f < bf: bf = f; bpid = pid
            net.release(r["path"], r["start_slot"], bw)
    return mapper.execute_path(src, dst, bw, bpid)

def run(topo, slots, reqs, bwlist, seed):
    arr, ht = (0.25, 8.0) if topo == "nsfnet" else (0.30, 20.0)
    samples = gen_data(topo, slots, 3000, seed)
    model = train(samples, seed)
    net = OpticalNetwork(topology=topo, num_slots=slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = HybridEncoder(net, k=3)
    selector = HybridSelector(model, encoder, "max_prob")
    rng = np.random.RandomState(seed+100)
    reqs_list = [{"src":rng.randint(0,net.NUM_NODES),"dst":rng.randint(0,net.NUM_NODES),"bw":bwlist[rng.randint(0,len(bwlist))]} for _ in range(reqs)]
    for r in reqs_list:
        while r["dst"] == r["src"]: r["dst"] = rng.randint(0, net.NUM_NODES)
    base = sim(net, mapper, selector, reqs_list, shortest, ht, arr, seed)
    frag = sim(net, mapper, selector, reqs_list, fragaware, ht, arr, seed)
    return base, frag, (base-frag)/base*100 if base > 1e-9 else 0

print("3000 samples / 5 epochs validation")
for topo, slots, reqs, bws, label in [
    ("nsfnet", 32, 2000, [1,2,4,8], "NSFNET"),
    ("random100", 128, 2000, [2,4,8,16], "Random100"),
]:
    print(f"\n{label}:")
    print(f"  {'Seed':>6} {'Base%':>8} {'Frag%':>8} {'Imp%':>8}")
    imps = []
    for seed in [42, 123, 456]:
        base, frag, imp = run(topo, slots, reqs, bws, seed)
        imps.append(imp)
        print(f"  {seed:>6} {base*100:>7.2f}% {frag*100:>7.2f}% {imp:>+7.1f}%")
    print(f"  {'Mean':>6} {'':>8} {'':>8} {np.mean(imps):>+7.1f}%")
    print(f"  {'Std':>6} {'':>8} {'':>8} {np.std(imps):>7.1f}%")
