"""R Disagreement Diagnostic: PPO-R vs BC-PPO-R vs DeepRMSA vs KSP-BF.

Runs all R backends on parallel env copies under the same fixed Agent-C,
tracking per-request action choices and outcomes for disagreement analysis.

Usage::

    # Smoke
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_r_disagreement_diagnostic \\
        --seeds 42 --episodes 1 --requests_per_episode 10

    # Full (S20, 5 seeds x 5 eps = matches recent eval setting)
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_r_disagreement_diagnostic \\
        --num_slots 20 --num_splits 5 --split_profile complex5_v2_lite \\
        --seeds 42,123,456,789,2024 --episodes 5 --requests_per_episode 80
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation, build_agent_r_observation,
    decode_agent_c_action, decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ======================================================================
# Model loaders
# ======================================================================

def _load_c(path, device="cpu"):
    ckpt = load_checkpoint(path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    fm = ckpt_args.get("agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"))
    idim = ckpt.get("input_dim", 17)
    if idim >= 24 and fm == "default": fm = "enhanced"
    a = PPOAgentC(input_dim=idim, hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
                  device=device, feature_mode=fm)
    a.policy_net.load_state_dict(ckpt["model_state"]); a.policy_net.eval()
    a.checkpoint_args = ckpt_args
    return a

def _load_r_ppo(path, mod_reg, device="cpu"):
    ckpt = load_checkpoint(path, map_location=device)
    fm_r = ckpt.get("agent_r_feature_mode", ckpt.get("args", {}).get("agent_r_feature_mode", "default"))
    a = PPOAgentR(input_dim=ckpt.get("input_dim", 11), mod_registry=mod_reg,
                  hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
                  device=device, feature_mode=fm_r)
    a.policy_net.load_state_dict(ckpt["model_state"]); a.policy_net.eval()
    return a

def _load_r_deep(path, env_proto, mod_reg, device="cpu"):
    ckpt = load_checkpoint(path, map_location=device)
    a = DeepRMSAAgent(
        num_nodes=env_proto.net.NUM_NODES, num_slots=env_proto.net.num_slots,
        k_path=ckpt.get("k_path", 5), m_blocks=ckpt.get("m_blocks", 10),
        mod_registry=mod_reg, gamma=ckpt.get("gamma", 0.95), device=device,
    )
    a.load_state_dict(ckpt); a.eval()
    return a


# ======================================================================
# Main diagnostic
# ======================================================================

def _fmt_pct(c):
    ct = Counter(c); t = sum(ct.values())
    if t <= 0: return "none"
    return ", ".join(f"{k}={v/t:.1%}" for k, v in ct.most_common(6))


def main():
    parser = argparse.ArgumentParser(description="R Disagreement Diagnostic")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--ppo_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/joint_mappo_v2_cshape_metro24_fs002_r_best.pt")
    parser.add_argument("--bc_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--deep_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/deep_rmsa_snap24_reach_mixed.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/r_disagreement_diagnostic.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/r_disagreement_diagnostic.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    mod_reg = ModulationRegistry.from_profile("default")

    # Load C
    print(f"Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_c(args.agent_c_checkpoint, args.device)
    print(f"  dim={agent_c.input_dim} fm={agent_c.feature_mode}")

    # Build prototype env
    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots, num_servers=args.num_servers,
        seed=42, slot_bw_hz=1.25e9, guard_band_fs=1, modulation_profile="default",
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy, k=args.k_paths,
    )
    num_servers = args.num_servers
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    # Load R backends
    r_backends = []

    print(f"PPO-R: {args.ppo_r_checkpoint}")
    r_backends.append({"name": "PPO-R", "agent": _load_r_ppo(args.ppo_r_checkpoint, mod_reg, args.device), "type": "ppo"})

    print(f"BC-PPO-R: {args.bc_r_checkpoint}")
    r_backends.append({"name": "BC-PPO-R", "agent": _load_r_ppo(args.bc_r_checkpoint, mod_reg, args.device), "type": "ppo"})

    # DeepRMSA: skip if k/m mismatch (checkpoint trained with specific params)
    deep_skip = False
    deep_ckpt = load_checkpoint(args.deep_r_checkpoint, map_location=args.device)
    deep_k = deep_ckpt.get("k_path", args.k_paths)
    deep_m = deep_ckpt.get("m_blocks", args.max_blocks)
    print(f"DeepRMSA: {args.deep_r_checkpoint}  (k={deep_k} m={deep_m})")
    if deep_k != args.k_paths or deep_m != args.max_blocks:
        print(f"  SKIP: k/m mismatch (ckpt k={deep_k} m={deep_m} vs args k={args.k_paths} m={args.max_blocks})")
        print(f"  PPO-R ≈ DeepRMSA already established in fixed_c_r_compare (0.224 vs 0.225 at k=3)")
        deep_skip = True
    if not deep_skip:
        deep_env_proto = make_env(
            topology=args.topology, num_slots=args.num_slots, num_servers=args.num_servers,
            seed=42, slot_bw_hz=1.25e9, guard_band_fs=1, modulation_profile="default",
            max_blocks=deep_m, block_sort_strategy=args.block_sort_strategy, k=deep_k,
        )
        r_backends.append({"name": "DeepRMSA", "agent": _load_r_deep(args.deep_r_checkpoint, deep_env_proto, mod_reg, args.device), "type": "deep"})

    r_backends.append({"name": "KSP-BF", "agent": None, "type": "ksp_bf"})

    num_r = len(r_backends)

    # Pre-generate episodes
    all_eps = {}
    for seed in seeds:
        rng = np.random.RandomState(seed); eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode, arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_eps[seed] = eps

    print(f"\n{'=' * 80}")
    print(f"R Disagreement Diagnostic: {args.num_slots} slots, {args.requests_per_episode} req/ep")
    print(f"Seeds: {seeds}  Eps/seed: {args.episodes}")
    print(f"R backends: {[b['name'] for b in r_backends]}")
    print(f"{'=' * 80}")

    # --- Accumulators ---
    totals = [0] * num_r; blockeds = [0] * num_r; successes = [0] * num_r
    total_delay = [0.0] * num_r; total_fs = [0.0] * num_r
    total_waste = [0.0] * num_r; total_path = [0.0] * num_r
    reason_ctrs = [Counter() for _ in range(num_r)]
    mod_ctrs = [Counter() for _ in range(num_r)]

    # Disagreement tracking (pairwise, using backend 0=PPO-R as reference)
    # For each pair (0, j): same_action, diff_action_ok, ...
    pair_same = {(0, j): 0 for j in range(1, num_r)}
    pair_diff = {(0, j): 0 for j in range(1, num_r)}
    pair_ref_win = {(0, j): 0 for j in range(1, num_r)}  # PPO-R ok, other fail
    pair_other_win = {(0, j): 0 for j in range(1, num_r)}  # other ok, PPO-R fail
    pair_both_ok_diff = {(0, j): 0 for j in range(1, num_r)}  # both ok, different action
    pair_both_fail = {(0, j): 0 for j in range(1, num_r)}
    # Action-level diff breakdown (path/mod/block different)
    diff_path = {(0, j): 0 for j in range(1, num_r)}
    diff_mod = {(0, j): 0 for j in range(1, num_r)}
    diff_block = {(0, j): 0 for j in range(1, num_r)}

    total_req = 0
    both_fail_ctr = Counter()
    t_start = time.time()

    for seed in seeds:
        for ep_idx, requests in enumerate(all_eps[seed]):
            # Create parallel env copies
            envs = []
            for _ in range(num_r):
                e = make_env(
                    topology=args.topology, num_slots=args.num_slots,
                    num_servers=num_servers, seed=42,
                    slot_bw_hz=1.25e9, guard_band_fs=1, modulation_profile="default",
                    max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                    server_nodes=server_nodes, capacities=capacities, k=args.k_paths,
                )
                e.reset(requests)
                envs.append(e)

            ref_env = envs[0]  # PPO-R's env for Agent-C decisions

            for req in requests:
                obs_c = build_agent_c_observation(ref_env, req)
                action_idx_c = agent_c.select_action(obs_c, deterministic=True)
                action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(action_idx_c, num_servers)
                split_id, server_id = action_c

                # Record actions and outcomes for all R backends
                actions_this_req = []
                outcomes_this_req = []

                for i, (env, backend) in enumerate(zip(envs, r_backends)):
                    obs_r = build_agent_r_observation(env, req, split_id, server_id)
                    r_type = backend["type"]

                    if r_type == "ppo":
                        a_idx = backend["agent"].select_action(obs_r, deterministic=True)
                    elif r_type == "deep":
                        a_idx = backend["agent"].select_action(obs_r)
                    elif r_type == "ksp_bf":
                        a_idx = ksp_bf_action(obs_r)
                    else:
                        a_idx = None

                    n_mods = len(obs_r["mod_names"]); max_blk = env.max_blocks
                    if a_idx is not None and a_idx < len(obs_r["agent_r_mask"]) and obs_r["agent_r_mask"][a_idx]:
                        a_tuple = decode_agent_r_action(a_idx, n_mods, max_blk)
                    else:
                        a_tuple = (0, 0, 0); a_idx = None

                    _, _, _, info = env.step(action_c, a_tuple)
                    totals[i] += 1
                    ok = info.get("success", False)

                    if ok:
                        successes[i] += 1
                        total_delay[i] += float(info.get("delay_ms", 0))
                        total_fs[i] += float(info.get("num_slots", 0))
                        total_waste[i] += float(info.get("block_waste", 0))
                        total_path[i] += float(info.get("path_dist_km", 0))
                        mod_ctrs[i][info.get("modulation", "?")] += 1
                    else:
                        blockeds[i] += 1
                        reason_ctrs[i][info.get("reason", "?")] += 1

                    actions_this_req.append({"idx": a_idx, "tuple": a_tuple, "ok": ok, "reason": info.get("reason", "?")})
                    outcomes_this_req.append(ok)

                total_req += 1

                # Pairwise disagreement analysis (PPO-R = index 0 as reference)
                ref_act = actions_this_req[0]
                ref_ok = outcomes_this_req[0]
                for j in range(1, num_r):
                    other_act = actions_this_req[j]
                    other_ok = outcomes_this_req[j]
                    same = (ref_act["idx"] == other_act["idx"])
                    if same:
                        pair_same[(0, j)] += 1
                    else:
                        pair_diff[(0, j)] += 1
                        # action-level breakdown
                        rt = ref_act["tuple"]; ot = other_act["tuple"]
                        if rt[0] != ot[0]: diff_path[(0, j)] += 1
                        if rt[1] != ot[1]: diff_mod[(0, j)] += 1
                        if rt[2] != ot[2]: diff_block[(0, j)] += 1

                    if ref_ok and not other_ok: pair_ref_win[(0, j)] += 1
                    if not ref_ok and other_ok: pair_other_win[(0, j)] += 1
                    if ref_ok and other_ok and not same: pair_both_ok_diff[(0, j)] += 1
                    if not ref_ok and not other_ok:
                        pair_both_fail[(0, j)] += 1
                        both_fail_ctr[f"PPO-R={ref_act['reason']},{r_backends[j]['name']}={other_act['reason']}"] += 1

    elapsed = time.time() - t_start

    # --- Aggregate results ---
    agg_results = []
    for i, backend in enumerate(r_backends):
        n = max(totals[i], 1); s = max(successes[i], 1)
        agg_results.append({
            "name": backend["name"],
            "blocking_rate": blockeds[i] / n, "success_rate": successes[i] / n,
            "avg_delay_ms": total_delay[i] / s, "avg_fs": total_fs[i] / s,
            "avg_waste": total_waste[i] / s, "avg_path_km": total_path[i] / s,
            "reason_counter": dict(reason_ctrs[i]), "mod_counter": dict(mod_ctrs[i]),
        })

    # --- Console report ---
    print(f"\n{'=' * 80}")
    print(f"RESULTS  (n={total_req} requests, {elapsed:.0f}s)")
    print(f"{'=' * 80}")
    hdr = f"{'R Backend':15s} {'Blocking':>9s} {'Delay':>7s} {'FS':>6s} {'Waste':>7s} {'PathKm':>7s} {'NSB%':>7s}"
    print(hdr); print("-" * 70)
    for r in agg_results:
        nsb = r["reason_counter"].get("no_suitable_block", 0) / max(sum(r["reason_counter"].values()), 1)
        print(f"{r['name']:15s} {r['blocking_rate']:9.4f} {r['avg_delay_ms']:6.1f} "
              f"{r['avg_fs']:6.2f} {r['avg_waste']:7.3f} {r['avg_path_km']:7.1f} {nsb:6.1%}")

    # --- Disagreement report (PPO-R as reference) ---
    print(f"\n{'=' * 80}")
    print("DISAGREEMENT ANALYSIS (reference = PPO-R)")
    print(f"{'=' * 80}")
    for j in range(1, num_r):
        name_j = r_backends[j]["name"]
        n = total_req
        print(f"\n  PPO-R vs {name_j}:")
        print(f"    Same action:       {pair_same[(0,j)]/n:.1%} ({pair_same[(0,j)]}/{n})")
        print(f"    Different action:  {pair_diff[(0,j)]/n:.1%} ({pair_diff[(0,j)]}/{n})")
        print(f"      path different:  {diff_path[(0,j)]/pair_diff[(0,j)]:.1%}" if pair_diff[(0,j)] > 0 else "      path different:  N/A")
        print(f"      mod different:   {diff_mod[(0,j)]/pair_diff[(0,j)]:.1%}" if pair_diff[(0,j)] > 0 else "      mod different:   N/A")
        print(f"      block different: {diff_block[(0,j)]/pair_diff[(0,j)]:.1%}" if pair_diff[(0,j)] > 0 else "      block different: N/A")
        print(f"    PPO-R win:         {pair_ref_win[(0,j)]/n:.1%} ({pair_ref_win[(0,j)]}/{n}) — PPO ok, {name_j} fail")
        print(f"    {name_j} win:       {pair_other_win[(0,j)]/n:.1%} ({pair_other_win[(0,j)]}/{n}) — {name_j} ok, PPO fail")
        print(f"    Both ok, diff act: {pair_both_ok_diff[(0,j)]/n:.1%} ({pair_both_ok_diff[(0,j)]}/{n})")
        print(f"    Both fail:         {pair_both_fail[(0,j)]/n:.1%} ({pair_both_fail[(0,j)]}/{n})")

    # --- Failure reasons ---
    print(f"\n{'=' * 80}")
    print("FAILURE REASONS")
    print(f"{'=' * 80}")
    for r in agg_results:
        print(f"  {r['name']:15s}: {_fmt_pct(r['reason_counter'])}")

    # --- JSON ---
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    def _clean(obj):
        if isinstance(obj, dict): return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        return obj
    pair_data = {}
    for j in range(1, num_r):
        name_j = r_backends[j]["name"]
        pair_data[f"PPO-R_vs_{name_j}"] = {
            "same_action": pair_same[(0, j)], "diff_action": pair_diff[(0, j)],
            "diff_path": diff_path[(0, j)], "diff_mod": diff_mod[(0, j)], "diff_block": diff_block[(0, j)],
            "ref_win": pair_ref_win[(0, j)], "other_win": pair_other_win[(0, j)],
            "both_ok_diff": pair_both_ok_diff[(0, j)], "both_fail": pair_both_fail[(0, j)],
        }
    out_json.write_text(json.dumps({
        "args": vars(args), "total_requests": total_req, "elapsed_s": elapsed,
        "results": _clean(agg_results),
        "disagreement": _clean(pair_data),
        "both_fail_reasons": _clean(dict(both_fail_ctr.most_common(20))),
    }, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # --- Markdown ---
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# R Disagreement Diagnostic\n\n")
    md.append(f"**Slots:** {args.num_slots}  **k:** {args.k_paths}  "
              f"**Req/ep:** {args.requests_per_episode}  **Seeds:** {args.seeds}  "
              f"**Eps/seed:** {args.episodes}\n\n")
    md.append(f"**Agent-C:** `{args.agent_c_checkpoint}`\n\n")
    md.append(f"**Total requests:** {total_req}  **Time:** {elapsed:.0f}s\n\n")

    md.append("## Blocking Comparison\n\n")
    md.append("| R Backend | Blocking | Delay | FS | Waste | PathKm | NSB% |\n")
    md.append("|-----------|----------|-------|----|-------|--------|------|\n")
    for r in agg_results:
        nsb = r["reason_counter"].get("no_suitable_block", 0) / max(sum(r["reason_counter"].values()), 1)
        md.append(f"| {r['name']} | {r['blocking_rate']:.4f} | {r['avg_delay_ms']:.1f} | "
                  f"{r['avg_fs']:.2f} | {r['avg_waste']:.3f} | {r['avg_path_km']:.1f} | {nsb:.1%} |\n")

    md.append("\n## Disagreement Analysis (ref = PPO-R)\n\n")
    for j in range(1, num_r):
        name_j = r_backends[j]["name"]
        n = total_req
        md.append(f"### PPO-R vs {name_j}\n\n")
        md.append(f"| Metric | Value |\n|---|---|\n")
        md.append(f"| Same action | {pair_same[(0,j)]/n:.1%} ({pair_same[(0,j)]}/{n}) |\n")
        md.append(f"| Different action | {pair_diff[(0,j)]/n:.1%} ({pair_diff[(0,j)]}/{n}) |\n")
        md.append(f"| PPO-R win (ok, other fail) | {pair_ref_win[(0,j)]/n:.1%} |\n")
        md.append(f"| {name_j} win (ok, PPO fail) | {pair_other_win[(0,j)]/n:.1%} |\n")
        md.append(f"| Both ok, different action | {pair_both_ok_diff[(0,j)]/n:.1%} |\n")
        md.append(f"| Both fail | {pair_both_fail[(0,j)]/n:.1%} |\n\n")

    md.append("## Failure Reasons\n\n")
    for r in agg_results:
        md.append(f"- **{r['name']}**: {_fmt_pct(r['reason_counter'])}\n")

    md.append("\n## Modulation Distribution\n\n")
    for r in agg_results:
        md.append(f"- **{r['name']}**: {_fmt_pct(r['mod_counter'])}\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")


if __name__ == "__main__":
    main()
