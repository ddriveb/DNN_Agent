"""Joint evaluation: Agent-C DQN + Agent-R DQN.

Agent-C selects (split, server); Agent-R selects (path, modulation, block).
Both use trained checkpoints.

Single seed:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_joint \
        --episodes 20 --requests_per_episode 20 --seed 123

Multi-seed robustness check:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_joint \
        --episodes 20 --requests_per_episode 20 \
        --seeds 42,123,456,789,2024
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
from collections import Counter
from typing import List, Dict
from scipy import stats

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _compute_greedy_action(obs_c, valid, key):
    if len(valid) == 0:
        return None
    best = valid[0]
    best_val = obs_c["candidate_features"][best][key]
    if best_val == float('inf'):
        best_val = 1e9
    for idx in valid[1:]:
        val = obs_c["candidate_features"][idx][key]
        if val == float('inf'):
            val = 1e9
        if val < best_val:
            best_val = val
            best = idx
    return int(best)


def _spectrum_greedy_action(obs_c, valid):
    if len(valid) == 0:
        return None
    best = valid[0]
    best_fs = obs_c["candidate_features"][best]["best_fs_estimate"]
    best_count = obs_c["candidate_features"][best]["feasible_count"]
    for idx in valid[1:]:
        fs = obs_c["candidate_features"][idx]["best_fs_estimate"]
        count = obs_c["candidate_features"][idx]["feasible_count"]
        if fs is not None and (best_fs is None or fs < best_fs):
            best = idx
            best_fs = fs
            best_count = count
        elif fs == best_fs and count > best_count:
            best = idx
            best_count = count
    return int(best)


def evaluate_method(env: SMDPEnv,
                    agent_c: AgentC,
                    agent_r: AgentR,
                    requests: list,
                    method_name: str,
                    waste_coef: float = 0.8,
                    rng: np.random.RandomState = None) -> dict:
    """Evaluate one joint method on a fixed request sequence."""
    env.reset(requests)
    total_reward = 0.0
    blocked = 0
    successes = 0
    total_delay = 0.0
    total_waste = 0.0
    total_fs = 0.0
    total_path_len = 0.0
    fs_list = []
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    mod_counter = Counter()
    no_valid_action_count = 0

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        mask = obs_c["agent_c_mask"]
        valid = np.where(mask)[0]

        if method_name.startswith("Agent-C-DQN"):
            action_idx_c = agent_c.select_action(obs_c, epsilon=0.0)
        elif method_name.startswith("Compute-Greedy"):
            action_idx_c = _compute_greedy_action(obs_c, valid, "edge_compute_ms")
        elif method_name.startswith("Spectrum-Greedy"):
            action_idx_c = _spectrum_greedy_action(obs_c, valid)
        elif method_name.startswith("Random-valid-C"):
            _rng = rng if rng is not None else np.random
            action_idx_c = int(_rng.choice(valid)) if len(valid) > 0 else None
        else:
            raise ValueError(f"Unknown method: {method_name}")

        if action_idx_c is None:
            no_valid_action_count += 1
            action_c = (0, 0)
        else:
            action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

        split_id, server_id = action_c
        split_counter[split_id] += 1
        server_counter[server_id] += 1

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        if "+ Agent-R-DQN" in method_name:
            action_idx_r = agent_r.select_action(obs_r, epsilon=0.0)
        else:
            action_idx_r = ksp_bf_action(obs_r)

        if action_idx_r is None:
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )

        _, _, done, info = env.step(action_c, action_r)
        server = env.mec.servers[server_id]
        reward = compute_agent_c_reward(
            info, req.deadline_ms, waste_coef, server.utilization
        )
        total_reward += reward

        if info.get("success", False):
            successes += 1
            fs = info.get("num_slots", 0)
            total_delay += info.get("delay_ms", 0.0)
            total_waste += info.get("block_waste", 0.0)
            total_fs += fs
            total_path_len += info.get("path_dist_km", 0.0)
            fs_list.append(fs)
            mod_counter[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reason_counter[reason] += 1

    n = len(requests)
    return {
        "blocking_rate": blocked / n,
        "success_rate": successes / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / successes if successes > 0 else 0.0,
        "avg_block_waste": total_waste / successes if successes > 0 else 0.0,
        "avg_fs": total_fs / successes if successes > 0 else 0.0,
        "avg_path_len_km": total_path_len / successes if successes > 0 else 0.0,
        "fs_list": fs_list,
        "reason_counter": reason_counter,
        "split_counter": split_counter,
        "server_counter": server_counter,
        "mod_counter": mod_counter,
        "no_valid_action_rate": no_valid_action_count / n,
    }


def _fs_histogram(fs_list):
    if not fs_list:
        return "N/A"
    bins = {"1": 0, "2": 0, "3-4": 0, "5-8": 0, "9-16": 0, ">16": 0}
    for fs in fs_list:
        if fs == 1:
            bins["1"] += 1
        elif fs == 2:
            bins["2"] += 1
        elif fs <= 4:
            bins["3-4"] += 1
        elif fs <= 8:
            bins["5-8"] += 1
        elif fs <= 16:
            bins["9-16"] += 1
        else:
            bins[">16"] += 1
    total = len(fs_list)
    return ", ".join(f"{k}:{v/total:.1%}" for k, v in bins.items() if v > 0)


METHODS = [
    "Agent-C-DQN + Agent-R-DQN",
    "Agent-C-DQN + KSP-BF",
    "Compute-Greedy + Agent-R-DQN",
    "Compute-Greedy + KSP-BF",
    "Spectrum-Greedy + KSP-BF",
    "Random-valid-C + KSP-BF",
]


def _run_single_seed(env, agent_c, agent_r, args):
    rng = np.random.RandomState(args.seed)
    all_results = {m: [] for m in METHODS}

    for ep in range(args.episodes):
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env, rng, src, args.requests_per_episode,
            arrival_interval=args.arrival_interval,
            holding_min=args.holding_min,
            holding_max=args.holding_max,
            deadline_min=args.deadline_min,
            deadline_max=args.deadline_max,
            size_min_mb=args.size_min_mb,
            size_max_mb=args.size_max_mb,
            edge_cost_min=args.edge_cost_min,
            edge_cost_max=args.edge_cost_max,
            num_splits=args.num_splits,
        )
        for method in METHODS:
            result = evaluate_method(env, agent_c, agent_r, requests, method, args.waste_coef, rng=rng)
            all_results[method].append(result)
    return all_results


def _aggregate_across_seeds(seed_results: List[Dict[str, list]]):
    metrics = ["blocking_rate", "success_rate", "avg_reward",
               "avg_delay_ms", "avg_block_waste", "avg_fs", "avg_path_len_km"]

    per_seed_means = {m: [] for m in METHODS}
    per_seed_episode_blocks = {m: [] for m in METHODS}

    for single_seed in seed_results:
        for method in METHODS:
            means = {k: np.mean([r[k] for r in single_seed[method]]) for k in metrics}
            per_seed_means[method].append(means)
            per_seed_episode_blocks[method].append(
                [r["blocking_rate"] for r in single_seed[method]]
            )

    overall = {}
    for method in METHODS:
        overall[method] = {}
        for k in metrics:
            vals = [s[k] for s in per_seed_means[method]]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            n = len(vals)
            ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n)) if n > 1 else (mean, mean)
            overall[method][k] = (mean, std, ci_low, ci_high)

    # Win count: Agent-C-DQN + Agent-R-DQN has lowest blocking per seed
    win_count = 0
    target = "Agent-C-DQN + Agent-R-DQN"
    baseline_methods = [m for m in METHODS if m != target]
    for i in range(len(seed_results)):
        target_blk = per_seed_means[target][i]["blocking_rate"]
        others_blk = [per_seed_means[m][i]["blocking_rate"] for m in baseline_methods]
        if target_blk <= min(others_blk):
            win_count += 1

    aggregated = {m: {"reasons": Counter(), "splits": Counter(),
                      "servers": Counter(), "mods": Counter(), "fs_list": []}
                  for m in METHODS}
    for single_seed in seed_results:
        for method in METHODS:
            for r in single_seed[method]:
                aggregated[method]["reasons"].update(r["reason_counter"])
                aggregated[method]["splits"].update(r["split_counter"])
                aggregated[method]["servers"].update(r["server_counter"])
                aggregated[method]["mods"].update(r["mod_counter"])
                aggregated[method]["fs_list"].extend(r["fs_list"])

    return per_seed_means, per_seed_episode_blocks, overall, win_count, aggregated


def _print_single_seed(all_results, args):
    means = {}
    for method in METHODS:
        means[method] = {k: np.mean([r[k] for r in all_results[method]])
                         for k in ["blocking_rate", "success_rate", "avg_reward",
                                   "avg_delay_ms", "avg_block_waste", "avg_fs", "avg_path_len_km"]}

    print("\n" + "=" * 120)
    print("Joint Evaluation: Agent-C + Agent-R")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print("=" * 120)

    header = (f"{'Method':<30} {'BlkRate':>8} {'SuccRate':>9} {'AvgRwd':>8} "
              f"{'Delay(ms)':>10} {'Waste':>7} {'AvgFS':>7} {'Path(km)':>9}")
    print(header)
    print("-" * 120)

    for method in METHODS:
        m = means[method]
        print(f"{method:<30} {m['blocking_rate']:>8.3f} {m['success_rate']:>9.3f} "
              f"{m['avg_reward']:>8.3f} {m['avg_delay_ms']:>10.2f} "
              f"{m['avg_block_waste']:>7.3f} {m['avg_fs']:>7.2f} {m['avg_path_len_km']:>9.2f}")
    print("-" * 120)

    print("\nDiagnostics")
    print("-" * 120)
    for method in METHODS:
        all_fs = []
        all_reasons = Counter()
        all_splits = Counter()
        all_servers = Counter()
        all_mods = Counter()
        no_valid_total = 0.0
        for r in all_results[method]:
            all_fs.extend(r["fs_list"])
            all_reasons.update(r["reason_counter"])
            all_splits.update(r["split_counter"])
            all_servers.update(r["server_counter"])
            all_mods.update(r["mod_counter"])
            no_valid_total += r["no_valid_action_rate"]

        n_total = args.episodes * args.requests_per_episode
        print(f"\n{method}:")
        print(f"  AvgFS: {np.mean(all_fs):.2f}" if all_fs else "  AvgFS: N/A")
        print(f"  FS histogram: {_fs_histogram(all_fs)}")
        print(f"  No-valid-action rate: {no_valid_total/len(all_results[method]):.3f}")
        print(f"  Splits: s0={all_splits.get(0,0)/n_total:.1%}, s1={all_splits.get(1,0)/n_total:.1%}, s2={all_splits.get(2,0)/n_total:.1%}")
        print(f"  Servers: sv0={all_servers.get(0,0)/n_total:.1%}, sv1={all_servers.get(1,0)/n_total:.1%}")
        if all_mods:
            total_mods = sum(all_mods.values())
            mod_str = ", ".join(f"{k}={v/total_mods:.1%}" for k, v in sorted(all_mods.items()))
            print(f"  Modulations: {mod_str}")
        if all_reasons:
            total_failures = sum(all_reasons.values())
            print(f"  Failure reasons (out of {total_failures}):")
            for reason, count in all_reasons.most_common(6):
                print(f"    {reason}: {count} ({count/total_failures:.1%})")
    print("=" * 120)


def _print_multiseed(overall, per_seed_episode_blocks, win_count, aggregated, seeds, args):
    target = "Agent-C-DQN + Agent-R-DQN"
    baseline_methods = [m for m in METHODS if m != target]

    print("\n" + "=" * 130)
    print("Joint Multi-Seed Evaluation: Agent-C + Agent-R")
    print(f"Seeds: {seeds} | Episodes/seed: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print("=" * 130)

    header = (f"{'Method':<30} {'Blocking':>14} {'Success':>14} {'Reward':>14} "
              f"{'Delay(ms)':>14} {'Waste':>14} {'AvgFS':>14} {'Path(km)':>14}")
    print(header)
    print("-" * 130)

    for method in METHODS:
        m = overall[method]
        def _fmt(metric):
            mean, std, ci_l, ci_h = m[metric]
            return f"{mean:.3f} [{ci_l:.3f},{ci_h:.3f}]"
        blk = _fmt("blocking_rate")
        succ = _fmt("success_rate")
        rwd = f"{m['avg_reward'][0]:+.3f} [{m['avg_reward'][2]:+.3f},{m['avg_reward'][3]:+.3f}]"
        dly = f"{m['avg_delay_ms'][0]:.1f} [{m['avg_delay_ms'][2]:.1f},{m['avg_delay_ms'][3]:.1f}]"
        wst = f"{m['avg_block_waste'][0]:.3f} [{m['avg_block_waste'][2]:.3f},{m['avg_block_waste'][3]:.3f}]"
        afs = f"{m['avg_fs'][0]:.2f} [{m['avg_fs'][2]:.2f},{m['avg_fs'][3]:.2f}]"
        pth = f"{m['avg_path_len_km'][0]:.1f} [{m['avg_path_len_km'][2]:.1f},{m['avg_path_len_km'][3]:.1f}]"
        print(f"{method:<30} {blk:>14} {succ:>14} {rwd:>14} {dly:>14} {wst:>14} {afs:>14} {pth:>14}")
    print("-" * 130)
    print(f"\n{target} win count (lowest blocking per seed): {win_count} / {len(seeds)}")

    # Significance: target vs each baseline
    print(f"\nSignificance Tests: {target} vs Baselines")
    print("-" * 130)
    dqn_blocks = np.concatenate(per_seed_episode_blocks[target])
    for baseline in baseline_methods:
        base_blocks = np.concatenate(per_seed_episode_blocks[baseline])
        t_stat, t_p = stats.ttest_rel(dqn_blocks, base_blocks)
        try:
            w_stat, w_p = stats.wilcoxon(dqn_blocks, base_blocks)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        diff = dqn_blocks - base_blocks
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        print(f"  (Episode-level) vs {baseline:<25} t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}, Cohen's d={cohens_d:+.3f}")

    dqn_seed_blocks = [np.mean(ep_blocks) for ep_blocks in per_seed_episode_blocks[target]]
    for baseline in baseline_methods:
        base_seed_blocks = [np.mean(ep_blocks) for ep_blocks in per_seed_episode_blocks[baseline]]
        t_stat, t_p = stats.ttest_rel(dqn_seed_blocks, base_seed_blocks)
        try:
            w_stat, w_p = stats.wilcoxon(dqn_seed_blocks, base_seed_blocks)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        diff = np.array(dqn_seed_blocks) - np.array(base_seed_blocks)
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        print(f"  (Seed-level)    vs {baseline:<25} t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}, Cohen's d={cohens_d:+.3f}")
    print("-" * 130)

    # Failure / split / mod diagnostics
    print("\nFailure Reason Distribution")
    print("-" * 130)
    for method in METHODS:
        reasons = aggregated[method]["reasons"]
        total = sum(reasons.values())
        if total == 0:
            print(f"{method:<30} No failures")
            continue
        parts = []
        for r in ["server_overload", "no_suitable_block", "fs_too_large", "deadline_infeasible"]:
            cnt = reasons.get(r, 0)
            parts.append(f"{r}={cnt} ({cnt/total:.1%})")
        print(f"{method:<30} {', '.join(parts)}")

    print("\nSplit / Server / Modulation Distribution")
    print("-" * 130)
    n_total = len(seeds) * args.episodes * args.requests_per_episode
    for method in METHODS:
        splits = aggregated[method]["splits"]
        servers = aggregated[method]["servers"]
        mods = aggregated[method]["mods"]
        s0 = splits.get(0, 0) / n_total
        s1 = splits.get(1, 0) / n_total
        s2 = splits.get(2, 0) / n_total
        sv0 = servers.get(0, 0) / n_total
        sv1 = servers.get(1, 0) / n_total
        mod_str = ", ".join(f"{k}={v/sum(mods.values()):.1%}" for k, v in sorted(mods.items())) if mods else "N/A"
        print(f"{method:<30} splits={s0:.1%}/{s1:.1%}/{s2:.1%} servers={sv0:.1%}/{sv1:.1%} mods={mod_str}")

    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="Joint Agent-C + Agent-R evaluation")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=20)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed eval")
    parser.add_argument("--topology", type=str, default="net1")
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--checkpoint_c", type=str, default=None)
    parser.add_argument("--checkpoint_r", type=str, default=None)
    parser.add_argument("--capacities", type=str, default=None,
                        help="Comma-separated MEC capacities in GFLOPS, e.g. 50,50")
    args = parser.parse_args()

    cap_list = None
    if args.capacities is not None:
        cap_list = [float(c.strip()) for c in args.capacities.split(",")]
    env = make_env(
        args.topology, args.num_slots, args.num_servers, args.seed,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        capacities=cap_list,
        modulation_profile=args.modulation_profile,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    device = 'cpu'

    # Load Agent-C
    agent_c = AgentC(input_dim=17, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device=device)
    if args.checkpoint_c is not None:
        ckpt_c_path = Path(args.checkpoint_c)
    else:
        ckpt_c_path = Path(__file__).resolve().parents[2] / "checkpoints" / "agent_c_best.pt"
    if ckpt_c_path.exists():
        ckpt = load_checkpoint(ckpt_c_path, map_location=device)
        agent_c.q_net.load_state_dict(ckpt["model_state"])
        agent_c.target_net.load_state_dict(ckpt["target_state"])
        print(f"Loaded Agent-C from {ckpt_c_path}")
    else:
        print("WARNING: No Agent-C checkpoint found, using random-init")

    # Load Agent-R
    agent_r = AgentR(input_dim=11, mod_registry=mod_reg, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device=device)
    if args.checkpoint_r is not None:
        ckpt_r_path = Path(args.checkpoint_r)
    else:
        ckpt_r_path = Path(__file__).resolve().parents[2] / "checkpoints" / "agent_r_smoke.pt"
    if ckpt_r_path.exists():
        ckpt = load_checkpoint(ckpt_r_path, map_location=device)
        agent_r.q_net.load_state_dict(ckpt["model_state"])
        agent_r.target_net.load_state_dict(ckpt["target_state"])
        print(f"Loaded Agent-R from {ckpt_r_path}")
    else:
        print("WARNING: No Agent-R checkpoint found, using random-init")

    if args.seeds is not None:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seed_list = [args.seed]

    if len(seed_list) == 1:
        args.seed = seed_list[0]
        all_results = _run_single_seed(env, agent_c, agent_r, args)
        _print_single_seed(all_results, args)
    else:
        seed_results = []
        for s in seed_list:
            args.seed = s
            seed_results.append(_run_single_seed(env, agent_c, agent_r, args))

        per_seed_means, per_seed_episode_blocks, overall, win_count, aggregated = _aggregate_across_seeds(seed_results)
        _print_multiseed(overall, per_seed_episode_blocks, win_count, aggregated, seed_list, args)


if __name__ == "__main__":
    main()
