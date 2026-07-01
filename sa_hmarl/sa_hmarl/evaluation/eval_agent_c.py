"""Formal evaluation script for Agent-C DQN vs baselines (fixed KSP-BF RMSA).

Supports single-seed or multi-seed evaluation.

Single seed (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_agent_c \
        --episodes 20 --requests_per_episode 20

Multi-seed robustness check:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_agent_c \
        --episodes 20 --requests_per_episode 20 \
        --seeds 42,123,456,789,2024

Stress evaluation example:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_agent_c \
        --episodes 20 --requests_per_episode 20 \
        --arrival_interval 0.1 --holding_min 6 --holding_max 15 \
        --size_min_mb 20.0 --size_max_mb 80.0 --deadline_min 20 --deadline_max 60 \
        --slot_bw_hz 1.25e9 --guard_band_fs 1
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
    """Select action minimizing key among valid actions."""
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
    """Select action with minimum best_fs_estimate; tie-break by max feasible_count."""
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
                    agent: AgentC,
                    requests: list,
                    method_name: str,
                    waste_coef: float = 0.8) -> dict:
    """Evaluate one Agent-C method on a fixed request sequence.

    RMSA is always KSP-BF.

    Returns dict with standard metrics plus diagnostic fields:
        - fs_list, reason_counter, split_counter, server_counter
    """
    env.reset(requests)
    total_reward = 0.0
    blocked = 0
    successes = 0
    total_delay = 0.0
    total_waste = 0.0
    total_fs = 0.0
    fs_list = []
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    no_valid_action_count = 0

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        mask = obs_c["agent_c_mask"]
        valid = np.where(mask)[0]

        if method_name == "Agent-C-DQN":
            action_idx_c = agent.select_action(obs_c, epsilon=0.0)
        elif method_name == "Compute-Greedy":
            action_idx_c = _compute_greedy_action(obs_c, valid, "edge_compute_ms")
        elif method_name == "Spectrum-Greedy":
            action_idx_c = _spectrum_greedy_action(obs_c, valid)
        elif method_name == "Random-valid-C":
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

        # Fixed RMSA: KSP-BF
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
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
            fs_list.append(fs)
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reason_counter[reason] += 1

    n = len(requests)
    result = {
        "blocking_rate": blocked / n,
        "success_rate": successes / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / successes if successes > 0 else 0.0,
        "avg_block_waste": total_waste / successes if successes > 0 else 0.0,
        "avg_fs": total_fs / successes if successes > 0 else 0.0,
        "fs_list": fs_list,
        "reason_counter": reason_counter,
        "split_counter": split_counter,
        "server_counter": server_counter,
        "no_valid_action_rate": no_valid_action_count / n,
    }
    return result


def _fs_histogram(fs_list):
    """Build a simple FS histogram string."""
    if not fs_list:
        return "N/A"
    bins = {
        "1": 0, "2": 0, "3-4": 0, "5-8": 0, "9-16": 0, ">16": 0
    }
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


def _distribution_str(counter, total):
    """Pretty-print a Counter distribution."""
    if not counter:
        return "N/A"
    return ", ".join(
        f"{k}={v} ({v/total:.1%})" for k, v in sorted(counter.items())
    )


def _run_single_seed(env: SMDPEnv, agent: AgentC, args) -> Dict[str, list]:
    """Run evaluation for one seed and return per-method list of results."""
    rng = np.random.RandomState(args.seed)
    all_results = {
        "Agent-C-DQN": [],
        "Compute-Greedy": [],
        "Spectrum-Greedy": [],
        "Random-valid-C": [],
    }

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

        for method in all_results.keys():
            result = evaluate_method(env, agent, requests, method, args.waste_coef, rng=rng)
            all_results[method].append(result)

    return all_results


def _aggregate_across_seeds(seed_results: List[Dict[str, list]], methods: List[str]):
    """Aggregate results across multiple seeds.

    Returns:
        per_seed_means: {method: [mean_dict_per_seed]}
        per_seed_episode_blocks: {method: [[ep_blocks_per_seed]]}
        overall_mean_std: {method: {metric: (mean, std, ci_low, ci_high)}}
        win_count: int
        aggregated_counters: {method: {reasons, splits, servers, fs_list}}
    """
    metrics_to_avg = [
        "blocking_rate", "success_rate", "avg_reward",
        "avg_delay_ms", "avg_block_waste", "avg_fs",
    ]

    per_seed_means = {m: [] for m in methods}
    per_seed_episode_blocks = {m: [] for m in methods}

    for single_seed in seed_results:
        for method in methods:
            means = {k: np.mean([r[k] for r in single_seed[method]])
                     for k in metrics_to_avg}
            per_seed_means[method].append(means)
            per_seed_episode_blocks[method].append(
                [r["blocking_rate"] for r in single_seed[method]]
            )

    overall_mean_std = {}
    for method in methods:
        overall_mean_std[method] = {}
        for k in metrics_to_avg:
            vals = [s[k] for s in per_seed_means[method]]
            mean = np.mean(vals)
            std = np.std(vals, ddof=1)
            n = len(vals)
            ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n)) if n > 1 else (mean, mean)
            overall_mean_std[method][k] = (mean, std, ci_low, ci_high)

    # Win count: Agent-C-DQN has lowest blocking_rate per seed
    win_count = 0
    for i in range(len(seed_results)):
        dqn_blk = per_seed_means["Agent-C-DQN"][i]["blocking_rate"]
        others_blk = [per_seed_means[m][i]["blocking_rate"] for m in methods if m != "Agent-C-DQN"]
        if dqn_blk <= min(others_blk):
            win_count += 1

    # Aggregate counters across ALL episodes of ALL seeds
    aggregated = {m: {"reasons": Counter(), "splits": Counter(),
                      "servers": Counter(), "fs_list": []}
                  for m in methods}
    for single_seed in seed_results:
        for method in methods:
            for r in single_seed[method]:
                aggregated[method]["reasons"].update(r["reason_counter"])
                aggregated[method]["splits"].update(r["split_counter"])
                aggregated[method]["servers"].update(r["server_counter"])
                aggregated[method]["fs_list"].extend(r["fs_list"])

    return per_seed_means, per_seed_episode_blocks, overall_mean_std, win_count, aggregated


def _print_single_seed_results(all_results: Dict[str, list], args):
    """Print results for a single seed (backward-compatible format)."""
    print("\n" + "=" * 100)
    print("Agent-C Evaluation Results (fixed KSP-BF RMSA)")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}, "
          f"Topology: {args.topology}, Slots: {args.num_slots}")
    print(f"Slot BW: {args.slot_bw_hz/1e9:.2f} GHz, Guard: {args.guard_band_fs}, Splits: {args.num_splits}")
    print("=" * 100)
    header = (f"{'Method':<18} {'BlkRate':>8} {'SuccRate':>9} {'AvgRwd':>8} "
              f"{'Delay(ms)':>10} {'Waste':>7} {'AvgFS':>7}")
    print(header)
    print("-" * 100)

    for method, results in all_results.items():
        avg = {k: np.mean([r[k] for r in results]) for k in results[0].keys()
               if k not in ("fs_list", "reason_counter", "split_counter", "server_counter")}
        print(f"{method:<18} {avg['blocking_rate']:>8.3f} {avg['success_rate']:>9.3f} "
              f"{avg['avg_reward']:>8.3f} {avg['avg_delay_ms']:>10.2f} "
              f"{avg['avg_block_waste']:>7.3f} {avg['avg_fs']:>7.2f}")

    print("-" * 100)

    # Diagnostic section
    print("\nDiagnostics")
    print("-" * 100)
    for method, results in all_results.items():
        all_fs = []
        all_reasons = Counter()
        all_splits = Counter()
        all_servers = Counter()
        no_valid_total = 0.0
        for r in results:
            all_fs.extend(r["fs_list"])
            all_reasons.update(r["reason_counter"])
            all_splits.update(r["split_counter"])
            all_servers.update(r["server_counter"])
            no_valid_total += r["no_valid_action_rate"]

        avg_no_valid = no_valid_total / len(results)
        n_total = args.episodes * args.requests_per_episode
        print(f"\n{method}:")
        print(f"  AvgFS (success only): {np.mean(all_fs):.2f}" if all_fs else "  AvgFS: N/A")
        print(f"  FS histogram: {_fs_histogram(all_fs)}")
        print(f"  No-valid-action rate: {avg_no_valid:.3f}")
        print(f"  Split choices: {_distribution_str(all_splits, n_total)}")
        print(f"  Server choices: {_distribution_str(all_servers, n_total)}")
        if all_reasons:
            total_failures = sum(all_reasons.values())
            print(f"  Failure reasons (out of {total_failures} failures):")
            for reason, count in all_reasons.most_common(6):
                print(f"    {reason}: {count} ({count/total_failures:.1%})")

    print("=" * 100)


def _print_multiseed_results(overall_mean_std, per_seed_episode_blocks, win_count, aggregated, seeds, args):
    """Print aggregated multi-seed results with 95% CI and significance tests."""
    methods = ["Agent-C-DQN", "Compute-Greedy", "Spectrum-Greedy", "Random-valid-C"]

    print("\n" + "=" * 120)
    print("Agent-C Multi-Seed Evaluation Results (fixed KSP-BF RMSA)")
    print(f"Seeds: {seeds} | Episodes/seed: {args.episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"Topology: {args.topology}, Slots: {args.num_slots}, "
          f"Slot BW: {args.slot_bw_hz/1e9:.2f} GHz, Guard: {args.guard_band_fs}, Splits: {args.num_splits}")
    print("=" * 120)

    # Main metrics table with 95% CI
    header = (f"{'Method':<18} {'Blocking':>14} {'Success':>14} {'Reward':>14} "
              f"{'Delay(ms)':>14} {'Waste':>14} {'AvgFS':>14}")
    print(header)
    print("-" * 120)

    for method in methods:
        m = overall_mean_std[method]
        def _fmt(metric):
            mean, std, ci_l, ci_h = m[metric]
            return f"{mean:.3f} [{ci_l:.3f},{ci_h:.3f}]"
        blk = _fmt("blocking_rate")
        succ = _fmt("success_rate")
        rwd = f"{m['avg_reward'][0]:+.3f} [{m['avg_reward'][2]:+.3f},{m['avg_reward'][3]:+.3f}]"
        dly = f"{m['avg_delay_ms'][0]:.1f} [{m['avg_delay_ms'][2]:.1f},{m['avg_delay_ms'][3]:.1f}]"
        wst = f"{m['avg_block_waste'][0]:.3f} [{m['avg_block_waste'][2]:.3f},{m['avg_block_waste'][3]:.3f}]"
        afs = f"{m['avg_fs'][0]:.2f} [{m['avg_fs'][2]:.2f},{m['avg_fs'][3]:.2f}]"
        print(f"{method:<18} {blk:>14} {succ:>14} {rwd:>14} {dly:>14} {wst:>14} {afs:>14}")

    print("-" * 120)
    print(f"\nAgent-C-DQN win count (lowest blocking per seed): {win_count} / {len(seeds)}")

    # Significance tests: Agent-C-DQN vs each baseline
    # (A) Episode-level paired test (high power, large N)
    print("\nSignificance Tests (Episode-level paired, high power)")
    print("-" * 120)
    dqn_blocks = np.concatenate(per_seed_episode_blocks["Agent-C-DQN"])
    for baseline in ["Spectrum-Greedy", "Random-valid-C", "Compute-Greedy"]:
        base_blocks = np.concatenate(per_seed_episode_blocks[baseline])
        t_stat, t_p = stats.ttest_rel(dqn_blocks, base_blocks)
        try:
            w_stat, w_p = stats.wilcoxon(dqn_blocks, base_blocks)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        diff = dqn_blocks - base_blocks
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        print(f"  vs {baseline:<16} t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}, Cohen's d={cohens_d:+.3f}")

    # (B) Seed-level paired test (strict, independent samples = seeds)
    print("\nSignificance Tests (Seed-level paired, strict)")
    print("-" * 120)
    dqn_seed_blocks = [np.mean(ep_blocks) for ep_blocks in per_seed_episode_blocks["Agent-C-DQN"]]
    for baseline in ["Spectrum-Greedy", "Random-valid-C", "Compute-Greedy"]:
        base_seed_blocks = [np.mean(ep_blocks) for ep_blocks in per_seed_episode_blocks[baseline]]
        t_stat, t_p = stats.ttest_rel(dqn_seed_blocks, base_seed_blocks)
        try:
            w_stat, w_p = stats.wilcoxon(dqn_seed_blocks, base_seed_blocks)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        diff = np.array(dqn_seed_blocks) - np.array(base_seed_blocks)
        cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        print(f"  vs {baseline:<16} t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}, Cohen's d={cohens_d:+.3f}")
    print("-" * 120)

    # Failure reason aggregation
    print("\nFailure Reason Distribution (aggregated across all seeds)")
    print("-" * 120)
    for method in methods:
        reasons = aggregated[method]["reasons"]
        total = sum(reasons.values())
        if total == 0:
            print(f"{method:<18} No failures")
            continue
        parts = []
        for r in ["server_overload", "no_suitable_block", "fs_too_large", "deadline_infeasible"]:
            cnt = reasons.get(r, 0)
            parts.append(f"{r}={cnt} ({cnt/total:.1%})")
        print(f"{method:<18} {', '.join(parts)}")

    # Split / server mean distribution
    print("\nSplit / Server Choice Distribution (mean across all seeds)")
    print("-" * 120)
    n_total = len(seeds) * args.episodes * args.requests_per_episode
    header2 = f"{'Method':<18} {'split0':>8} {'split1':>8} {'split2':>8} {'server0':>8} {'server1':>8}"
    print(header2)
    print("-" * 120)
    for method in methods:
        splits = aggregated[method]["splits"]
        servers = aggregated[method]["servers"]
        s0 = splits.get(0, 0) / n_total
        s1 = splits.get(1, 0) / n_total
        s2 = splits.get(2, 0) / n_total
        sv0 = servers.get(0, 0) / n_total
        sv1 = servers.get(1, 0) / n_total
        print(f"{method:<18} {s0:>8.1%} {s1:>8.1%} {s2:>8.1%} {sv0:>8.1%} {sv1:>8.1%}")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent-C vs baselines")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Single seed (used when --seeds is not provided)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds for multi-seed evaluation, e.g. 42,123,456")
    parser.add_argument("--topology", type=str, default="net1")
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Agent-C checkpoint. If omitted, uses "
                             "<package_root>/checkpoints/agent_c_best.pt")
    parser.add_argument("--ablation", action="store_true",
                        help="Use ablation model (no spectrum summary, input_dim=7)")
    parser.add_argument("--zero_spectrum", action="store_true",
                        help="Zero-spectrum ablation (keep 17-dim input but set spectrum summary to 0)")
    args = parser.parse_args()

    env = make_env(
        args.topology, args.num_slots, args.num_servers, args.seed,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )
    device = 'cpu'

    # Load Agent-C
    agent = AgentC(
        input_dim=17,
        hidden_dims=(128, 64),
        gamma=0.95,
        epsilon=0.0,
        device=device,
        ablation=args.ablation,
        zero_spectrum=args.zero_spectrum,
    )

    # Resolve checkpoint path
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        package_root = Path(__file__).resolve().parents[2]
        ckpt_dir = package_root / "checkpoints"
        # Priority: best > last > smoke (legacy)
        best_path = ckpt_dir / "agent_c_best.pt"
        last_path = ckpt_dir / "agent_c_last.pt"
        smoke_path = ckpt_dir / "agent_c_smoke.pt"
        if best_path.exists():
            ckpt_path = best_path
        elif last_path.exists():
            ckpt_path = last_path
        else:
            ckpt_path = smoke_path

    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        agent.q_net.load_state_dict(ckpt["model_state"])
        agent.target_net.load_state_dict(ckpt["target_state"])
        print(f"Loaded checkpoint from {ckpt_path}")
        if "best_eval_reward" in ckpt:
            print(f"  (checkpoint best_reward={ckpt['best_eval_reward']:+.3f} @ ep={ckpt.get('best_episode', '?')})")
    else:
        print("WARNING: No checkpoint found, using random-init Agent-C")

    # Determine seed list
    if args.seeds is not None:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seed_list = [args.seed]

    methods = ["Agent-C-DQN", "Compute-Greedy", "Spectrum-Greedy", "Random-valid-C"]

    if len(seed_list) == 1:
        # Single-seed mode (backward compatible)
        args.seed = seed_list[0]
        all_results = _run_single_seed(env, agent, args)
        _print_single_seed_results(all_results, args)
    else:
        # Multi-seed mode
        seed_results = []
        for s in seed_list:
            args.seed = s
            single = _run_single_seed(env, agent, args)
            seed_results.append(single)

        per_seed_means, per_seed_episode_blocks, overall_mean_std, win_count, aggregated = _aggregate_across_seeds(
            seed_results, methods
        )
        _print_multiseed_results(overall_mean_std, per_seed_episode_blocks, win_count, aggregated, seed_list, args)


if __name__ == "__main__":
    main()
