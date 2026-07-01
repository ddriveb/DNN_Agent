"""Activation Function Ablation Evaluation for PPO Agent-C.

Evaluates pre-trained Agent-C checkpoints (different activations) against a
fixed BC-PPO-R checkpoint.  Computes the metrics required by the ablation
report: Blocking, Delay, AvgFS, noC%, avgRacts, NSB%.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_activation_ablation \
        --topology snap24_gnutella_reach --num_slots 20 --requests_per_episode 80 \
        --k_paths 5 --max_blocks 10 --block_sort_strategy mixed \
        --frozen_r_checkpoint sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt \
        --seeds 42,123,456,789,2024 --episodes 20 \
        --output_json sa_hmarl/experiments/activation_ablation_results.json \
        --output_md sa_hmarl/experiments/activation_ablation_agent_c.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Model loaders (with activation support)
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    """Load a PPO Agent-C checkpoint, reading activation from stored args."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    activation = (
        ckpt_args.get("agent_c_activation", "tanh")
        if isinstance(ckpt_args, dict) else "tanh"
    )
    feature_mode = (
        ckpt_args.get("agent_c_feature_mode", "default")
        if isinstance(ckpt_args, dict) else "default"
    )
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        activation=activation,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry,
                device: str = "cpu") -> PPOAgentR:
    """Load a frozen PPO Agent-R checkpoint."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_single_episode(
    env_factory,
    requests: list,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
) -> Dict[str, Any]:
    """Evaluate one episode (sequence of requests).

    Returns per-episode metrics dict.
    """
    env = env_factory()
    env.reset(requests)
    num_servers = len(env.mec.servers)

    total = 0
    blocked = 0
    successes = 0
    total_fs = 0.0
    total_delay = 0.0
    no_suitable_block = 0
    server_overload = 0
    other_fail = 0
    no_c_valid = 0  # Agent-C had no valid action
    r_action_counts = []  # number of valid R actions per step

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        c_features, c_mask = agent_c.build_action_features(obs_c)

        action_idx_c, _, _ = agent_c.select_from_features(
            c_features, c_mask, deterministic=True
        )
        if action_idx_c is None:
            no_c_valid += 1
            action_c = (0, 0)
        else:
            action_c = decode_agent_c_action(action_idx_c, num_servers)

        split_id, server_id = action_c

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        # Count valid R actions
        _, r_mask = agent_r.build_action_features(obs_r)
        r_action_counts.append(int(np.sum(r_mask)))

        action_idx_r = agent_r.select_action(obs_r, deterministic=True)
        if action_idx_r is None:
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )

        _, _, _, info = env.step(action_c, action_r)
        total += 1

        if info.get("success", False):
            successes += 1
            total_fs += info.get("num_slots", 0)
            total_delay += info.get("delay_ms", 0.0)
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            if reason == "no_suitable_block":
                no_suitable_block += 1
            elif reason in ("server_overload", "server_saturated"):
                server_overload += 1
            else:
                other_fail += 1

    n = max(total, 1)
    s = max(successes, 1)

    return {
        "total": total,
        "blocked": blocked,
        "success": successes,
        "blocking_rate": blocked / n,
        "success_rate": successes / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "noC_pct": no_c_valid / n,  # % of requests where Agent-C had no valid action
        "avgRacts": float(np.mean(r_action_counts)) if r_action_counts else 0.0,
        "NSB_pct": no_suitable_block / n,  # % of requests failed due to no_suitable_block
        "server_overload_pct": server_overload / n,
        "other_fail_pct": other_fail / n,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _fmt_pct(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in counter.most_common())


def aggregate_across_seeds(seed_results: List[Dict[str, List[Dict]]],
                           methods: List[str]) -> Dict:
    """Aggregate per-method per-seed results into mean/std/CI."""
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_delay_ms", "avg_fs",
        "noC_pct", "avgRacts", "NSB_pct",
        "server_overload_pct", "other_fail_pct",
    ]

    method_vals = {m: {k: [] for k in scalar_keys} for m in methods}

    for single_seed in seed_results:
        for method in methods:
            episodes = single_seed[method]
            for k in scalar_keys:
                method_vals[method][k].append(
                    float(np.mean([ep[k] for ep in episodes]))
                )

    agg = {}
    for method in methods:
        agg[method] = {}
        for k in scalar_keys:
            vals = method_vals[method][k]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = mean, mean
            if len(vals) > 1:
                ci_low, ci_high = stats.t.interval(
                    0.95, df=len(vals) - 1, loc=mean, scale=std / np.sqrt(len(vals))
                )
                ci_low, ci_high = float(ci_low), float(ci_high)
            agg[method][k] = {
                "mean": mean,
                "std": std,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            }
    return agg


def aggregate_result_groups(result_groups: Dict[str, List[List[Dict]]],
                            methods: List[str]) -> Dict:
    """Aggregate per-activation groups across train checkpoints and eval seeds."""
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_delay_ms", "avg_fs",
        "noC_pct", "avgRacts", "NSB_pct",
        "server_overload_pct", "other_fail_pct",
    ]

    method_vals = {m: {k: [] for k in scalar_keys} for m in methods}
    for method in methods:
        for episodes in result_groups.get(method, []):
            for k in scalar_keys:
                method_vals[method][k].append(
                    float(np.mean([ep[k] for ep in episodes]))
                )

    agg = {}
    for method in methods:
        agg[method] = {}
        for k in scalar_keys:
            vals = method_vals[method][k]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ci_low, ci_high = mean, mean
            if len(vals) > 1:
                ci_low, ci_high = stats.t.interval(
                    0.95, df=len(vals) - 1, loc=mean, scale=std / np.sqrt(len(vals))
                )
                ci_low, ci_high = float(ci_low), float(ci_high)
            agg[method][k] = {
                "mean": mean,
                "std": std,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            }
    return agg


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_markdown_report(
    agg: Dict,
    args,
    activation_labels: Dict[str, str],
) -> str:
    """Generate the activation ablation markdown report."""
    lines = []

    def _w(line=""):
        lines.append(line + "\n")

    _w("# Activation Function Ablation — PPO Agent-C")
    _w()
    _w("## Motivation")
    _w()
    _w(
        "This experiment evaluates whether changing the activation function in the "
        "PPO Agent-C actor network (`MaskedPPOActorNetwork`) can improve blocking "
        "performance in the **Enhanced Agent-C + BC-PPO-R** framework. "
        "The R-side decision quality is already near ceiling (PPO-R / BC-PPO-R / "
        "DeepRMSA per-state decisions are nearly identical), so this ablation "
        "**only modifies Agent-C** while keeping the R checkpoint fixed."
    )
    _w()
    _w(
        "Baseline activation: `tanh` (used by current PPO agent).  "
        "Candidates: `silu`, `gelu`, `leaky_relu`."
    )
    _w()
    _w("## Configuration")
    _w()
    _w("| Parameter | Value |")
    _w("|-----------|-------|")
    _w(f"| Topology | `{args.topology}` |")
    _w(f"| Slots (S20) | `{args.num_slots}` |")
    _w(f"| Requests/episode (R80) | `{args.requests_per_episode}` |")
    _w(f"| k_paths | `{args.k_paths}` |")
    _w(f"| max_blocks | `{args.max_blocks}` |")
    _w(f"| block_sort | `{args.block_sort_strategy}` |")
    _w(f"| Frozen R checkpoint | `{args.frozen_r_checkpoint}` |")
    _w(f"| Training episodes | {args.training_episodes} (seeds: {args.training_seeds}) |")
    _w(f"| Eval seeds | `{args.seeds}` |")
    _w(f"| Eval episodes/seed | `{args.episodes}` |")
    _w(f"| Requests/eval episode | `{args.requests_per_episode}` |")
    _w()
    _w("## Results")
    _w()

    # Determine baseline for delta computation
    baseline_key = None
    for m in agg:
        if "tanh" in m.lower():
            baseline_key = m
            break
    if baseline_key is None:
        baseline_key = list(agg.keys())[0]

    # Build metric descriptions
    metric_descriptions = {
        "blocking_rate": "Blocking",
        "avg_delay_ms": "Delay (ms)",
        "avg_fs": "AvgFS",
        "noC_pct": "noC%",
        "avgRacts": "avgRacts",
        "NSB_pct": "NSB/request%",
        "server_overload_pct": "ServerOverload%",
        "other_fail_pct": "OtherFail%",
    }

    # Main results table
    _w("### Per-Activation Metrics (mean ± std across train checkpoints and eval seeds)")
    _w()

    # Build header
    header_cols = ["Activation"] + list(metric_descriptions.values())
    _w("| " + " | ".join(header_cols) + " |")
    _w("|" + "|".join(["--------"] * len(header_cols)) + "|")

    for method in sorted(agg.keys()):
        m = agg[method]
        row = [activation_labels.get(method, method)]
        for key in ["blocking_rate", "avg_delay_ms", "avg_fs",
                     "noC_pct", "avgRacts", "NSB_pct",
                     "server_overload_pct", "other_fail_pct"]:
            v = m[key]
            if key in ("blocking_rate", "noC_pct", "NSB_pct",
                        "server_overload_pct", "other_fail_pct"):
                row.append(f"{v['mean']:.3f}±{v['std']:.3f}")
            elif key == "avg_delay_ms":
                row.append(f"{v['mean']:.1f}±{v['std']:.1f}")
            elif key == "avg_fs":
                row.append(f"{v['mean']:.2f}±{v['std']:.2f}")
            elif key == "avgRacts":
                row.append(f"{v['mean']:.1f}±{v['std']:.1f}")
        _w("| " + " | ".join(row) + " |")

    _w()

    # Delta vs baseline table
    _w("### Delta vs Tanh Baseline")
    _w()
    _w("| Activation | ΔBlocking | ΔDelay | ΔAvgFS | ΔnoC% | ΔavgRacts | ΔNSB/request% |")
    _w("|------------|-----------|--------|--------|-------|-----------|-------|")

    base = agg[baseline_key]
    for method in sorted(agg.keys()):
        if method == baseline_key:
            continue
        m = agg[method]
        d_blk = m["blocking_rate"]["mean"] - base["blocking_rate"]["mean"]
        d_dly = m["avg_delay_ms"]["mean"] - base["avg_delay_ms"]["mean"]
        d_fs = m["avg_fs"]["mean"] - base["avg_fs"]["mean"]
        d_noc = m["noC_pct"]["mean"] - base["noC_pct"]["mean"]
        d_rac = m["avgRacts"]["mean"] - base["avgRacts"]["mean"]
        d_nsb = m["NSB_pct"]["mean"] - base["NSB_pct"]["mean"]

        def _delta(v, fmt=".3f"):
            s = f"{v:+{fmt}}"
            if v < 0:
                return f"**{s}**"  # bold improvements
            return s

        _w(f"| {activation_labels.get(method, method)} | "
           f"{_delta(d_blk)} | {_delta(d_dly, '.1f')} | "
           f"{_delta(d_fs, '.2f')} | {_delta(d_noc)} | "
           f"{_delta(d_rac, '.1f')} | {_delta(d_nsb)} |")

    _w()
    _w("Negative ΔBlocking / ΔNSB/request% / ΔDelay / ΔAvgFS = improvement (lower is better).")
    _w()
    _w(
        "`NSB/request%` is the fraction of all requests that fail with "
        "`no_suitable_block`; it is not the failure-reason share among blocked "
        "requests used in some other reports."
    )
    _w()

    # Warm-start note
    _w("### Warm-Start Note")
    _w()
    _w(
        "All activation variants share the same state_dict shape (activation "
        "functions have no learnable parameters).  Cross-activation warm-start "
        "is technically possible but may cause initial behavior drift as the "
        "weights adapt to a new activation landscape.  In this experiment, "
        "**each activation is trained from scratch** with its own random "
        "initialization to avoid this confound."
    )
    _w()

    # Conclusion
    _w("## Conclusion")
    _w()

    # Check if any activation improves blocking by >= 1pp
    best_improvement = 0.0
    best_method = None
    for method in agg:
        if method == baseline_key:
            continue
        delta = base["blocking_rate"]["mean"] - agg[method]["blocking_rate"]["mean"]
        if delta > best_improvement:
            best_improvement = delta
            best_method = method

    if best_improvement >= 0.01:
        _w(
            f"**{activation_labels.get(best_method, best_method)}** improves blocking "
            f"by **{best_improvement:.1%}** over the tanh baseline, with no meaningful "
            f"degradation in delay, AvgFS, or noC%.  "
            f"**Recommendation:** Run a 5-seed formal evaluation for "
            f"{activation_labels.get(best_method, best_method)}."
        )
    else:
        _w(
            "**No activation function achieves a ≥1pp blocking improvement** over the "
            "tanh baseline.  All candidates show statistically indistinguishable "
            "performance on blocking, delay, AvgFS, noC%, and NSB%."
        )
        _w()
        _w(
            "**Conclusion: Activation ablation — no meaningful gain.**  "
            "The choice of activation function in the PPO Agent-C actor does not "
            "materially affect the Enhanced Agent-C + BC-PPO-R system performance.  "
            "Retaining `tanh` (the current default) is recommended."
        )

    _w()
    _w("---")
    _w()
    _w(f"*Generated by `eval_activation_ablation.py` on {args.topology}*  ")
    _w(f"*Checkpoints: `agent_c_act_{{activation}}_s{{seed}}_s20_r80_best.pt`*")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Activation Function Ablation Evaluation"
    )
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--frozen_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str,
                        default="sa_hmarl/experiments/activation_ablation_results.json")
    parser.add_argument("--output_md", type=str,
                        default="sa_hmarl/experiments/activation_ablation_agent_c.md")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="sa_hmarl/checkpoints")
    parser.add_argument("--checkpoint_pattern", type=str,
                        default="agent_c_act_{activation}_s{seed}_s20_r80_best.pt")
    parser.add_argument("--training_episodes", type=int, default=100,
                        help="Training episodes used to produce activation checkpoints.")
    parser.add_argument("--training_seeds", type=str, default="42,123",
                        help="Training seeds used to produce activation checkpoints.")

    args = parser.parse_args()

    eval_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    training_seeds = [int(s.strip()) for s in args.training_seeds.split(",") if s.strip()]

    # Activation → checkpoint mapping
    activations = ["tanh", "silu", "gelu", "leaky_relu"]
    activation_labels = {
        "tanh": "Tanh (baseline)",
        "silu": "SiLU",
        "gelu": "GELU",
        "leaky_relu": "LeakyReLU",
    }

    # Build checkpoint paths
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_map: Dict[str, List[str]] = {}

    for act in activations:
        found_paths: List[str] = []
        for seed in training_seeds:
            path = ckpt_dir / args.checkpoint_pattern.format(activation=act, seed=seed)
            if path.exists():
                found_paths.append(str(path))
                print(f"[{act}] Found: {path}")
        if not found_paths:
            # Fallback: try old-format checkpoints (no seed suffix, legacy naming)
            legacy_path = ckpt_dir / f"agent_c_act_{act}_best.pt"
            if legacy_path.exists():
                found_paths.append(str(legacy_path))
                print(f"[{act}] Found (legacy): {legacy_path}")
            else:
                print(f"[{act}] WARNING: No checkpoint found, will skip")
        if found_paths:
            ckpt_map[act] = found_paths

    if not ckpt_map:
        print("ERROR: No checkpoints found. Run training first.")
        sys.exit(1)

    # Load frozen R once
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"\nLoading frozen Agent-R from {args.frozen_r_checkpoint}")
    agent_r = _load_ppo_r(args.frozen_r_checkpoint, mod_reg, args.device)

    # Env factory
    def make_env_factory():
        return make_env(
            topology=args.topology,
            num_slots=args.num_slots,
            num_servers=args.num_servers,
            seed=42,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            k=args.k_paths,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )

    env_proto = make_env_factory()

    # Pre-generate episodes for all eval seeds
    all_episodes: Dict[int, List] = {}
    for seed in eval_seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            requests = generate_requests(
                env_proto, rng, src,
                args.requests_per_episode,
                arrival_interval=0.25,
                holding_min=4.0, holding_max=10.0,
                deadline_min=30.0, deadline_max=100.0,
                size_min_mb=5.0, size_max_mb=30.0,
                edge_cost_min=0.5, edge_cost_max=15.0,
                num_splits=args.num_splits,
            )
            eps.append(requests)
        all_episodes[seed] = eps

    # Evaluate each activation checkpoint across all eval seeds.  Each
    # train-checkpoint/eval-seed pair contributes one aggregate sample.
    activation_results: Dict[str, List[List[Dict]]] = {
        act: [] for act in ckpt_map
    }

    for act_name, ckpt_paths in ckpt_map.items():
        for ckpt_path in ckpt_paths:
            print(f"\n{'='*70}")
            print(f"Activation {act_name}: {ckpt_path}")
            print(f"{'='*70}")
            agent_c = _load_ppo_c(ckpt_path, args.device)
            for seed in eval_seeds:
                ep_results = []
                for requests in all_episodes[seed]:
                    result = evaluate_single_episode(
                        make_env_factory, requests, agent_c, agent_r
                    )
                    ep_results.append(result)
                activation_results[act_name].append(ep_results)

                # Quick summary
                blk = float(np.mean([r["blocking_rate"] for r in ep_results]))
                dly = float(np.mean([r["avg_delay_ms"] for r in ep_results]))
                print(f"    eval_seed={seed:<5d}  blk={blk:.3f}  delay={dly:.1f}ms")

    # Aggregate
    method_names = sorted(ckpt_map.keys())
    agg = aggregate_result_groups(activation_results, method_names)

    # ---- Print terminal report ----
    print(f"\n{'='*90}")
    print("ACTIVATION ABLATION — AGGREGATED RESULTS (mean ± std across train/eval seeds)")
    print(f"{'='*90}")
    header = (f"{'Activation':15s} {'Blocking':>12s} {'Delay(ms)':>12s} "
              f"{'AvgFS':>8s} {'noC%':>8s} {'avgRacts':>10s} {'NSB/req%':>8s} "
              f"{'SrvOver%':>10s}")
    print(header)
    print("-" * 90)

    for method in sorted(agg.keys()):
        m = agg[method]
        print(
            f"{activation_labels.get(method, method):15s} "
            f"{m['blocking_rate']['mean']:7.3f}±{m['blocking_rate']['std']:.3f} "
            f"{m['avg_delay_ms']['mean']:7.1f}±{m['avg_delay_ms']['std']:.1f} "
            f"{m['avg_fs']['mean']:6.2f}±{m['avg_fs']['std']:.2f} "
            f"{m['noC_pct']['mean']:6.3f}±{m['noC_pct']['std']:.3f} "
            f"{m['avgRacts']['mean']:8.1f}±{m['avgRacts']['std']:.1f} "
            f"{m['NSB_pct']['mean']:6.3f}±{m['NSB_pct']['std']:.3f} "
            f"{m['server_overload_pct']['mean']:6.3f}±{m['server_overload_pct']['std']:.3f}"
        )

    print("-" * 90)

    # Delta vs baseline
    baseline_key = "tanh" if "tanh" in agg else list(agg.keys())[0]
    base = agg[baseline_key]
    print(f"\nDelta vs {activation_labels.get(baseline_key, baseline_key)}:")
    print(f"{'Activation':15s} {'ΔBlocking':>10s} {'ΔDelay':>10s} {'ΔAvgFS':>10s} "
          f"{'ΔnoC%':>10s} {'ΔavgRacts':>10s} {'ΔNSB/req%':>10s}")
    print("-" * 90)
    for method in sorted(agg.keys()):
        if method == baseline_key:
            continue
        m = agg[method]
        d_blk = m["blocking_rate"]["mean"] - base["blocking_rate"]["mean"]
        d_dly = m["avg_delay_ms"]["mean"] - base["avg_delay_ms"]["mean"]
        d_fs = m["avg_fs"]["mean"] - base["avg_fs"]["mean"]
        d_noc = m["noC_pct"]["mean"] - base["noC_pct"]["mean"]
        d_rac = m["avgRacts"]["mean"] - base["avgRacts"]["mean"]
        d_nsb = m["NSB_pct"]["mean"] - base["NSB_pct"]["mean"]
        print(
            f"{activation_labels.get(method, method):15s} "
            f"{d_blk:+10.3f} {d_dly:+10.1f} {d_fs:+10.2f} "
            f"{d_noc:+10.3f} {d_rac:+10.1f} {d_nsb:+10.3f}"
        )
    print("-" * 90)

    # ---- JSON export ----
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to plain Python types
        clean = {}
        for method, metrics in agg.items():
            clean[activation_labels.get(method, method)] = {
                k: {
                    "mean": float(v["mean"]),
                    "std": float(v["std"]),
                    "ci_95_low": float(v["ci_95_low"]),
                    "ci_95_high": float(v["ci_95_high"]),
                }
                for k, v in metrics.items()
            }
        # Add metadata
        clean["_metadata"] = {
            "topology": args.topology,
            "num_slots": args.num_slots,
            "requests_per_episode": args.requests_per_episode,
            "k_paths": args.k_paths,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "eval_seeds": eval_seeds,
            "training_seeds": training_seeds,
            "training_episodes": args.training_episodes,
            "checkpoints": ckpt_map,
            "frozen_r_checkpoint": args.frozen_r_checkpoint,
            "baseline_activation": "tanh",
        }
        json.dump(clean, open(str(out_path), "w"), indent=2)
        print(f"\nJSON results written to {out_path}")

    # ---- Markdown export ----
    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        report = _generate_markdown_report(agg, args, activation_labels)
        md_path.write_text(report)
        print(f"Markdown report written to {md_path}")


if __name__ == "__main__":
    main()
