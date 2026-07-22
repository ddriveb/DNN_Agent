"""Evaluate native DeepRMSA adapters on the locked PAPER_PRIMARY_K50 traces."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.evaluation.optical_only_rmsa_env import OpticalOnlyRMSAEnv
from sa_hmarl.evaluation.optical_only_rmsa_evaluator import _generate_trace, _hash_requests
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_deeprmsa_optical_k50 import PROTOCOL_ID, protocol_config


DEFAULT_BASELINE = Path("sa_hmarl/experiments/v135_optical_only_rmsa_audit/pilot.json")
DEFAULT_OUTPUT = Path("sa_hmarl/experiments/deeprmsa_adapted_paper_primary_k50")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_agent(path: Path, device: str) -> tuple[DeepRMSAAgent, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Checkpoint is not {PROTOCOL_ID}: {path}")
    expected = protocol_config()
    if ckpt.get("protocol_config") != expected:
        raise ValueError("Checkpoint protocol config does not match the locked evaluator")
    training_args = ckpt.get("training_args", {})
    agent = DeepRMSAAgent(
        num_nodes=int(ckpt["num_nodes"]),
        num_slots=int(ckpt["num_slots"]),
        k_path=int(ckpt["k_path"]),
        m_blocks=int(ckpt["m_blocks"]),
        mod_registry=ModulationRegistry.from_profile("default"),
        gamma=float(ckpt.get("gamma", 0.95)),
        lr=float(training_args.get("lr", 1e-5)),
        entropy_coef=float(training_args.get("entropy_coef", 0.01)),
        value_loss_coef=float(training_args.get("value_loss_coef", 0.5)),
        max_grad_norm=float(training_args.get("max_grad_norm", 40.0)),
        num_layers=5,
        layer_size=128,
        device=device,
    )
    agent.load_state_dict(ckpt)
    agent.eval()
    return agent, ckpt


def evaluate_agent(
    agent: DeepRMSAAgent,
    seed: int,
    warmup: int,
    evaluated: int,
    arrival_interval: float,
) -> Dict[str, Any]:
    config = protocol_config()
    requests = _generate_trace(seed, arrival_interval, warmup, evaluated, config)
    env = OpticalOnlyRMSAEnv(
        topology=config["topology"], num_slots=100, k_paths=50, max_blocks=10,
        path_sort_strategy="hops", block_sort_strategy="start_asc",
        mod_registry=ModulationRegistry.from_profile("default"),
        slot_bw_hz=12.5e9, guard_band_fs=1, seed=seed,
    )
    env.reset(requests)
    admitted = blocked = no_valid = invalid_execution = 0
    fs_sum = slot_hops = path_km = hops = 0.0

    for idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs = env.build_observation(req)
        action = agent.select_action(obs)
        measured = idx >= warmup
        if action is None:
            if measured:
                blocked += 1
                no_valid += 1
            continue
        step = env.step(action, obs, req.holding_time)
        if not measured:
            continue
        if step["success"]:
            admitted += 1
            fs_sum += step["required_fs"]
            slot_hops += step["required_fs"] * step["hop_count"]
            path_km += step["path_length_km"]
            hops += step["hop_count"]
        else:
            blocked += 1
            invalid_execution += 1

    max_release = max(r.arrival_time + r.holding_time for r in requests)
    env.advance_time(max_release + 1.0)
    total = admitted + blocked
    return {
        "seed": seed,
        "evaluated_requests": total,
        "admitted": admitted,
        "blocked": blocked,
        "blocking_rate": blocked / max(total, 1),
        "r_no_valid_action": no_valid,
        "invalid_execution": invalid_execution,
        "avg_fs": fs_sum / max(admitted, 1),
        "avg_slot_hops": slot_hops / max(admitted, 1),
        "avg_path_length_km": path_km / max(admitted, 1),
        "avg_hop_count": hops / max(admitted, 1),
        "request_trace_hash": _hash_requests(requests),
        "final_active_connections": len(env.active_connections),
        "final_utilization": env.get_utilization(),
    }


def paired_stats(baseline: List[float], method: List[float]) -> Dict[str, Any]:
    # Positive differences mean that the adapted DeepRMSA method is better.
    diff = np.asarray(baseline, dtype=float) - np.asarray(method, dtype=float)
    rng = np.random.RandomState(13550)
    indices = rng.randint(0, len(diff), size=(20000, len(diff)))
    means = diff[indices].mean(axis=1)
    observed = float(diff.mean())
    sign_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(diff)):
        sign_means.append(float(np.mean(diff * np.asarray(signs))))
    permutation_p = (sum(x >= observed - 1e-15 for x in sign_means) + 1) / (len(sign_means) + 1)
    return {
        "mean_baseline_minus_method": observed,
        "mean_difference_pp": 100.0 * observed,
        "bootstrap_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "paired_sign_permutation_p_one_sided": float(permutation_p),
        "wins_ties_losses": {
            "wins": int(np.sum(diff > 0)),
            "ties": int(np.sum(diff == 0)),
            "losses": int(np.sum(diff < 0)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--baseline_json", default=str(DEFAULT_BASELINE))
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--evaluated", type=int, default=10000)
    parser.add_argument("--arrival_interval", type=float, default=0.025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.labels):
        parser.error("--checkpoints and --labels must have equal lengths")

    baseline_payload = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    if baseline_payload["config"] != protocol_config():
        raise ValueError("Existing KSP/Strict pilot config is not the locked protocol")
    baseline_results = baseline_payload["results"]
    seeds = [int(x) for x in baseline_results["ksp_ff_highest"].keys()]
    output: Dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_config": protocol_config(),
        "seeds": seeds,
        "warmup": args.warmup,
        "evaluated": args.evaluated,
        "arrival_interval": args.arrival_interval,
        "baseline_json": args.baseline_json,
        "baselines": {},
        "methods": {},
        "comparisons": {},
    }

    for baseline_name in ("ksp_ff_highest", "strict_v13", "ppo_r_top1"):
        rates = [
            baseline_results[baseline_name][str(seed)]["blocking_rate"]
            for seed in seeds
        ]
        output["baselines"][baseline_name] = {
            "mean_blocking_rate": float(np.mean(rates)),
            "std_blocking_rate": float(np.std(rates, ddof=1)),
        }

    for label, checkpoint in zip(args.labels, args.checkpoints):
        path = Path(checkpoint)
        agent, ckpt = load_agent(path, args.device)
        per_seed: Dict[str, Any] = {}
        for seed in seeds:
            row = evaluate_agent(agent, seed, args.warmup, args.evaluated, args.arrival_interval)
            expected_hash = baseline_results["ksp_ff_highest"][str(seed)]["request_trace_hash"]
            if row["request_trace_hash"] != expected_hash:
                raise AssertionError(f"Request trace mismatch for seed {seed}")
            if row["admitted"] + row["blocked"] != args.evaluated:
                raise AssertionError(f"Conservation mismatch for seed {seed}")
            if row["invalid_execution"] != 0 or row["final_utilization"] != 0.0:
                raise AssertionError(f"Mechanical invariant failed for seed {seed}")
            per_seed[str(seed)] = row
            print(f"[{label}] seed={seed} blocking={row['blocking_rate']:.3%}", flush=True)
        rates = [per_seed[str(seed)]["blocking_rate"] for seed in seeds]
        output["methods"][label] = {
            "checkpoint": str(path),
            "checkpoint_sha256": _sha256(path),
            "implementation_label": ckpt.get("implementation_label"),
            "k_path": int(ckpt["k_path"]),
            "m_blocks": int(ckpt["m_blocks"]),
            "training_seed": ckpt.get("training_args", {}).get("seed"),
            "selected_epoch": ckpt.get("epoch"),
            "per_seed": per_seed,
            "mean_blocking_rate": float(np.mean(rates)),
            "std_blocking_rate": float(np.std(rates, ddof=1)),
        }
        for baseline_name in ("ksp_ff_highest", "strict_v13", "ppo_r_top1"):
            baseline_rates = [baseline_results[baseline_name][str(seed)]["blocking_rate"] for seed in seeds]
            output["comparisons"][f"{label}_vs_{baseline_name}"] = paired_stats(baseline_rates, rates)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "DEEPRMSA_K50_RESULTS.json"
    json_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    lines = [
        "# DeepRMSA-Adapted PAPER_PRIMARY_K50 Results", "",
        "> These are native COST239/K=50 PyTorch DeepRMSA-style adapter results, not an original DeepRMSA reproduction.", "",
        "| Method | K | M | Mean blocking | Std |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline_labels = {
        "ksp_ff_highest": "Formal KSP-FF K=50",
        "strict_v13": "Strict v1.3 zero-shot",
        "ppo_r_top1": "PPO-R Top-1 zero-shot",
    }
    for name, label in baseline_labels.items():
        row = output["baselines"][name]
        lines.append(
            f"| {label} | 50 | n/a | {100*row['mean_blocking_rate']:.4f}% | "
            f"{100*row['std_blocking_rate']:.4f}% |"
        )
    for label, row in output["methods"].items():
        lines.append(
            f"| {label} | {row['k_path']} | {row['m_blocks']} | "
            f"{100*row['mean_blocking_rate']:.4f}% | {100*row['std_blocking_rate']:.4f}% |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, row in output["comparisons"].items():
        ci = row["bootstrap_ci95"]
        lines.append(
            f"- `{name}`: baseline-method={row['mean_difference_pp']:.4f} pp, "
            f"95% CI [{100*ci[0]:.4f}, {100*ci[1]:.4f}] pp, "
            f"p={row['paired_sign_permutation_p_one_sided']:.6f}."
        )
    (out_dir / "DEEPRMSA_K50_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
