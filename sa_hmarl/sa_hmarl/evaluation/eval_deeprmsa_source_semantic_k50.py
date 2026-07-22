"""Evaluate the unmasked DeepRMSA source-semantic port on locked K=50 traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from sa_hmarl.agents.deep_rmsa_source_semantic_agent import DeepRMSASourceSemanticAgent
from sa_hmarl.evaluation.eval_deeprmsa_optical_k50 import paired_stats
from sa_hmarl.evaluation.optical_only_rmsa_env import OpticalOnlyRMSAEnv
from sa_hmarl.evaluation.optical_only_rmsa_evaluator import _generate_trace, _hash_requests
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_deeprmsa_optical_k50 import protocol_config
from sa_hmarl.training.train_deeprmsa_source_semantic_k50 import (
    IMPLEMENTATION_LABEL,
    PROTOCOL_ID,
)


DEFAULT_BASELINE = Path("sa_hmarl/experiments/v135_optical_only_rmsa_audit/pilot.json")
DEFAULT_OUTPUT = Path("sa_hmarl/experiments/deeprmsa_source_semantic_paper_primary_k50")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_agent(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Unexpected checkpoint protocol: {checkpoint.get('protocol_id')}")
    if checkpoint.get("protocol_config") != protocol_config():
        raise ValueError("Checkpoint protocol does not match PAPER_PRIMARY_K50")
    if checkpoint.get("implementation_label") != IMPLEMENTATION_LABEL:
        raise ValueError("Checkpoint is not the source-semantic implementation")
    agent = DeepRMSASourceSemanticAgent(
        num_nodes=int(checkpoint["num_nodes"]), num_slots=int(checkpoint["num_slots"]),
        k_path=int(checkpoint["k_path"]), m_blocks=int(checkpoint["m_blocks"]),
        mod_registry=ModulationRegistry.from_profile("default"), gamma=0.95,
        lr=float(checkpoint["training_args"].get("lr", 1e-5)), entropy_coef=0.01,
        value_loss_coef=1.0, max_grad_norm=40.0, num_layers=5, layer_size=128,
        device=device,
    )
    agent.load_state_dict(checkpoint)
    agent.eval()
    return agent, checkpoint


def evaluate(agent, seed: int, warmup: int, evaluated: int,
             arrival_interval: float) -> Dict[str, Any]:
    config = protocol_config()
    requests = _generate_trace(seed, arrival_interval, warmup, evaluated, config)
    env = OpticalOnlyRMSAEnv(
        topology=config["topology"], num_slots=config["num_slots"],
        k_paths=config["k_paths"], max_blocks=config["max_blocks"],
        path_sort_strategy=config["path_sort_strategy"],
        block_sort_strategy=config["block_sort_strategy"],
        mod_registry=ModulationRegistry.from_profile(config["modulation_profile"]),
        slot_bw_hz=config["slot_bw_hz"], guard_band_fs=config["guard_band_fs"],
        seed=seed,
    )
    env.reset(requests)
    admitted = blocked = invalid_choice = invalid_execution = 0
    for index, request in enumerate(requests):
        env.advance_time(request.arrival_time)
        observation = env.build_observation(request)
        action = agent.select_action(observation)
        measured = index >= warmup
        if action is None:
            if measured:
                blocked += 1
                invalid_choice += 1
            continue
        result = env.step(action, observation, request.holding_time)
        if measured:
            admitted += int(result["success"])
            blocked += int(not result["success"])
            invalid_execution += int(not result["success"])

    env.advance_time(max(req.arrival_time + req.holding_time for req in requests) + 1.0)
    return {
        "seed": seed, "evaluated_requests": admitted + blocked,
        "admitted": admitted, "blocked": blocked,
        "blocking_rate": blocked / max(admitted + blocked, 1),
        "invalid_policy_choice": invalid_choice,
        "invalid_execution": invalid_execution,
        "request_trace_hash": _hash_requests(requests),
        "final_active_connections": len(env.active_connections),
        "final_utilization": env.get_utilization(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline_json", default=str(DEFAULT_BASELINE))
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--evaluated", type=int, default=10000)
    parser.add_argument("--arrival_interval", type=float, default=0.025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    baseline_path = Path(args.baseline_json)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["config"] != protocol_config():
        raise ValueError("Baseline protocol mismatch")
    if float(baseline["arrival_interval"]) != args.arrival_interval:
        raise ValueError("Baseline arrival interval mismatch")
    baseline_results = baseline["results"]
    seeds = [int(seed) for seed in baseline_results["ksp_ff_highest"]]

    checkpoint_path = Path(args.checkpoint)
    agent, checkpoint = load_agent(checkpoint_path, args.device)
    per_seed: Dict[str, Any] = {}
    for seed in seeds:
        row = evaluate(agent, seed, args.warmup, args.evaluated, args.arrival_interval)
        expected = baseline_results["ksp_ff_highest"][str(seed)]
        if row["request_trace_hash"] != expected["request_trace_hash"]:
            raise AssertionError(f"Request trace mismatch for seed {seed}")
        if row["evaluated_requests"] != args.evaluated:
            raise AssertionError(f"Conservation mismatch for seed {seed}")
        if row["invalid_execution"] != 0:
            raise AssertionError(f"Decoded action failed execution for seed {seed}")
        if row["final_active_connections"] or row["final_utilization"] != 0.0:
            raise AssertionError(f"Final release invariant failed for seed {seed}")
        per_seed[str(seed)] = row
        print(
            f"[source-semantic] seed={seed} blocking={row['blocking_rate']:.3%} "
            f"invalid_choice={row['invalid_policy_choice']}", flush=True,
        )

    rates = [per_seed[str(seed)]["blocking_rate"] for seed in seeds]
    output: Dict[str, Any] = {
        "protocol_id": PROTOCOL_ID, "protocol_config": protocol_config(),
        "implementation_label": IMPLEMENTATION_LABEL,
        "upstream_commit": checkpoint.get("upstream_commit"),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "training_seed": checkpoint["training_args"]["seed"],
        "selected_epoch": checkpoint["epoch"], "seeds": seeds,
        "warmup": args.warmup, "evaluated": args.evaluated,
        "arrival_interval": args.arrival_interval, "per_seed": per_seed,
        "mean_blocking_rate": float(np.mean(rates)),
        "std_blocking_rate": float(np.std(rates, ddof=1)),
        "comparisons": {},
    }
    for name in ("ksp_ff_highest", "strict_v13", "ppo_r_top1"):
        baseline_rates = [baseline_results[name][str(seed)]["blocking_rate"] for seed in seeds]
        output["comparisons"][name] = paired_stats(baseline_rates, rates)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "DEEPRMSA_SOURCE_SEMANTIC_RESULTS.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    comparison = output["comparisons"]["ksp_ff_highest"]
    lines = [
        "# DeepRMSA Source-Semantic K=50 Result", "",
        "> This is an unmasked PyTorch protocol port, not unchanged execution of the upstream TensorFlow source.", "",
        f"- Mean blocking: **{100*output['mean_blocking_rate']:.4f}%**",
        f"- Standard deviation: {100*output['std_blocking_rate']:.4f}%",
        f"- Formal KSP-FF minus method: {comparison['mean_difference_pp']:.4f} pp",
        f"- KSP comparison wins/ties/losses: {comparison['wins_ties_losses']}",
        f"- Selected training epoch: {output['selected_epoch']}", "",
        "The policy has no legal-action mask. An unavailable selected action is counted as blocking.",
    ]
    (output_dir / "DEEPRMSA_SOURCE_SEMANTIC_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
