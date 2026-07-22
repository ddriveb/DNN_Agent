"""Train the unmasked, source-semantic DeepRMSA port on PAPER_PRIMARY_K50."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from sa_hmarl.agents.deep_rmsa_source_semantic_agent import DeepRMSASourceSemanticAgent
from sa_hmarl.evaluation.optical_only_rmsa_env import OpticalOnlyRMSAEnv
from sa_hmarl.evaluation.optical_only_rmsa_evaluator import _generate_trace
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_deeprmsa_optical_k50 import protocol_config


PROTOCOL_ID = "DEEPRMSA_SOURCE_SEMANTIC_PAPER_PRIMARY_K50"
IMPLEMENTATION_LABEL = "DeepRMSA source-semantic unmasked PyTorch port"
DEFAULT_OUTPUT = Path("sa_hmarl/experiments/deeprmsa_source_semantic_paper_primary_k50")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_env(seed: int) -> OpticalOnlyRMSAEnv:
    config = protocol_config()
    return OpticalOnlyRMSAEnv(
        topology=config["topology"], num_slots=config["num_slots"],
        k_paths=config["k_paths"], max_blocks=config["max_blocks"],
        path_sort_strategy=config["path_sort_strategy"],
        block_sort_strategy=config["block_sort_strategy"],
        mod_registry=ModulationRegistry.from_profile(config["modulation_profile"]),
        slot_bw_hz=config["slot_bw_hz"], guard_band_fs=config["guard_band_fs"],
        seed=seed,
    )


def build_agent(seed: int, args: argparse.Namespace) -> DeepRMSASourceSemanticAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return DeepRMSASourceSemanticAgent(
        num_nodes=11, num_slots=100, k_path=50, m_blocks=1,
        mod_registry=ModulationRegistry.from_profile("default"),
        gamma=0.95, lr=args.lr, entropy_coef=0.01, value_loss_coef=1.0,
        max_grad_norm=40.0, num_layers=5, layer_size=128, device=args.device,
    )


def train_epoch(agent, env, requests, warmup: int) -> Dict[str, float]:
    env.reset(requests)
    agent.train()
    agent.clear_buffer()
    admitted = blocked = invalid_choice = 0
    for index, request in enumerate(requests):
        env.advance_time(request.arrival_time)
        observation = env.build_observation(request)
        action = agent.select_action(observation)
        measured = index >= warmup
        if action is None:
            if measured:
                blocked += 1
                invalid_choice += 1
                agent.store_transition(-1.0, index == len(requests) - 1)
            continue
        result = env.step(action, observation, request.holding_time)
        reward = 1.0 if result["success"] else -1.0
        if measured:
            admitted += int(result["success"])
            blocked += int(not result["success"])
            agent.store_transition(reward, index == len(requests) - 1)
    loss = agent.optimize(bootstrap_value=0.0)
    return {
        "admitted": admitted, "blocked": blocked,
        "invalid_policy_choice": invalid_choice,
        "blocking_rate": blocked / max(admitted + blocked, 1),
        **(loss or {}),
    }


def evaluate(agent, env, seeds: List[int], warmup: int, evaluated: int,
             arrival_interval: float) -> Dict[str, Any]:
    agent.eval()
    rows: Dict[str, Any] = {}
    for seed in seeds:
        requests = _generate_trace(seed, arrival_interval, warmup, evaluated, protocol_config())
        env.reset(requests)
        admitted = blocked = invalid_choice = 0
        for index, request in enumerate(requests):
            env.advance_time(request.arrival_time)
            observation = env.build_observation(request)
            action = agent.select_action(observation)
            if action is None:
                if index >= warmup:
                    blocked += 1
                    invalid_choice += 1
                continue
            result = env.step(action, observation, request.holding_time)
            if index >= warmup:
                admitted += int(result["success"])
                blocked += int(not result["success"])
        rows[str(seed)] = {
            "admitted": admitted, "blocked": blocked,
            "invalid_policy_choice": invalid_choice,
            "blocking_rate": blocked / max(admitted + blocked, 1),
        }
    rates = [row["blocking_rate"] for row in rows.values()]
    return {"per_seed": rows, "mean_blocking_rate": float(np.mean(rates))}


def save_checkpoint(path: Path, agent, args, epoch: int, validation: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **agent.state_dict(), "protocol_id": PROTOCOL_ID,
        "protocol_config": protocol_config(), "training_args": vars(args),
        "epoch": epoch, "validation": validation,
        "implementation_label": IMPLEMENTATION_LABEL,
        "upstream_commit": "6708e9a023df1ec05bfdc77804b6829e33cacfe4",
    }, path)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(args.seed, args)
    train_env = build_env(args.seed)
    validation_env = build_env(args.seed)
    best = float("inf")
    best_path = output_dir / "best.pt"
    history = []
    started = time.perf_counter()
    validation_seeds = [int(value) for value in args.validation_seeds.split(",")]
    for epoch in range(1, args.epochs + 1):
        requests = _generate_trace(
            args.seed * 100000 + epoch, args.arrival_interval,
            args.train_warmup, args.train_requests, protocol_config(),
        )
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train": train_epoch(agent, train_env, requests, args.train_warmup),
        }
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate(
                agent, validation_env, validation_seeds, args.validation_warmup,
                args.validation_requests, args.arrival_interval,
            )
            row["validation"] = validation
            if validation["mean_blocking_rate"] < best:
                best = validation["mean_blocking_rate"]
                save_checkpoint(best_path, agent, args, epoch, validation)
            print(
                f"[source-semantic seed={args.seed}] epoch={epoch:04d} "
                f"train_blk={row['train']['blocking_rate']:.3%} "
                f"val_blk={validation['mean_blocking_rate']:.3%} best={best:.3%}",
                flush=True,
            )
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "protocol_id": PROTOCOL_ID, "implementation_label": IMPLEMENTATION_LABEL,
        "best_validation_blocking_rate": best, "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path),
        "elapsed_seconds": time.perf_counter() - started, "args": vars(args),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--epochs", type=int, default=80)
    result.add_argument("--train_warmup", type=int, default=500)
    result.add_argument("--train_requests", type=int, default=1500)
    result.add_argument("--arrival_interval", type=float, default=0.025)
    result.add_argument("--eval_every", type=int, default=5)
    result.add_argument("--validation_seeds", default="42001,42002,42003")
    result.add_argument("--validation_warmup", type=int, default=1000)
    result.add_argument("--validation_requests", type=int, default=3000)
    result.add_argument("--lr", type=float, default=1e-5)
    result.add_argument("--device", default="cpu")
    result.add_argument("--output_dir", default=str(DEFAULT_OUTPUT / "checkpoints"))
    return result


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(train(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
