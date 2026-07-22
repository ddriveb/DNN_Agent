"""Native optical-only training for the PAPER_PRIMARY_K50 DeepRMSA adapter.

This is deliberately separate from ``train_deep_rmsa.py`` because that script
uses the coupled SMDP/MEC environment.  The model remains the repository's
PyTorch DeepRMSA actor-critic adaptation, but its topology, traffic, spectrum,
and path support match the certified optical-only comparison protocol.
"""
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

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.evaluation.optical_only_rmsa_env import OpticalOnlyRMSAEnv
from sa_hmarl.evaluation.optical_only_rmsa_evaluator import _generate_trace
from sa_hmarl.network.modulation import ModulationRegistry


PROTOCOL_ID = "DEEPRMSA_ADAPTED_TO_PAPER_PRIMARY_K50"
DEFAULT_OUTPUT = Path("sa_hmarl/experiments/deeprmsa_adapted_paper_primary_k50")


def protocol_config() -> Dict[str, Any]:
    return {
        "topology": "xlron_cost239_ptrnet_real",
        "num_slots": 100,
        "k_paths": 50,
        "max_blocks": 10,
        "path_sort_strategy": "hops",
        "block_sort_strategy": "start_asc",
        "modulation_profile": "default",
        "slot_bw_hz": 12.5e9,
        "guard_band_fs": 1,
        "mean_holding_time": 10.0,
        "bitrate_min_gbps": 25,
        "bitrate_max_gbps": 100,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_agent(seed: int, m_blocks: int, device: str, args: argparse.Namespace) -> DeepRMSAAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return DeepRMSAAgent(
        num_nodes=11,
        num_slots=100,
        k_path=50,
        m_blocks=m_blocks,
        mod_registry=ModulationRegistry.from_profile("default"),
        gamma=args.gamma,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        num_layers=5,
        layer_size=128,
        device=device,
    )


def _credit_forced_block(agent: DeepRMSAAgent, reward: float) -> None:
    """Fold a no-action blocking reward into the preceding decision.

    With a legal-action mask, a fully blocked state has no policy action to
    store.  Folding its discounted reward into the preceding transition keeps
    the long-term cost visible without inventing a fallback RMSA action.
    """
    if not agent.episode_buffer:
        return
    state, action, previous_reward, value, mask, done = agent.episode_buffer[-1]
    agent.episode_buffer[-1] = (
        state,
        action,
        float(previous_reward) + agent.gamma * float(reward),
        value,
        mask,
        done,
    )


def run_training_episode(
    agent: DeepRMSAAgent,
    env: OpticalOnlyRMSAEnv,
    requests,
    warmup: int,
) -> Dict[str, float]:
    env.reset(requests)
    agent.train()
    agent.clear_buffer()
    admitted = blocked = no_valid = 0

    for idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs = env.build_observation(req)
        action = agent.select_action(obs)
        in_training_window = idx >= warmup

        if action is None:
            if in_training_window:
                blocked += 1
                no_valid += 1
                _credit_forced_block(agent, -1.0)
            continue

        step = env.step(action, obs, req.holding_time)
        reward = 1.0 if step["success"] else -1.0
        if in_training_window:
            if step["success"]:
                admitted += 1
            else:
                blocked += 1
            agent.store_transition(reward, idx == len(requests) - 1)

    loss = agent.optimize(bootstrap_value=0.0)
    total = admitted + blocked
    return {
        "admitted": admitted,
        "blocked": blocked,
        "no_valid": no_valid,
        "blocking_rate": blocked / max(total, 1),
        **(loss or {}),
    }


def evaluate(
    agent: DeepRMSAAgent,
    env: OpticalOnlyRMSAEnv,
    seeds: List[int],
    warmup: int,
    evaluated: int,
    arrival_interval: float,
) -> Dict[str, Any]:
    agent.eval()
    config = protocol_config()
    per_seed: Dict[str, Any] = {}
    for seed in seeds:
        requests = _generate_trace(seed, arrival_interval, warmup, evaluated, config)
        env.reset(requests)
        admitted = blocked = no_valid = 0
        for idx, req in enumerate(requests):
            env.advance_time(req.arrival_time)
            obs = env.build_observation(req)
            action = agent.select_action(obs)
            if action is None:
                if idx >= warmup:
                    blocked += 1
                    no_valid += 1
                continue
            step = env.step(action, obs, req.holding_time)
            if idx >= warmup:
                if step["success"]:
                    admitted += 1
                else:
                    blocked += 1
        per_seed[str(seed)] = {
            "admitted": admitted,
            "blocked": blocked,
            "r_no_valid_action": no_valid,
            "evaluated": admitted + blocked,
            "blocking_rate": blocked / max(admitted + blocked, 1),
        }
    rates = [row["blocking_rate"] for row in per_seed.values()]
    return {
        "per_seed": per_seed,
        "mean_blocking_rate": float(np.mean(rates)),
        "std_blocking_rate": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
    }


def save_checkpoint(
    path: Path,
    agent: DeepRMSAAgent,
    args: argparse.Namespace,
    epoch: int,
    validation: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **agent.state_dict(),
        "protocol_id": PROTOCOL_ID,
        "protocol_config": protocol_config(),
        "training_args": vars(args),
        "epoch": epoch,
        "validation": validation,
        "implementation_label": "PyTorch DeepRMSA-style masked A2C adapter",
    }, path)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    config = protocol_config()
    output_dir = Path(args.output_dir) / f"m{args.m_blocks}" / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(args.seed, args.m_blocks, args.device, args)
    train_env = OpticalOnlyRMSAEnv(**{
        "topology": config["topology"],
        "num_slots": config["num_slots"],
        "k_paths": config["k_paths"],
        "max_blocks": config["max_blocks"],
        "path_sort_strategy": config["path_sort_strategy"],
        "block_sort_strategy": config["block_sort_strategy"],
        "mod_registry": ModulationRegistry.from_profile(config["modulation_profile"]),
        "slot_bw_hz": config["slot_bw_hz"],
        "guard_band_fs": config["guard_band_fs"],
        "seed": args.seed,
    })
    val_env = OpticalOnlyRMSAEnv(**{
        "topology": config["topology"], "num_slots": 100, "k_paths": 50,
        "max_blocks": 10, "path_sort_strategy": "hops",
        "block_sort_strategy": "start_asc",
        "mod_registry": ModulationRegistry.from_profile("default"),
        "slot_bw_hz": 12.5e9, "guard_band_fs": 1, "seed": args.seed,
    })

    history = []
    best_rate = float("inf")
    best_path = output_dir / "best.pt"
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        trace_seed = args.seed * 100000 + epoch
        requests = _generate_trace(
            trace_seed,
            args.arrival_interval,
            args.train_warmup,
            args.train_requests,
            config,
        )
        metrics = run_training_episode(agent, train_env, requests, args.train_warmup)
        row: Dict[str, Any] = {"epoch": epoch, "train": metrics}

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            validation = evaluate(
                agent,
                val_env,
                [int(x) for x in args.validation_seeds.split(",")],
                args.validation_warmup,
                args.validation_requests,
                args.arrival_interval,
            )
            row["validation"] = validation
            if validation["mean_blocking_rate"] < best_rate:
                best_rate = validation["mean_blocking_rate"]
                save_checkpoint(best_path, agent, args, epoch, validation)
            print(
                f"[m={args.m_blocks} seed={args.seed}] epoch={epoch:04d} "
                f"train_blk={metrics['blocking_rate']:.3%} "
                f"val_blk={validation['mean_blocking_rate']:.3%} best={best_rate:.3%}",
                flush=True,
            )
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=str), encoding="utf-8"
        )

    summary = {
        "protocol_id": PROTOCOL_ID,
        "protocol_config": config,
        "implementation_label": "PyTorch DeepRMSA-style masked A2C adapter",
        "m_blocks": args.m_blocks,
        "seed": args.seed,
        "best_validation_blocking_rate": best_rate,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path),
        "elapsed_seconds": time.perf_counter() - t0,
        "args": vars(args),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_blocks", type=int, choices=[1, 10], default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train_warmup", type=int, default=500)
    parser.add_argument("--train_requests", type=int, default=1500)
    parser.add_argument("--arrival_interval", type=float, default=0.025)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--validation_seeds", default="42001,42002,42003")
    parser.add_argument("--validation_warmup", type=int, default=1000)
    parser.add_argument("--validation_requests", type=int, default=3000)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT / "checkpoints"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(train(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
