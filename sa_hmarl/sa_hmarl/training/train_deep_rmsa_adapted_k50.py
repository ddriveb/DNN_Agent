"""Training script for topology-matched adapted DeepRMSA K=50 hops.

Fixed-C / all-OD COST239 pure RMSA. Trains from scratch; does not load any
existing snap24/NSFNET/Germany/Japan checkpoint. Checkpoint selection is by
validation blocking rate on held-out validation seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_hmarl.agents.deep_rmsa_adapted_k50_agent import DeepRMSAAdaptedK50Agent
from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
from sa_hmarl.evaluation.diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od import (
    _generate_all_od_requests,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


PROTOCOL_ID = "DEEPRMSA_ADAPTED_K50_COST239_FIXED_C_ALL_OD"
IMPLEMENTATION_LABEL = "Topology-matched adapted DeepRMSA K=50 hops"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_threading():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")


def _build_agent(env, args: argparse.Namespace) -> DeepRMSAAdaptedK50Agent:
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    return DeepRMSAAdaptedK50Agent(
        num_nodes=env.net.NUM_NODES,
        num_slots=args.num_slots,
        k_path=args.k_paths_r,
        num_modulations=mod_reg.num_formats,
        m_blocks=args.max_blocks,
        mod_registry=mod_reg,
        gamma=args.gamma,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        num_layers=args.num_layers,
        layer_size=args.layer_size,
        device=args.device,
    )


def _make_env(seed: int, args: argparse.Namespace):
    return make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy_r,
        path_sort_strategy=args.path_sort_strategy_r,
        k=args.k_paths_r,
    )


def _generate_requests_for_seed(seed: int, num_requests: int, warmup: int, args: argparse.Namespace):
    rng = np.random.RandomState(seed)
    env_for_nodes = _make_env(42, args)
    server_node_ids = [int(s.node_id) for s in env_for_nodes.mec.servers]
    return _generate_all_od_requests(
        num_nodes=env_for_nodes.net.NUM_NODES,
        server_node_ids=server_node_ids,
        rng=rng,
        num_requests=warmup + num_requests,
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
        split_profile=args.split_profile,
        poisson_arrivals=False,
        exponential_holding=False,
    )


def _evaluate_agent(
    agent: DeepRMSAAdaptedK50Agent,
    env,
    requests: List[Any],
    fixed_split_id: int,
) -> Tuple[int, int, float, Dict[str, Any]]:
    """Evaluate agent in greedy mode on a fixed request sequence."""
    agent.eval()
    env.reset(requests)
    node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks

    admitted = 0
    blocked = 0
    illegal_action_count = 0
    path_idx_counts: Dict[int, int] = {}
    mod_counts: Dict[str, int] = {}
    required_fs_values: List[int] = []
    block_start_values: List[int] = []
    block_waste_values: List[float] = []
    path_hops_values: List[int] = []
    path_km_values: List[float] = []

    for step_idx, req in enumerate(requests):
        if step_idx < 500:
            is_warmup = True
        else:
            is_warmup = False
        env.advance_time(req.arrival_time)
        dst_node = int(getattr(req, "_dst_node", req.src_node))
        server_id = node_to_server.get(dst_node)
        if server_id is None:
            server_id = min(node_to_server.values(), key=lambda i: abs(i - dst_node))

        env.k = 50
        env.path_sort_strategy = "hops"
        env.block_sort_strategy = "start_asc"
        obs_r = build_agent_r_observation(env, req, fixed_split_id, server_id)
        action_idx = agent.select_action(obs_r)

        if action_idx is None:
            if not is_warmup:
                blocked += 1
            env.reject_next_request(req.req_id, "r_no_valid_action")
            continue

        # Verify the action is legal (should always be true due to mask).
        if not obs_r["agent_r_mask"][action_idx]:
            illegal_action_count += 1
            if not is_warmup:
                blocked += 1
            env.reject_next_request(req.req_id, "illegal_action")
            continue

        r_action = decode_agent_r_action(action_idx, num_mods, max_blocks)
        _, _, _, info = env.step((fixed_split_id, server_id), r_action)
        success = bool(info.get("success", False))

        if not is_warmup:
            if success:
                admitted += 1
                path_idx, mod_idx, block_idx = r_action
                path_feats = obs_r.get("path_features", [])
                if path_feats and 0 <= path_idx < len(path_feats):
                    path_hops_values.append(int(path_feats[path_idx].get("hop_count", 0)))
                    path_km_values.append(float(path_feats[path_idx].get("path_length_km", 0.0)))
                mod_names = obs_r.get("mod_names", [])
                if mod_names and 0 <= mod_idx < len(mod_names):
                    mod_name = str(mod_names[mod_idx])
                    mod_counts[mod_name] = mod_counts.get(mod_name, 0) + 1
                path_idx_counts[path_idx] = path_idx_counts.get(path_idx, 0) + 1
                blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
                if block_idx < len(blocks):
                    block_start_values.append(int(blocks[block_idx][0]))
                    block_size = int(blocks[block_idx][1])
                    req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
                    required_fs = int(req_fs) if req_fs is not None else 0
                    required_fs_values.append(required_fs)
                    block_waste_values.append((block_size - required_fs) / max(block_size, 1))
            else:
                blocked += 1

    total = admitted + blocked
    blocking_rate = blocked / max(total, 1)
    metrics = {
        "admitted": admitted,
        "blocked": blocked,
        "total": total,
        "blocking_rate": blocking_rate,
        "illegal_action_count": illegal_action_count,
        "path_idx_distribution": {str(k): v / max(admitted, 1) for k, v in path_idx_counts.items()},
        "mod_distribution": {k: v / max(admitted, 1) for k, v in mod_counts.items()},
        "avg_required_fs": float(np.mean(required_fs_values)) if required_fs_values else 0.0,
        "avg_block_start": float(np.mean(block_start_values)) if block_start_values else 0.0,
        "avg_block_waste": float(np.mean(block_waste_values)) if block_waste_values else 0.0,
        "avg_hops": float(np.mean(path_hops_values)) if path_hops_values else 0.0,
        "avg_path_km": float(np.mean(path_km_values)) if path_km_values else 0.0,
    }
    return admitted, blocked, blocking_rate, metrics


def _train_epoch(
    agent: DeepRMSAAdaptedK50Agent,
    env,
    requests: List[Any],
    fixed_split_id: int,
    warmup: int,
) -> Dict[str, Any]:
    """Train one epoch on a fixed request sequence."""
    agent.train()
    agent.clear_buffer()
    env.reset(requests)
    node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}
    num_mods = env.mod_reg.num_formats
    max_blocks = env.max_blocks

    admitted = 0
    blocked = 0
    invalid_choice = 0
    total_reward = 0.0

    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < warmup
        env.advance_time(req.arrival_time)
        dst_node = int(getattr(req, "_dst_node", req.src_node))
        server_id = node_to_server.get(dst_node)
        if server_id is None:
            server_id = min(node_to_server.values(), key=lambda i: abs(i - dst_node))

        env.k = 50
        env.path_sort_strategy = "hops"
        env.block_sort_strategy = "start_asc"
        obs_r = build_agent_r_observation(env, req, fixed_split_id, server_id)
        action_idx = agent.select_action(obs_r)

        if action_idx is None:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            reward = -1.0
            if not is_warmup:
                blocked += 1
        else:
            r_action = decode_agent_r_action(action_idx, num_mods, max_blocks)
            _, _, _, info = env.step((fixed_split_id, server_id), r_action)
            success = bool(info.get("success", False))
            reward = 1.0 if success else -1.0
            if not is_warmup:
                if success:
                    admitted += 1
                else:
                    blocked += 1

        total_reward += reward
        is_terminal = (step_idx + 1) >= len(requests)
        agent.store_transition(reward, is_terminal)

    loss = agent.optimize(bootstrap_value=0.0)
    total = admitted + blocked
    return {
        "admitted": admitted,
        "blocked": blocked,
        "total": total,
        "blocking_rate": blocked / max(total, 1),
        "avg_reward": total_reward / max(len(requests) - warmup, 1),
        "invalid_policy_choice": invalid_choice,
        **(loss or {}),
    }


def _save_checkpoint(path: Path, agent: DeepRMSAAdaptedK50Agent, args: argparse.Namespace, epoch: int, validation: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **agent.state_dict(),
            "protocol_id": PROTOCOL_ID,
            "protocol_config": {
                "topology": args.topology,
                "num_slots": args.num_slots,
                "num_servers": args.num_servers,
                "fixed_split_id": args.fixed_split_id,
                "arrival_interval": args.arrival_interval,
                "k_paths_r": args.k_paths_r,
                "path_sort_strategy_r": args.path_sort_strategy_r,
                "block_sort_strategy_r": args.block_sort_strategy_r,
                "max_blocks": args.max_blocks,
                "modulation_profile": args.modulation_profile,
            },
            "training_args": vars(args),
            "epoch": epoch,
            "validation": validation,
            "implementation_label": IMPLEMENTATION_LABEL,
        },
        path,
    )


def train(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = output_dir / f"seed_{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = _make_env(args.seed, args)
    agent = _build_agent(env, args)
    validation_env = _make_env(args.seed + 100000, args)

    best_blocking = float("inf")
    best_path = seed_dir / "best.pt"
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()
    validation_seeds = [int(v) for v in args.validation_seeds.split(",")]

    print("=" * 70)
    print(IMPLEMENTATION_LABEL)
    print(f"Output: {output_dir}")
    print(f"State dim: {agent.state_dim}, Action dim: {agent.n_actions}")
    print(f"Trainable parameters: {agent.count_parameters()}")
    print(f"Train epochs: {args.epochs}, requests/epoch: {args.train_requests}, warmup: {args.train_warmup}")
    print(f"Validation every {args.eval_every} epochs on seeds: {validation_seeds}")
    print(f"Validation requests: {args.validation_requests}, warmup: {args.validation_warmup}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_requests = _generate_requests_for_seed(
            args.seed * 100000 + epoch, args.train_requests, args.train_warmup, args
        )
        train_row = _train_epoch(agent, env, train_requests, args.fixed_split_id, args.train_warmup)
        row: Dict[str, Any] = {"epoch": epoch, "train": train_row}

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            validation_results = {}
            blocking_rates = []
            for val_seed in validation_seeds:
                val_requests = _generate_requests_for_seed(
                    val_seed, args.validation_requests, args.validation_warmup, args
                )
                _, _, br, metrics = _evaluate_agent(agent, validation_env, val_requests, args.fixed_split_id)
                validation_results[str(val_seed)] = metrics
                blocking_rates.append(br)
            mean_blocking = float(np.mean(blocking_rates))
            validation = {"per_seed": validation_results, "mean_blocking_rate": mean_blocking}
            row["validation"] = validation

            if mean_blocking < best_blocking:
                best_blocking = mean_blocking
                best_epoch = epoch
                _save_checkpoint(best_path, agent, args, epoch, validation)

            print(
                f"[seed={args.seed}] epoch={epoch:04d} "
                f"train_blk={train_row['blocking_rate']:.3%} "
                f"val_blk={mean_blocking:.3%} best={best_blocking:.3%} (ep {best_epoch})",
                flush=True,
            )
        else:
            print(
                f"[seed={args.seed}] epoch={epoch:04d} "
                f"train_blk={train_row['blocking_rate']:.3%}",
                flush=True,
            )

        history.append(row)
        (seed_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "protocol_id": PROTOCOL_ID,
        "implementation_label": IMPLEMENTATION_LABEL,
        "best_validation_blocking_rate": best_blocking,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path) if best_path.exists() else None,
        "parameter_count": agent.count_parameters(),
        "elapsed_seconds": time.perf_counter() - started,
        "args": vars(args),
    }
    (seed_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train adapted DeepRMSA K=50 hops on COST239 fixed-C/all-OD")
    # Environment / traffic config (locked to source-of-truth diagnosis)
    p.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    p.add_argument("--num_slots", type=int, default=320)
    p.add_argument("--num_servers", type=int, default=4)
    p.add_argument("--fixed_split_id", type=int, default=0)
    p.add_argument("--arrival_interval", type=float, default=0.3)
    p.add_argument("--holding_min", type=float, default=20.0)
    p.add_argument("--holding_max", type=float, default=30.0)
    p.add_argument("--deadline_min", type=float, default=30.0)
    p.add_argument("--deadline_max", type=float, default=100.0)
    p.add_argument("--size_min_mb", type=float, default=5.0)
    p.add_argument("--size_max_mb", type=float, default=30.0)
    p.add_argument("--edge_cost_min", type=float, default=0.1)
    p.add_argument("--edge_cost_max", type=float, default=2.2)
    p.add_argument("--split_profile", default="default3")
    p.add_argument("--num_splits", type=int, default=3)
    # R-side config
    p.add_argument("--k_paths_r", type=int, default=50)
    p.add_argument("--path_sort_strategy_r", default="hops")
    p.add_argument("--block_sort_strategy_r", default="start_asc")
    p.add_argument("--max_blocks", type=int, default=10)
    p.add_argument("--modulation_profile", default="default")
    # Training hyperparameters
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--train_warmup", type=int, default=500)
    p.add_argument("--train_requests", type=int, default=1500)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--validation_seeds", default="6001,6002,6003")
    p.add_argument("--validation_warmup", type=int, default=500)
    p.add_argument("--validation_requests", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_loss_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=40.0)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--layer_size", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--output_dir",
        default="sa_hmarl/experiments/strict_v13_vs_ksp_ff_vs_deeprmsa_cost239_fixed_c_all_od/deep_rmsa_adapted",
    )
    return p


def main() -> int:
    _set_threading()
    args = parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
