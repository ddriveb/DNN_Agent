#!/usr/bin/env python3
"""Stage D — train the local source-semantic DeepRMSA (paper-standard).

Upstream-faithful training schedule (DeepRMSA_Agent.py / Deep_RMSA_A3C.py):
- K=5 km-ordered candidate paths, M=1, 5-dim unmasked policy, 5x128 ELU.
- Streaming requests at the paper load for the topology.
- First 3000 requests: warmup (no experience stored).
- Overlapping window updates: update on 2*batch-1 = 399 transitions, drop the
  oldest batch = 200 after each update; bootstrap value 0.0.
- Inverted epsilon schedule: sample from the policy with probability epsilon,
  argmax otherwise; epsilon = 1.0 initially, decays by 1e-5 per gradient
  update, floor 0.05.
- Reward: +1 admit, -1 block (including invalid unmasked path choices).

Validation (argmax policy) every `validate_every` requests; best checkpoint by
mean validation blocking rate is kept.

Usage:
    python train_deeprmsa_paper.py --topology nsfnet --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parents[1]))

from paper_deeprmsa_agent import (
    PaperDeepRMSAAgent,
    build_agent_obs,
    decode_flat_action,
    make_deeprmsa_decide,
)
from paper_rmsa_core import (
    PaperRMSAEnv,
    generate_paper_requests,
    load_run_config,
    paper_modulation_registry,
    run_simulation,
)

CFG = load_run_config()
DCFG = CFG["methods"]["deeprmsa_local"]
MEAS = CFG["measurement"]


def build_agent(topology: str, seed: int, device: str = "cpu", lr: float = None,
                normalize_advantages: bool = None, grad_clip: float = None,
                entropy_coef: float = None) -> PaperDeepRMSAAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if normalize_advantages is None:
        normalize_advantages = bool(DCFG.get("normalize_advantages", False))
    return PaperDeepRMSAAgent(
        num_nodes=CFG["topologies"][topology]["num_nodes"],
        num_slots=CFG["spectrum"]["num_slots"],
        k_path=DCFG["k_paths"],
        m_blocks=DCFG["m_blocks"],
        mod_registry=paper_modulation_registry(),
        gamma=DCFG["gamma"],
        lr=DCFG["lr"] if lr is None else lr,
        entropy_coef=DCFG["entropy_coef"] if entropy_coef is None else entropy_coef,
        value_loss_coef=DCFG["value_loss_coef"],
        max_grad_norm=DCFG["max_grad_norm"] if grad_clip is None else grad_clip,
        num_layers=DCFG["num_layers"],
        layer_size=DCFG["layer_size"],
        device=device,
        normalize_advantages=normalize_advantages,
    )


def validate(agent: PaperDeepRMSAAgent, topology: str, val_env: PaperRMSAEnv,
             mod_reg, seeds: List[int], warmup: int, evaluated: int,
             arrival_interval: float) -> Dict[str, Any]:
    agent.eval()
    topo_cfg = CFG["topologies"][topology]
    per_seed: Dict[str, float] = {}
    for seed in seeds:
        requests = generate_paper_requests(
            topo_cfg["num_nodes"], seed, warmup + evaluated, arrival_interval,
            mean_holding_time=CFG["traffic"]["mean_holding_time"],
        )
        res = run_simulation(
            topo_cfg["topology_key"], requests,
            make_deeprmsa_decide(agent, mod_reg),
            k_paths=DCFG["k_paths"], path_sort=DCFG["path_sort"],
            warmup=warmup, num_slots=CFG["spectrum"]["num_slots"], env=val_env,
        )
        per_seed[str(seed)] = res["blocking_rate"]
    rates = list(per_seed.values())
    return {"per_seed": per_seed, "mean_blocking_rate": float(np.mean(rates))}


def save_checkpoint(path: Path, agent: PaperDeepRMSAAgent, topology: str,
                    seed: int, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **agent.state_dict(),
        "protocol_id": CFG["protocol_id"],
        "topology": topology,
        "training_seed": seed,
        "agent_class": "PaperDeepRMSAAgent",
        "meta": meta,
    }, path)


def train(topology: str, seed: int, total_requests: int, device: str = "cpu",
          lr: float = None, normalize_advantages: bool = None,
          grad_clip: float = None, entropy_coef: float = None) -> Dict[str, Any]:
    torch.set_num_threads(1)
    topo_cfg = CFG["topologies"][topology]
    mod_reg = paper_modulation_registry()
    if normalize_advantages is None:
        normalize_advantages = bool(DCFG.get("normalize_advantages", False))
    agent = build_agent(topology, seed, device, lr=lr,
                        normalize_advantages=normalize_advantages, grad_clip=grad_clip,
                        entropy_coef=entropy_coef)
    agent.train()

    env = PaperRMSAEnv(topo_cfg["topology_key"], CFG["spectrum"]["num_slots"], mod_reg=mod_reg)
    val_env = PaperRMSAEnv(topo_cfg["topology_key"], CFG["spectrum"]["num_slots"], mod_reg=mod_reg)
    arrival_interval = CFG["traffic"]["mean_holding_time"] / topo_cfg["load_erlang"]

    out_dir = EXP_DIR / "checkpoints" / topology / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = DCFG["batch_size"]                # 200
    window = 2 * batch - 1                    # 399
    epsilon = DCFG["epsilon_start"]
    eps_decay = DCFG["epsilon_decay_per_update"]
    eps_floor = DCFG["epsilon_floor"]
    train_warmup = DCFG["train_warmup_requests"]
    validate_every = DCFG["validate_every_requests"]
    val_seeds = DCFG["validation_seeds"]

    num_mods = mod_reg.num_formats
    max_blocks = 10

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_meta: Dict[str, Any] = {}
    updates = 0
    time_offset = 0.0
    episode_idx = 0
    global_idx = 0
    admitted = blocked = invalid_choice = 0
    ep_admitted = ep_blocked = 0

    t0 = time.perf_counter()
    episode_size = 1000
    while global_idx < total_requests:
        episode_idx += 1
        chunk_seed = seed * 100000 + episode_idx
        chunk = generate_paper_requests(
            topo_cfg["num_nodes"], chunk_seed, episode_size, arrival_interval,
            mean_holding_time=CFG["traffic"]["mean_holding_time"],
        )
        chunk = [replace(r, arrival_time=r.arrival_time + time_offset) for r in chunk]

        for req in chunk:
            if global_idx >= total_requests:
                break
            env.advance_time(req.arrival_time)
            view = env.build_view(req, DCFG["k_paths"], DCFG["path_sort"])
            obs = build_agent_obs(
                view, mod_reg, num_slots=CFG["spectrum"]["num_slots"],
                max_blocks=max_blocks,
                slot_bw_hz=CFG["spectrum"]["slot_bw_hz"],
                guard_band_fs=CFG["spectrum"]["guard_band_fs"],
            )
            # Upstream inverted epsilon-greedy: sample w.p. epsilon else argmax.
            agent.training = random.random() < epsilon
            flat = agent.select_action(obs)
            agent.training = True

            success = False
            if flat is None:
                invalid_choice += int(global_idx >= train_warmup)
            else:
                path_idx, _m, _b = decode_flat_action(flat, num_mods, max_blocks)
                if path_idx < len(view["paths"]):
                    start = view["paths"][path_idx]["first_fit_start"]
                    if start is not None:
                        result = env.commit(req, view["paths"][path_idx], start)
                        success = bool(result["success"])
            reward = 1.0 if success else -1.0

            if global_idx >= train_warmup:
                admitted += int(success)
                blocked += int(not success)
                ep_admitted += int(success)
                ep_blocked += int(not success)
                agent.store_transition(reward, False)
                if len(agent.episode_buffer) == window:
                    tail = agent.episode_buffer[batch:]
                    agent.optimize(bootstrap_value=0.0)
                    agent.episode_buffer.extend(tail)
                    epsilon = max(epsilon - eps_decay, eps_floor)
                    updates += 1

            global_idx += 1
            time_offset = req.arrival_time

            if global_idx % validate_every == 0 or global_idx == total_requests:
                val = validate(
                    agent, topology, val_env, mod_reg, val_seeds,
                    DCFG["validation_warmup"], DCFG["validation_requests"],
                    arrival_interval,
                )
                train_blk = ep_blocked / max(ep_admitted + ep_blocked, 1)
                row = {
                    "requests": global_idx, "updates": updates,
                    "epsilon": epsilon, "train_blocking": train_blk,
                    "val_blocking": val["mean_blocking_rate"],
                    "val_per_seed": val["per_seed"],
                    "elapsed_s": time.perf_counter() - t0,
                }
                history.append(row)
                if val["mean_blocking_rate"] < best_val:
                    best_val = val["mean_blocking_rate"]
                    best_meta = {"requests": global_idx, "updates": updates,
                                 "val_blocking": best_val}
                    save_checkpoint(out_dir / "best.pt", agent, topology, seed, best_meta)
                print(
                    f"[{topology} seed={seed}] req={global_idx:7d} upd={updates:4d} "
                    f"eps={epsilon:.3f} train_blk={train_blk:.3%} "
                    f"val_blk={val['mean_blocking_rate']:.3%} best={best_val:.3%}",
                    flush=True,
                )
                ep_admitted = ep_blocked = 0
                agent.train()

    save_checkpoint(out_dir / "final.pt", agent, topology, seed,
                    {"requests": global_idx, "updates": updates})
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "protocol_id": CFG["protocol_id"],
        "topology": topology,
        "training_seed": seed,
        "lr_effective": DCFG["lr"] if lr is None else lr,
        "normalize_advantages": normalize_advantages,
        "grad_clip_effective": DCFG["max_grad_norm"] if grad_clip is None else grad_clip,
        "entropy_coef_effective": DCFG["entropy_coef"] if entropy_coef is None else entropy_coef,
        "total_requests": global_idx,
        "gradient_updates": updates,
        "final_epsilon": epsilon,
        "train_admitted": admitted,
        "train_blocked": blocked,
        "train_invalid_choice": invalid_choice,
        "train_blocking_rate": blocked / max(admitted + blocked, 1),
        "best_validation_blocking_rate": best_val,
        "best_meta": best_meta,
        "best_checkpoint": str(out_dir / "best.pt"),
        "final_checkpoint": str(out_dir / "final.pt"),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, choices=list(CFG["topologies"].keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total_requests", type=int, default=DCFG["total_train_requests"])
    parser.add_argument("--lr", type=float, default=None,
                        help="override RUN_CONFIG lr (compute-budget adaptation)")
    parser.add_argument("--no_norm", action="store_true",
                        help="disable per-window advantage normalization")
    parser.add_argument("--norm", action="store_true",
                        help="enable per-window advantage normalization (overrides RUN_CONFIG)")
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="override RUN_CONFIG max_grad_norm")
    parser.add_argument("--entropy", type=float, default=None,
                        help="override RUN_CONFIG entropy_coef")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    norm = False if args.no_norm else (True if args.norm else None)
    summary = train(args.topology, args.seed, args.total_requests, args.device,
                    lr=args.lr, normalize_advantages=norm,
                    grad_clip=args.grad_clip, entropy_coef=args.entropy)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
