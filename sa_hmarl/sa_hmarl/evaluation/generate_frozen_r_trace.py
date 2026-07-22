"""Generate a frozen C-side trace for R-only discriminative validation.

The trace records, for every request in an episode, the C-side decision and the
C-context that the origin R-side method observed.  Later replay engines apply
the same C-side decision sequence to every R-side method and overwrite the
observed c_context with the recorded one, removing the C->R closed-loop coupling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation import eval_strict_v13_multitopology_cside_verified as ev
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
)
from sa_hmarl.evaluation.eval_long_horizon_system_comparison import (
    generate_long_horizon_requests,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


SCHEMA_VERSION = "frozen-trace-v1.0"

ORIGIN_R_MODES = ("ksp_ff_highest", "ppo_r_top1", "strict_v13")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _serialize_split(sp) -> Dict[str, Any]:
    return {
        "split_id": int(sp.split_id),
        "intermediate_size_mb": float(sp.intermediate_size_mb),
        "local_compute_cost": float(sp.local_compute_cost),
        "edge_compute_cost": float(sp.edge_compute_cost),
    }


def _serialize_request(req) -> Dict[str, Any]:
    return {
        "req_id": int(req.req_id),
        "src_node": int(req.src_node),
        "arrival_time": float(req.arrival_time),
        "holding_time": float(req.holding_time),
        "deadline_ms": float(req.deadline_ms),
        "splits": [_serialize_split(sp) for sp in req.splits],
    }


def _c_context_hash(contexts: List[Optional[Dict[str, Any]]]) -> str:
    payload = []
    for ctx in contexts:
        if ctx is None:
            payload.append(None)
        else:
            payload.append({k: v for k, v in sorted(ctx.items())})
    return _hash_text(json.dumps(payload, sort_keys=True, default=str))


def _config_hash(args) -> str:
    cfg = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    return _hash_text(json.dumps(cfg, sort_keys=True, default=str))


def generate_trace(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    root = Path(__file__).resolve().parents[3]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    # Agent-C is only used if we later support ppo_c origins; for df_c origins it is None.
    agent_c = None
    if args.c_mode != "df_c":
        agent_c = _load_ppo_c(args.c_checkpoint, args.device)

    ranker = None
    if args.origin_r_mode in ev.RANKER_MODES:
        if not args.ranker_spec:
            raise ValueError(
                f"--ranker_spec is required for origin_r_mode={args.origin_r_mode}"
            )
        name, path = args.ranker_spec.split("=", 1)
        ranker = ev._load_ranker_model(path, args.device)

    total_requests = args.warmup_requests + args.requests_per_episode
    rng = np.random.RandomState(args.seed)
    env = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        args.seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy_r,
        path_sort_strategy=args.path_sort_strategy_r,
        k=args.k_paths_r,
    )
    requests = generate_long_horizon_requests(
        num_nodes=env.net.NUM_NODES,
        rng=rng,
        num_requests=total_requests,
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
        poisson_arrivals=args.poisson_arrivals,
        exponential_holding=args.exponential_holding,
    )

    request_trace_hash = ev._hash_requests(requests)
    config_hash = _config_hash(args)

    env.reset(requests)
    num_servers = len(env.mec.servers)
    max_blocks = env.max_blocks
    num_mods = env.mod_reg.num_formats

    records: List[Dict[str, Any]] = []
    context_list: List[Optional[Dict[str, Any]]] = []

    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)

        # C-side under K_C.
        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        c_idx, split_id, server_id, c_valid, c_info = ev._select_c_action(
            args.c_mode, agent_c, env, req, obs_c, num_servers
        )

        if not c_valid:
            env.reject_next_request(req.req_id, "c_no_valid_action")
            records.append({
                "req_id": int(req.req_id),
                "arrival_time": float(req.arrival_time),
                "holding_time": float(req.holding_time),
                "deadline_ms": float(req.deadline_ms),
                "src_node": int(req.src_node),
                "splits": [_serialize_split(sp) for sp in req.splits],
                "origin_c_valid": False,
                "origin_c_decision": None,
                "origin_c_context": None,
                "origin_r_action": None,
            })
            context_list.append(None)
            continue

        # R-side under K_R.
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        c_context = obs_r["c_context"]

        # Use PPO-R top-1 as the audit reference for the origin; for replay
        # agreement measurement we store whatever the origin R method selects.
        audit_idx, _ = ev._select_r_action_from_obs(agent_r, obs_r, max_blocks)
        r_idx, r_valid, _ = ev._select_r_action(
            args.origin_r_mode,
            agent_r,
            ranker,
            env,
            req,
            obs_c,
            obs_r,
            audit_idx,
            split_id,
            server_id,
        )

        if r_valid:
            r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
            env.step((split_id, server_id), r_action)
            origin_r_action = int(r_idx)
        else:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            origin_r_action = None

        records.append({
            "req_id": int(req.req_id),
            "arrival_time": float(req.arrival_time),
            "holding_time": float(req.holding_time),
            "deadline_ms": float(req.deadline_ms),
            "src_node": int(req.src_node),
            "splits": [_serialize_split(sp) for sp in req.splits],
            "origin_c_valid": True,
            "origin_c_decision": {"split_id": int(split_id), "server_id": int(server_id)},
            "origin_c_context": {k: v for k, v in sorted(c_context.items())},
            "origin_r_action": origin_r_action,
        })
        context_list.append({k: v for k, v in sorted(c_context.items())})

    if len(env.event_queue) != 0:
        raise RuntimeError(f"episode ended with {len(env.event_queue)} events remaining")

    trace = {
        "schema_version": SCHEMA_VERSION,
        "origin": {"c_mode": args.c_mode, "r_mode": args.origin_r_mode},
        "topology": args.topology,
        "seed": args.seed,
        "warmup_requests": args.warmup_requests,
        "requests_per_episode": args.requests_per_episode,
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "config_hash": config_hash,
        "request_trace_hash": request_trace_hash,
        "c_context_hash": _c_context_hash(context_list),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requests": records,
    }
    return trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--c_mode", default="df_c")
    parser.add_argument("--c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--origin_r_mode", required=True, choices=ORIGIN_R_MODES)
    parser.add_argument("--ranker_spec", default=None, help="name=path for origin ranker (required if origin_r_mode is a ranker mode)")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths_c", type=int, default=5)
    parser.add_argument("--k_paths_r", type=int, default=50)
    parser.add_argument("--path_sort_strategy_c", default="hops")
    parser.add_argument("--path_sort_strategy_r", default="hops")
    parser.add_argument("--block_sort_strategy_c", default="start_asc")
    parser.add_argument("--block_sort_strategy_r", default="start_asc")
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--warmup_requests", type=int, default=1000)
    parser.add_argument("--requests_per_episode", type=int, default=5000)
    parser.add_argument("--poisson_arrivals", action="store_true")
    parser.add_argument("--exponential_holding", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v135_r_only_frozen_trace_audit/traces")
    parser.add_argument("--output_json", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.origin_r_mode in ev.RANKER_MODES and not args.ranker_spec:
        parser.error(f"--ranker_spec required when origin_r_mode={args.origin_r_mode}")

    trace = generate_trace(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_json is None:
        args.output_json = f"origin_{args.origin_r_mode}_seed_{args.seed}.json"
    out_path = out_dir / args.output_json
    out_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    print(f"[trace] Wrote {out_path}")
    print(f"[trace] request_trace_hash={trace['request_trace_hash']}")
    print(f"[trace] c_context_hash={trace['c_context_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
