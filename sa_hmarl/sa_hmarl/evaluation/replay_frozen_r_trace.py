"""Replay a frozen C-side trace with multiple R-side methods.

The C-side decision sequence is read from the trace and applied unchanged.  Each
R-side method is replayed from a fresh environment so that the comparison is
over the fixed C-side policy, not over a shared branching rollout.  The
C-context observed by each R method is overwritten with the recorded origin
context so that C-side feature differences do not confound the R-side ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.evaluation import eval_strict_v13_multitopology_cside_verified as ev
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_r_action_from_obs,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


SCHEMA_VERSION = "frozen-trace-replay-v1.0"

R_MODES = ("ksp_ff_highest", "ppo_r_top1", "strict_v13", "v135_afterstate", "v135_afterstate_explicit")


@dataclass
class ReplayResult:
    r_mode: str
    origin: Dict[str, str]
    topology: str
    seed: int
    evaluated_requests: int = 0
    admitted: int = 0
    blocked: int = 0
    r_reached: int = 0
    r_decisions: int = 0
    c_no_valid_action: int = 0
    r_no_valid_action: int = 0
    server_overload: int = 0
    no_suitable_block: int = 0
    deadline_failure: int = 0
    other_failure: int = 0
    delay_sum: float = 0.0
    fs_sum: float = 0.0
    delay_ms_list: List[float] = field(default_factory=list)
    selected_actions: List[Dict[str, Any]] = field(default_factory=list)
    request_id_mismatch_count: int = 0
    consistency_error_count: int = 0
    queue_conservation_error: bool = False

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.evaluated_requests, 1))

    def avg_delay_ms(self) -> float:
        return float(np.mean(self.delay_ms_list)) if self.delay_ms_list else 0.0

    def avg_fs(self) -> float:
        return float(self.fs_sum / max(self.admitted, 1))

    def to_summary_dict(
        self,
        request_trace_hash: str,
        c_context_hash: str,
        selected_action_hash: str,
        config_hash: str,
    ) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "topology": self.topology,
            "seed": self.seed,
            "origin_c_mode": self.origin.get("c_mode"),
            "origin_r_mode": self.origin.get("r_mode"),
            "r_mode": self.r_mode,
            "evaluated_requests": self.evaluated_requests,
            "admitted": self.admitted,
            "blocked": self.blocked,
            "blocking_rate": self.blocking_rate(),
            "r_reached": self.r_reached,
            "r_decisions": self.r_decisions,
            "c_no_valid_action": self.c_no_valid_action,
            "r_no_valid_action": self.r_no_valid_action,
            "server_overload": self.server_overload,
            "no_suitable_block": self.no_suitable_block,
            "deadline_failure": self.deadline_failure,
            "other_failure": self.other_failure,
            "avg_delay_ms": self.avg_delay_ms(),
            "avg_fs": self.avg_fs(),
            "request_id_mismatch_count": self.request_id_mismatch_count,
            "consistency_error_count": self.consistency_error_count,
            "queue_conservation_error": self.queue_conservation_error,
            "request_trace_hash": request_trace_hash,
            "c_context_hash": c_context_hash,
            "selected_action_hash": selected_action_hash,
            "config_hash": config_hash,
        }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_hash(config: Dict[str, Any]) -> str:
    return _hash_text(json.dumps(config, sort_keys=True, default=str))


def _deserialize_request(rec: Dict[str, Any]) -> Any:
    """Rebuild a lightweight request-like object from a trace record."""
    from sa_hmarl.env.request import DNNRequest, SplitProfile

    splits = [
        SplitProfile(
            split_id=int(sp["split_id"]),
            intermediate_size_mb=float(sp["intermediate_size_mb"]),
            local_compute_cost=float(sp["local_compute_cost"]),
            edge_compute_cost=float(sp["edge_compute_cost"]),
        )
        for sp in rec["splits"]
    ]
    return DNNRequest(
        req_id=int(rec["req_id"]),
        src_node=int(rec["src_node"]),
        arrival_time=float(rec["arrival_time"]),
        holding_time=float(rec["holding_time"]),
        deadline_ms=float(rec["deadline_ms"]),
        splits=splits,
    )


def _load_rankers(ranker_specs: List[str], device: str) -> Dict[str, Dict[str, Any]]:
    rankers: Dict[str, Dict[str, Any]] = {}
    for spec in ranker_specs:
        if "=" not in spec:
            raise ValueError(f"Ranker spec must be name=path, got {spec}")
        name, path = spec.split("=", 1)
        rankers[name] = ev._load_ranker_model(path, device)
    return rankers


def _make_env_from_trace(trace: Dict[str, Any], args: argparse.Namespace):
    """Build a fresh environment matching the trace configuration."""
    cfg = trace.get("config", {})
    return make_env(
        trace["topology"],
        int(cfg.get("num_slots", args.num_slots)),
        int(cfg.get("num_servers", args.num_servers)),
        int(trace["seed"]),
        modulation_profile=cfg.get("modulation_profile", args.modulation_profile),
        max_blocks=int(cfg.get("max_blocks", args.max_blocks)),
        block_sort_strategy=cfg.get("block_sort_strategy_r", args.block_sort_strategy_r),
        path_sort_strategy=cfg.get("path_sort_strategy_r", args.path_sort_strategy_r),
        k=int(cfg.get("k_paths_r", args.k_paths_r)),
    )


def replay_one_method(
    trace: Dict[str, Any],
    args: argparse.Namespace,
    agent_r,
    ranker: Optional[Dict[str, Any]],
    r_mode: str,
) -> ReplayResult:
    origin = trace["origin"]
    topology = trace["topology"]
    seed = int(trace["seed"])
    warmup = int(trace.get("warmup_requests", args.warmup_requests))

    requests = [_deserialize_request(r) for r in trace["requests"]]
    request_trace_hash = trace["request_trace_hash"]
    c_context_hash = trace["c_context_hash"]
    config_hash = _config_hash(trace.get("config", {}))

    env = _make_env_from_trace(trace, args)
    env.reset(requests)
    num_servers = len(env.mec.servers)
    max_blocks = env.max_blocks
    num_mods = env.mod_reg.num_formats

    res = ReplayResult(r_mode=r_mode, origin=origin, topology=topology, seed=seed)

    for step_idx, rec in enumerate(trace["requests"]):
        req = requests[step_idx]
        is_warmup = step_idx < warmup

        # Request-sync check.
        if not env.event_queue:
            raise RuntimeError(f"event queue empty before req {req.req_id}")
        head_arrival, _, head_req = env.event_queue[0]
        if int(head_req.req_id) != int(req.req_id):
            res.request_id_mismatch_count += 1
            raise RuntimeError(
                f"request id mismatch at step {step_idx}: loop {req.req_id}, head {head_req.req_id}"
            )
        if float(head_arrival) != float(req.arrival_time):
            res.request_id_mismatch_count += 1
            raise RuntimeError(
                f"arrival_time mismatch at step {step_idx}: loop {req.arrival_time}, head {head_arrival}"
            )

        env.advance_time(req.arrival_time)

        origin_c_valid = rec["origin_c_valid"]

        if not origin_c_valid:
            env.reject_next_request(req.req_id, "c_no_valid_action")
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.c_no_valid_action += 1
                res.selected_actions.append({
                    "c_idx": None, "split_id": -1, "server_id": -1, "r_idx": None,
                    "c_valid": False, "r_valid": False,
                })
            continue

        split_id = int(rec["origin_c_decision"]["split_id"])
        server_id = int(rec["origin_c_decision"]["server_id"])
        recorded_context = rec["origin_c_context"]
        flat_c = split_id * num_servers + server_id

        # Rebuild C observation only to check whether the recorded action is still legal.
        env.k = int(trace["config"].get("k_paths_c", args.k_paths_c))
        env.path_sort_strategy = trace["config"].get("path_sort_strategy_c", args.path_sort_strategy_c)
        env.block_sort_strategy = trace["config"].get("block_sort_strategy_c", args.block_sort_strategy_c)
        obs_c = build_agent_c_observation(env, req)
        c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if not (0 <= flat_c < len(c_mask) and c_mask[flat_c]):
            env.reject_next_request(req.req_id, "c_no_valid_action")
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.c_no_valid_action += 1
                res.selected_actions.append({
                    "c_idx": int(flat_c), "split_id": split_id, "server_id": server_id,
                    "r_idx": None, "c_valid": False, "r_valid": False,
                })
            continue

        # Build R observation under K_R and overwrite C-context.
        env.k = int(trace["config"].get("k_paths_r", args.k_paths_r))
        env.path_sort_strategy = trace["config"].get("path_sort_strategy_r", args.path_sort_strategy_r)
        env.block_sort_strategy = trace["config"].get("block_sort_strategy_r", args.block_sort_strategy_r)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        obs_r["c_context"] = recorded_context

        if not is_warmup:
            res.r_reached += 1

        mask = np.asarray(obs_r.get("agent_r_mask", []), dtype=bool)
        if not mask.any():
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.r_no_valid_action += 1
                res.selected_actions.append({
                    "c_idx": int(flat_c), "split_id": split_id, "server_id": server_id,
                    "r_idx": None, "c_valid": True, "r_valid": False,
                })
            continue

        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, max_blocks)
        r_idx, r_valid, _ = ev._select_r_action(
            r_mode,
            agent_r,
            ranker,
            env,
            req,
            obs_c,
            obs_r,
            ppo_idx,
            split_id,
            server_id,
        )

        if not r_valid:
            env.reject_next_request(req.req_id, "r_no_valid_action")
            if not is_warmup:
                res.evaluated_requests += 1
                res.blocked += 1
                res.r_no_valid_action += 1
                res.selected_actions.append({
                    "c_idx": int(flat_c), "split_id": split_id, "server_id": server_id,
                    "r_idx": None, "c_valid": True, "r_valid": False,
                })
            continue

        r_action = decode_agent_r_action(r_idx, num_mods, max_blocks)
        try:
            ev._validate_selected_r_action(obs_r, r_idx, r_action, env.mod_reg)
        except ev.ActionObservationConsistencyError:
            res.consistency_error_count += 1
            raise

        _, _, _, info = env.step((split_id, server_id), r_action)
        if info.get("req_id") != req.req_id:
            res.request_id_mismatch_count += 1
            raise RuntimeError(
                f"env.step consumed wrong request: expected {req.req_id}, got {info.get('req_id')}"
            )

        if not info.get("success", False):
            reason = info.get("reason", "unknown")
            if reason in ("server_saturated", "server_overload"):
                res.server_overload += 1
            elif reason == "no_suitable_block":
                res.no_suitable_block += 1
            elif reason in ("deadline_infeasible", "deadline_failure"):
                res.deadline_failure += 1
            else:
                res.other_failure += 1
                res.consistency_error_count += 1
                raise ev.ActionObservationConsistencyError(
                    f"impossible failure {reason} after legal action {r_idx}"
                )

        if not is_warmup:
            res.evaluated_requests += 1
            res.r_decisions += 1
            res.selected_actions.append({
                "c_idx": int(flat_c), "split_id": split_id, "server_id": server_id,
                "r_idx": int(r_idx), "c_valid": True, "r_valid": True,
            })
            if info.get("success", False):
                res.admitted += 1
                delay_ms = float(info.get("delay_ms", 0.0))
                num_slots = float(info.get("num_slots", 0.0))
                res.delay_sum += delay_ms
                res.fs_sum += num_slots
                res.delay_ms_list.append(delay_ms)
            else:
                res.blocked += 1

    if len(env.event_queue) != 0:
        res.queue_conservation_error = True
        raise RuntimeError(f"episode ended with {len(env.event_queue)} events remaining")

    selected_action_hash = _hash_text(
        json.dumps(res.selected_actions, sort_keys=True, default=str)
    )
    return res.to_summary_dict(
        request_trace_hash, c_context_hash, selected_action_hash, config_hash
    )


def replay_trace(
    trace: Dict[str, Any],
    args: argparse.Namespace,
    agent_r,
    rankers: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    modes = [args.r_mode] if args.r_mode else R_MODES
    rows = []
    for r_mode in modes:
        ranker = rankers.get(r_mode) if r_mode in ev.RANKER_MODES else None
        row = replay_one_method(trace, args, agent_r, ranker, r_mode)
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, help="Path to frozen trace JSON")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranker_specs", action="append", default=[], help="name=path specs")
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
    parser.add_argument("--warmup_requests", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v135_r_only_frozen_trace_audit/replay")
    parser.add_argument("--output_json", default="results.json")
    parser.add_argument("--r_mode", default=None, help="Replay a single R method instead of all five.")
    return parser


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")

    parser = build_parser()
    args = parser.parse_args()

    trace_path = Path(args.trace)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    cfg = trace.get("config", {})
    mod_reg = ModulationRegistry.from_profile(
        cfg.get("modulation_profile", args.modulation_profile)
    )
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rankers = _load_rankers(args.ranker_specs, args.device)

    rows = replay_trace(trace, args, agent_r, rankers)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "trace_path": str(trace_path),
        "trace_origin": trace["origin"],
        "request_trace_hash": trace["request_trace_hash"],
        "c_context_hash": trace["c_context_hash"],
        "config_hash": _config_hash(trace.get("config", {})),
        "results": rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_json
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[replay] Wrote {out_path}")
    for row in rows:
        print(
            f"[replay] origin={row['origin_r_mode']:18s} r_mode={row['r_mode']:25s} "
            f"blocking={row['blocking_rate']:.4%} "
            f"admitted={row['admitted']:4d} blocked={row['blocked']:4d} "
            f"c_no={row['c_no_valid_action']:3d} r_no={row['r_no_valid_action']:3d} "
            f"consistency_errors={row['consistency_error_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
