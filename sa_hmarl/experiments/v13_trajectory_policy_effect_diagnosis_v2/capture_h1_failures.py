#!/usr/bin/env python3
"""Capture all H=1 non-zero records from the old horizon-counting bug."""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sa_hmarl"))

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.evaluation.optical_only_rmsa_env import OpticalOnlyRMSAEnv, generate_od_requests
from sa_hmarl.evaluation.optical_only_rmsa_evaluator import RANKER_CKPTS, _hash_requests
from sa_hmarl.network.modulation import ModulationRegistry

SNAPSHOT_PATH = Path("sa_hmarl/experiments/v135_r_only_standard_and_blocking_diagnosis/strict_snapshots.pkl")
ORIG_SEED = 3030
NUM_FUTURE_TRACES = 5


def _fork_env(env: OpticalOnlyRMSAEnv) -> OpticalOnlyRMSAEnv:
    new_env = OpticalOnlyRMSAEnv(
        topology=env.topology, num_slots=env.num_slots, k_paths=env.k_paths,
        max_blocks=env.max_blocks, path_sort_strategy=env.path_sort_strategy,
        block_sort_strategy=env.block_sort_strategy, mod_registry=env.mod_reg,
        slot_bw_hz=env.slot_bw_hz, guard_band_fs=env.guard_band_fs, seed=env.seed,
    )
    new_env.time = env.time
    new_env._counter = env._counter
    new_env._release_heap = copy.deepcopy(env._release_heap)
    new_env.active_connections = copy.deepcopy(env.active_connections)
    new_env.allocations = env.allocations
    new_env.releases = env.releases
    new_env.net.link_states = {k: v.copy() for k, v in env.net.link_states.items()}
    return new_env


def _generate_future(current_time, req_id, total, trace_idx):
    remaining = max(total - (req_id + 1), 0)
    if remaining == 0:
        return []
    rng = np.random.RandomState(ORIG_SEED * 100000 + req_id * 7 + trace_idx * 13 + 999)
    future = generate_od_requests(
        num_nodes=11, rng=rng, num_requests=remaining, arrival_interval=0.025,
        mean_holding_time=10.0, bitrate_min_gbps=25, bitrate_max_gbps=100,
    )
    first = future[0].arrival_time
    return [
        type(future[0])(
            req_id=r.req_id + req_id + 1, src_node=r.src_node, dst_node=r.dst_node,
            bitrate_gbps=r.bitrate_gbps, arrival_time=current_time + (r.arrival_time - first) + 0.025,
            holding_time=r.holding_time,
        ) for r in future
    ]


def _old_continue(env, future, horizon):
    # BUGGY: horizon interpreted as number of future requests.
    blocked = 0
    for req in future[:horizon]:
        env.advance_time(req.arrival_time)
        obs = env.build_observation(req)
        action = ksp_ff_highest_mod_action(obs)
        if action is None:
            blocked += 1
            continue
        info = env.step(action, obs, req.holding_time)
        if not info["success"]:
            blocked += 1
    return blocked


def main():
    with open(SNAPSHOT_PATH, "rb") as f:
        snapshots = pickle.load(f)
    mod_reg = ModulationRegistry.from_profile("default")
    total = 13000
    failures = []
    eligible = 0
    for req_id in sorted(snapshots.keys()):
        snap = snapshots[req_id]
        env = snap["env"]
        obs = snap["obs"]
        req = snap["request"]
        a_strict = snap["strict_action"]
        a_ksp = ksp_ff_highest_mod_action(obs)
        if a_strict is None or a_ksp is None or a_strict == a_ksp:
            continue
        mask = np.asarray(obs["agent_r_mask"], dtype=bool)
        if not (mask[a_strict] and mask[a_ksp]):
            continue
        env_strict = _fork_env(env)
        env_ksp = _fork_env(env)
        info_strict = env_strict.step(a_strict, obs, req.holding_time)
        info_ksp = env_ksp.step(a_ksp, obs, req.holding_time)
        if not (info_strict["success"] and info_ksp["success"]):
            continue
        eligible += 1
        for trace_idx in range(NUM_FUTURE_TRACES):
            future = _generate_future(env.time, req_id, total, trace_idx)
            if not future:
                continue
            fs = _old_continue(_fork_env(env_strict), future, 1)
            fk = _old_continue(_fork_env(env_ksp), future, 1)
            delta = fs - fk
            if delta != 0:
                failures.append({
                    "snapshot_id": req_id,
                    "future_trace_id": trace_idx,
                    "future_trace_hash": _hash_requests(future[:1]),
                    "a_strict": int(a_strict),
                    "a_ksp": int(a_ksp),
                    "first_success_strict": True,
                    "first_success_ksp": True,
                    "blocks_strict_h1": fs,
                    "blocks_ksp_h1": fk,
                    "delta_b_h1": delta,
                    "future_req_id_0": future[0].req_id,
                })
    out = {
        "n_eligible_snapshots": eligible,
        "n_h1_nonzero_records": len(failures),
        "fraction_h1_nonzero": len(failures) / max(eligible * NUM_FUTURE_TRACES, 1),
        "records": failures,
    }
    out_path = Path(__file__).resolve().parent / "H1_FAILURE_RECORDS.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Eligible: {eligible}, H1 non-zero records: {len(failures)}, wrote {out_path}")


if __name__ == "__main__":
    main()
