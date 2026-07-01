"""Future-aware Oracle-R action selection.

For each candidate (path, modulation, block) that is physically feasible,
simulate the allocation on a cloned environment, compute future RMSA action
availability for upcoming requests, and select the action that succeeds while
minimising future feasibility degradation.

This is an *oracle* upper bound — it is too expensive for production use but
establishes what a perfect future-aware R policy could achieve.  If effective,
use it as a teacher for BC pretraining + PPO fine-tuning.
"""
from __future__ import annotations

import copy
import heapq
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.env.request import DNNRequest


# ---------------------------------------------------------------------------
# Helpers (duplicated from train_agent_r_fragaware_ppo.py to keep the oracle
# self-contained; kept in sync manually)
# ---------------------------------------------------------------------------

def _future_requests(env: SMDPEnv, horizon: int) -> List:
    if horizon <= 0:
        return []
    return sorted(env.event_queue)[:horizon]


def _advance_without_current_allocation(env: SMDPEnv) -> SMDPEnv:
    """Return a copy advanced to the current request arrival without allocation."""
    env_copy = copy.deepcopy(env)
    if not env_copy.event_queue:
        return env_copy
    arrival_time, _, _ = heapq.heappop(env_copy.event_queue)
    env_copy.advance_time(arrival_time)
    return env_copy


def _estimate_future_feasibility(
    env: SMDPEnv,
    agent_c: Any,
    horizon: int = 5,
    c_policy: str = "agent",
) -> float:
    """Estimate future RMSA action availability for the next H requests.

    For each upcoming request, advance a cloned environment to that request's
    arrival time, use the fixed C policy to choose split/server, then count
    valid R actions.  We do **not** allocate these future requests.

    Returns the mean number of valid R actions across the horizon.
    """
    horizon_events = _future_requests(env, horizon)
    if not horizon_events:
        return 0.0

    env_eval = copy.deepcopy(env)
    counts: List[float] = []
    for arrival_time, _, future_req in horizon_events:
        env_eval.advance_time(arrival_time)
        try:
            obs_c = build_agent_c_observation(env_eval, future_req)
        except Exception:
            counts.append(0.0)
            continue

        # Choose C action
        if c_policy == "agent" and agent_c is not None:
            from sa_hmarl.agents.ppo_agents import PPOAgentC
            if isinstance(agent_c, PPOAgentC):
                features_c, mask_c = agent_c.build_action_features(obs_c)
                # Apply risk mask if available
                ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
                if ckpt_args:
                    from sa_hmarl.env.c_action_risk import (
                        apply_agent_c_risk_mask,
                        risk_kwargs_from_args,
                    )
                    risk_kwargs = risk_kwargs_from_args(ckpt_args)
                    min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
                    mask_c = apply_agent_c_risk_mask(
                        obs_c, mask_c,
                        num_slots_total=int(ckpt_args.get("num_slots", 24)),
                        min_valid_after_mask=min_valid,
                        **risk_kwargs,
                    )
                action_idx_c, _, _ = agent_c.select_from_features(
                    features_c, mask_c, deterministic=True
                )
            else:
                # Fallback: use argmax over mask
                mask_c = obs_c.get("agent_c_mask", np.ones(len(obs_c.get("candidate_features", [])), dtype=bool))
                valid = np.where(mask_c)[0]
                action_idx_c = int(valid[0]) if len(valid) > 0 else None
        else:
            # greedy: pick minimum edge_compute_ms
            mask_c = obs_c.get("agent_c_mask", np.ones(len(obs_c.get("candidate_features", [])), dtype=bool))
            valid = np.where(mask_c)[0]
            if len(valid) == 0:
                counts.append(0.0)
                continue
            best_idx = int(valid[0])
            best_val = float(obs_c["candidate_features"][best_idx].get("edge_compute_ms", 1e9))
            for idx in valid[1:]:
                idx = int(idx)
                val = float(obs_c["candidate_features"][idx].get("edge_compute_ms", 1e9))
                if val < best_val:
                    best_idx = idx
                    best_val = val
            action_idx_c = best_idx

        if action_idx_c is None:
            counts.append(0.0)
            continue

        try:
            num_servers = len(env_eval.mec.servers)
            split_id, server_id = decode_agent_c_action(action_idx_c, num_servers)
            obs_r = build_agent_r_observation(env_eval, future_req, split_id, server_id)
            counts.append(float(np.sum(obs_r["agent_r_mask"])))
        except Exception:
            counts.append(0.0)

    return float(np.mean(counts)) if counts else 0.0


# ---------------------------------------------------------------------------
# Single-action simulation
# ---------------------------------------------------------------------------

def _simulate_r_allocation(
    env_sim: SMDPEnv,
    request: DNNRequest,
    split_id: int,
    server_id: int,
    path_idx: int,
    mod_idx: int,
    block_idx: int,
    obs_r: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Simulate a single R action allocation on *env_sim* (already a copy).

    Replicates the key steps of ``SMDPEnv._handle_arrival`` without mutating
    the original environment.

    Returns:
        (success, info_dict)
    """
    split = request.splits[split_id]
    server = env_sim.mec.servers[server_id]

    # --- 1. Server admission ---
    if not server.allocate_task(split.edge_compute_cost):
        return False, {"reason": "server_overload"}

    # --- 2. Get path ---
    paths = obs_r.get("candidate_paths", [])
    if path_idx < 0 or path_idx >= len(paths):
        server.release_task(split.edge_compute_cost)
        return False, {"reason": "invalid_path"}
    path = paths[path_idx]

    # --- 3. Modulation feasibility ---
    if mod_idx < 0 or mod_idx >= env_sim.mod_reg.num_formats:
        server.release_task(split.edge_compute_cost)
        return False, {"reason": "invalid_modulation"}
    mod = env_sim.mod_reg[mod_idx]
    path_dist = env_sim.net.path_length_km(path)
    if path_dist > mod.reach_km:
        server.release_task(split.edge_compute_cost)
        return False, {"reason": "modulation_reach"}

    # --- 4. FS demand (from pre-computed observation) ---
    req_fs = None
    try:
        req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    except (IndexError, KeyError):
        pass
    if req_fs is None or req_fs <= 0:
        server.release_task(split.edge_compute_cost)
        return False, {"reason": "deadline_infeasible"}

    if req_fs > env_sim.net.num_slots:
        server.release_task(split.edge_compute_cost)
        return False, {"reason": "fs_too_large"}

    # --- 5. Get candidate blocks from the *simulated* environment state ---
    # The blocks in obs_r were computed on env_before (time-advanced).
    # But after we allocated the server task, the spectrum might be unchanged.
    # We re-extract blocks to be safe.
    blocks = env_sim.net.get_candidate_blocks(
        path, req_fs,
        max_candidates=env_sim.max_blocks,
        sort_by=env_sim.block_sort_strategy,
    )
    if block_idx < 0 or block_idx >= len(blocks):
        server.release_task(split.edge_compute_cost)
        return False, {"reason": "no_suitable_block"}

    block = blocks[block_idx]
    start_slot = block.start_slot
    block_size = block.size

    # --- 6. Allocate optical resources ---
    allocated = env_sim.net.allocate(path, start_slot, req_fs)
    if not allocated:
        server.release_task(split.edge_compute_cost)
        return False, {"reason": "allocation_failed"}

    # --- 7. Record active connection ---
    release_time = env_sim.time + request.holding_time
    env_sim.active_connections.append({
        "release_time": release_time,
        "path": path,
        "start_slot": start_slot,
        "num_slots": req_fs,
        "server_id": server_id,
        "compute_cost": split.edge_compute_cost,
    })

    # --- 8. Compute delay ---
    num_hops = env_sim.net.path_num_hops(path)
    # Default constants (matching event_env.py)
    DEFAULT_PROP_SPEED_KM_S = 2.0e5
    DEFAULT_PROC_PER_HOP_S = 1e-6
    DEFAULT_SETUP_TIME_S = 1e-6
    prop_delay_ms = (path_dist / DEFAULT_PROP_SPEED_KM_S) * 1000.0
    proc_delay_ms = num_hops * DEFAULT_PROC_PER_HOP_S * 1000.0
    setup_delay_ms = DEFAULT_SETUP_TIME_S * 1000.0
    edge_compute_ms = SMDPEnv._estimate_compute_ms(server, split.edge_compute_cost)
    local_compute_ms = SMDPEnv._estimate_local_compute_ms(server, split.local_compute_cost)
    # edge_compute_ms was estimated *before* allocate_task, so for the delay calc
    # we re-estimate using the post-allocation state; but the difference is tiny,
    # so we just use the same formula.
    total_delay_ms = (
        prop_delay_ms + proc_delay_ms + setup_delay_ms
        + (local_compute_ms if local_compute_ms != float('inf') else 0.0)
        + (edge_compute_ms if edge_compute_ms != float('inf') else 0.0)
    )

    return True, {
        "success": True,
        "delay_ms": total_delay_ms,
        "path_dist_km": path_dist,
        "num_hops": num_hops,
        "start_slot": start_slot,
        "num_slots": req_fs,
        "modulation": mod.name,
        "block_size": block_size,
        "block_waste": (block_size - req_fs) / max(block_size, 1),
    }


# ---------------------------------------------------------------------------
# Oracle-R main entry point
# ---------------------------------------------------------------------------

def oracle_r_select(
    env: SMDPEnv,
    request: DNNRequest,
    action_c: Tuple[int, int],
    agent_c: Any,
    *,
    future_feas_coef: float = 1.0,
    future_feas_bonus_coef: float = 0.1,
    future_feas_horizon: int = 5,
    future_feas_norm: float = 60.0,
    waste_coef: float = 0.8,
    c_policy: str = "agent",
    verbose: bool = False,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Tuple[int, int, int]]]:
    """Oracle-R: simulate each candidate R action and select the best.

    For every valid (path, modulation, block) combination:
      1. Simulate the allocation on a cloned environment.
      2. Estimate future RMSA action availability for upcoming requests.
      3. Score = immediate_reward - coef * future_drop + bonus * future_gain.

    Args:
        env: Current environment (before time-advance for this request).
        request: The current DNN request to handle.
        action_c: (split_id, server_id) chosen by Agent-C.
        agent_c: Fixed Agent-C policy (PPOAgentC) for future feasibility estimation.
        future_feas_coef: Penalty weight for reducing future action availability.
        future_feas_bonus_coef: Bonus weight for increasing future availability.
        future_feas_horizon: Number of upcoming requests to consider.
        future_feas_norm: Normalisation factor for feasibility drops.
        waste_coef: Penalty for block waste in immediate reward.
        c_policy: "agent" or "greedy" for future feasibility C choices.
        verbose: If True, print per-candidate scoring details.

    Returns:
        (best_action_idx, debug_info, obs_r_before, decoded_action_r) 4-tuple.
        - best_action_idx: flat index into obs_r_before's action space, or None.
        - debug_info: dict with scoring details, or None.
        - obs_r_before: the R observation built on the time-advanced state
          (use this for correct decoding of best_action_idx).
        - decoded_action_r: (path_idx, mod_idx, block_idx) pre-decoded, or None.
    """
    split_id, server_id = action_c

    # --- Build the "before" state (time advanced, no allocation) ---
    env_before = _advance_without_current_allocation(env)

    # --- Build R observation on the time-advanced state ---
    obs_r = build_agent_r_observation(env_before, request, split_id, server_id)
    mask = obs_r["agent_r_mask"]

    if not np.any(mask):
        if verbose:
            print("  [oracle_r] no valid R actions in mask")
        return None, None, obs_r, None

    # --- Check server admission (shared across all R candidates) ---
    split = request.splits[split_id]
    server = env_before.mec.servers[server_id]
    if split.edge_compute_cost > server.available_compute:
        if verbose:
            print("  [oracle_r] server would reject — all R actions fail")
        return None, None, obs_r, None

    # --- Pre-compute future_before (baseline without this allocation) ---
    future_before = _estimate_future_feasibility(
        env_before, agent_c, horizon=future_feas_horizon, c_policy=c_policy,
    )

    num_modulations = len(obs_r["mod_names"])
    num_blocks = getattr(env_before, "max_blocks", 10)
    valid_indices = np.where(mask)[0]

    best_action: Optional[int] = None
    best_score: float = -float("inf")
    best_info: Dict[str, Any] = {}

    if verbose:
        t0 = _time.time()
        print(
            f"  [oracle_r] evaluating {len(valid_indices)} candidates "
            f"(future_before={future_before:.2f}, horizon={future_feas_horizon})"
        )

    for i, flat_idx in enumerate(valid_indices):
        flat_idx = int(flat_idx)
        path_idx = flat_idx // (num_modulations * num_blocks)
        remainder = flat_idx % (num_modulations * num_blocks)
        mod_idx = remainder // num_blocks
        block_idx = remainder % num_blocks

        # --- Simulate allocation ---
        env_sim = copy.deepcopy(env_before)
        success, sim_info = _simulate_r_allocation(
            env_sim, request, split_id, server_id,
            path_idx, mod_idx, block_idx, obs_r,
        )

        if not success:
            if verbose and i < 5:
                print(
                    f"    [{i}] ({path_idx},{mod_idx},{block_idx}) "
                    f"FAIL: {sim_info.get('reason', '?')}"
                )
            continue

        # --- Compute future feasibility after allocation ---
        future_after = _estimate_future_feasibility(
            env_sim, agent_c, horizon=future_feas_horizon, c_policy=c_policy,
        )

        # --- Score ---
        waste = float(sim_info.get("block_waste", 0.0))
        immediate_reward = 1.0 - waste_coef * waste

        norm = max(float(future_feas_norm), 1.0)
        future_drop = max(future_before - future_after, 0.0) / norm
        future_gain = max(future_after - future_before, 0.0) / norm

        score = (
            immediate_reward
            - future_feas_coef * future_drop
            + future_feas_bonus_coef * future_gain
        )

        if verbose and i < 5:
            print(
                f"    [{i}] ({path_idx},{mod_idx},{block_idx}) "
                f"reward={immediate_reward:.3f} future={future_before:.1f}→{future_after:.1f} "
                f"drop={future_drop:.3f} gain={future_gain:.3f} score={score:.4f}"
            )

        if score > best_score:
            best_score = score
            best_action = flat_idx
            best_info = {
                "score": float(score),
                "immediate_reward": float(immediate_reward),
                "future_before": float(future_before),
                "future_after": float(future_after),
                "future_drop": float(future_drop),
                "future_gain": float(future_gain),
                "path_idx": path_idx,
                "mod_idx": mod_idx,
                "block_idx": block_idx,
                "sim_info": sim_info,
            }

    if verbose:
        elapsed = _time.time() - t0
        n_success = sum(
            1 for _ in valid_indices
            if best_action is not None  # at least one succeeded
        )
        print(
            f"  [oracle_r] best=({best_info.get('path_idx')},{best_info.get('mod_idx')},"
            f"{best_info.get('block_idx')}) score={best_score:.4f} "
            f"({elapsed:.1f}s for {len(valid_indices)} candidates)"
        )

    if best_action is None:
        return None, None, obs_r, None

    # Pre-decode the action so callers don't need to use obs_r for decoding
    decoded = (
        best_info.get("path_idx", 0),
        best_info.get("mod_idx", 0),
        best_info.get("block_idx", 0),
    )
    return best_action, best_info, obs_r, decoded


# ---------------------------------------------------------------------------
# Batch data collection helper
# ---------------------------------------------------------------------------

def collect_oracle_episode(
    env_proto: SMDPEnv,
    requests: List[DNNRequest],
    agent_c: Any,
    *,
    future_feas_coef: float = 1.0,
    future_feas_bonus_coef: float = 0.1,
    future_feas_horizon: int = 5,
    future_feas_norm: float = 60.0,
    waste_coef: float = 0.8,
    c_policy: str = "agent",
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one episode with Oracle-R, collecting (obs, teacher_action) pairs.

    Returns:
        samples: list of dicts with keys:
            - features: np.ndarray (num_actions, input_dim)
            - mask: np.ndarray (num_actions,) bool
            - teacher_action: int (flat index)
        stats: dict with episode-level metrics.
    """
    from sa_hmarl.training.utils import compute_reward, make_env

    env = make_env(
        topology=env_proto.topology if hasattr(env_proto, 'topology') else "snap24_gnutella_reach",
        num_slots=env_proto.net.num_slots,
        num_servers=len(env_proto.mec.servers),
        seed=42,
        slot_bw_hz=1.25e9,
        guard_band_fs=1,
        modulation_profile="default",
        max_blocks=getattr(env_proto, "max_blocks", 10),
        block_sort_strategy=getattr(env_proto, "block_sort_strategy", "mixed"),
        server_nodes=[s.node_id for s in env_proto.mec.servers],
        capacities=[s.compute_capacity for s in env_proto.mec.servers],
    )
    env.reset(requests)

    samples: List[Dict[str, Any]] = []
    total = blocked = success = 0
    total_delay = total_fs = total_waste = 0.0
    reasons: Dict[str, int] = {}
    mods: Dict[str, int] = {}

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        if c_policy == "agent" and agent_c is not None:
            from sa_hmarl.agents.ppo_agents import PPOAgentC
            if isinstance(agent_c, PPOAgentC):
                features_c, mask_c = agent_c.build_action_features(obs_c)
                ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
                if ckpt_args:
                    from sa_hmarl.env.c_action_risk import (
                        apply_agent_c_risk_mask,
                        risk_kwargs_from_args,
                    )
                    risk_kwargs = risk_kwargs_from_args(ckpt_args)
                    min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
                    mask_c = apply_agent_c_risk_mask(
                        obs_c, mask_c,
                        num_slots_total=int(ckpt_args.get("num_slots", 24)),
                        min_valid_after_mask=min_valid,
                        **risk_kwargs,
                    )
                action_idx_c, _, _ = agent_c.select_from_features(
                    features_c, mask_c, deterministic=True
                )
            else:
                mask_c = obs_c.get("agent_c_mask")
                valid = np.where(mask_c)[0] if mask_c is not None else np.array([])
                action_idx_c = int(valid[0]) if len(valid) > 0 else None
        else:
            mask_c = obs_c.get("agent_c_mask")
            valid = np.where(mask_c)[0] if mask_c is not None else np.array([])
            if len(valid) == 0:
                action_idx_c = None
            else:
                best_idx = int(valid[0])
                best_val = float(obs_c["candidate_features"][best_idx].get("edge_compute_ms", 1e9))
                for idx in valid[1:]:
                    idx = int(idx)
                    val = float(obs_c["candidate_features"][idx].get("edge_compute_ms", 1e9))
                    if val < best_val:
                        best_idx = idx
                        best_val = val
                action_idx_c = best_idx

        action_c = (
            (0, 0)
            if action_idx_c is None
            else decode_agent_c_action(action_idx_c, len(env.mec.servers))
        )

        # Oracle-R selection
        oracle_action, oracle_info, obs_r_before, decoded_action_r = oracle_r_select(
            env, req, action_c, agent_c,
            future_feas_coef=future_feas_coef,
            future_feas_bonus_coef=future_feas_bonus_coef,
            future_feas_horizon=future_feas_horizon,
            future_feas_norm=future_feas_norm,
            waste_coef=waste_coef,
            c_policy=c_policy,
            verbose=verbose,
        )

        # Use obs_r_before for consistency with oracle's action indexing
        split_id, server_id = action_c
        if obs_r_before is None:
            obs_r_before = build_agent_r_observation(env, req, split_id, server_id)

        if oracle_action is not None:
            samples.append({
                "obs_r": obs_r_before,
                "teacher_action": oracle_action,
                "oracle_info": oracle_info,
            })

        # Execute the Oracle's action in the real environment
        if decoded_action_r is not None:
            action_r = decoded_action_r
        elif oracle_action is not None and obs_r_before is not None:
            action_r = decode_agent_r_action(
                oracle_action, len(obs_r_before["mod_names"]), getattr(env, "max_blocks", 10)
            )
        else:
            action_r = (0, 0, 0)

        _, _, _, info = env.step(action_c, action_r)
        total += 1

        if info.get("success", False):
            success += 1
            total_delay += float(info.get("delay_ms", 0.0))
            total_fs += float(info.get("num_slots", 0))
            total_waste += float(info.get("block_waste", 0.0))
            mods[info.get("modulation", "unknown")] = mods.get(info.get("modulation", "unknown"), 0) + 1
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1

    n = max(total, 1)
    s = max(success, 1)
    stats = {
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "reason_counter": reasons,
        "mod_counter": mods,
    }
    return samples, stats
