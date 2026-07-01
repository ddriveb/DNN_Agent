"""Phase 0: All-Block Oracle-R Diagnostic.

Tests whether the top-N (max_blocks=10) candidate block truncation
is a performance bottleneck.  The All-Block Oracle enumerates ALL
feasible contiguous blocks on every (path, mod) combination, not
just the top-N, and picks the one minimising a multi-objective
spectrum-preservation cost.

Usage::

    # Smoke
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_allblock_oracle_r \\
        --seeds 42 --episodes 1 --requests_per_episode 5

    # Full
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_allblock_oracle_r
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.spectrum_blocks import extract_contiguous_blocks
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Spectrum metrics (pre/post allocation)
# ---------------------------------------------------------------------------

def _path_lfb_and_frag(net, path: List[int]) -> Tuple[int, float, int, int]:
    """Return (lfb, frag_index, total_free, free_blocks) for a path."""
    avail = net.get_available_slots(path)
    total_free = int(np.sum(avail))
    lfb = net._max_consecutive(avail)
    free_blocks = net._count_blocks(avail)
    if total_free == 0:
        frag = 1.0
    elif total_free == net.num_slots:
        frag = 0.0
    else:
        frag = 1.0 - (lfb / total_free)
    return lfb, frag, total_free, free_blocks


# ---------------------------------------------------------------------------
# Oracle cost function
# ---------------------------------------------------------------------------

def _oracle_cost(
    info: Dict[str, Any],
    req_deadline_ms: float,
    lfb_drop: float,
    frag_increase: float,
) -> float:
    """Multi-objective spectrum-preservation cost (lower = better)."""
    if not info.get("success", False):
        return 100.0  # failure penalty
    fs_used = float(info.get("num_slots", 0))
    waste = float(info.get("block_waste", 0.0))
    delay = float(info.get("delay_ms", 0.0))
    deadline_violation = 1.0 if delay > req_deadline_ms else 0.0
    return (
        5.0 * deadline_violation
        + 1.0 * fs_used
        + 0.5 * waste
        + 0.3 * lfb_drop
        + 0.1 * frag_increase
        + 0.01 * delay
    )


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    fm = ckpt_args.get("agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"))
    idim = ckpt.get("input_dim", 17)
    if idim >= 24 and fm == "default":
        fm = "enhanced"
    agent = PPOAgentC(
        input_dim=idim, hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=fm,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    fm_r = ckpt.get("agent_r_feature_mode", ckpt.get("args", {}).get("agent_r_feature_mode", "default"))
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=fm_r,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def _eval_episode(
    agent_c: PPOAgentC, agent_r: PPOAgentR,
    requests: List, env_proto, args, ep_idx: int, seed: int,
) -> Dict[str, Any]:
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities, k=args.k_paths,
    )
    env.reset(requests)

    total = 0
    # BC-PPO-R
    bc_blocked = 0; bc_success = 0
    bc_total_delay = bc_total_fs = bc_total_waste = bc_total_path = 0.0
    bc_reason_ctr = Counter(); bc_mod_ctr = Counter()
    bc_split_ctr = Counter(); bc_server_ctr = Counter()
    # All-Block Oracle
    or_blocked = 0; or_success = 0
    or_total_delay = or_total_fs = or_total_waste = or_total_path = 0.0
    or_reason_ctr = Counter(); or_mod_ctr = Counter()
    # Comparison
    same_count = 0; help_count = 0; hurt_count = 0
    outside_topN_count = 0; outside_topN_help = 0
    # Oracle perf
    oracle_time_total = 0.0; oracle_cand_total = 0
    # C stats
    noc_count = 0; c_act_counts = []

    per_req_log: List[Dict] = []

    for i, req in enumerate(requests):
        obs_c = build_agent_c_observation(env, req)
        n_valid_c = int(obs_c["agent_c_mask"].sum())
        c_act_counts.append(n_valid_c)

        # Agent-C selection
        action_idx_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
        if action_idx_c is None or n_valid_c == 0:
            noc_count += 1
            _, _, _, info = env.step((0, 0), (0, 0, 0))
            total += 1
            if not info.get("success", False):
                bc_blocked += 1; bc_reason_ctr[info.get("reason", "unknown")] += 1
            else:
                bc_success += 1; d = float(info.get("delay_ms", 0)); bc_total_delay += d
                bc_total_fs += float(info.get("num_slots", 0)); bc_total_waste += float(info.get("block_waste", 0))
                bc_total_path += float(info.get("path_dist_km", 0))
            or_blocked += 1; or_reason_ctr["no_valid_c_action"] += 1
            per_req_log.append({"req": i, "no_valid_c": True})
            continue

        split_id, server_id = decode_agent_c_action(action_idx_c, num_servers)
        bc_split_ctr[f"split{split_id}"] += 1; bc_server_ctr[f"s{server_id}"] += 1

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask: np.ndarray = obs_r["agent_r_mask"]
        num_mods = len(obs_r["mod_names"]); max_blk_env = env.max_blocks
        paths = obs_r["candidate_paths"]
        fmask_pm = obs_r["feasible_mask_per_path_mod"]
        reqfs_pm = obs_r["required_fs_per_path_mod"]
        num_paths = len(paths)

        # --- Pre-allocation spectrum state (for LFB/frag delta) ---
        pre_lfb, pre_frag = {}, {}
        for p in range(num_paths):
            if p < len(paths):
                lfb, frag, _, _ = _path_lfb_and_frag(env.net, paths[p])
                pre_lfb[p] = lfb; pre_frag[p] = frag

        # --- BC-PPO-R on real env ---
        env_pre = copy.deepcopy(env)
        bc_idx = agent_r.select_action(obs_r, deterministic=True)
        bc_tuple: Tuple[int, int, int] = (0, 0, 0)
        bc_is_valid = (bc_idx is not None and bc_idx < len(r_mask) and r_mask[bc_idx])
        if bc_is_valid:
            bc_tuple = decode_agent_r_action(bc_idx, num_mods, max_blk_env)
        _, _, _, bc_info = env.step((split_id, server_id), bc_tuple)
        total += 1
        bc_ok = bc_info.get("success", False)

        if bc_ok:
            bc_success += 1; d = float(bc_info.get("delay_ms", 0))
            bc_total_delay += d; bc_total_fs += float(bc_info.get("num_slots", 0))
            bc_total_waste += float(bc_info.get("block_waste", 0))
            bc_total_path += float(bc_info.get("path_dist_km", 0))
            bc_mod_ctr[bc_info.get("modulation", "unknown")] += 1
        else:
            bc_blocked += 1; bc_reason_ctr[bc_info.get("reason", "unknown")] += 1

        # --- Build the top-N candidate set (the blocks BC-PPO-R could choose) ---
        topN_keys: set = set()
        for p in range(num_paths):
            for m in range(num_mods):
                if not fmask_pm[p][m]: continue
                try: req_fs = reqfs_pm[p][m]
                except IndexError: continue
                if req_fs is None or req_fs <= 0: continue
                blocks_pm = obs_r["candidate_blocks_per_path_mod"]
                blist = blocks_pm[p][m] if p < len(blocks_pm) and m < len(blocks_pm[p]) else []
                for b_idx, blk in enumerate(blist[:max_blk_env]):
                    if isinstance(blk, (tuple, list)) and len(blk) >= 2 and int(blk[1]) >= req_fs:
                        topN_keys.add((p, m, int(blk[0]), int(blk[1])))  # (path, mod, start, size)

        # --- All-Block Oracle on pre-step copy ---
        best_oracle_cost = float("inf")
        best_oracle_tuple: Tuple[int, int, int] = (0, 0, 0)
        oracle_found = False
        oracle_is_outside = False

        if len(paths) > 0:
            t0 = time.time()
            for p in range(num_paths):
                path_p = paths[p]
                avail = env_pre.net.get_available_slots(path_p)
                for m in range(num_mods):
                    if not fmask_pm[p][m]: continue
                    try: req_fs = reqfs_pm[p][m]
                    except IndexError: continue
                    if req_fs is None or req_fs <= 0: continue

                    # Get ALL feasible blocks (no top-N truncation)
                    all_blocks = extract_contiguous_blocks(avail, min_size=req_fs)
                    oracle_cand_total += len(all_blocks)

                    for start_slot, block_size in all_blocks:
                        # Try on deep copy
                        env_try = copy.deepcopy(env_pre)
                        _, _, _, info_try = env_try.step(
                            (split_id, server_id), (p, m, 0),
                        )
                        # Override block: env.step processes block_idx; we need to
                        # simulate with the specific all-block.  Since step() takes
                        # block_idx and looks up from candidate_blocks, we can't
                        # directly pass (start_slot, size).  We need to use a
                        # different approach.

            # --- The approach above is WRONG.  env.step() takes (path_idx, mod_idx, block_idx)
            # and looks up blocks from get_candidate_blocks which is top-N.
            # To use ALL blocks, we need a lower-level allocation.
            # Let me restart the oracle logic properly below. ---
            oracle_time_total += time.time() - t0

    # --- The function above has a structural issue: env.step() uses get_candidate_blocks
    # internally, which is limited to max_blocks.  We need to bypass this.
    # Let me rewrite with direct spectrum allocation. ---

    n = max(total, 1); s_bc = max(bc_success, 1)
    return {
        "bc_blocking": bc_blocked / n,
        "placeholder": True,  # Will be replaced by v2
    }


def _select_agent_c(agent_c, obs_c, num_slots):
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask, num_slots_total=num_slots,
            min_valid_after_mask=min_valid, **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


# ---------------------------------------------------------------------------
# Proper version: bypass get_candidate_blocks for oracle
# ---------------------------------------------------------------------------

def _allocate_direct(
    env, path: List[int], start_slot: int, req_fs: int, mod_idx: int,
    server_id: int, split, request_deadline_ms: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Direct spectrum allocation — bypasses get_candidate_blocks.

    Handles server admission, modulation check, FS calc, and direct
    optical allocation at a specific start_slot.  Used by the All-Block
    Oracle to test blocks outside the top-N candidate list.
    """
    server = env.mec.servers[server_id]
    edge_ms = env._estimate_compute_ms(server, split.edge_compute_cost)
    local_ms = env._estimate_local_compute_ms(server, split.local_compute_cost)

    if edge_ms == float("inf"):
        return False, {"success": False, "reason": "server_saturated"}

    if not server.allocate_task(split.edge_compute_cost):
        return False, {"success": False, "reason": "server_overload"}

    mod = env.mod_reg[mod_idx]
    path_dist = env.net.path_length_km(path)
    if path_dist > mod.reach_km:
        server.release_task(split.edge_compute_cost)
        return False, {"success": False, "reason": "modulation_reach"}

    num_hops = env.net.path_num_hops(path)
    edge_lengths = {}
    for u, v in zip(path[:-1], path[1:]):
        key = (min(u, v), max(u, v))
        edge_lengths[key] = env.net.G.edges[key]["length_km"]

    queue_delay_s = server._queue_delay_ema / 1000.0
    feasible, fs_req, t_data_max = env.fs_calc.compute_full(
        data_bits=split.intermediate_data_bits,
        deadline_s=request_deadline_ms / 1000.0,
        ctrl_delay_s=0.0,
        local_compute_s=local_ms / 1000.0,
        edge_compute_s=edge_ms / 1000.0,
        queue_delay_s=queue_delay_s,
        path=path, edge_lengths=edge_lengths, modulation=mod,
    )

    if not feasible:
        server.release_task(split.edge_compute_cost)
        return False, {"success": False, "reason": "deadline_infeasible"}

    if fs_req > env.net.num_slots:
        server.release_task(split.edge_compute_cost)
        return False, {"success": False, "reason": "fs_too_large"}

    # Direct allocation at specified start_slot
    allocated = env.net.allocate(path, start_slot, fs_req)
    if not allocated:
        server.release_task(split.edge_compute_cost)
        return False, {"success": False, "reason": "allocation_failed"}

    # Compute delay
    prop_ms = (path_dist / 200000.0) * 1000.0  # ~speed of light in fibre
    proc_ms = num_hops * 1e-6 * 1000.0  # negligible
    setup_ms = 0.1  # 0.1ms setup
    trans_ms = prop_ms + proc_ms + setup_ms
    total_delay_ms = trans_ms + edge_ms + local_ms

    return True, {
        "success": True,
        "delay_ms": total_delay_ms,
        "num_slots": fs_req,
        "block_waste": 1.0 - (fs_req / max(fs_req, 1)),  # simplified
        "path_dist_km": path_dist,
        "modulation": mod.name,
        "start_slot": start_slot,
    }


def _eval_episode_v2(
    agent_c: PPOAgentC, agent_r: PPOAgentR,
    requests: List, env_proto, args, ep_idx: int, seed: int,
) -> Dict[str, Any]:
    """Proper evaluation: BC-PPO-R vs All-Block Oracle.

    BC-PPO-R drives the real env.  Oracle enumerates all blocks on
    deep copies and picks the cost-minimising action.
    """
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities, k=args.k_paths,
    )
    env.reset(requests)

    total = 0
    bc_blocked = 0; bc_success = 0
    bc_total_delay = bc_total_fs = bc_total_waste = bc_total_path = 0.0
    bc_reason_ctr = Counter(); bc_mod_ctr = Counter()
    bc_split_ctr = Counter(); bc_server_ctr = Counter()
    or_blocked = 0; or_success = 0
    or_total_delay = or_total_fs = or_total_waste = or_total_path = 0.0
    or_reason_ctr = Counter(); or_mod_ctr = Counter()
    same_count = 0; help_count = 0; hurt_count = 0
    outside_topN = 0; outside_topN_help = 0
    oracle_time_total = 0.0; oracle_cand_total = 0
    noc_count = 0; c_act_counts = []
    per_req_log: List[Dict] = []

    for i, req in enumerate(requests):
        obs_c = build_agent_c_observation(env, req)
        n_valid_c = int(obs_c["agent_c_mask"].sum())
        c_act_counts.append(n_valid_c)

        action_idx_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
        if action_idx_c is None or n_valid_c == 0:
            noc_count += 1
            _, _, _, info = env.step((0, 0), (0, 0, 0))
            total += 1
            if not info.get("success", False):
                bc_blocked += 1; bc_reason_ctr[info.get("reason", "unknown")] += 1
            else:
                bc_success += 1; d = float(info.get("delay_ms", 0))
                bc_total_delay += d; bc_total_fs += float(info.get("num_slots", 0))
                bc_total_waste += float(info.get("block_waste", 0))
                bc_total_path += float(info.get("path_dist_km", 0))
            or_blocked += 1; or_reason_ctr["no_valid_c_action"] += 1
            per_req_log.append({"req": i, "no_valid_c": True})
            continue

        split_id, server_id = decode_agent_c_action(action_idx_c, num_servers)
        bc_split_ctr[f"split{split_id}"] += 1; bc_server_ctr[f"s{server_id}"] += 1
        split = req.splits[split_id]

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask: np.ndarray = obs_r["agent_r_mask"]
        num_mods = len(obs_r["mod_names"]); max_blk_env = env.max_blocks
        paths = obs_r["candidate_paths"]
        num_paths = len(paths)
        fmask_pm = obs_r["feasible_mask_per_path_mod"]
        reqfs_pm = obs_r["required_fs_per_path_mod"]
        blocks_pm = obs_r["candidate_blocks_per_path_mod"]
        path_features = obs_r["path_features"]

        # --- Pre-allocation LFB/frag per path ---
        pre_lfb = {}; pre_frag = {}
        for p in range(num_paths):
            lfb_val, frag_val, _, _ = _path_lfb_and_frag(env.net, paths[p])
            pre_lfb[p] = lfb_val; pre_frag[p] = frag_val

        # --- Build top-N key set for outside-topN tracking ---
        topN_keys: set = set()
        for p in range(num_paths):
            for m in range(num_mods):
                if not fmask_pm[p][m]: continue
                try: rfs = reqfs_pm[p][m]
                except IndexError: continue
                if rfs is None or rfs <= 0: continue
                blist = blocks_pm[p][m] if p < len(blocks_pm) and m < len(blocks_pm[p]) else []
                for b_idx, blk in enumerate(blist[:max_blk_env]):
                    if isinstance(blk, (tuple, list)) and len(blk) >= 2 and int(blk[1]) >= rfs:
                        topN_keys.add((p, m, int(blk[0])))

        # --- BC-PPO-R on real env ---
        env_pre = copy.deepcopy(env)
        bc_idx = agent_r.select_action(obs_r, deterministic=True)
        bc_tuple: Tuple[int, int, int] = (0, 0, 0)
        bc_p, bc_m, bc_start = -1, -1, -1
        bc_is_valid = (bc_idx is not None and bc_idx < len(r_mask) and r_mask[bc_idx])
        if bc_is_valid:
            bc_tuple = decode_agent_r_action(bc_idx, num_mods, max_blk_env)
            bc_p, bc_m, _ = bc_tuple
            # Get the actual block chosen by BC
            blist_bc = blocks_pm[bc_p][bc_m] if bc_p < len(blocks_pm) and bc_m < len(blocks_pm[bc_p]) else []
            bc_block_idx = bc_tuple[2]
            if bc_block_idx < len(blist_bc):
                bc_start = int(blist_bc[bc_block_idx][0]) if isinstance(blist_bc[bc_block_idx], (tuple, list)) else -1

        _, _, _, bc_info = env.step((split_id, server_id), bc_tuple)
        total += 1
        bc_ok = bc_info.get("success", False)
        bc_reason = bc_info.get("reason", "unknown")

        if bc_ok:
            bc_success += 1; d = float(bc_info.get("delay_ms", 0))
            bc_total_delay += d; bc_total_fs += float(bc_info.get("num_slots", 0))
            bc_total_waste += float(bc_info.get("block_waste", 0))
            bc_total_path += float(bc_info.get("path_dist_km", 0))
            bc_mod_ctr[bc_info.get("modulation", "unknown")] += 1
        else:
            bc_blocked += 1; bc_reason_ctr[bc_reason] += 1

        # --- All-Block Oracle on pre-step copy ---
        best_cost = float("inf")
        best_or_p = best_or_m = best_or_start = 0
        oracle_found = False

        t0 = time.time()
        for p in range(num_paths):
            if p >= len(paths): continue
            avail = env_pre.net.get_available_slots(paths[p])
            for m in range(num_mods):
                if not fmask_pm[p][m]: continue
                try: rfs = reqfs_pm[p][m]
                except IndexError: continue
                if rfs is None or rfs <= 0: continue

                all_blocks = extract_contiguous_blocks(avail, min_size=rfs)
                oracle_cand_total += len(all_blocks)

                for start_slot, blk_size in all_blocks:
                    env_try = copy.deepcopy(env_pre)
                    ok, info_try = _allocate_direct(
                        env_try, paths[p], start_slot, rfs, m,
                        server_id, split, req.deadline_ms,
                    )
                    # Post-allocation LFB/frag
                    post_lfb, post_frag, _, _ = _path_lfb_and_frag(env_try.net, paths[p])
                    lfb_drop = max(0.0, float(pre_lfb.get(p, 0)) - float(post_lfb))
                    frag_inc = max(0.0, float(post_frag) - float(pre_frag.get(p, 0)))

                    cost = _oracle_cost(info_try, req.deadline_ms, lfb_drop, frag_inc)
                    if cost < best_cost:
                        best_cost = cost
                        best_or_p = p; best_or_m = m; best_or_start = start_slot
                        oracle_found = True
        oracle_time_total += time.time() - t0

        # Oracle outcome
        or_ok = False; or_reason = "no_valid_block"; or_score = 100.0
        if oracle_found:
            env_or = copy.deepcopy(env_pre)
            ok, or_info = _allocate_direct(
                env_or, paths[best_or_p], best_or_start,
                reqfs_pm[best_or_p][best_or_m], best_or_m,
                server_id, split, req.deadline_ms,
            )
            or_ok = ok; or_reason = or_info.get("reason", "unknown")

            if or_ok:
                or_success += 1; d = float(or_info.get("delay_ms", 0))
                or_total_delay += d; or_total_fs += float(or_info.get("num_slots", 0))
                or_total_waste += float(or_info.get("block_waste", 0))
                or_total_path += float(or_info.get("path_dist_km", 0))
                or_mod_ctr[or_info.get("modulation", "unknown")] += 1
            else:
                or_blocked += 1; or_reason_ctr[or_reason] += 1
        else:
            or_blocked += 1; or_reason_ctr["no_valid_block"] += 1

        # --- Comparison ---
        oracle_topN_key = (best_or_p, best_or_m, best_or_start)
        is_outside = oracle_found and (oracle_topN_key not in topN_keys)
        if is_outside:
            outside_topN += 1
            if not bc_ok and or_ok:
                outside_topN_help += 1

        if bc_ok and not or_ok:
            hurt_count += 1
        if not bc_ok and or_ok:
            help_count += 1
        # Same action: same (path, mod, start_slot)
        if bc_is_valid and oracle_found and bc_p == best_or_p and bc_m == best_or_m and bc_start == best_or_start:
            same_count += 1

        per_req_log.append({
            "req": i, "n_valid_c": n_valid_c,
            "bc_ok": bc_ok, "bc_reason": bc_reason,
            "bc_p": bc_p, "bc_m": bc_m, "bc_start": bc_start,
            "or_ok": or_ok, "or_reason": or_reason,
            "or_p": best_or_p, "or_m": best_or_m, "or_start": best_or_start,
            "or_outside_topN": is_outside,
            "same_action": (bc_is_valid and oracle_found and bc_p == best_or_p and bc_m == best_or_m and bc_start == best_or_start),
            "help": (not bc_ok and or_ok),
            "hurt": (bc_ok and not or_ok),
        })

    n = max(total, 1); s_bc = max(bc_success, 1); s_or = max(or_success, 1)
    return {
        "total": total,
        "bc_blocking": bc_blocked / n, "bc_success_rate": bc_success / n,
        "bc_avg_delay_ms": bc_total_delay / s_bc, "bc_avg_fs": bc_total_fs / s_bc,
        "bc_avg_waste": bc_total_waste / s_bc, "bc_avg_path_km": bc_total_path / s_bc,
        "bc_reason_counter": dict(bc_reason_ctr), "bc_mod_counter": dict(bc_mod_ctr),
        "bc_split_counter": dict(bc_split_ctr), "bc_server_counter": dict(bc_server_ctr),
        "or_blocking": or_blocked / n, "or_success_rate": or_success / n,
        "or_avg_delay_ms": or_total_delay / s_or, "or_avg_fs": or_total_fs / s_or,
        "or_avg_waste": or_total_waste / s_or, "or_avg_path_km": or_total_path / s_or,
        "or_reason_counter": dict(or_reason_ctr), "or_mod_counter": dict(or_mod_ctr),
        "oracle_help_count": help_count, "oracle_hurt_count": hurt_count,
        "same_action_count": same_count,
        "oracle_help_rate": help_count / n, "oracle_hurt_rate": hurt_count / n,
        "same_action_rate": same_count / n,
        "outside_topN_count": outside_topN,
        "outside_topN_rate": outside_topN / n,
        "outside_topN_help": outside_topN_help,
        "no_valid_c_count": noc_count, "no_valid_c_rate": noc_count / n,
        "avg_valid_c_actions": float(np.mean(c_act_counts)) if c_act_counts else 0.0,
        "avg_oracle_candidates_per_req": float(oracle_cand_total) / n,
        "oracle_time_s": oracle_time_total,
        "per_req_log": per_req_log,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(c: Dict) -> str:
    ct = Counter(c); t = sum(ct.values())
    if t <= 0: return "none"
    return ", ".join(f"{k}={v / t:.1%}" for k, v in ct.most_common())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="All-Block Oracle-R Diagnostic")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/allblock_oracle_r_eval.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/allblock_oracle_r_eval.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"  input_dim={agent_c.input_dim} fm={agent_c.feature_mode}")
    print(f"Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )

    all_eps: Dict[int, List] = {}
    src_nodes: Dict[int, int] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed); eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_eps[seed] = eps
        src_nodes[seed] = eps[0][0].src_node if eps and eps[0] else -1

    print(f"\n{'=' * 80}")
    print("All-Block Oracle-R Diagnostic")
    print(f"Slots: {args.num_slots}  k: {args.k_paths}  max_blocks: {args.max_blocks}")
    print(f"Seeds: {seeds}  Eps/seed: {args.episodes}  Req/ep: {args.requests_per_episode}")
    print(f"Source nodes: {src_nodes}")
    print(f"{'=' * 80}")

    t_start = time.time()
    ep_results = []
    for seed in seeds:
        for ep_idx, requests in enumerate(all_eps[seed]):
            res = _eval_episode_v2(agent_c, agent_r, requests, env_proto, args, ep_idx, seed)
            ep_results.append(res)
            print(f"  seed={seed:4d} ep={ep_idx}: "
                  f"BC blk={res['bc_blocking']:.3f}  OR blk={res['or_blocking']:.3f}  "
                  f"same={res['same_action_rate']:.1%}  "
                  f"help={res['oracle_help_count']}/{res['total']}  "
                  f"outTopN={res['outside_topN_count']}/{res['total']} "
                  f"outHelp={res['outside_topN_help']}  "
                  f"avgCands={res['avg_oracle_candidates_per_req']:.0f}")

    total_elapsed = time.time() - t_start

    # Aggregate
    scalar_keys = [
        "bc_blocking", "bc_avg_delay_ms", "bc_avg_fs", "bc_avg_waste", "bc_avg_path_km",
        "or_blocking", "or_avg_delay_ms", "or_avg_fs", "or_avg_waste", "or_avg_path_km",
        "oracle_help_rate", "oracle_hurt_rate", "same_action_rate",
        "outside_topN_rate", "no_valid_c_rate", "avg_valid_c_actions",
        "avg_oracle_candidates_per_req",
    ]
    agg: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(r[key]) for r in ep_results]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    for key in ["bc_reason_counter", "bc_mod_counter", "bc_split_counter", "bc_server_counter",
                "or_reason_counter", "or_mod_counter"]:
        c = Counter()
        for r in ep_results:
            c.update(r.get(key, {}))
        agg[key] = dict(c)

    agg["total_requests"] = sum(r["total"] for r in ep_results)
    agg["total_oracle_help"] = sum(r["oracle_help_count"] for r in ep_results)
    agg["total_oracle_hurt"] = sum(r["oracle_hurt_count"] for r in ep_results)
    agg["total_same_action"] = sum(r["same_action_count"] for r in ep_results)
    agg["total_outside_topN"] = sum(r["outside_topN_count"] for r in ep_results)
    agg["total_outside_topN_help"] = sum(r["outside_topN_help"] for r in ep_results)
    agg["total_oracle_time_s"] = sum(r["oracle_time_s"] for r in ep_results)
    agg["delta_blocking"] = agg["bc_blocking"] - agg["or_blocking"]

    # Per-seed
    eps_per_seed = args.episodes
    per_seed_agg = {}
    for si, seed in enumerate(seeds):
        start = si * eps_per_seed
        seed_res = ep_results[start:start + eps_per_seed]
        ps: Dict[str, Any] = {}
        for key in ["bc_blocking", "or_blocking", "oracle_help_rate", "same_action_rate",
                     "outside_topN_rate", "avg_oracle_candidates_per_req"]:
            ps[key] = float(np.mean([float(r[key]) for r in seed_res]))
        ps["outside_topN_help"] = sum(r["outside_topN_help"] for r in seed_res)
        per_seed_agg[seed] = ps

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    d_blk = agg["delta_blocking"]
    print(f"  BC-PPO-R blocking:            {agg['bc_blocking']:.4f} ± {agg['bc_blocking_std']:.4f}")
    print(f"  All-Block Oracle blocking:    {agg['or_blocking']:.4f} ± {agg['or_blocking_std']:.4f}")
    print(f"  Δblocking (BC − oracle):      {d_blk:+.4f}")
    print(f"  Same action rate:             {agg['same_action_rate']:.1%}")
    print(f"  Oracle help rate:             {agg['oracle_help_rate']:.1%} ({agg['total_oracle_help']}/{agg['total_requests']})")
    print(f"  Oracle hurt rate:             {agg['oracle_hurt_rate']:.1%} ({agg['total_oracle_hurt']}/{agg['total_requests']})")
    print(f"  Outside-topN rate:            {agg['outside_topN_rate']:.1%} ({agg['total_outside_topN']}/{agg['total_requests']})")
    print(f"  Outside-topN help:            {agg['total_outside_topN_help']} (BC fail→OR ok via outside block)")
    print(f"  Avg all-blocks per request:   {agg['avg_oracle_candidates_per_req']:.0f}")
    print(f"  BC delay: {agg['bc_avg_delay_ms']:.1f}ms  OR delay: {agg['or_avg_delay_ms']:.1f}ms")
    print(f"  BC FS:    {agg['bc_avg_fs']:.2f}       OR FS:    {agg['or_avg_fs']:.2f}")
    print(f"  Total oracle time: {agg['total_oracle_time_s']:.0f}s")

    print(f"\n  Per-seed:")
    for seed in seeds:
        ps = per_seed_agg[seed]
        print(f"    seed={seed:4d}: BC={ps['bc_blocking']:.4f}  OR={ps['or_blocking']:.4f}  "
              f"Δ={ps['bc_blocking']-ps['or_blocking']:+.4f}  "
              f"outTopN={ps['outside_topN_rate']:.1%}  outHelp={ps['outside_topN_help']}")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    def _clean(obj):
        if isinstance(obj, dict): return {str(k): _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    out_json.write_text(json.dumps({
        "args": vars(args), "source_nodes": src_nodes, "total_time_s": total_elapsed,
        "aggregate": _clean(agg), "per_seed": _clean(per_seed_agg),
    }, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# All-Block Oracle-R Diagnostic\n\n")
    md.append(f"**Slots:** {args.num_slots}  **k:** {args.k_paths}  "
              f"**max_blocks (BC):** {args.max_blocks}  **Arrival:** {args.arrival_interval}s\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}  "
              f"**Req/ep:** {args.requests_per_episode}  **Source nodes:** {src_nodes}\n\n")
    md.append(f"**Agent-C:** `{args.agent_c_checkpoint}`  \n")
    md.append(f"**Agent-R:** `{args.agent_r_checkpoint}`  \n\n")
    md.append(f"**Total time:** {total_elapsed:.0f}s  **Oracle time:** {agg['total_oracle_time_s']:.0f}s\n\n")

    md.append("## Core Result\n\n")
    md.append("| Metric | BC-PPO-R | All-Block Oracle | Δ |\n")
    md.append("|--------|----------|------------------|---|\n")
    md.append(f"| Blocking | {agg['bc_blocking']:.4f} | {agg['or_blocking']:.4f} | **{d_blk:+.4f}** |\n")
    md.append(f"| Delay (ms) | {agg['bc_avg_delay_ms']:.1f} | {agg['or_avg_delay_ms']:.1f} | |\n")
    md.append(f"| Avg FS | {agg['bc_avg_fs']:.2f} | {agg['or_avg_fs']:.2f} | |\n")
    md.append(f"| Avg Waste | {agg['bc_avg_waste']:.3f} | {agg['or_avg_waste']:.3f} | |\n")
    md.append(f"| Avg Path (km) | {agg['bc_avg_path_km']:.1f} | {agg['or_avg_path_km']:.1f} | |\n\n")

    md.append("## Oracle Gap Metrics\n\n")
    md.append(f"- **Same action rate:** {agg['same_action_rate']:.1%} "
              f"({agg['total_same_action']}/{agg['total_requests']})\n")
    md.append(f"- **Oracle help rate:** {agg['oracle_help_rate']:.1%} "
              f"(BC fail → oracle succeed, {agg['total_oracle_help']}/{agg['total_requests']})\n")
    md.append(f"- **Oracle hurt rate:** {agg['oracle_hurt_rate']:.1%} "
              f"(BC succeed → oracle fail, {agg['total_oracle_hurt']}/{agg['total_requests']})\n")
    md.append(f"- **Outside-topN rate:** {agg['outside_topN_rate']:.1%} "
              f"(oracle chose block outside top-{args.max_blocks}, {agg['total_outside_topN']}/{agg['total_requests']})\n")
    md.append(f"- **Outside-topN success gain:** {agg['total_outside_topN_help']} "
              f"(BC failed but oracle succeeded via outside-topN block)\n")
    md.append(f"- **Avg all-blocks per request:** {agg['avg_oracle_candidates_per_req']:.0f} "
              f"(vs top-{args.max_blocks})\n\n")

    md.append("## Per-Seed Breakdown\n\n")
    md.append("| Seed | Src | BC Blk | OR Blk | ΔBlk | Help% | Same% | OutTopN% | OutHelp |\n")
    md.append("|------|-----|--------|--------|------|-------|-------|----------|--------|\n")
    for seed in seeds:
        ps = per_seed_agg[seed]
        md.append(f"| {seed} | {src_nodes[seed]} | {ps['bc_blocking']:.4f} | {ps['or_blocking']:.4f} | "
                  f"{ps['bc_blocking']-ps['or_blocking']:+.4f} | {ps['oracle_help_rate']:.1%} | "
                  f"{ps['same_action_rate']:.1%} | {ps['outside_topN_rate']:.1%} | {ps['outside_topN_help']} |\n")

    # Verdict
    md.append("\n## Verdict\n\n")
    if d_blk >= 0.02 and agg["total_outside_topN_help"] > 0:
        md.append("### ✓ Top-N block truncation IS a bottleneck\n\n")
        md.append(f"All-Block Oracle reduces blocking by {d_blk:.1%} (≥ 2pp), "
                  f"with {agg['total_outside_topN_help']} successes from outside-topN blocks.\n\n")
        md.append("**→ Proceed to AllFeasibleBlock-R architecture.**\n")
    elif d_blk > 0.005:
        md.append(f"### ~ Marginal improvement (Δblk = {d_blk:+.4f})\n\n")
        md.append(f"Outside-topN help = {agg['total_outside_topN_help']}. ")
        md.append("Top-N truncation is NOT the main bottleneck.\n")
    elif agg["total_outside_topN_help"] > 0:
        md.append(f"### ~ Oracle blocking similar, but outside-topN blocks help in {agg['total_outside_topN_help']} cases\n\n")
        md.append("Top-N expansion could help at the margin but is not the main lever.\n")
    else:
        md.append(f"### ✗ Top-N block truncation is NOT a bottleneck\n\n")
        md.append(f"Δblocking = {d_blk:+.4f}, outside-topN help = {agg['total_outside_topN_help']}.\n")
        md.append("All blocks beyond top-N provide zero additional successful allocations.\n\n")
        md.append("**Stop R action-space expansion.  The bottleneck is elsewhere.**\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # Verdict
    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    print(f"BC blk={agg['bc_blocking']:.4f}  Oracle blk={agg['or_blocking']:.4f}  "
          f"Δ={d_blk:+.4f}  outTopNHelp={agg['total_outside_topN_help']}")
    if d_blk >= 0.02 and agg["total_outside_topN_help"] > 0:
        print("✓ TOP-N IS BOTTLENECK → proceed to AllFeasibleBlock-R")
    elif agg["total_outside_topN_help"] > 0:
        print("~ MARGINAL — top-N expansion helps but is not the main lever")
    else:
        print("✗ TOP-N NOT BOTTLENECK — stop R action-space expansion")


if __name__ == "__main__":
    main()
