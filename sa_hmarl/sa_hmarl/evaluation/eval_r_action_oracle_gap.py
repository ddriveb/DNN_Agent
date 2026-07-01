"""R-Action Oracle Gap Diagnostic.

For each Agent-C (split, server) choice, enumerates ALL valid R actions
from the mask, simulates each on a deep copy, and picks the one-step
optimal action via a composite score.  Compares BC-PPO-R against this
oracle to quantify the remaining R-side learning potential.

Usage::

    # Smoke
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_r_action_oracle_gap \\
        --seeds 42 --episodes 1 --requests_per_episode 5

    # Full
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_r_action_oracle_gap
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
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Oracle score (lower = better)
# ---------------------------------------------------------------------------

def _oracle_score(info: Dict[str, Any], req_deadline_ms: float) -> float:
    """Composite one-step score for an R action outcome."""
    if not info.get("success", False):
        return 100.0  # failed → heavy penalty
    delay = float(info.get("delay_ms", 0.0))
    deadline_violation = 1.0 if delay > req_deadline_ms else 0.0
    fs_val = float(info.get("num_slots", 0))
    waste = float(info.get("block_waste", 0.0))
    path_km = float(info.get("path_dist_km", 0.0))
    return (
        5.0 * deadline_violation
        + 1.0 * fs_val
        + 0.5 * waste
        + 0.01 * delay
        + 0.0001 * path_km
    )


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    fm = ckpt_args.get("agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"))
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
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
# Agent-C selection
# ---------------------------------------------------------------------------

def _select_agent_c(agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int) -> Optional[int]:
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
# Main evaluation per episode
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
    # BC-PPO-R stats
    bc_blocked = 0; bc_success = 0
    bc_total_delay = bc_total_fs = bc_total_waste = bc_total_path = 0.0
    bc_reason_ctr = Counter()
    bc_mod_ctr = Counter()
    # Oracle stats
    or_blocked = 0; or_success = 0
    or_total_delay = or_total_fs = or_total_waste = or_total_path = 0.0
    or_reason_ctr = Counter()
    or_mod_ctr = Counter()
    # Comparison stats
    oracle_help_count = 0   # BC failed, oracle succeeded
    oracle_hurt_count = 0   # BC succeeded, oracle failed
    same_action_count = 0   # BC action == oracle action
    bc_regret_sum = 0.0     # sum of (bc_score - oracle_score)
    bc_regret_count = 0     # count of comparisons where both had valid actions
    # Oracle enumeration stats
    total_oracle_candidates = 0
    total_oracle_time = 0.0
    # Per-request diagnostics
    per_request_diags: List[Dict[str, Any]] = []

    # C mask stats
    noc_count = 0

    for i, req in enumerate(requests):
        obs_c = build_agent_c_observation(env, req)

        # Agent-C selection
        action_idx_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
        if action_idx_c is None:
            noc_count += 1
            action_c = (0, 0)
        else:
            action_c = decode_agent_c_action(action_idx_c, num_servers)
        split_id, server_id = action_c

        # Build R observation
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask: np.ndarray = obs_r["agent_r_mask"]
        valid_r = np.where(r_mask)[0]
        n_valid_r = len(valid_r)
        total_oracle_candidates += n_valid_r

        # BC-PPO-R selection
        bc_action_idx = agent_r.select_action(obs_r, deterministic=True)
        bc_action_tuple: Tuple[int, int, int] = (0, 0, 0)
        if bc_action_idx is not None and bc_action_idx < len(r_mask) and r_mask[bc_action_idx]:
            bc_action_tuple = decode_agent_r_action(
                bc_action_idx, len(obs_r["mod_names"]), env.max_blocks,
            )
        # else: BC action is invalid → (0,0,0) → will fail

        # Oracle: enumerate all valid R actions, pick best by score
        oracle_action_idx: Optional[int] = None
        oracle_action_tuple: Tuple[int, int, int] = (0, 0, 0)
        oracle_best_score = float("inf")
        oracle_found_any = False

        if n_valid_r > 0:
            t0 = time.time()
            num_mods = len(obs_r["mod_names"])
            max_blk = env.max_blocks
            for aidx in valid_r:
                aidx = int(aidx)
                p, m, b = decode_agent_r_action(aidx, num_mods, max_blk)
                env_try = copy.deepcopy(env)
                _, _, _, info_try = env_try.step(
                    (split_id, server_id), (p, m, b),
                )
                score = _oracle_score(info_try, req.deadline_ms)
                if score < oracle_best_score:
                    oracle_best_score = score
                    oracle_action_idx = aidx
                    oracle_action_tuple = (p, m, b)
                    oracle_found_any = True
            total_oracle_time += time.time() - t0

        # Compute BC score for the same action (using deep copy to not double-advance)
        if bc_action_tuple != (0, 0, 0):
            env_bc_try = copy.deepcopy(env)
            _, _, _, bc_info_try = env_bc_try.step(
                (split_id, server_id), bc_action_tuple,
            )
            bc_score = _oracle_score(bc_info_try, req.deadline_ms)
        else:
            bc_score = 100.0  # invalid BC action

        # Execute BC-PPO-R on the REAL env
        _, _, _, bc_info = env.step(action_c, bc_action_tuple)
        total += 1

        bc_success_flag = bc_info.get("success", False)
        bc_reason = bc_info.get("reason", "unknown")

        if bc_success_flag:
            bc_success += 1
            d = float(bc_info.get("delay_ms", 0.0))
            bc_total_delay += d
            bc_total_fs += float(bc_info.get("num_slots", 0))
            bc_total_waste += float(bc_info.get("block_waste", 0.0))
            bc_total_path += float(bc_info.get("path_dist_km", 0.0))
            bc_mod_ctr[bc_info.get("modulation", "unknown")] += 1
        else:
            bc_blocked += 1
            bc_reason_ctr[bc_reason] += 1

        # Oracle outcome (from deep copy, does not affect env)
        if oracle_found_any:
            env_or_try = copy.deepcopy(env)
            _, _, _, or_info = env_or_try.step(
                (split_id, server_id), oracle_action_tuple,
            )
            # NOTE: env_or_try state differs from real env because BC already consumed
            # the request.  We use the pre-step deep copy (env_before_step state).
            # Actually we need to use the pre-BC-step env.  Let's re-do:
            # The env_bc_try was a pre-step copy.  We use that for oracle comparison.
            env_pre_step = copy.deepcopy(env)
            # Actually env already consumed the request via BC step.  We need
            # to compare using the PRE-STEP state.
            # Let me restructure: save pre-step env, run BC on it (discard outcome for
            # score comparison), run oracle on another copy.
            # We already have env_bc_try which was a pre-step copy where we ran BC.
            # We need another pre-step copy for oracle.
            # Let me fix this below by restructuring.
            pass

        # --- Actually, the above has a bug: env_bc_try consumed its request.
        # I need to restructure: make TWO pre-step copies, one for BC, one for oracle.
        # Let me fix this in the code above by restructuring.
        # For now, let me redo the comparison logic cleanly.

    # --- Wait, the above has a structural issue.  Let me rewrite this function
    # properly below. ---

    # This function is structurally flawed.  Let me rewrite it with proper pre-step
    # deep copies for BC-vs-oracle comparison.

    n = max(total, 1); s_bc = max(bc_success, 1)
    # Note: the rest of this function won't be reached due to the structural
    # issue noted above.  I'll rewrite the function entirely.
    return {
        "bc_blocking": bc_blocked / n if n > 0 else 0,
        "placeholder": True,
    }


# ---------------------------------------------------------------------------
# Proper per-episode evaluation (rewrite)
# ---------------------------------------------------------------------------

def _eval_episode_v2(
    agent_c: PPOAgentC, agent_r: PPOAgentR,
    requests: List, env_proto, args, ep_idx: int, seed: int,
) -> Dict[str, Any]:
    """Evaluate one episode: BC-PPO-R vs one-step R oracle.

    The BC-PPO-R actions drive the REAL environment forward.
    Oracle actions are evaluated on pre-step deep copies for
    fair comparison (same env state before the C+R decision).
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
    # BC (real env)
    bc_blocked = 0; bc_success = 0
    bc_total_delay = bc_total_fs = bc_total_waste = bc_total_path = 0.0
    bc_reason_ctr = Counter()
    bc_mod_ctr = Counter()
    bc_split_ctr = Counter(); bc_server_ctr = Counter()
    # Oracle (pre-step copy)
    or_blocked = 0; or_success = 0
    or_total_delay = or_total_fs = or_total_waste = or_total_path = 0.0
    or_reason_ctr = Counter()
    or_mod_ctr = Counter()
    # Comparison
    help_count = 0; hurt_count = 0; same_count = 0
    regret_sum = 0.0; regret_n = 0
    # Oracle perf
    oracle_time_total = 0.0; oracle_cand_total = 0
    # C mask
    noc_count = 0; c_act_counts = []

    per_req_log: List[Dict] = []

    for i, req in enumerate(requests):
        obs_c = build_agent_c_observation(env, req)
        action_idx_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
        n_valid_c = int(obs_c["agent_c_mask"].sum())
        c_act_counts.append(n_valid_c)
        if action_idx_c is None or n_valid_c == 0:
            noc_count += 1
            # No valid C action → skip (request blocked at C level)
            _, _, _, info = env.step((0, 0), (0, 0, 0))
            total += 1
            if not info.get("success", False):
                bc_blocked += 1
                bc_reason_ctr[info.get("reason", "unknown")] += 1
            else:
                bc_success += 1
            or_blocked += 1  # Oracle also can't help if C fails
            or_reason_ctr["no_valid_c_action"] += 1
            per_req_log.append({"req": i, "no_valid_c": True})
            continue

        split_id, server_id = decode_agent_c_action(action_idx_c, num_servers)
        bc_split_ctr[f"split{split_id}"] += 1
        bc_server_ctr[f"s{server_id}"] += 1

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask: np.ndarray = obs_r["agent_r_mask"]
        valid_r = np.where(r_mask)[0]
        n_valid_r = len(valid_r)
        oracle_cand_total += n_valid_r
        num_mods = len(obs_r["mod_names"]); max_blk = env.max_blocks

        # --- Pre-step deep copies for oracle comparison ---
        env_pre = copy.deepcopy(env)

        # --- BC-PPO-R on real env ---
        bc_idx = agent_r.select_action(obs_r, deterministic=True)
        bc_tuple: Tuple[int, int, int] = (0, 0, 0)
        bc_valid = (bc_idx is not None and bc_idx < len(r_mask) and r_mask[bc_idx])
        if bc_valid:
            bc_tuple = decode_agent_r_action(bc_idx, num_mods, max_blk)

        _, _, _, bc_info = env.step((split_id, server_id), bc_tuple)
        total += 1
        bc_ok = bc_info.get("success", False)
        bc_reason = bc_info.get("reason", "unknown")
        bc_score_val = _oracle_score(bc_info, req.deadline_ms)

        if bc_ok:
            bc_success += 1
            d = float(bc_info.get("delay_ms", 0.0))
            bc_total_delay += d; bc_total_fs += float(bc_info.get("num_slots", 0))
            bc_total_waste += float(bc_info.get("block_waste", 0.0))
            bc_total_path += float(bc_info.get("path_dist_km", 0.0))
            bc_mod_ctr[bc_info.get("modulation", "unknown")] += 1
        else:
            bc_blocked += 1
            bc_reason_ctr[bc_reason] += 1

        # --- Oracle on pre-step copy ---
        oracle_idx: Optional[int] = None
        oracle_tuple: Tuple[int, int, int] = (0, 0, 0)
        oracle_found = False
        oracle_score_best = float("inf")

        if n_valid_r > 0:
            t0 = time.time()
            for aidx in valid_r:
                aidx = int(aidx)
                p, m, b = decode_agent_r_action(aidx, num_mods, max_blk)
                env_try = copy.deepcopy(env_pre)
                _, _, _, info_try = env_try.step((split_id, server_id), (p, m, b))
                s = _oracle_score(info_try, req.deadline_ms)
                if s < oracle_score_best:
                    oracle_score_best = s
                    oracle_idx = aidx; oracle_tuple = (p, m, b)
                    oracle_found = True
            oracle_time_total += time.time() - t0

        # Oracle outcome
        or_ok = False; or_reason = "no_valid_r_actions"
        or_score_val = 100.0
        if oracle_found:
            env_or = copy.deepcopy(env_pre)
            _, _, _, or_info = env_or.step((split_id, server_id), oracle_tuple)
            or_ok = or_info.get("success", False)
            or_reason = or_info.get("reason", "unknown")
            or_score_val = _oracle_score(or_info, req.deadline_ms)

            if or_ok:
                or_success += 1
                d = float(or_info.get("delay_ms", 0.0))
                or_total_delay += d; or_total_fs += float(or_info.get("num_slots", 0))
                or_total_waste += float(or_info.get("block_waste", 0.0))
                or_total_path += float(or_info.get("path_dist_km", 0.0))
                or_mod_ctr[or_info.get("modulation", "unknown")] += 1
            else:
                or_blocked += 1
                or_reason_ctr[or_reason] += 1
        else:
            or_blocked += 1
            or_reason_ctr["no_valid_r_actions"] += 1

        # --- Comparison ---
        if bc_ok and not or_ok:
            hurt_count += 1
        if not bc_ok and or_ok:
            help_count += 1
        if bc_idx == oracle_idx:
            same_count += 1

        # Regret: BC score - oracle score (positive = BC worse)
        if oracle_found:
            regret_sum += max(0.0, bc_score_val - oracle_score_best)
            regret_n += 1

        per_req_log.append({
            "req": i,
            "n_valid_c": n_valid_c, "n_valid_r": n_valid_r,
            "bc_action": bc_idx, "bc_ok": bc_ok, "bc_reason": bc_reason, "bc_score": bc_score_val,
            "or_action": oracle_idx, "or_ok": or_ok, "or_reason": or_reason, "or_score": oracle_score_best,
            "same_action": bc_idx == oracle_idx,
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
        "bc_regret_mean": regret_sum / max(regret_n, 1),
        "no_valid_c_count": noc_count, "no_valid_c_rate": noc_count / n,
        "avg_valid_c_actions": float(np.mean(c_act_counts)) if c_act_counts else 0.0,
        "avg_oracle_candidates": float(oracle_cand_total) / n,
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
    parser = argparse.ArgumentParser(description="R-Action Oracle Gap Diagnostic")
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
                        default="sa_hmarl/experiments/r_action_oracle_gap_s20.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/r_action_oracle_gap_s20.md")
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

    # Pre-generate episodes
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
    print("R-Action Oracle Gap Diagnostic")
    print(f"Slots: {args.num_slots}  k: {args.k_paths}  Arrival: {args.arrival_interval}s")
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
                  f"help={res['oracle_help_count']}/{res['total']} "
                  f"hurt={res['oracle_hurt_count']}/{res['total']} "
                  f"same={res['same_action_rate']:.1%} "
                  f"regret={res['bc_regret_mean']:.3f}")

    total_elapsed = time.time() - t_start

    # Aggregate
    scalar_keys = [
        "bc_blocking", "bc_success_rate", "bc_avg_delay_ms", "bc_avg_fs",
        "bc_avg_waste", "bc_avg_path_km",
        "or_blocking", "or_success_rate", "or_avg_delay_ms", "or_avg_fs",
        "or_avg_waste", "or_avg_path_km",
        "oracle_help_rate", "oracle_hurt_rate", "same_action_rate",
        "bc_regret_mean", "no_valid_c_rate", "avg_valid_c_actions",
        "avg_oracle_candidates", "oracle_time_s",
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
    agg["total_oracle_time_s"] = sum(r["oracle_time_s"] for r in ep_results)
    agg["delta_blocking"] = agg["bc_blocking"] - agg["or_blocking"]

    # Per-seed aggregate
    per_seed = {}
    for seed in seeds:
        seed_eps = [r for r in ep_results if r.get("_seed", -1) == seed]
        # Actually seed isn't stored in result.  Let me recompute.
        pass
    # Recompute per-seed from ep_results with seed tracking
    per_seed_agg: Dict[int, Dict] = {}
    for seed in seeds:
        # We need to match ep_results to seeds.  The order is: for seed in seeds: for ep in range(episodes):
        # So results are grouped by seed.
        pass

    # Simpler: recompute per seed by indexing
    eps_per_seed = args.episodes
    per_seed_agg = {}
    for si, seed in enumerate(seeds):
        start = si * eps_per_seed
        seed_res = ep_results[start:start + eps_per_seed]
        ps: Dict[str, Any] = {}
        for key in ["bc_blocking", "or_blocking", "oracle_help_rate", "same_action_rate",
                     "bc_regret_mean", "no_valid_c_rate"]:
            ps[key] = float(np.mean([float(r[key]) for r in seed_res]))
        per_seed_agg[seed] = ps

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    d_blk = agg["delta_blocking"]
    print(f"  BC-PPO-R blocking:         {agg['bc_blocking']:.4f} ± {agg['bc_blocking_std']:.4f}")
    print(f"  R one-step oracle blocking:{agg['or_blocking']:.4f} ± {agg['or_blocking_std']:.4f}")
    print(f"  Δblocking (BC − oracle):   {d_blk:+.4f}")
    print(f"  Oracle help rate:          {agg['oracle_help_rate']:.1%}")
    print(f"  Oracle hurt rate:          {agg['oracle_hurt_rate']:.1%}")
    print(f"  Same action rate:          {agg['same_action_rate']:.1%}")
    print(f"  BC regret mean:            {agg['bc_regret_mean']:.4f}")
    print(f"  BC delay: {agg['bc_avg_delay_ms']:.1f}ms  Oracle delay: {agg['or_avg_delay_ms']:.1f}ms")
    print(f"  BC FS:    {agg['bc_avg_fs']:.2f}     Oracle FS:    {agg['or_avg_fs']:.2f}")
    print(f"  Total oracle time: {agg['total_oracle_time_s']:.0f}s")

    print(f"\n  Per-seed BC blocking:")
    for seed in seeds:
        ps = per_seed_agg[seed]
        print(f"    seed={seed:4d}: BC={ps['bc_blocking']:.4f}  OR={ps['or_blocking']:.4f}  "
              f"Δ={ps['bc_blocking']-ps['or_blocking']:+.4f}  help={ps['oracle_help_rate']:.1%}")

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
    md.append("# R-Action Oracle Gap — S20 Diagnostic\n\n")
    md.append(f"**Slots:** {args.num_slots}  **k:** {args.k_paths}  "
              f"**Arrival:** {args.arrival_interval}s  **Req/ep:** {args.requests_per_episode}\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}  "
              f"**Source nodes:** {src_nodes}\n\n")
    md.append(f"**Agent-C:** `{args.agent_c_checkpoint}`  \n")
    md.append(f"**Agent-R:** `{args.agent_r_checkpoint}`  \n\n")
    md.append(f"**Total time:** {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)  "
              f"**Oracle time:** {agg['total_oracle_time_s']:.0f}s\n\n")

    md.append("## Core Result\n\n")
    md.append("| Metric | BC-PPO-R | R Oracle | Δ |\n")
    md.append("|--------|----------|----------|---|\n")
    md.append(f"| Blocking | {agg['bc_blocking']:.4f} | {agg['or_blocking']:.4f} | "
              f"**{d_blk:+.4f}** |\n")
    md.append(f"| Delay (ms) | {agg['bc_avg_delay_ms']:.1f} | {agg['or_avg_delay_ms']:.1f} | |\n")
    md.append(f"| Avg FS | {agg['bc_avg_fs']:.2f} | {agg['or_avg_fs']:.2f} | |\n")
    md.append(f"| Avg Waste | {agg['bc_avg_waste']:.3f} | {agg['or_avg_waste']:.3f} | |\n")
    md.append(f"| Avg Path (km) | {agg['bc_avg_path_km']:.1f} | {agg['or_avg_path_km']:.1f} | |\n\n")

    md.append("## Oracle Gap Metrics\n\n")
    md.append(f"- **Oracle help rate:** {agg['oracle_help_rate']:.1%} "
              f"(BC failed → oracle succeeded, {agg['total_oracle_help']}/{agg['total_requests']})\n")
    md.append(f"- **Oracle hurt rate:** {agg['oracle_hurt_rate']:.1%} "
              f"(BC succeeded → oracle failed, {agg['total_oracle_hurt']}/{agg['total_requests']})\n")
    md.append(f"- **Same action rate:** {agg['same_action_rate']:.1%} "
              f"(BC chose same action as oracle, {agg['total_same_action']}/{agg['total_requests']})\n")
    md.append(f"- **BC regret mean:** {agg['bc_regret_mean']:.4f} (BC score − oracle score)\n\n")

    md.append("## Per-Seed Breakdown\n\n")
    md.append("| Seed | Src | BC Blk | OR Blk | ΔBlk | Help% | Same% | Regret |\n")
    md.append("|------|-----|--------|--------|------|-------|-------|--------|\n")
    for seed in seeds:
        ps = per_seed_agg[seed]
        md.append(f"| {seed} | {src_nodes[seed]} | {ps['bc_blocking']:.4f} | {ps['or_blocking']:.4f} | "
                  f"{ps['bc_blocking']-ps['or_blocking']:+.4f} | {ps['oracle_help_rate']:.1%} | "
                  f"{ps['same_action_rate']:.1%} | {ps['bc_regret_mean']:.4f} |\n")

    md.append("\n## BC-PPO-R Failure Reasons\n\n")
    md.append(f"{_fmt_pct(agg.get('bc_reason_counter', {}))}\n\n")
    md.append("## Oracle Failure Reasons\n\n")
    md.append(f"{_fmt_pct(agg.get('or_reason_counter', {}))}\n")

    # Verdict
    md.append("\n## Verdict\n\n")
    if d_blk >= 0.02:
        md.append(f"### ✓ R has learning headroom (Δblk = {d_blk:+.4f} ≥ 2pp)\n\n")
        md.append("**Recommendation:** Oracle imitation / DAgger to close the gap.\n")
    elif d_blk > 0.005:
        md.append(f"### ~ Marginal R headroom (Δblk = {d_blk:+.4f}, < 2pp)\n\n")
        md.append("Consider oracle imitation if other metrics (delay/FS) improve.\n")
    else:
        md.append(f"### ✗ R is at ceiling (Δblk = {d_blk:+.4f})\n\n")
        md.append("BC-PPO-R is already near-optimal within the R action mask. "
                  "Stop R-only optimization.  Shift to C/R joint or scenario design.\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # Console verdict
    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    print(f"BC-PPO-R blk={agg['bc_blocking']:.4f}  Oracle blk={agg['or_blocking']:.4f}  "
          f"Δ={d_blk:+.4f}  help={agg['oracle_help_rate']:.1%}  same={agg['same_action_rate']:.1%}")
    if d_blk >= 0.02:
        print("✓ R HAS learning headroom → oracle imitation / DAgger")
    elif d_blk > 0.005:
        print("~ MARGINAL headroom — consider DAgger if other metrics improve")
    else:
        print("✗ R AT CEILING — stop R-only optimization")


if __name__ == "__main__":
    main()
