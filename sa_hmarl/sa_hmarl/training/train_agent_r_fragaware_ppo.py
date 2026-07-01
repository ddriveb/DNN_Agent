"""Fine-tune Agent-R with PPO on frag-aware mixed-block actions.

This path starts from an existing PPO-R/BC-PPO-R checkpoint trained on the
standard 11-D action features, expands the actor to 17-D frag-aware features,
and fine-tunes it in the real SA-HMARL environment.  Agent-C is fixed, so only
Agent-R learns.

Example:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.train_agent_r_fragaware_ppo
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.env.r_frag_aware import compute_r_action_frag_metrics
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode",
        ckpt.get("agent_c_feature_mode", "default"),
    )
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _select_c_agent(agent_c: PPOAgentC, obs_c: Dict[str, Any]) -> Optional[int]:
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c,
            mask,
            num_slots_total=int(ckpt_args.get("num_slots", 24)),
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


def _select_c_greedy(obs_c: Dict[str, Any]) -> Optional[int]:
    mask = obs_c["agent_c_mask"]
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    best_idx = int(valid[0])
    best_val = float(obs_c["candidate_features"][best_idx].get("edge_compute_ms", 1e9))
    if not np.isfinite(best_val):
        best_val = 1e9
    for idx in valid[1:]:
        idx = int(idx)
        val = float(obs_c["candidate_features"][idx].get("edge_compute_ms", 1e9))
        if not np.isfinite(val):
            val = 1e9
        if val < best_val:
            best_idx = idx
            best_val = val
    return best_idx


def _warm_start_r_actor(
    agent_r: PPOAgentR,
    ckpt_path: str,
    device: str = "cpu",
) -> None:
    """Load old PPO-R weights, including 11-D -> 17-D first-layer transfer."""
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    old_state = ckpt["model_state"]
    new_state = agent_r.policy_net.state_dict()
    transferred = {}

    for name, new_tensor in new_state.items():
        old_tensor = old_state.get(name)
        if old_tensor is None:
            transferred[name] = new_tensor
            continue
        if old_tensor.shape == new_tensor.shape:
            transferred[name] = old_tensor
            continue
        # First linear layer weight: [hidden, old_input] -> [hidden, new_input]
        if (
            old_tensor.ndim == 2
            and new_tensor.ndim == 2
            and old_tensor.shape[0] == new_tensor.shape[0]
            and old_tensor.shape[1] < new_tensor.shape[1]
        ):
            patched = new_tensor.clone()
            patched[:, : old_tensor.shape[1]] = old_tensor
            transferred[name] = patched
        else:
            transferred[name] = new_tensor

    agent_r.policy_net.load_state_dict(transferred)
    print(
        f"Warm-started Agent-R from {ckpt_path} "
        f"(old input_dim={ckpt.get('input_dim')}, new input_dim={agent_r.input_dim})"
    )


def _frag_reward_penalty(obs_r: Dict[str, Any], action_idx: Optional[int], args) -> float:
    if action_idx is None:
        return 0.0
    m = compute_r_action_frag_metrics(obs_r, action_idx)
    return (
        args.frag_delta_coef * m["delta_frag_est"]
        + args.frag_large_block_coef * m["large_block_risk"]
        + args.frag_lfb_drop_coef * m["lfb_drop_ratio"]
        + args.frag_waste_coef * m["leftover_ratio"]
        - args.frag_exact_fit_bonus * m["exact_fit"]
    )


def _future_requests(env, horizon: int):
    if horizon <= 0:
        return []
    return sorted(env.event_queue)[:horizon]


def _advance_without_current_allocation(env):
    """Return a copy advanced to the current request arrival without allocation.

    Before env.step() is called, the current request is still at the head of
    event_queue and resources may not yet be released to its arrival time.
    This helper builds the counterfactual state: time is advanced and expired
    resources are released, but the current request is not allocated.
    """
    env_copy = copy.deepcopy(env)
    if not env_copy.event_queue:
        return env_copy
    arrival_time, _, _ = heapq.heappop(env_copy.event_queue)
    env_copy.advance_time(arrival_time)
    return env_copy


def _estimate_future_feasibility(env, agent_c, args) -> float:
    """Estimate future RMSA action availability for the next H requests.

    For each upcoming request, advance a cloned environment to that request's
    arrival time, use the fixed C policy to choose split/server, then count
    valid R actions.  We do not allocate these future requests; the metric is a
    lightweight opportunity estimate under current active connections.
    """
    horizon_events = _future_requests(env, args.future_feas_horizon)
    if not horizon_events:
        return 0.0

    env_eval = copy.deepcopy(env)
    counts = []
    for arrival_time, _, future_req in horizon_events:
        env_eval.advance_time(arrival_time)
        obs_c = build_agent_c_observation(env_eval, future_req)
        if args.c_policy == "agent":
            action_idx_c = _select_c_agent(agent_c, obs_c)
        else:
            action_idx_c = _select_c_greedy(obs_c)
        action_c = (
            (0, 0)
            if action_idx_c is None
            else decode_agent_c_action(action_idx_c, len(env_eval.mec.servers))
        )
        split_id, server_id = action_c
        try:
            obs_r = build_agent_r_observation(env_eval, future_req, split_id, server_id)
            counts.append(float(np.sum(obs_r["agent_r_mask"])))
        except Exception:
            counts.append(0.0)

    if args.future_feas_metric == "sum":
        return float(np.sum(counts))
    return float(np.mean(counts)) if counts else 0.0


def _future_feas_reward(before: float, after: float, args) -> float:
    if args.future_feas_coef <= 0 and args.future_feas_bonus_coef <= 0:
        return 0.0
    denom = max(float(args.future_feas_norm), 1.0)
    drop = max(before - after, 0.0) / denom
    gain = max(after - before, 0.0) / denom
    return -args.future_feas_coef * drop + args.future_feas_bonus_coef * gain


def _discounted_returns(rewards: List[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


def _make_eval_requests(env_proto, args, seeds: List[int]):
    episodes_by_seed = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.eval_episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            episodes.append(
                generate_requests(
                    env_proto,
                    rng,
                    src,
                    num_requests=args.requests_per_episode,
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
                )
            )
        episodes_by_seed[seed] = episodes
    return episodes_by_seed


def evaluate(agent_c, agent_r: PPOAgentR, episodes_by_seed, env_proto, args):
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    total = blocked = success = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    total_future_feas = 0.0
    reasons = Counter()
    mods = Counter()

    for episodes in episodes_by_seed.values():
        for requests in episodes:
            env = make_env(
                topology=args.topology,
                num_slots=args.num_slots,
                num_servers=args.num_servers,
                seed=42,
                slot_bw_hz=args.slot_bw_hz,
                guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                server_nodes=server_nodes,
                capacities=capacities,
            )
            env.reset(requests)
            for req in requests:
                obs_c = build_agent_c_observation(env, req)
                if args.c_policy == "agent":
                    action_idx_c = _select_c_agent(agent_c, obs_c)
                else:
                    action_idx_c = _select_c_greedy(obs_c)
                action_c = (
                    (0, 0)
                    if action_idx_c is None
                    else decode_agent_c_action(action_idx_c, num_servers)
                )
                split_id, server_id = action_c

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                action_idx_r = agent_r.select_action(obs_r, deterministic=True)
                action_r = (
                    (0, 0, 0)
                    if action_idx_r is None
                    else decode_agent_r_action(
                        action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                    )
                )
                _, _, _, info = env.step(action_c, action_r)
                total += 1
                reward = compute_reward(info, args.waste_coef) - _frag_reward_penalty(
                    obs_r, action_idx_r, args
                )
                # Evaluation reports the shaped reward consistently with
                # training only when future-feasibility shaping is requested.
                if args.future_feas_eval_reward and args.future_feas_coef > 0:
                    total_future_feas += 0.0
                total_reward += reward
                if info.get("success", False):
                    success += 1
                    total_delay += float(info.get("delay_ms", 0.0))
                    total_fs += float(info.get("num_slots", 0))
                    total_waste += float(info.get("block_waste", 0.0))
                    total_path += float(info.get("path_dist_km", 0.0))
                    mods[info.get("modulation", "unknown")] += 1
                else:
                    blocked += 1
                    reasons[info.get("reason", "unknown")] += 1

    n = max(total, 1)
    s = max(success, 1)
    return {
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "avg_future_feas_reward": total_future_feas / n,
        "reason_counter": dict(reasons),
        "mod_counter": dict(mods),
    }


def _save_checkpoint(agent_r: PPOAgentR, path: Path, args, metrics: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": agent_r.policy_net.state_dict(),
            "input_dim": agent_r.input_dim,
            "hidden_dims": agent_r.hidden_dims,
            "agent_r_feature_mode": agent_r.feature_mode,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "modulation_profile": args.modulation_profile,
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=args.seed,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
    )
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    # Auto-set input_dim if not explicitly overridden
    r_input_dim = args.agent_r_input_dim
    if r_input_dim == 17 and args.agent_r_feature_mode == "c_aware":
        r_input_dim = 19  # 11 base + 8 c_context
    agent_r = PPOAgentR(
        input_dim=r_input_dim,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        feature_mode=args.agent_r_feature_mode,
    )
    _warm_start_r_actor(agent_r, args.warm_start_r, args.device)
    ref_policy = None
    if args.ref_kl_coef > 0:
        ref_policy = copy.deepcopy(agent_r.policy_net).to(args.device)
        ref_policy.eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    eval_seeds = [int(s.strip()) for s in args.eval_seeds.split(",") if s.strip()]
    eval_sets = _make_eval_requests(env_proto, args, eval_seeds)

    best_metric = float("inf")
    best_metrics = {}
    ckpt_prefix = Path(args.checkpoint_prefix)

    print("=" * 90)
    print("Frag-Aware PPO-R Fine-Tuning")
    print(
        f"topology={args.topology} slots={args.num_slots} max_blocks={args.max_blocks} "
        f"block_sort={args.block_sort_strategy} feature={args.agent_r_feature_mode}"
    )
    print(f"C policy={args.c_policy}, warm_start_r={args.warm_start_r}")
    print(f"Reference KL regularization: coef={args.ref_kl_coef}")
    print(
        f"Future feasibility reward: coef={args.future_feas_coef} "
        f"bonus={args.future_feas_bonus_coef} horizon={args.future_feas_horizon} "
        f"metric={args.future_feas_metric} norm={args.future_feas_norm}"
    )
    print("=" * 90)

    for episode in range(args.episodes):
        env = make_env(
            topology=args.topology,
            num_slots=args.num_slots,
            num_servers=args.num_servers,
            seed=args.seed + episode,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env,
            rng,
            src,
            num_requests=args.requests_per_episode,
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
        )
        env.reset(requests)

        features_list = []
        masks_list = []
        actions = []
        old_log_probs = []
        rewards = []
        blocked = 0
        mods = Counter()
        reasons = Counter()
        future_rewards = []
        future_drops = []

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            if args.c_policy == "agent":
                action_idx_c = _select_c_agent(agent_c, obs_c)
            else:
                action_idx_c = _select_c_greedy(obs_c)
            action_c = (
                (0, 0)
                if action_idx_c is None
                else decode_agent_c_action(action_idx_c, len(env.mec.servers))
            )
            split_id, server_id = action_c

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_features, r_mask = agent_r.build_action_features(obs_r)
            action_idx_r, logp_r, _ = agent_r.select_from_features(
                r_features, r_mask, deterministic=False
            )
            action_r = (
                (0, 0, 0)
                if action_idx_r is None
                else decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), env.max_blocks)
            )

            future_before = 0.0
            if args.future_feas_coef > 0 or args.future_feas_bonus_coef > 0:
                env_before = _advance_without_current_allocation(env)
                future_before = _estimate_future_feasibility(env_before, agent_c, args)

            _, _, _, info = env.step(action_c, action_r)

            reward = compute_reward(info, args.waste_coef) - _frag_reward_penalty(
                obs_r, action_idx_r, args
            )
            if args.future_feas_coef > 0 or args.future_feas_bonus_coef > 0:
                future_after = _estimate_future_feasibility(env, agent_c, args)
                f_reward = _future_feas_reward(future_before, future_after, args)
                reward += f_reward
                future_rewards.append(f_reward)
                future_drops.append(max(future_before - future_after, 0.0) / max(args.future_feas_norm, 1.0))

            if args.fs_penalty_coef > 0 and info.get("success", False):
                reward -= args.fs_penalty_coef * (
                    float(info.get("num_slots", 0)) / max(args.num_slots, 1)
                )

            if action_idx_r is not None:
                features_list.append(r_features)
                masks_list.append(r_mask)
                actions.append(action_idx_r)
                old_log_probs.append(logp_r)
                rewards.append(float(reward))

            if info.get("success", False):
                mods[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reasons[info.get("reason", "unknown")] += 1

        if len(actions) > 1:
            returns = _discounted_returns(rewards, args.gamma)
            advantages = returns.copy()
            if args.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss, entropy, kl, _ = agent_r.optimize_ppo(
                features_list,
                masks_list,
                actions,
                old_log_probs,
                advantages,
                clip_coef=args.clip_coef,
                epochs=args.ppo_epochs,
                target_kl=args.target_kl,
                reference_policy=ref_policy,
                ref_kl_coef=args.ref_kl_coef,
            )
        else:
            loss = entropy = kl = 0.0

        if episode % args.eval_freq == 0 or episode == args.episodes - 1:
            metrics = evaluate(agent_c, agent_r, eval_sets, env_proto, args)
            score = metrics["blocking_rate"]
            if args.best_metric == "objective":
                score = metrics["blocking_rate"] + args.delay_coef * (
                    metrics["avg_delay_ms"] / max(args.deadline_max, 1.0)
                )
            if score < best_metric:
                best_metric = score
                best_metrics = metrics
                _save_checkpoint(agent_r, ckpt_prefix.with_name(ckpt_prefix.name + "_best.pt"), args, metrics)
                best_mark = "*"
            else:
                best_mark = ""
            print(
                f"Ep {episode:4d} | train_blk={blocked / max(len(requests), 1):.3f} "
                f"train_r={np.mean(rewards) if rewards else 0.0:+.3f} "
                f"loss={loss:+.3f} ent={entropy:.3f} kl={kl:.4f} | "
                f"eval_blk={metrics['blocking_rate']:.3f} "
                f"eval_r={metrics['avg_reward']:+.3f} "
                f"delay={metrics['avg_delay_ms']:.2f} fs={metrics['avg_fs']:.2f} "
                f"best={best_metric:.4f}{best_mark}"
            )
            if future_rewards:
                print(
                    f"         future_feas: r={np.mean(future_rewards):+.4f} "
                    f"drop={np.mean(future_drops):.4f}"
                )
            if mods:
                mod_total = sum(mods.values())
                mod_str = ", ".join(f"{k}={v / mod_total:.1%}" for k, v in mods.most_common())
                print(f"         train_mods: {mod_str}")
            if reasons:
                reason_str = ", ".join(f"{k}={v}" for k, v in reasons.most_common())
                print(f"         train_fail: {reason_str}")

    final_metrics = evaluate(agent_c, agent_r, eval_sets, env_proto, args)
    _save_checkpoint(agent_r, ckpt_prefix.with_name(ckpt_prefix.name + "_last.pt"), args, final_metrics)
    summary = {
        "best_metric": best_metric,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "args": vars(args),
    }
    out = ckpt_prefix.with_name(ckpt_prefix.name + "_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved best/last checkpoints with prefix {ckpt_prefix}")
    print(f"Saved summary: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed",
                        choices=["size_desc", "waste_asc", "start_asc", "center_asc", "mixed"])
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.20)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--c_policy", type=str, default="agent", choices=["agent", "greedy"])
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--warm_start_r", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--agent_r_feature_mode", type=str, default="frag_aware",
                        choices=["default", "frag_aware", "c_aware"])
    parser.add_argument("--agent_r_input_dim", type=int, default=17,
                        help="Base input dim (auto-set: 11 default, 17 frag_aware, 19 c_aware)")
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--frag_waste_coef", type=float, default=0.1)
    parser.add_argument("--frag_delta_coef", type=float, default=0.3)
    parser.add_argument("--frag_large_block_coef", type=float, default=0.2)
    parser.add_argument("--frag_lfb_drop_coef", type=float, default=0.2)
    parser.add_argument("--frag_exact_fit_bonus", type=float, default=0.05)
    parser.add_argument("--fs_penalty_coef", type=float, default=0.02)
    parser.add_argument("--future_feas_coef", type=float, default=0.0,
                        help="Penalty coefficient for reducing future R valid-action availability.")
    parser.add_argument("--future_feas_bonus_coef", type=float, default=0.0,
                        help="Optional bonus coefficient for increasing future R valid-action availability.")
    parser.add_argument("--future_feas_horizon", type=int, default=5)
    parser.add_argument("--future_feas_metric", type=str, default="mean",
                        choices=["mean", "sum"])
    parser.add_argument("--future_feas_norm", type=float, default=60.0)
    parser.add_argument("--future_feas_eval_reward", action="store_true",
                        help="Reserved: keep eval reward accounting compatible with future-feas shaping.")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--ref_kl_coef", type=float, default=0.0,
                        help="KL penalty to keep fine-tuned R close to the warm-start BC policy.")
    parser.add_argument("--normalize_advantage", action="store_true", default=True)
    parser.add_argument("--eval_freq", type=int, default=25)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--eval_seeds", type=str, default="42,123")
    parser.add_argument("--best_metric", type=str, default="blocking", choices=["blocking", "objective"])
    parser.add_argument("--delay_coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_fragaware_mixed10_ft")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
