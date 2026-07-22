"""Root-cause diagnostic for v1.3 post-decision ranker on COST239.

Collects per-decision statistics needed to answer:
- Are H-step labels noisy / short-horizon biased?
- How much oracle headroom exists among all legal R actions?
- Does the training candidate pool (ppo_r_topk_only) cover the oracle action?
- Does the ranker select actions with low regret under longer horizons?
- How does conditional R-side headroom vary with C-side server pressure?
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action, ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _rollout_future,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
    _snapshot_before_r_decision,
    _spectrum_stats,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import (
    _load_ranker_policy_from_checkpoint,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


@dataclass
class CandidateOutcome:
    action_idx: int
    path_idx: int
    mod_idx: int
    block_idx: int
    current_success: bool
    future_blocked_h5: int
    future_overload_h5: int
    future_nsb_h5: int
    return_h5: float
    return_h20: Optional[float] = None
    return_h50: Optional[float] = None
    future_blocked_h20: Optional[int] = None
    future_overload_h20: Optional[int] = None
    future_blocked_h50: Optional[int] = None
    future_overload_h50: Optional[int] = None
    ranker_score: Optional[float] = None


@dataclass
class StateDiagnostic:
    seed: int
    episode: int
    request: int
    server_utilization: float
    selected_valid_r_count: int
    raw_r_count: int
    split_id: int
    server_id: int

    all_legal_outcomes: List[CandidateOutcome] = field(default_factory=list)
    train_candidates: List[int] = field(default_factory=list)
    online_candidates: List[int] = field(default_factory=list)
    ksp_highest_action: Optional[int] = None
    ksp_plain_action: Optional[int] = None
    ppo_action: Optional[int] = None
    ranker_train_action: Optional[int] = None
    ranker_online_action: Optional[int] = None

    def oracle_idx(self, horizon: str = "h5") -> Optional[int]:
        succ = [o for o in self.all_legal_outcomes if o.current_success]
        if not succ:
            return None
        key = {
            "h5": lambda o: o.return_h5,
            "h20": lambda o: o.return_h20 if o.return_h20 is not None else -1e9,
            "h50": lambda o: o.return_h50 if o.return_h50 is not None else -1e9,
        }[horizon]
        return max(succ, key=key).action_idx

    def best_return(self, horizon: str = "h5") -> Optional[float]:
        succ = [o for o in self.all_legal_outcomes if o.current_success]
        if not succ:
            return None
        key = {
            "h5": lambda o: o.return_h5,
            "h20": lambda o: o.return_h20 if o.return_h20 is not None else -1e9,
            "h50": lambda o: o.return_h50 if o.return_h50 is not None else -1e9,
        }[horizon]
        return max(succ, key=key).return_h5 if horizon == "h5" else max(succ, key=key).return_h20 if horizon == "h20" else max(succ, key=key).return_h50

    def action_return(self, action_idx: int, horizon: str = "h5") -> Optional[float]:
        for o in self.all_legal_outcomes:
            if o.action_idx == action_idx:
                return getattr(o, f"return_{horizon}", None)
        return None


def _compute_return(info: Dict[str, Any], future: Dict[str, Any], args) -> float:
    """Return label matching the current dataset coefficient defaults."""
    current_blocked = 0.0 if info.get("success", False) else 1.0
    return float(
        -args.return_current_block_coef * current_blocked
        - args.return_future_block_coef * float(future.get("blocked", 0))
        - args.return_future_nsb_coef * float(future.get("no_suitable_block", 0))
        - args.return_future_server_overload_coef * float(future.get("server_overload", 0))
        - args.return_delay_coef * float(future.get("delay_mean", 0.0))
        - args.return_fs_coef * float(future.get("avg_fs", 0.0))
    )


def _ppo_topk_actions(agent_r, obs_r, legal: List[int], top_k: int) -> List[int]:
    features, mask = agent_r.build_action_features(obs_r)
    logits = agent_r.policy_net(
        torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
    ).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    if not legal:
        return []
    local_idx = {int(a): i for i, a in enumerate(legal)}
    scores = np.asarray([logits[local_idx[a]] for a in legal])
    order = np.argsort(-scores, kind="stable")[: min(top_k, len(legal))]
    return [int(legal[i]) for i in order]


def _legalctx48_candidates(
    agent_r,
    obs_r,
    r_features: np.ndarray,
    legal: List[int],
    max_blocks: int,
    max_candidates: int = 48,
    ppo_top_k: int = 8,
    num_random: int = 5,
    min_candidates: int = 15,
    candidate_seed: int = 12345,
) -> List[int]:
    num_mods = len(obs_r["mod_names"])
    legal_set = set(legal)

    # PPO top-K
    ppo_top = _ppo_topk_actions(agent_r, obs_r, legal, ppo_top_k)
    candidates = list(ppo_top)

    legal_arr = np.asarray(legal, dtype=int)
    path_indices = legal_arr // (num_mods * max_blocks)
    block_indices = legal_arr % max_blocks
    by_path: Dict[int, List[Tuple[int, int]]] = {}
    for a, pidx, bidx in zip(legal, path_indices, block_indices):
        by_path.setdefault(int(pidx), []).append((int(a), int(bidx)))

    for actions in by_path.values():
        action_ids = [a for a, _ in actions]
        feats = r_features[action_ids]
        block_sizes = feats[:, 2]  # lfb index in base features? wrong; use feature order
        # Use r_features directly; indices from generate script
        block_sizes = feats[:, 8]
        required_fs = feats[:, 7]
        block_wastes = feats[:, 9]
        block_idx_arr = np.asarray([bidx for _, bidx in actions], dtype=int)
        candidates.append(action_ids[int(np.argmin(block_idx_arr))])
        candidates.append(action_ids[int(np.argmin(np.abs(block_sizes - required_fs)))])
        candidates.append(action_ids[int(np.argmax(block_sizes))])
        candidates.append(action_ids[int(np.argmin(block_wastes))])

    path_lengths = r_features[legal, 0]
    candidates.append(int(legal[int(np.argmin(path_lengths))]))

    seen = set(candidates)
    rng = np.random.RandomState(candidate_seed)
    remaining = [a for a in legal if a not in seen]
    if remaining:
        n_random = min(num_random, len(remaining))
        candidates.extend([int(a) for a in rng.choice(remaining, size=n_random, replace=False)])

    deduped = []
    seen = set()
    for a in candidates:
        if a not in seen and a in legal_set:
            seen.add(a)
            deduped.append(a)
    if len(deduped) < min_candidates and len(legal) <= max_candidates:
        deduped = list(dict.fromkeys(legal))
    core = deduped[:max_candidates]

    # Fill to max_candidates with highest PPO logit
    if len(core) < max_candidates:
        core_set = set(core)
        remaining = [a for a in legal if a not in core_set]
        if remaining:
            remaining_sorted = _ppo_topk_actions(agent_r, obs_r, remaining, len(remaining))
            core.extend(remaining_sorted[: max_candidates - len(core)])
    return core


def _run_state_rollouts(
    env_snapshot,
    req,
    obs_c,
    obs_r,
    requests,
    step_idx,
    legal: List[int],
    agent_c,
    agent_r,
    num_servers: int,
    args,
) -> List[CandidateOutcome]:
    outcomes = []
    num_mods = len(obs_r["mod_names"])
    max_blocks = env_snapshot.max_blocks
    for r_idx in legal:
        branch = copy.deepcopy(env_snapshot)
        action_r = decode_agent_r_action(int(r_idx), num_mods, max_blocks)
        _, _, _, info = branch.step((obs_c["selected_split"], obs_c["selected_server"]), action_r)
        future5 = _rollout_future(
            copy.deepcopy(branch), requests, step_idx + 1, 5,
            agent_c, agent_r, num_servers, args.util_threshold, args.alpha,
        )
        ret5 = _compute_return(info, future5, args)
        path_idx, mod_idx, block_idx = action_r
        outcomes.append(CandidateOutcome(
            action_idx=int(r_idx),
            path_idx=path_idx,
            mod_idx=mod_idx,
            block_idx=block_idx,
            current_success=bool(info.get("success", False)),
            future_blocked_h5=int(future5.get("blocked", 0)),
            future_overload_h5=int(future5.get("server_overload", 0)),
            future_nsb_h5=int(future5.get("no_suitable_block", 0)),
            return_h5=ret5,
        ))
    return outcomes


def _score_candidates(
    ranker_policy: CounterfactualRRankerPolicy,
    env,
    req,
    obs_c,
    obs_r,
    agent_r,
    candidate_actions: List[int],
    split_id: int,
    server_id: int,
) -> np.ndarray:
    r_features, _ = agent_r.build_action_features(obs_r)
    feats = ranker_policy.model.feature_names
    # Build features only for the requested candidate actions using the ranker policy helper.
    # The helper expects a list and returns scores aligned with candidate_actions.
    _, scores = ranker_policy.score_legal_actions(
        env, req, obs_c, obs_r, agent_r, split_id, server_id
    )
    # Re-align to candidate_actions because score_legal_actions may have reordered/truncated.
    # Actually it returns candidate_actions and scores in the same order.
    return candidate_actions, scores


def _evaluate_state(
    env_snapshot,
    req,
    obs_c,
    obs_r,
    r_features,
    requests,
    step_idx,
    agent_c,
    agent_r,
    ranker_policy,
    env,
    args,
) -> StateDiagnostic:
    num_servers = len(env_snapshot.mec.servers)
    split_id, server_id = decode_agent_c_action(obs_c["c_action_idx"], num_servers)
    legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()

    diag = StateDiagnostic(
        seed=args.seed,
        episode=0,
        request=step_idx,
        server_utilization=float(obs_c["server_utilizations"][server_id]),
        selected_valid_r_count=int(obs_c["feasible_counts"][split_id][server_id]),
        raw_r_count=len(legal),
        split_id=split_id,
        server_id=server_id,
    )

    # H=5 returns for all legal actions (needed for oracle/recall under the short teacher).
    diag.all_legal_outcomes = _run_state_rollouts(
        env_snapshot, req, obs_c, obs_r, requests, step_idx, legal,
        agent_c, agent_r, num_servers, args,
    )

    # Training candidate pool: ppo_r_topk_only, max 30 (same as current checkpoint metadata).
    train_candidates = _ppo_topk_actions(agent_r, obs_r, legal, args.train_ppo_top_k)
    train_candidates = train_candidates[: args.train_max_candidates]
    diag.train_candidates = train_candidates

    # Online candidate pool: legalctx48 + ensure KSP (v1.3 deployment in heuristic-C report).
    online_candidates = _legalctx48_candidates(
        agent_r, obs_r, r_features, legal, env.max_blocks,
        max_candidates=args.online_max_candidates,
        ppo_top_k=args.online_ppo_top_k,
        num_random=args.online_num_random,
        min_candidates=args.online_min_candidates,
    )
    ksp_h = ksp_ff_highest_mod_action(obs_r)
    if ksp_h is not None and int(ksp_h) in legal and int(ksp_h) not in online_candidates:
        online_candidates = [int(ksp_h)] + [a for a in online_candidates if a != int(ksp_h)]
        online_candidates = online_candidates[: args.online_max_candidates]
    diag.online_candidates = online_candidates
    diag.ksp_highest_action = int(ksp_h) if ksp_h is not None else None
    ksp_p = ksp_ff_action(obs_r)
    diag.ksp_plain_action = int(ksp_p) if ksp_p is not None else None

    ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
    diag.ppo_action = int(ppo_idx)

    # Score training candidates with ranker and pick top.
    if train_candidates and ranker_policy is not None:
        _, train_scores = ranker_policy.score_legal_actions(
            env, req, obs_c, obs_r, agent_r, split_id, server_id
        )
        # train_scores correspond to the candidates returned; restrict to train_candidates subset.
        train_set = set(train_candidates)
        cand_list, score_list = [], []
        for c, s in zip(_, train_scores):
            if c in train_set:
                cand_list.append(c)
                score_list.append(float(s))
        if cand_list:
            diag.ranker_train_action = cand_list[int(np.argmax(score_list))]

    # Score online candidates similarly.
    if online_candidates and ranker_policy is not None:
        _, online_scores = ranker_policy.score_legal_actions(
            env, req, obs_c, obs_r, agent_r, split_id, server_id
        )
        online_set = set(online_candidates)
        cand_list, score_list = [], []
        for c, s in zip(_, online_scores):
            if c in online_set:
                cand_list.append(c)
                score_list.append(float(s))
        if cand_list:
            diag.ranker_online_action = cand_list[int(np.argmax(score_list))]

    # Compute H=20 and H=50 returns for training candidates only (expensive).
    out_map = {o.action_idx: o for o in diag.all_legal_outcomes}
    for r_idx in train_candidates:
        o = out_map.get(r_idx)
        if o is None:
            continue
        branch = copy.deepcopy(env_snapshot)
        action_r = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = branch.step((split_id, server_id), action_r)
        future20 = _rollout_future(
            copy.deepcopy(branch), requests, step_idx + 1, 20,
            agent_c, agent_r, num_servers, args.util_threshold, args.alpha,
        )
        o.return_h20 = _compute_return(info, future20, args)
        o.future_blocked_h20 = int(future20.get("blocked", 0))
        o.future_overload_h20 = int(future20.get("server_overload", 0))
        future50 = _rollout_future(
            copy.deepcopy(branch), requests, step_idx + 1, 50,
            agent_c, agent_r, num_servers, args.util_threshold, args.alpha,
        )
        o.return_h50 = _compute_return(info, future50, args)
        o.future_blocked_h50 = int(future50.get("blocked", 0))
        o.future_overload_h50 = int(future50.get("server_overload", 0))

    return diag


def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=args.seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        path_sort_strategy=args.path_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    ranker_policy = None
    if args.ranker_checkpoint:
        ranker_policy = _load_ranker_policy_from_checkpoint(args.ranker_checkpoint, args.device)
        ranker_policy.candidate_mode = "all_legal"  # we will feed explicit candidate lists

    rng = np.random.RandomState(args.seed)
    src = int(rng.randint(0, env_proto.net.NUM_NODES))
    requests = generate_requests(
        env_proto, rng, src, args.requests_per_episode,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )

    env = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=args.seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        path_sort_strategy=args.path_sort_strategy,
        k=args.k_paths,
    )
    env.reset(requests)

    diagnostics: List[StateDiagnostic] = []
    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        if step_idx < args.warmup_requests:
            # Run PPO-R to advance real trajectory during warmup.
            obs_c = build_agent_c_observation(env, req)
            c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            action_r = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), action_r)
            continue

        if len(diagnostics) >= args.max_states:
            # Still need to advance env? No, we can stop.
            break

        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        obs_c_dict = dict(obs_c)
        obs_c_dict["c_action_idx"] = c_idx
        obs_c_dict["selected_split"] = split_id
        obs_c_dict["selected_server"] = server_id
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        if len(legal) < 2:
            # advance with PPO-R to keep trajectory consistent
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            action_r = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), action_r)
            continue

        snapshot = _snapshot_before_r_decision(env, req.req_id)
        diag = _evaluate_state(
            snapshot, req, obs_c_dict, obs_r, r_features, requests, step_idx,
            agent_c, agent_r, ranker_policy, env, args,
        )
        diagnostics.append(diag)

        # Advance real env with PPO-R.
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        action_r = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), action_r)

    return _aggregate(diagnostics, args)


def _aggregate(diagnostics: List[StateDiagnostic], args) -> Dict[str, Any]:
    n = len(diagnostics)

    def _spearman_h5_h20():
        rhos = []
        for d in diagnostics:
            xs, ys = [], []
            for o in d.all_legal_outcomes:
                if o.return_h20 is not None:
                    xs.append(o.return_h5)
                    ys.append(o.return_h20)
            if len(xs) >= 2:
                rho, _ = stats.spearmanr(xs, ys)
                if not np.isnan(rho):
                    rhos.append(float(rho))
        return float(np.mean(rhos)) if rhos else 0.0

    def _spearman_h5_h50():
        rhos = []
        for d in diagnostics:
            xs, ys = [], []
            for o in d.all_legal_outcomes:
                if o.return_h50 is not None:
                    xs.append(o.return_h5)
                    ys.append(o.return_h50)
            if len(xs) >= 2:
                rho, _ = stats.spearmanr(xs, ys)
                if not np.isnan(rho):
                    rhos.append(float(rho))
        return float(np.mean(rhos)) if rhos else 0.0

    def _bucket(items, key_fn, bins):
        out = {b: [] for b in bins}
        for it in items:
            k = key_fn(it)
            for b in bins:
                if k <= b:
                    out[b].append(it)
                    break
        return out

    # Oracle headroom and candidate recall.
    oracle_h5 = []
    oracle_h20 = []
    oracle_h50 = []
    train_recall_h5 = []
    online_recall_h5 = []
    ksp_h_in_train = []
    ksp_h_in_online = []
    ksp_p_in_train = []
    ksp_p_in_online = []
    label_ranges_h5 = []
    label_ranges_h20 = []

    for d in diagnostics:
        rets5 = [o.return_h5 for o in d.all_legal_outcomes if o.current_success]
        if rets5:
            label_ranges_h5.append(float(max(rets5) - min(rets5)))
        rets20 = [o.return_h20 for o in d.all_legal_outcomes if o.current_success and o.return_h20 is not None]
        if rets20:
            label_ranges_h20.append(float(max(rets20) - min(rets20)))

        oracle5 = d.oracle_idx("h5")
        if oracle5 is not None:
            train_recall_h5.append(int(oracle5 in d.train_candidates))
            online_recall_h5.append(int(oracle5 in d.online_candidates))
            best_ret = d.best_return("h5")
            ksp_h_ret = d.action_return(d.ksp_highest_action, "h5") if d.ksp_highest_action is not None else None
            if best_ret is not None and ksp_h_ret is not None:
                oracle_h5.append(float(best_ret - ksp_h_ret))
            ppo_ret = d.action_return(d.ppo_action, "h5")
            if best_ret is not None and ppo_ret is not None:
                oracle_h5.append(float(best_ret - ppo_ret))

        oracle20 = d.oracle_idx("h20")
        if oracle20 is not None:
            best_ret = d.best_return("h20")
            ksp_h_ret = d.action_return(d.ksp_highest_action, "h20") if d.ksp_highest_action is not None else None
            if best_ret is not None and ksp_h_ret is not None:
                oracle_h20.append(float(best_ret - ksp_h_ret))
            ranker_ret = d.action_return(d.ranker_train_action, "h20")
            if best_ret is not None and ranker_ret is not None:
                oracle_h20.append(float(best_ret - ranker_ret))

        oracle50 = d.oracle_idx("h50")
        if oracle50 is not None:
            best_ret = d.best_return("h50")
            ksp_h_ret = d.action_return(d.ksp_highest_action, "h50") if d.ksp_highest_action is not None else None
            if best_ret is not None and ksp_h_ret is not None:
                oracle_h50.append(float(best_ret - ksp_h_ret))
            ranker_ret = d.action_return(d.ranker_train_action, "h50")
            if best_ret is not None and ranker_ret is not None:
                oracle_h50.append(float(best_ret - ranker_ret))

        if d.ksp_highest_action is not None:
            ksp_h_in_train.append(int(d.ksp_highest_action in d.train_candidates))
            ksp_h_in_online.append(int(d.ksp_highest_action in d.online_candidates))
        if d.ksp_plain_action is not None:
            ksp_p_in_train.append(int(d.ksp_plain_action in d.train_candidates))
            ksp_p_in_online.append(int(d.ksp_plain_action in d.online_candidates))

    # Bucket by server utilization.
    util_bins = [0.7, 0.8, 0.9, 1.0]
    by_util = _bucket(diagnostics, lambda d: d.server_utilization, util_bins)
    util_summary = {}
    for b, group in by_util.items():
        gaps = []
        for d in group:
            best = d.best_return("h5")
            ksp = d.action_return(d.ksp_highest_action, "h5") if d.ksp_highest_action is not None else None
            if best is not None and ksp is not None:
                gaps.append(float(best - ksp))
        util_summary[f"<={b}"] = {
            "states": len(group),
            "mean_oracle_headroom_h5": float(np.mean(gaps)) if gaps else 0.0,
        }

    summary = {
        "states": n,
        "mean_label_range_h5": float(np.mean(label_ranges_h5)) if label_ranges_h5 else 0.0,
        "median_label_range_h5": float(np.median(label_ranges_h5)) if label_ranges_h5 else 0.0,
        "nonzero_label_range_h5": float(np.mean([r > 1e-6 for r in label_ranges_h5])) if label_ranges_h5 else 0.0,
        "mean_label_range_h20_train": float(np.mean(label_ranges_h20)) if label_ranges_h20 else 0.0,
        "spearman_h5_h20": _spearman_h5_h20(),
        "spearman_h5_h50": _spearman_h5_h50(),
        "oracle_headroom_h5_vs_ksp_ppo": float(np.mean(oracle_h5)) if oracle_h5 else 0.0,
        "oracle_headroom_h20_vs_ksp": float(np.mean(oracle_h20[::2])) if oracle_h20 else 0.0,
        "oracle_headroom_h20_vs_ranker": float(np.mean(oracle_h20[1::2])) if len(oracle_h20) > 1 else 0.0,
        "oracle_headroom_h50_vs_ksp": float(np.mean(oracle_h50[::2])) if oracle_h50 else 0.0,
        "oracle_headroom_h50_vs_ranker": float(np.mean(oracle_h50[1::2])) if len(oracle_h50) > 1 else 0.0,
        "train_recall_h5_oracle": float(np.mean(train_recall_h5)) if train_recall_h5 else 0.0,
        "online_recall_h5_oracle": float(np.mean(online_recall_h5)) if online_recall_h5 else 0.0,
        "ksp_highest_in_train": float(np.mean(ksp_h_in_train)) if ksp_h_in_train else 0.0,
        "ksp_highest_in_online": float(np.mean(ksp_h_in_online)) if ksp_h_in_online else 0.0,
        "ksp_plain_in_train": float(np.mean(ksp_p_in_train)) if ksp_p_in_train else 0.0,
        "ksp_plain_in_online": float(np.mean(ksp_p_in_online)) if ksp_p_in_online else 0.0,
        "util_bucket_summary": util_summary,
    }

    return {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "summary": summary,
        "elapsed_seconds": time.time() - args._start_time,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranker_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt")
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=50)
    parser.add_argument("--path_sort_strategy", default="hops", choices=["km", "hops"])
    parser.add_argument("--block_sort_strategy", default="start_asc")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3030)
    parser.add_argument("--requests_per_episode", type=int, default=3000)
    parser.add_argument("--warmup_requests", type=int, default=2000)
    parser.add_argument("--max_states", type=int, default=50)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--train_ppo_top_k", type=int, default=50)
    parser.add_argument("--train_max_candidates", type=int, default=30)
    parser.add_argument("--online_max_candidates", type=int, default=48)
    parser.add_argument("--online_ppo_top_k", type=int, default=8)
    parser.add_argument("--online_num_random", type=int, default=5)
    parser.add_argument("--online_min_candidates", type=int, default=15)

    parser.add_argument("--return_current_block_coef", type=float, default=3.0)
    parser.add_argument("--return_future_block_coef", type=float, default=4.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=3.0)
    parser.add_argument("--return_future_server_overload_coef", type=float, default=0.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.03)
    parser.add_argument("--return_fs_coef", type=float, default=0.05)

    parser.add_argument("--output_json", default="sa_hmarl/experiments/v13_postdecision_root_cause_diagnostic.json")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    report = _run_diagnostic(args)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
