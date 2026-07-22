"""Shared PPO support helpers kept separate from archived DQN trainers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentR
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_frozen_reference(
    ckpt_path: str,
    mod_reg: ModulationRegistry,
    device: str,
) -> Optional[PPOAgentR]:
    """Load a frozen PPO Agent-R reference policy for shaping/BC penalties."""
    path = Path(ckpt_path)
    if not path.exists():
        print(f"WARNING: reference PPO Agent-R checkpoint not found: {path}")
        return None

    ckpt = load_checkpoint(path, map_location=device)
    ref_agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=ckpt.get("agent_r_feature_mode", "default"),
    )
    ref_agent.policy_net.load_state_dict(ckpt["model_state"])
    for param in ref_agent.policy_net.parameters():
        param.requires_grad = False
    ref_agent.policy_net.eval()
    print(f"Loaded frozen PPO Agent-R reference from {path}")
    return ref_agent


def _get_reference_mod_name(
    ref_agent: PPOAgentR,
    obs_r: Dict,
    mod_names: List[str],
) -> Optional[str]:
    """Return the modulation name chosen by the frozen PPO Agent-R reference."""
    action_idx = ref_agent.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return None
    _, mod_idx, _ = decode_agent_r_action(
        action_idx, len(mod_names), 5
    )
    if mod_idx < 0 or mod_idx >= len(mod_names):
        return None
    return mod_names[mod_idx]


def _build_eval_set(env, rng: np.random.RandomState, args) -> List[List]:
    """Build a fixed validation set of request episodes for periodic PPO eval."""
    eval_episodes_list = []
    split_profile = getattr(args, "split_profile", "default3")
    for _ in range(args.eval_episodes):
        src = rng.randint(0, env.net.NUM_NODES)
        requests = generate_requests(
            env,
            rng,
            src,
            args.requests_per_episode,
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
            split_profile=split_profile,
            traffic_mode=getattr(args, "traffic_mode", "iid"),
            regime_stay_prob=getattr(args, "regime_stay_prob", 0.9),
        )
        eval_episodes_list.append(requests)
    return eval_episodes_list
