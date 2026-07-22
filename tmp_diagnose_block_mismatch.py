#!/usr/bin/env python3
"""Diagnostic: why does obs_r block_start differ from env.step start_slot?"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "sa_hmarl"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import copy
import argparse
import numpy as np
import torch

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import (
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_r,
    _select_r_action_from_obs,
)
from sa_hmarl.evaluation.diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od import (
    _generate_all_od_requests,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5 import (
    _ppo_r_topk_actions,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env


def main():
    args = argparse.Namespace(
        topology="xlron_cost239_ptrnet_real",
        num_slots=320,
        num_servers=4,
        k_paths_r=50,
        path_sort_strategy_r="hops",
        block_sort_strategy_r="start_asc",
        modulation_profile="default",
        max_blocks=10,
        fixed_split_id=0,
        warmup_requests=500,
        smoke_eval_requests=100,
        agent_r_checkpoint="sa_hmarl/checkpoints/ppo_r_cost239_k50_hops_fixed_c.pt",
        device="cpu",
    )

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    requests, _ = _generate_all_od_requests(7201, 0.30, args)

    env = make_env(args.topology, args.num_slots, args.num_servers, 7201,
                   modulation_profile=args.modulation_profile)
    env.k = args.k_paths_r
    env.path_sort_strategy = args.path_sort_strategy_r
    env.block_sort_strategy = args.block_sort_strategy_r
    env.max_blocks = args.max_blocks
    env.reset(requests)

    node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}
    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        if step_idx < args.warmup_requests:
            dst_node = int(getattr(req, "_dst_node", req.src_node))
            server_id = node_to_server.get(dst_node)
            split_id = args.fixed_split_id
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
            if r_mask.any():
                r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, args.max_blocks)
                r_action = decode_agent_r_action(int(r_idx), env.mod_reg.num_formats, env.max_blocks)
                env.step((split_id, server_id), r_action)
            else:
                env.reject_next_request(req.req_id, "r_no_valid_action")
            continue

        dst_node = int(getattr(req, "_dst_node", req.src_node))
        server_id = node_to_server.get(dst_node)
        split_id = args.fixed_split_id
        obs_r = build_agent_r_observation(env, req, split_id, server_id)

        print(f"\n=== step_idx={step_idx} req_id={req.req_id} ===")
        print(f"env.block_sort_strategy = {env.block_sort_strategy!r}")
        print(f"env.max_blocks = {env.max_blocks}")
        print(f"env.time = {env.time}")

        # Pick a legal action from Top-30 and compare
        candidates = _ppo_r_topk_actions(agent_r, obs_r, 30)
        if not candidates:
            print("No legal candidates")
            continue
        r_idx = int(candidates[0])
        path_idx, mod_idx, block_idx = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), args.max_blocks)
        print(f"r_idx={r_idx} path_idx={path_idx} mod_idx={mod_idx} block_idx={block_idx}")

        obs_blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        obs_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
        print(f"obs_fs={obs_fs}")
        print(f"obs_blocks (len={len(obs_blocks)}) = {obs_blocks}")
        if block_idx < len(obs_blocks):
            obs_start_slot, obs_block_size = obs_blocks[block_idx]
            print(f"obs_start_slot={obs_start_slot} obs_block_size={obs_block_size}")

        # Deep path
        env_cand = copy.deepcopy(env)
        r_action = decode_agent_r_action(r_idx, env.mod_reg.num_formats, env.max_blocks)
        _, _, _, info = env_cand.step((split_id, server_id), r_action)
        print(f"deep info: {info}")
        print(f"env_cand.block_sort_strategy = {env_cand.block_sort_strategy!r}")

        if info.get("success"):
            print(f"deep start_slot={info.get('start_slot')} num_slots={info.get('num_slots')}")
            if block_idx < len(obs_blocks):
                if info.get("start_slot") != obs_start_slot:
                    print(f"MISMATCH: obs_start_slot={obs_start_slot} != deep_start_slot={info.get('start_slot')}")
                    print("Exiting at first mismatch for diagnosis.")
                    return

        if step_idx > args.warmup_requests + 5:
            print("No mismatch in first 5 post-warmup requests.")
            return


if __name__ == "__main__":
    main()
