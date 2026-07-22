import copy
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
from sa_hmarl.evaluation.diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od import _generate_all_od_requests
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5 import _ppo_r_topk_actions
from sa_hmarl.evaluation.generate_v20_sliding_window_afterstate_dataset import _global_spectrum
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_r, _select_r_action_from_obs
from sa_hmarl.training.utils import make_env

seed = 7201
warmup = 500
eval_requests = 100
topology = "xlron_cost239_ptrnet_real"
num_slots = 320
num_servers = 4
k_paths_r = 50
fixed_split_id = 0
agent_r_checkpoint = "sa_hmarl/checkpoints/agent_r_mixed.pt"
device = "cpu"

env = make_env(topology, num_slots, num_servers, seed, modulation_profile="default")
env.k = k_paths_r
env.path_sort_strategy = "hops"
env.block_sort_strategy = "start_asc"
agent_r = _load_ppo_r(agent_r_checkpoint, env.mod_reg, device)
rng = np.random.RandomState(seed)
requests = _generate_all_od_requests(
    num_nodes=env.net.NUM_NODES,
    server_node_ids=[int(s.node_id) for s in env.mec.servers],
    rng=rng,
    num_requests=warmup + eval_requests,
    arrival_interval=0.3,
    holding_min=20.0,
    holding_max=30.0,
    deadline_min=30.0,
    deadline_max=100.0,
    size_min_mb=5.0,
    size_max_mb=30.0,
    edge_cost_min=0.1,
    edge_cost_max=2.2,
    num_splits=3,
    split_profile="default3",
)
env.reset(requests)

node_to_server = {int(s.node_id): i for i, s in enumerate(env.mec.servers)}
max_mismatch = 0.0
checked = 0
first_bad = None

for step_idx, req in enumerate(requests):
    env.advance_time(req.arrival_time)
    if step_idx < warmup:
        dst_node = int(getattr(req, "_dst_node", req.src_node))
        server_id = node_to_server.get(dst_node, 0)
        obs_r = build_agent_r_observation(env, req, fixed_split_id, server_id)
        r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        if r_idx is None:
            env.reject_next_request(req.req_id, "r_no_valid_action")
        else:
            r_action = decode_agent_r_action(int(r_idx), env.mod_reg.num_formats, env.max_blocks)
            env.step((fixed_split_id, server_id), r_action)
        continue

    if step_idx >= warmup + eval_requests:
        break

    dst_node = int(getattr(req, "_dst_node", req.src_node))
    server_id = node_to_server.get(dst_node, 0)
    obs_r = build_agent_r_observation(env, req, fixed_split_id, server_id)
    r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
    if r_idx is None:
        env.reject_next_request(req.req_id, "r_no_valid_action")
        continue

    candidate_actions = _ppo_r_topk_actions(agent_r, obs_r, 30)
    if not candidate_actions:
        env.step((fixed_split_id, server_id), decode_agent_r_action(int(r_idx), env.mod_reg.num_formats, env.max_blocks))
        continue

    env_deep = copy.deepcopy(env)
    deep_results = []
    for r_idx in candidate_actions:
        env_cand = copy.deepcopy(env_deep)
        r_action = decode_agent_r_action(int(r_idx), env.mod_reg.num_formats, env.max_blocks)
        env_cand.step((fixed_split_id, server_id), r_action)
        deep_results.append({"r_idx": int(r_idx), "afterstate": _global_spectrum(env_cand)})

    fast_results = []
    paths = obs_r["candidate_paths"]
    required_fs_per_path_mod = obs_r["required_fs_per_path_mod"]
    candidate_blocks = obs_r["candidate_blocks_per_path_mod"]
    for r_idx in candidate_actions:
        path_idx, mod_idx, block_idx = decode_agent_r_action(int(r_idx), env.mod_reg.num_formats, env.max_blocks)
        fs_req = required_fs_per_path_mod[path_idx][mod_idx]
        blocks = candidate_blocks[path_idx][mod_idx]
        start_slot, _ = blocks[block_idx]
        path = paths[path_idx]
        env.net.allocate(path, start_slot, fs_req)
        fast = _global_spectrum(env)
        env.net.release(path, start_slot, fs_req)
        fast_results.append({"r_idx": int(r_idx), "afterstate": fast})

    for d, f in zip(deep_results, fast_results):
        for key in ("phi_spec", "free_ratio", "lfb", "frag"):
            diff = abs(d["afterstate"][key] - f["afterstate"][key])
            if diff > max_mismatch:
                max_mismatch = diff
                if first_bad is None:
                    first_bad = (step_idx, d["r_idx"], key, d["afterstate"][key], f["afterstate"][key], diff)
    checked += 1

    r_action = decode_agent_r_action(int(r_idx), env.mod_reg.num_formats, env.max_blocks)
    env.step((fixed_split_id, server_id), r_action)

print(f"checked={checked} max_mismatch={max_mismatch:.6e}")
if first_bad:
    print(f"first_bad: step={first_bad[0]} r_idx={first_bad[1]} key={first_bad[2]} deep={first_bad[3]} fast={first_bad[4]} diff={first_bad[5]}")
