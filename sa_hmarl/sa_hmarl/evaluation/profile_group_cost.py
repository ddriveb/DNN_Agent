"""Profile cost of one counterfactual group on COST239."""
import copy
import time
import numpy as np
from sa_hmarl.env.observation_builder import build_agent_c_observation, build_agent_r_observation, decode_agent_c_action, decode_agent_r_action
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r, _select_c_action_from_obs, _select_r_action_from_obs, _snapshot_before_r_decision
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env

args_dict = dict(
    topology="xlron_cost239_ptrnet_real", num_slots=320, num_servers=4, k_paths=50,
    path_sort_strategy="hops", max_blocks=10, block_sort_strategy="start_asc",
    split_profile="default3", num_splits=3, seed=3030, requests_per_episode=2100,
    arrival_interval=0.0625, holding_min=20.0, holding_max=30.0, deadline_min=30.0, deadline_max=100.0,
    size_min_mb=5.0, size_max_mb=30.0, edge_cost_min=0.1, edge_cost_max=2.2,
    modulation_profile="default", device="cpu",
)

env = make_env(
    args_dict["topology"], args_dict["num_slots"], args_dict["num_servers"], args_dict["seed"],
    modulation_profile=args_dict["modulation_profile"],
    max_blocks=args_dict["max_blocks"], block_sort_strategy=args_dict["block_sort_strategy"],
    path_sort_strategy=args_dict["path_sort_strategy"], k=args_dict["k_paths"],
)
mod_reg = env.mod_reg
agent_c = _load_ppo_c("sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt", "cpu")
agent_r = _load_ppo_r("sa_hmarl/checkpoints/agent_r_mixed.pt", mod_reg, "cpu")
rng = np.random.RandomState(args_dict["seed"])
src = int(rng.randint(0, env.net.NUM_NODES))
requests = generate_requests(env, rng, src, args_dict["requests_per_episode"], args_dict["arrival_interval"],
    args_dict["holding_min"], args_dict["holding_max"], args_dict["deadline_min"], args_dict["deadline_max"],
    args_dict["size_min_mb"], args_dict["size_max_mb"], args_dict["edge_cost_min"], args_dict["edge_cost_max"],
    args_dict["num_splits"], args_dict["split_profile"])
env.reset(requests)

for step_idx, req in enumerate(requests):
    env.advance_time(req.arrival_time)
    obs_c = build_agent_c_observation(env, req)
    c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
    split_id, server_id = decode_agent_c_action(c_idx, args_dict["num_servers"])
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()
    ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
    action_r = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
    env.step((split_id, server_id), action_r)
    if step_idx == 2098:
        break

req = requests[2099]
env.advance_time(req.arrival_time)
obs_c = build_agent_c_observation(env, req)
c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
split_id, server_id = decode_agent_c_action(c_idx, args_dict["num_servers"])
obs_r = build_agent_r_observation(env, req, split_id, server_id)
legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()
print("legal", len(legal))

t0 = time.perf_counter()
snapshot = _snapshot_before_r_decision(env, req.req_id)
t1 = time.perf_counter()
print(f"snapshot deepcopy: {t1-t0:.3f}s")

branch = copy.deepcopy(snapshot)
t2 = time.perf_counter()
print(f"branch deepcopy: {t2-t1:.3f}s")

# apply one action
action_r = decode_agent_r_action(legal[0], len(obs_r["mod_names"]), env.max_blocks)
_,_,_,info = branch.step((split_id, server_id), action_r)
t3 = time.perf_counter()
print(f"apply action: {t3-t2:.3f}s")

# future rollout H=5 manually
future_env = copy.deepcopy(branch)
t4 = time.perf_counter()
print(f"future deepcopy: {t4-t3:.3f}s")
for offset in range(5):
    t = 2100 + offset
    if t >= len(requests): break
    freq = requests[t]
    future_env.advance_time(freq.arrival_time)
    fobs_c = build_agent_c_observation(future_env, freq)
    fc_idx,_,_ = _select_c_action_from_obs(agent_c, fobs_c, future_env.net.num_slots)
    fsplit, fserver = decode_agent_c_action(fc_idx, args_dict["num_servers"])
    fobs_r = build_agent_r_observation(future_env, freq, fsplit, fserver)
    if int(np.asarray(fobs_r["agent_r_mask"], dtype=bool).sum()) == 0:
        future_env.step((fsplit, fserver), (0,0,0))
        continue
    fppo_idx,_ = _select_r_action_from_obs(agent_r, fobs_r, future_env.max_blocks)
    fr_action = decode_agent_r_action(fppo_idx, len(fobs_r["mod_names"]), future_env.max_blocks)
    future_env.step((fsplit, fserver), fr_action)
t5 = time.perf_counter()
print(f"future H=5 rollout: {t5-t4:.3f}s")
print("info success", info.get("success"))
