"""Profile one full group generation step (candidate selection, features, rollout H=5)."""
import copy
import time
import numpy as np
from sa_hmarl.env.observation_builder import build_agent_c_observation, build_agent_r_observation, decode_agent_c_action, decode_agent_r_action
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r, _select_c_action_from_obs, _select_r_action_from_obs, _snapshot_before_r_decision
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import _select_candidate_actions, _build_candidate_feature, _rollout_future
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env

env = make_env('xlron_cost239_ptrnet_real', 320, 4, 3030, modulation_profile='default', max_blocks=10, block_sort_strategy='start_asc', path_sort_strategy='hops', k=50)
mod_reg = env.mod_reg
agent_c = _load_ppo_c('sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt', 'cpu')
agent_r = _load_ppo_r('sa_hmarl/checkpoints/agent_r_mixed.pt', mod_reg, 'cpu')
rng = np.random.RandomState(3030)
src = int(rng.randint(0, env.net.NUM_NODES))
requests = generate_requests(env, rng, src, 2100, 0.0625, 20, 30, 30, 100, 5, 30, 0.1, 2.2, 3, 'default3')
env.reset(requests)
for step_idx, req in enumerate(requests):
    env.advance_time(req.arrival_time)
    obs_c = build_agent_c_observation(env, req)
    c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
    split_id, server_id = decode_agent_c_action(c_idx, 4)
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    legal = np.flatnonzero(np.asarray(obs_r['agent_r_mask'], dtype=bool)).tolist()
    ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
    action_r = decode_agent_r_action(ppo_idx, len(obs_r['mod_names']), env.max_blocks)
    env.step((split_id, server_id), action_r)
    if step_idx == 2098:
        break

req = requests[2099]
env.advance_time(req.arrival_time)
obs_c = build_agent_c_observation(env, req)
c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
split_id, server_id = decode_agent_c_action(c_idx, 4)
obs_r = build_agent_r_observation(env, req, split_id, server_id)
r_features, r_mask = agent_r.build_action_features(obs_r)
legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
print('legal', len(legal))

class Args:
    candidate_mode = 'poststate_anchor48'
    max_candidates = 48
    ppo_top_k = 50
    min_candidates = 15
    num_random_candidates = 5
    candidate_seed = 12345
    ensure_ksp_action = False
    feature_schema = 'poststate_v1'
    horizon = 5
    gamma = 0.99
    util_threshold = 0.95
    alpha = 0.3
    viability_phi_coef = 0.0
    return_current_block_coef = 3.0
    return_future_block_coef = 4.0
    return_future_nsb_coef = 3.0
    return_future_server_overload_coef = 4.0
    return_delay_coef = 0.03
    return_fs_coef = 0.05
    path_penalty_coef = 0.0
    fs_penalty_coef = 0.0
    return_mode = 'v1'
    return_future_optical_coef = None
    return_future_overload_coef = None
    return_future_other_coef = None
    _feature_names = None
args = Args()

t0 = time.perf_counter()
snapshot = _snapshot_before_r_decision(env, req.req_id)
t1 = time.perf_counter()
candidates = _select_candidate_actions(env, agent_r, obs_r, r_features, legal, env.max_blocks, args)
t2 = time.perf_counter()
print('candidates', len(candidates))
for a in candidates[:3]:
    _build_candidate_feature(env, req, obs_c, obs_r, r_features, a, split_id, server_id, args)
t3 = time.perf_counter()
base_rng = np.random.RandomState(3030 + 2099 + 12345)
base_state = base_rng.get_state()
for a in candidates[:3]:
    branch = copy.deepcopy(snapshot)
    r_action = decode_agent_r_action(int(a), len(obs_r['mod_names']), env.max_blocks)
    branch.step((split_id, server_id), r_action)
    cand_rng = np.random.RandomState(0)
    cand_rng.set_state(base_state)
    _rollout_future(copy.deepcopy(branch), requests, 2100, args.horizon, agent_c, agent_r, 4, args.util_threshold, args.alpha, rng=cand_rng, gamma=args.gamma)
t4 = time.perf_counter()
print(f'snapshot {t1-t0:.3f}s select {t2-t1:.3f}s features(3) {t3-t2:.3f}s rollout(3) {t4-t3:.3f}s')
print(f'est per group {t2-t1 + (t3-t2)*len(candidates)/3 + (t4-t3)*len(candidates)/3:.3f}s')
