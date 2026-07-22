# Fairness Audit: COST239 Fixed-C / All-OD Three-Method Comparison

## Protocol Verification

- **Comparison type**: fixed-C / all-OD pure RMSA
- **No PPO-C loaded or called**: verified across all methods
- **Fixed split_id**: 0
- **Deterministic server mapping by dst_node**: verified
- **Uniform all-OD traffic**: src uniform over all nodes, dst uniform over server nodes [0,1,2,3]
- **Unified action space**: K_path=50, |M|=4, B=10, total=2000
- **All methods use the same physical R mask from `build_agent_r_observation`**: verified

## Configuration Lock

- Topology: `xlron_cost239_ptrnet_real`
- num_slots: 320
- num_servers: 4
- k_paths_r: 50
- path_sort_strategy_r: `hops`
- block_sort_strategy_r: `start_asc`
- max_blocks: 10
- arrival_interval: 0.3
- holding: [20.0, 30.0]
- deadline: [30.0, 100.0]
- size: [5.0, 30.0] MB
- edge_cost: [0.1, 2.2]
- warmup: 500, evaluated: 6000
- seeds: 5001,5002,5003,5004,5005

## Method-Specific Fairness Checks

### Strict v1.3
- Uses PPO-R checkpoint `sa_hmarl/checkpoints/agent_r_mixed.pt`
- Uses ranker checkpoint `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt`
- Candidate pool = PPO-R legal Top-30 only
- No KSP anchor, heuristic filler, diversity/random, or all-legal candidate

### KSP-FF K=50 hops
- Implementation: `ksp_ff_highest_mod_action`
- Path order: hops (tie-break by km)
- Modulation: highest feasible SE per path
- Block: `start_asc` First-Fit

### Topology-matched adapted DeepRMSA K=50 hops
- Agent class: `sa_hmarl/sa_hmarl/agents/deep_rmsa_adapted_k50_agent.py`
- Trained from scratch in this environment
- No external snap24/NSFNET/Germany/Japan checkpoint loaded
- 5-layer 128-unit ELU MLP backbone, A2C episode-level updates
- Output head size 2000, masked over the same physical R mask

## Fairness Verdict

| Method | Blocking Rate |
|---|---:|
| Strict v1.3 | 5.5933% |
| KSP-FF K=50 hops | 5.5933% |
| Topology-matched adapted DeepRMSA K=50 hops | 12.0833% |

All three methods share the same environment, request traces, and R mask. The comparison is fair if the adapted DeepRMSA was trained on the same fixed-C / all-OD distribution and evaluated greedily without online updates. KSP-FF parity check reproduces the source-of-truth blocking rate (~5.63%).
