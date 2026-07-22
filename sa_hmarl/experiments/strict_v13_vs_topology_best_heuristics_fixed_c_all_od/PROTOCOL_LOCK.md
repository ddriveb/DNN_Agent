# Protocol Lock: Multi-Topology Strict v1.3 vs Heuristics

## Scope

This experiment compares Strict v1.3 (frozen cross-topology transfer) against KSP-FF K=50 hops and FF-KSP K=50 hops under fixed-C / all-OD pure RMSA.

## Fixed parameters

- Topologies: `xlron_nsfnet_deeprmsa, xlron_usnet_gcnrmsa, xlron_jpn48`
- num_slots: 320
- arrival_interval: 0.3
- holding_min: 20.0
- holding_max: 30.0
- warmup_requests: 500
- requests_per_episode (eval): 6000
- seeds: 6101,6102,6103,6104,6105
- k_paths_r: 50
- path_sort_strategy_r: `hops`
- block_sort_strategy_r: `start_asc`
- max_blocks: 10
- fixed_split_id: 0
- num_splits: 3
- split_profile: default3
- modulation_profile: default

## Method definitions

- **Strict v1.3 frozen cross-topology transfer**: PPO-R Top-30 legal actions, reranked by 25-dim ranker features.
- **KSP-FF K=50 hops**: `ksp_ff_highest_mod_action` (path-first, highest feasible modulation per path).
- **FF-KSP K=50 hops**: `ff_ksp_highest_mod_action` (spectrum-start-first, highest feasible modulation per path).

## Topology-specific comparison

| Topology | Main heuristic | Auxiliary heuristic |
|---|---|---|
| xlron_nsfnet_deeprmsa | KSP-FF K=50 hops | FF-KSP K=50 hops |
| xlron_usnet_gcnrmsa | FF-KSP K=50 hops | KSP-FF K=50 hops |
| xlron_jpn48 | FF-KSP K=50 hops | KSP-FF K=50 hops |

## Non-goals

- No PPO-C is loaded or called.
- No network is trained or fine-tuned.
- No checkpoint is modified.
