# Diagnosis: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / all-OD)

## Protocol Lock

- **Comparison**: Strict v1.3 vs KSP-FF K=50 hops
- Topology: `xlron_german17`
- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)
- Traffic matrix: uniform all-OD
- fixed_split_id: 0
- server_node_ids: [0, 1, 2, 3]
- R-side K_path: 50
- R-side path_sort_strategy: `hops`
- R-side block_sort_strategy: `start_asc`
- max_blocks: 10
- KSP-FF implementation: `ksp_ff_highest_mod_action` (distance-adaptive)
- Seeds: 6101,6102,6103,6104,6105
- Warmup: 500, Evaluated: 6000

**Strict v1.3 details**:

- Candidate set: PPO-R legal Top-30 only (`ppo_r_topk_only`)
- Feature dim: 25 pre-decision state-action features
- Label: H=5, gamma=1.0 common-future counterfactual rollout
- Model: MLP 25 -> 128 -> 64 -> 1, SiLU, dropout=0
- Strict loss: listwise KL + SmoothL1, reg_weight=1.0, lambda_pair=0, lambda_hard=0
- Checkpoint: selected by validation regret
- Ranker gate: `all` (E=0/E=1 both invoke ranker)
- Fallback: PPO-R top-1 only when no candidates
- No KSP anchor, no heuristic filler, no diversity/random candidate, no all-legal candidate

**KSP-FF K=50 hops details**:

- Function: `ksp_ff_highest_mod_action`
- Path ordering: hops (then km within same hop count)
- Block ordering: start_asc
- Modulation: 4 default formats (BPSK/QPSK/8QAM/16QAM)
- `ksp_ff_action` (naive flat First-Fit / BPSK-first) is NOT used as the formal baseline

## Main Result

| Method | Blocking | Overload | NSB | R-no-valid | Deadline | Avg delay ms | Avg FS | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict v1.3 | 5.4767% | 0.0000% | 0.0000% | 5.4767% | 0.0000% | 16.01 | 2.46 | 1189.48 | 16.46 |
| KSP-FF K=50 hops | 5.4767% | 0.0000% | 0.0000% | 5.4767% | 0.0000% | 11.72 | 2.06 | 813.15 | 4.43 |

**Blocking difference**: Strict v1.3 - KSP-FF K=50 hops = **0.00 pp**

## Mechanism Summary

Across the evaluated seeds under fixed-C / fixed-OD, Strict v1.3 achieves a blocking rate of 5.4767% while KSP-FF K=50 hops achieves 5.4767%, a difference of 0.00 percentage points.

Among strict_win requests, the dominant KSP-FF K=50 hops failure reason is 'r_no_valid_action' (1 occurrences).

The mean distance from the most recent R-action divergence to a strict_win event is 1.0 requests.

Admitted-request path preferences differ: Strict v1.3 averages 5.13 hops / 1189.48 km, while KSP-FF K=50 hops averages 2.72 hops / 813.15 km.

Strict v1.3 uses an average of 2.46 FS per admitted request, whereas KSP-FF K=50 hops uses 2.06 FS.

KSP-FF K=50 hops action is present in Strict v1.3's PPO-R Top-30 candidate pool in 28.67% of evaluated requests.

## Evidence Tables

### Win/Loss Decomposition

| Outcome | Count |
|---|---:|
| strict_win | 1 |
| ksp_win | 1 |
| both_success | 28356 |
| both_block | 1642 |

### strict_win: KSP-FF K=50 hops failure reason breakdown

| Reason | Count |
|---|---:|
| r_no_valid_action | 1 |

### ksp_win: Strict v1.3 failure reason breakdown

| Reason | Count |
|---|---:|
| r_no_valid_action | 1 |

### Divergence timing

| Metric | Value |
|---|---:|
| Mean R-divergence distance before strict_win | 1.00 |
| Median R-divergence distance before strict_win | 1.00 |
| Mean R-divergence distance before ksp_win | 1.00 |
| Median R-divergence distance before ksp_win | 1.00 |

### Action preference deltas (admitted requests)

| Metric | Strict v1.3 | KSP-FF K=50 hops |
|---|---|---|
| Avg path idx | 11.21 | 0.00 |
| Avg required FS | 2.46 | 2.06 |
| Avg block start | 99.11 | 22.61 |
| Avg block waste | 0.9147 | 0.3063 |
| Avg hops | 5.13 | 2.72 |
| Avg path km | 1189.48 | 813.15 |

### Strict v1.3 candidate coverage

| Metric | Value |
|---|---:|
| KSP action in Strict Top-30 rate | 28.6700% |

## Causal Caution

Under fixed-C / fixed-OD, the R-side cannot change server selection. Any observed blocking difference is therefore attributable to optical spectrum trajectory management, not to server-choice coupling. If Strict v1.3 reduces server_overload relative to KSP-FF K=50 hops, this must reflect an indirect load imbalance caused by different success counts, not a direct server assignment.
