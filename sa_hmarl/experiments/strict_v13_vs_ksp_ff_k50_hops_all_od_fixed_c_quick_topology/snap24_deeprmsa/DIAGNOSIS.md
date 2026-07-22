# Diagnosis: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / all-OD)

## Protocol Lock

- **Comparison**: Strict v1.3 vs KSP-FF K=50 hops
- Topology: `snap24_gnutella_reach`
- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)
- Traffic matrix: uniform all-OD
- fixed_split_id: 0
- server_node_ids: [6, 1, 7, 12]
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
| Strict v1.3 | 25.7067% | 0.0000% | 0.0000% | 25.7067% | 0.0000% | 8.28 | 2.01 | 792.81 | 2.24 |
| KSP-FF K=50 hops | 25.3667% | 0.0000% | 0.0000% | 25.3667% | 0.0000% | 7.11 | 2.01 | 641.74 | 2.11 |

**Blocking difference**: Strict v1.3 - KSP-FF K=50 hops = **0.34 pp**

## Mechanism Summary

Across the evaluated seeds under fixed-C / fixed-OD, Strict v1.3 achieves a blocking rate of 25.7067% while KSP-FF K=50 hops achieves 25.3667%, a difference of 0.34 percentage points.

Request-level decomposition shows 1750 ksp_win requests versus 1648 strict_win requests.

Among strict_win requests, the dominant KSP-FF K=50 hops failure reason is 'r_no_valid_action' (1648 occurrences).

The mean distance from the most recent R-action divergence to a strict_win event is 1.2 requests.

Admitted-request path preferences differ: Strict v1.3 averages 2.39 hops / 792.81 km, while KSP-FF K=50 hops averages 1.96 hops / 641.74 km.

Strict v1.3 uses an average of 2.01 FS per admitted request, whereas KSP-FF K=50 hops uses 2.01 FS.

KSP-FF K=50 hops action is present in Strict v1.3's PPO-R Top-30 candidate pool in 0.00% of evaluated requests.

## Evidence Tables

### Win/Loss Decomposition

| Outcome | Count |
|---|---:|
| strict_win | 1648 |
| ksp_win | 1750 |
| both_success | 20640 |
| both_block | 5962 |

### strict_win: KSP-FF K=50 hops failure reason breakdown

| Reason | Count |
|---|---:|
| r_no_valid_action | 1648 |

### ksp_win: Strict v1.3 failure reason breakdown

| Reason | Count |
|---|---:|
| r_no_valid_action | 1750 |

### Divergence timing

| Metric | Value |
|---|---:|
| Mean R-divergence distance before strict_win | 1.24 |
| Median R-divergence distance before strict_win | 1.00 |
| Mean R-divergence distance before ksp_win | 1.21 |
| Median R-divergence distance before ksp_win | 1.00 |

### Action preference deltas (admitted requests)

| Metric | Strict v1.3 | KSP-FF K=50 hops |
|---|---|---|
| Avg path idx | 1.05 | 0.33 |
| Avg required FS | 2.01 | 2.01 |
| Avg block start | 7.67 | 7.18 |
| Avg block waste | 0.2398 | 0.2695 |
| Avg hops | 2.39 | 1.96 |
| Avg path km | 792.81 | 641.74 |

### Strict v1.3 candidate coverage

| Metric | Value |
|---|---:|
| KSP action in Strict Top-30 rate | 0.0000% |

## Causal Caution

Under fixed-C / fixed-OD, the R-side cannot change server selection. Any observed blocking difference is therefore attributable to optical spectrum trajectory management, not to server-choice coupling. If Strict v1.3 reduces server_overload relative to KSP-FF K=50 hops, this must reflect an indirect load imbalance caused by different success counts, not a direct server assignment.
