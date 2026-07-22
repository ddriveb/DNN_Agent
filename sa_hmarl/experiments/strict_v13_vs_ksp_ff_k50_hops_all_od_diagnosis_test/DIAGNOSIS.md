# Diagnosis: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / all-OD)

## Protocol Lock

- **Comparison**: Strict v1.3 vs KSP-FF K=50 hops
- Topology: `xlron_cost239_ptrnet_real`
- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)
- Traffic matrix: uniform all-OD
- fixed_split_id: 0
- server_node_ids: [0, 1, 2, 3]
- R-side K_path: 50
- R-side path_sort_strategy: `hops`
- R-side block_sort_strategy: `start_asc`
- max_blocks: 10
- KSP-FF implementation: `ksp_ff_highest_mod_action` (distance-adaptive)
- Seeds: 3030
- Warmup: 100, Evaluated: 2000

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
| Strict v1.3 | 5.3000% | 0.0000% | 0.0000% | 5.3000% | 0.0000% | 11.92 | 2.07 | 992.16 | 13.01 |
| KSP-FF K=50 hops | 5.3000% | 0.0000% | 0.0000% | 5.3000% | 0.0000% | 10.57 | 2.04 | 829.69 | 3.16 |

**Blocking difference**: Strict v1.3 - KSP-FF K=50 hops = **0.00 pp**

## Mechanism Summary

Across the evaluated seeds under fixed-C / fixed-OD, Strict v1.3 achieves a blocking rate of 5.3000% while KSP-FF K=50 hops achieves 5.3000%, a difference of 0.00 percentage points.

Admitted-request path preferences differ: Strict v1.3 averages 2.00 hops / 992.16 km, while KSP-FF K=50 hops averages 1.47 hops / 829.69 km.

Strict v1.3 uses an average of 2.07 FS per admitted request, whereas KSP-FF K=50 hops uses 2.04 FS.

KSP-FF K=50 hops action is present in Strict v1.3's PPO-R Top-30 candidate pool in 33.90% of evaluated requests.

## Evidence Tables

### Win/Loss Decomposition

| Outcome | Count |
|---|---:|
| strict_win | 0 |
| ksp_win | 0 |
| both_success | 1894 |
| both_block | 106 |

### strict_win: KSP-FF K=50 hops failure reason breakdown

| Reason | Count |
|---|---:|

### ksp_win: Strict v1.3 failure reason breakdown

| Reason | Count |
|---|---:|

### Divergence timing

| Metric | Value |
|---|---:|
| Mean R-divergence distance before strict_win | N/A |
| Median R-divergence distance before strict_win | N/A |
| Mean R-divergence distance before ksp_win | N/A |
| Median R-divergence distance before ksp_win | N/A |

### Action preference deltas (admitted requests)

| Metric | Strict v1.3 | KSP-FF K=50 hops |
|---|---|---|
| Avg path idx | 4.89 | 0.00 |
| Avg required FS | 2.07 | 2.04 |
| Avg block start | 64.33 | 8.18 |
| Avg block waste | 0.9647 | 0.4859 |
| Avg hops | 2.00 | 1.47 |
| Avg path km | 992.16 | 829.69 |

### Strict v1.3 candidate coverage

| Metric | Value |
|---|---:|
| KSP action in Strict Top-30 rate | 33.9000% |

## Causal Caution

Under fixed-C / fixed-OD, the R-side cannot change server selection. Any observed blocking difference is therefore attributable to optical spectrum trajectory management, not to server-choice coupling. If Strict v1.3 reduces server_overload relative to KSP-FF K=50 hops, this must reflect an indirect load imbalance caused by different success counts, not a direct server assignment.
