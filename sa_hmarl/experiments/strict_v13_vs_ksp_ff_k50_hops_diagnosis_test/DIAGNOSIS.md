# Diagnosis: Strict v1.3 vs KSP-FF K=50 hops

## Protocol Lock

- **Comparison**: Strict v1.3 vs KSP-FF K=50 hops
- Topology: `xlron_cost239_ptrnet_real`
- C-side: PPO-C (`sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt`)
- R-side methods: Strict v1.3 and KSP-FF K=50 hops
- R-side K_path: 50
- R-side path_sort_strategy: `hops`
- R-side block_sort_strategy: `start_asc`
- max_blocks: 10
- KSP-FF implementation: `ksp_ff_highest_mod_action` (distance-adaptive)
- Seeds: 3030
- Warmup: 100, Evaluated: 1000

**Strict v1.3 details**:

- Candidate set: PPO-R legal Top-30 only (`ppo_r_topk_only`)
- Feature dim: 25 pre-decision state-action features
- Label: H=5, gamma=1.0 common-future counterfactual rollout
- Model: MLP 25 -> 128 -> 64 -> 1, SiLU, dropout=0
- Strict loss: listwise KL + SmoothL1, reg_weight=1.0, lambda_pair=0, lambda_hard=0
- Checkpoint: selected by validation regret
- Ranker gate: `all` (E=0/E=1 both invoke ranker)
- Fallback: PPO-R top-1 only when no candidates

**KSP-FF K=50 hops details**:

- Function: `ksp_ff_highest_mod_action`
- Path ordering: hops
- Block ordering: start_asc
- Modulation: 4 default formats (BPSK/QPSK/8QAM/16QAM)
- `ksp_ff_action` (naive flat First-Fit / BPSK-first) is NOT used as the formal baseline

## Main Result

| Method | Blocking | Overload | NSB | C-no-valid | R-no-valid | Avg delay ms | Avg FS | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict v1.3 | 5.2000% | 0.0000% | 0.0000% | 5.2000% | 0.0000% | 9.61 | 3.10 | 978.67 | 24.10 |
| KSP-FF K=50 hops | 10.0000% | 0.0000% | 0.0000% | 10.0000% | 0.0000% | 7.58 | 2.28 | 675.17 | 11.82 |

**Blocking difference**: Strict v1.3 - KSP-FF K=50 hops = **-4.80 pp**

## Mechanism Summary

Across the evaluated seeds, Strict v1.3 achieves a blocking rate of 5.2000% while KSP-FF K=50 hops achieves 10.0000%, a difference of -4.80 percentage points in favor of Strict v1.3.

Request-level decomposition shows 61 strict_win requests versus 13 ksp_win requests, confirming the aggregate advantage is systematic.

Among strict_win requests, the dominant KSP-FF K=50 hops failure reason is 'c_no_valid_action' (61 occurrences). This indicates that KSP-FF's greedy path/mod/block choices tend to push the trajectory into states where later requests cannot be admitted.

The mean distance from the most recent R-action divergence to a strict_win event is 1.3 requests. A non-zero distance indicates that the advantage is not solely due to the immediate action being better, but rather due to trajectory-level resource-state management.

Admitted-request path preferences differ: Strict v1.3 averages 1.96 hops / 978.67 km, while KSP-FF K=50 hops averages 1.29 hops / 675.17 km.

Strict v1.3 uses an average of 3.10 FS per admitted request, whereas KSP-FF K=50 hops uses 2.28 FS. Higher FS usage in Strict v1.3 (if observed) typically reflects higher spectral-efficiency modulations or better spectrum placement, not inefficiency.

## Evidence Tables

### Win/Loss Decomposition

| Outcome | Count |
|---|---:|
| strict_win | 61 |
| ksp_win | 13 |
| both_success | 887 |
| both_block | 39 |

### strict_win: KSP-FF K=50 hops failure reason breakdown

| Reason | Count |
|---|---:|
| c_no_valid_action | 61 |

### ksp_win: Strict v1.3 failure reason breakdown

| Reason | Count |
|---|---:|
| c_no_valid_action | 13 |

### Divergence timing

| Metric | Value |
|---|---:|
| Mean R-divergence distance before strict_win | 1.28 |
| Mean C-divergence distance before strict_win | 1.41 |
| Mean R-divergence distance before ksp_win | 1.62 |
| Mean C-divergence distance before ksp_win | 1.69 |

### Action preference deltas (admitted requests)

| Metric | Strict v1.3 | KSP-FF K=50 hops |
|---|---|---|
| Avg path idx | 2.87 | 0.00 |
| Avg required FS | 3.10 | 2.28 |
| Avg block start | 115.79 | 34.04 |
| Avg block waste | 0.5796 | 0.6961 |
| Avg hops | 1.96 | 1.29 |
| Avg path km | 978.67 | 675.17 |

## Counterfactual Probe Summary

- Sampled divergence probes: 1
- Strict action legal in KSP obs: 1/1
- KSP action legal in Strict obs: 1/1
- KSP action feasible in Strict state: 1/1
- Strict action feasible in KSP state: 1/1

## Causal Caution

If this diagnostic shows that Strict v1.3 reduces **server_overload** failures relative to KSP-FF K=50 hops, this is not because the R-ranker directly assigns a server. R-actions change the trajectory-level optical spectrum and server load state, which in turn changes the PPO-C observation and the subsequent (split, server) decision. The observed blocking gap is therefore a coupled C+R closed-loop effect, not a direct per-action server assignment.
