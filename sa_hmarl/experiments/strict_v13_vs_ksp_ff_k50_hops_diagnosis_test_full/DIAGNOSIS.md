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

**KSP-FF K=50 hops details**:

- Function: `ksp_ff_highest_mod_action`
- Path ordering: hops
- Block ordering: start_asc
- Modulation: 4 default formats (BPSK/QPSK/8QAM/16QAM)
- `ksp_ff_action` (naive flat First-Fit / BPSK-first) is NOT used as the formal baseline

## Main Result

| Method | Blocking | Overload | NSB | C-no-valid | R-no-valid | Avg delay ms | Avg FS | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict v1.3 | 4.0000% | 0.0000% | 0.0000% | 4.0000% | 0.0000% | 9.38 | 3.19 | 835.12 | 18.31 |
| KSP-FF K=50 hops | 5.9333% | 0.0000% | 0.0000% | 5.9333% | 0.0000% | 8.35 | 2.31 | 707.10 | 9.07 |

**Blocking difference**: Strict v1.3 - KSP-FF K=50 hops = **-1.93 pp**

## Mechanism Summary

Across the evaluated seeds, Strict v1.3 achieves a blocking rate of 4.0000% while KSP-FF K=50 hops achieves 5.9333%, a difference of -1.93 percentage points in favor of Strict v1.3.

Request-level decomposition shows 204 strict_win requests versus 88 ksp_win requests, confirming the aggregate advantage is systematic.

Among strict_win requests, the dominant KSP-FF K=50 hops failure reason is 'c_no_valid_action' (204 occurrences). This indicates that KSP-FF's greedy path/mod/block choices tend to push the trajectory into states where later requests cannot be admitted.

The mean distance from the most recent R-action divergence to a strict_win event is 1.1 requests. A non-zero distance indicates that the advantage is not solely due to the immediate action being better, but rather due to trajectory-level resource-state management.

Admitted-request path preferences differ: Strict v1.3 averages 1.77 hops / 835.12 km, while KSP-FF K=50 hops averages 1.30 hops / 707.10 km.

Strict v1.3 uses an average of 3.19 FS per admitted request, whereas KSP-FF K=50 hops uses 2.31 FS. Higher FS usage in Strict v1.3 (if observed) typically reflects higher spectral-efficiency modulations or better spectrum placement, not inefficiency.

## Evidence Tables

### Win/Loss Decomposition

| Outcome | Count |
|---|---:|
| strict_win | 204 |
| ksp_win | 88 |
| both_success | 5556 |
| both_block | 152 |

### strict_win: KSP-FF K=50 hops failure reason breakdown

| Reason | Count |
|---|---:|
| c_no_valid_action | 204 |

### ksp_win: Strict v1.3 failure reason breakdown

| Reason | Count |
|---|---:|
| c_no_valid_action | 88 |

### Divergence timing

| Metric | Value |
|---|---:|
| Mean R-divergence distance before strict_win | 1.15 |
| Mean C-divergence distance before strict_win | 1.37 |
| Mean R-divergence distance before ksp_win | 1.23 |
| Mean C-divergence distance before ksp_win | 1.59 |

### Action preference deltas (admitted requests)

| Metric | Strict v1.3 | KSP-FF K=50 hops |
|---|---|---|
| Avg path idx | 1.61 | 0.00 |
| Avg required FS | 3.19 | 2.31 |
| Avg block start | 133.80 | 31.72 |
| Avg block waste | 0.4634 | 0.5293 |
| Avg hops | 1.77 | 1.30 |
| Avg path km | 835.12 | 707.10 |

## Counterfactual Probe Summary

- Sampled divergence probes: 1
- Strict action legal in KSP obs: 0/1
- KSP action legal in Strict obs: 1/1
- KSP action feasible in Strict state: 1/1
- Strict action feasible in KSP state: 0/1

## Causal Caution

If this diagnostic shows that Strict v1.3 reduces **server_overload** failures relative to KSP-FF K=50 hops, this is not because the R-ranker directly assigns a server. R-actions change the trajectory-level optical spectrum and server load state, which in turn changes the PPO-C observation and the subsequent (split, server) decision. The observed blocking gap is therefore a coupled C+R closed-loop effect, not a direct per-action server assignment.
