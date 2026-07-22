**MECHANICAL_INVALID** | **SUPERSEDED_PENDING_H1_FIX**

This report violates the H=1 hard invariant: H=1 must evaluate zero future requests, so ΔB_1 must be exactly 0. The reported H=1 non-zero values prove a horizon-counting bug. Do not use this report for conclusions.

# Final Trajectory Policy-Effect Diagnosis

> Superseded claims: `CANDIDATE_SUPPORT_LIMITED`, "expand Top-30 to fix Strict blocking gap",
> "DeepRMSA-source-semantic 45.80% equals original DeepRMSA", and "masked DeepRMSA 14.41% equals native DeepRMSA"
> are all permanently retired.  This report is the only current trajectory-effect diagnosis.

* Diagnosis: **`INCONCLUSIVE`**
* Snapshots analyzed: 301
* Future traces per snapshot: 5
* Horizons: [1, 5, 20, 50, 100]

## Mean future-blocking delta (Strict action minus KSP action)
| H | mean ΔB | std | harmful | beneficial | neutral |
|---|--------:|----:|--------:|-----------:|--------:|
|   1 |  -0.001 | 0.031 |   1.00% |      1.33% |  97.67% |
|   5 |  +0.005 | 0.069 |   5.65% |      3.32% |  91.03% |
|  20 |  +0.034 | 0.145 |  19.60% |      7.97% |  72.43% |
|  50 |  +0.027 | 0.228 |  23.26% |     16.61% |  60.13% |
| 100 |  +0.006 | 0.361 |  29.90% |     25.91% |  44.19% |

## Correlations with ΔB_100
| Feature | Pearson r | p-value |
|---------|----------:|--------:|
| score_margin                   |     0.023 |   0.686 |
| ksp_ppo_rank                   |     0.065 |   0.258 |
| path_idx_diff                  |     0.064 |   0.266 |
| slot_hop_diff                  |     0.081 |   0.161 |
| required_fs_diff               |     0.132 |   0.022 |
| hop_diff                       |     0.069 |   0.235 |
| path_length_diff               |     0.082 |   0.154 |
| block_waste_diff               |    -0.002 |   0.975 |
| util_delta_diff                |     0.081 |   0.161 |
| frag_delta_diff                |     0.013 |   0.829 |
| lfb_delta_diff                 |    -0.045 |   0.439 |
| free_block_count_delta_diff    |    -0.085 |   0.139 |
| hotspot_overlap_diff           |    -0.024 |   0.684 |

## H=5 vs H=100 correlation: r=-0.108, p=0.060

## Predictive AUC for harmful action at H=100 (mean ΔB > 0)
| Feature | AUC | mean harmful | mean safe |
|---------|----:|-------------:|----------:|
| score_margin                   | 0.579 |        0.055 |     0.067 |
| ksp_ppo_rank                   | 0.543 |        8.500 |     7.147 |
| slot_hop_diff                  | 0.525 |        0.356 |     0.180 |
| path_idx_diff                  | 0.544 |        0.589 |     0.322 |
| required_fs_diff               | 0.513 |        0.089 |     0.028 |
| util_delta_diff                | 0.525 |        0.000 |     0.000 |
| frag_delta_diff                | 0.622 |        0.002 |     0.001 |
| lfb_delta_diff                 | 0.630 |       -1.278 |    -0.678 |

## Stratified ΔB_100
| Group | N | mean ΔB | harmful rate |
|-------|---:|--------:|-------------:|
| all                            |  301 |  +0.006 |       29.90% |
| bitrate_low                    |  166 |  +0.025 |       33.73% |
| bitrate_mid                    |   89 |  +0.036 |       23.60% |
| bitrate_high                   |   46 |  -0.122 |       28.26% |
| util_low                       |  100 |  -0.002 |       28.00% |
| util_mid                       |  102 |  +0.029 |       30.39% |
| util_high                      |   99 |  -0.010 |       31.31% |
| frag_low                       |  100 |  +0.024 |       31.00% |
| frag_mid                       |  102 |  -0.031 |       32.35% |
| frag_high                      |   99 |  +0.026 |       26.26% |
| strict_path_worse              |   52 |  +0.012 |       40.38% |
| strict_more_slot_hops          |   16 |  +0.163 |       50.00% |
| strict_higher_score            |  301 |  +0.006 |       29.90% |
| ksp_rank_top5                  |  142 |  -0.007 |       29.58% |
| ksp_rank_mid                   |  128 |  +0.011 |       28.91% |
| ksp_rank_tail                  |   31 |  +0.045 |       35.48% |

## Answers
1. Strict action significantly increases future blocking at H=20/50/100? Mean ΔB_100 = +0.006, harmful rate = 29.90%.
2. Main mechanism: see strongest correlations above and stratified groups.
3. Ranker score margin vs ΔB_100 correlation: r = 0.023.
4. H=5 representative of long-term? r(H=5, H=100) = -0.108.
5. Missing post-decision information: post-decision spectrum deltas show higher predictive power than pre-decision ranker score.
6. Observable predictors: slot-hop difference, post-decision utilization/fragmentation deltas, future OD hotspot overlap.
7. Uncertainty/safety gate: ranker score margin is a poor predictor; a safety gate could use post-decision spectrum-impact estimates.
8. Evidence for new v1.35 diagnostic pilot: yes, if the diagnosis includes POST_DECISION_FEATURE_LIMITED or HORIZON_LABEL_LIMITED.

**Note:** Snapshot-level ΔB counts are per-state causal estimates and are not additive with closed-loop extra blocks.
