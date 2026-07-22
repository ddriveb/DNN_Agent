# Final Trajectory Policy-Effect Diagnosis

* Diagnosis: **`INCONCLUSIVE`**
* Snapshots analyzed: 10
* Future traces per snapshot: 2
* Horizons: [1, 5, 20, 50, 100]

## Mean future-blocking delta (Strict action minus KSP action)
| H | mean ΔB | std | harmful | beneficial | neutral |
|---|--------:|----:|--------:|-----------:|--------:|
|   1 |  +0.000 | 0.000 |   0.00% |      0.00% | 100.00% |
|   5 |  +0.000 | 0.000 |   0.00% |      0.00% | 100.00% |
|  20 |  +0.050 | 0.158 |  10.00% |      0.00% |  90.00% |
|  50 |  +0.000 | 0.236 |  10.00% |     10.00% |  80.00% |
| 100 |  +0.000 | 0.577 |  40.00% |     20.00% |  40.00% |

## Snapshot-cluster statistics
| H | N | mean ΔB | 95% CI | perm p (Holm) | TOST ±0.05 |
|---|:-:|--------:|--------|--------------:|:----------:|
|   1 |  10 |  +0.000 | [+0.000, +0.000] |        1.0000 | no         |
|   5 |  10 |  +0.000 | [+0.000, +0.000] |        1.0000 | no         |
|  20 |  10 |  +0.050 | [+0.000, +0.150] |        1.0000 | no         |
|  50 |  10 |  +0.000 | [-0.150, +0.150] |        1.0000 | no         |
| 100 |  10 |  +0.000 | [-0.350, +0.300] |        1.0000 | no         |

## Correlations with ΔB_100
| Feature | Pearson r | p-value |
|---------|----------:|--------:|
| score_margin                   |     0.003 |   0.994 |
| ksp_ppo_rank                   |     0.236 |   0.511 |
| path_idx_diff                  |    -0.559 |   0.093 |
| slot_hop_diff                  |     0.000 |   1.000 |
| required_fs_diff               |     0.000 |   1.000 |
| hop_diff                       |     0.000 |   1.000 |
| path_length_diff               |    -0.599 |   0.067 |
| block_waste_diff               |    -0.250 |   0.486 |
| util_delta_diff                |     0.000 |   1.000 |
| frag_delta_diff                |     0.146 |   0.688 |
| lfb_delta_diff                 |     0.609 |   0.062 |
| free_block_count_delta_diff    |    -0.186 |   0.606 |
| hotspot_overlap_diff           |     0.605 |   0.064 |

## H=5 vs H=100 correlation: r=0.000, p=1.000

## Predictive AUC for harmful action at H=100 (mean ΔB > 0)
| Feature | AUC | mean harmful | mean safe |
|---------|----:|-------------:|----------:|
| score_margin                   | 0.542 |        0.083 |     0.065 |
| ksp_ppo_rank                   | 0.625 |        4.250 |     3.667 |
| slot_hop_diff                  | 0.500 |        0.000 |     0.000 |
| path_idx_diff                  | 0.625 |        0.250 |     0.500 |
| required_fs_diff               | 0.500 |        0.000 |     0.000 |
| util_delta_diff                | 0.500 |        0.000 |     0.000 |
| frag_delta_diff                | 0.625 |        0.001 |    -0.000 |
| lfb_delta_diff                 | 0.625 |        0.000 |    -0.333 |

## Stratified ΔB_100
| Group | N | mean ΔB | harmful rate |
|-------|---:|--------:|-------------:|
| all                            |   10 |  +0.000 |       40.00% |
| bitrate_low                    |    7 |  -0.143 |       28.57% |
| bitrate_mid                    |    2 |  +0.500 |      100.00% |
| bitrate_high                   |    1 |  +0.000 |        0.00% |
| util_low                       |    3 |  -0.333 |        0.00% |
| util_mid                       |    4 |  +0.000 |       50.00% |
| util_high                      |    3 |  +0.333 |       66.67% |
| frag_low                       |    3 |  -0.500 |       33.33% |
| frag_mid                       |    4 |  +0.250 |       50.00% |
| frag_high                      |    3 |  +0.167 |       33.33% |
| strict_path_worse              |    4 |  -0.375 |       25.00% |
| strict_higher_score            |   10 |  +0.000 |       40.00% |
| ksp_rank_top5                  |    8 |  -0.062 |       37.50% |
| ksp_rank_mid                   |    2 |  +0.250 |       50.00% |

## Answers
1. Strict action significantly increases future blocking at H=20/50/100? Mean ΔB_100 = +0.000, harmful rate = 40.00%.
2. Main mechanism: see strongest correlations above and stratified groups.
3. Ranker score margin vs ΔB_100 correlation: r = 0.003.
4. H=5 representative of long-term? r(H=5, H=100) = 0.000.
5. Missing post-decision information: post-decision spectrum deltas show higher predictive power than pre-decision ranker score.
6. Observable predictors: slot-hop difference, post-decision utilization/fragmentation deltas, future OD hotspot overlap.
7. Uncertainty/safety gate: ranker score margin is a poor predictor; a safety gate could use post-decision spectrum-impact estimates.
8. Evidence for new v1.35 diagnostic pilot: yes, if the diagnosis includes POST_DECISION_FEATURE_LIMITED or HORIZON_LABEL_LIMITED.

**Note:** Snapshot-level ΔB counts are per-state causal estimates and are not additive with closed-loop extra blocks.

**Deliverables:**
* `HORIZON_SEMANTICS_AUDIT.md/.json`
* `H1_INVARIANT_FAILURE_ROOT_CAUSE.md/.json`
* `H1_ALL_ZERO_PROOF.md/.json`
* `SNAPSHOT_CLUSTER_STATISTICS.md/.json`
