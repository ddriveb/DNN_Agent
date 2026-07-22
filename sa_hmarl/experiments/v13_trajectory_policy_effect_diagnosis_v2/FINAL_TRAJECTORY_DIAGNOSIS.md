# Final Trajectory Policy-Effect Diagnosis

* Diagnosis: **`INCONCLUSIVE`**
* Snapshots analyzed: 301
* Future traces per snapshot: 5
* Horizons: [1, 5, 20, 50, 100]

## Mean future-blocking delta (Strict action minus KSP action)
| H | mean ΔB | std | harmful | beneficial | neutral |
|---|--------:|----:|--------:|-----------:|--------:|
|   1 |  +0.000 | 0.000 |   0.00% |      0.00% | 100.00% |
|   5 |  +0.000 | 0.067 |   3.32% |      3.32% |  93.36% |
|  20 |  +0.041 | 0.150 |  20.60% |      6.64% |  72.76% |
|  50 |  +0.020 | 0.210 |  21.26% |     16.61% |  62.13% |
| 100 |  +0.017 | 0.333 |  28.90% |     25.25% |  45.85% |

## Snapshot-cluster statistics
| H | N | mean ΔB | 95% CI | perm p (Holm) | TOST ±0.05 |
|---|:-:|--------:|--------|--------------:|:----------:|
|   1 | 301 |  +0.000 | [+0.000, +0.000] |        1.0000 | yes        |
|   5 | 301 |  +0.000 | [-0.008, +0.007] |        1.0000 | yes        |
|  20 | 301 |  +0.041 | [+0.023, +0.058] |        0.0002 | no         |
|  50 | 301 |  +0.020 | [-0.003, +0.044] |        0.4292 | yes        |
| 100 | 301 |  +0.017 | [-0.021, +0.054] |        1.0000 | yes        |

## Correlations with ΔB_100
| Feature | Pearson r | p-value |
|---------|----------:|--------:|
| score_margin                   |    -0.009 |   0.880 |
| ksp_ppo_rank                   |     0.012 |   0.842 |
| path_idx_diff                  |     0.133 |   0.021 |
| slot_hop_diff                  |     0.134 |   0.020 |
| required_fs_diff               |     0.178 |   0.002 |
| hop_diff                       |     0.118 |   0.041 |
| path_length_diff               |     0.119 |   0.040 |
| block_waste_diff               |    -0.017 |   0.772 |
| util_delta_diff                |     0.134 |   0.020 |
| frag_delta_diff                |     0.018 |   0.751 |
| lfb_delta_diff                 |    -0.100 |   0.084 |
| free_block_count_delta_diff    |    -0.039 |   0.502 |
| hotspot_overlap_diff           |     0.028 |   0.624 |

## H=5 vs H=100 correlation: r=-0.095, p=0.099

## Predictive AUC for harmful action at H=100 (mean ΔB > 0)
| Feature | AUC | mean harmful | mean safe |
|---------|----:|-------------:|----------:|
| score_margin                   | 0.574 |        0.059 |     0.065 |
| ksp_ppo_rank                   | 0.511 |        7.793 |     7.453 |
| slot_hop_diff                  | 0.552 |        0.644 |     0.065 |
| path_idx_diff                  | 0.560 |        0.874 |     0.210 |
| required_fs_diff               | 0.538 |        0.149 |     0.005 |
| util_delta_diff                | 0.552 |        0.000 |     0.000 |
| frag_delta_diff                | 0.626 |        0.002 |     0.001 |
| lfb_delta_diff                 | 0.625 |       -1.333 |    -0.664 |

## Stratified ΔB_100
| Group | N | mean ΔB | harmful rate |
|-------|---:|--------:|-------------:|
| all                            |  301 |  +0.017 |       28.90% |
| bitrate_low                    |  166 |  +0.028 |       31.93% |
| bitrate_mid                    |   89 |  +0.009 |       22.47% |
| bitrate_high                   |   46 |  -0.009 |       30.43% |
| util_low                       |  100 |  +0.002 |       27.00% |
| util_mid                       |  102 |  +0.031 |       29.41% |
| util_high                      |   99 |  +0.016 |       30.30% |
| frag_low                       |  100 |  +0.036 |       32.00% |
| frag_mid                       |  102 |  -0.018 |       28.43% |
| frag_high                      |   99 |  +0.032 |       26.26% |
| strict_path_worse              |   52 |  -0.000 |       42.31% |
| strict_more_slot_hops          |   16 |  +0.225 |       68.75% |
| strict_higher_score            |  301 |  +0.017 |       28.90% |
| ksp_rank_top5                  |  142 |  +0.006 |       28.87% |
| ksp_rank_mid                   |  128 |  +0.028 |       29.69% |
| ksp_rank_tail                  |   31 |  +0.019 |       25.81% |

## Answers
1. Strict action significantly increases future blocking? H=20 shows a small increase (mean ΔB=+0.041, Holm p=0.0002), but H=50 (mean ΔB=+0.020, Holm p=0.4292) and H=100 (mean ΔB=+0.017, Holm p=1.0000) are not significant after Holm correction and are TOST-equivalent to zero. No robust long-term harmful effect is detected; diagnosis = `INCONCLUSIVE`.
2. Main mechanism: the only significant mean effect appears at H=20; it does not persist at longer horizons.
3. Ranker score margin vs ΔB_100 correlation: r = -0.009 (no predictive power).
4. H=5 representative of long-term? r(H=5, H=100) = -0.095; short- and long-horizon effects are not aligned.
5. Missing post-decision information: post-decision spectrum deltas (e.g., utilization, slot-hop, path-index) correlate more strongly with ΔB than pre-decision ranker score.
6. Observable predictors: slot-hop difference, required-FS difference, post-decision utilization delta, and path-index difference show the strongest (but still modest) correlations.
7. Uncertainty/safety gate: ranker score margin is a poor predictor; a safety gate should use post-decision spectrum-impact estimates or a shorter-horizon value label.
8. Evidence for new v1.35 diagnostic pilot: limited; the H=20 signal suggests the training horizon/label may be mismatched, but the long-term effect is not statistically robust.

**Note:** Snapshot-level ΔB counts are per-state causal estimates and are not additive with closed-loop extra blocks.

**Deliverables:**
* `HORIZON_SEMANTICS_AUDIT.md/.json`
* `H1_INVARIANT_FAILURE_ROOT_CAUSE.md/.json`
* `H1_ALL_ZERO_PROOF.md/.json`
* `SNAPSHOT_CLUSTER_STATISTICS.md/.json`
