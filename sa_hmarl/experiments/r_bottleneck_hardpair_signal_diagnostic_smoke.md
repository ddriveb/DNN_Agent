# R-Side Bottleneck / Release / Hard-Pair Signal Diagnostic

- Scenario: `snap24_gnutella_reach`, 24 slots, 4 servers, k=5, max_blocks=10
- Load: arrival_interval=0.09, holding=4.0-14.0, size=5.0-30.0 MB
- Seeds: [42], episodes/seed=1, requests/episode=20
- H-step horizon H=5
- Checkpoints: C=`sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt`, ranker=`sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`

## Aggregate counts

| Metric | Value |
|---|---:|
| Total states | 20 |
| Multi-action (>=2 legal R) states | 20 (100.00%) |
| R-mask empty states | 0 |
| Legal R count mean/std/median | 4.00 / 0.00 / 4 |
| Total candidates evaluated | 80 |
| Current success rate | 100.00% |

## Return formulas

**G_v12** = -3·current_block -4·future_blocked -3·future_nsb -2·future_overload -0.03·future_delay_mean -0.05·future_avg_fs

**G_hard** = -4·current_block -5·future_blocked -4·future_nsb -2·future_overload -0.03·future_delay_mean -0.05·future_avg_fs -0.0005·path_km -0.0208·required_fs

## Correlations with G_v12

### Basic features
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| path_km | lower_better | N/A | N/A | N/A |
| required_fs_norm | lower_better | N/A | N/A | N/A |
| block_waste | lower_better | N/A | N/A | N/A |

### Bottleneck-edge features
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| edge_utilization_mean | lower_better | N/A | N/A | N/A |
| edge_utilization_max | lower_better | N/A | N/A | N/A |
| edge_free_slot_ratio_mean | higher_better | N/A | N/A | N/A |
| edge_free_slot_ratio_min | higher_better | N/A | N/A | N/A |
| edge_largest_free_block_mean | higher_better | N/A | N/A | N/A |
| edge_largest_free_block_min | higher_better | N/A | N/A | N/A |
| min_edge_lfb_margin | higher_better | N/A | N/A | N/A |
| bottleneck_lfb_pressure | lower_better | N/A | N/A | N/A |
| path_pressure_sum | lower_better | N/A | N/A | N/A |
| path_overlap_count | lower_better | N/A | N/A | N/A |
| path_overlap_ratio | lower_better | N/A | N/A | N/A |
| post_allocation_min_lfb | higher_better | N/A | N/A | N/A |
| post_allocation_frag_delta_on_path | lower_better | N/A | N/A | N/A |

### Release-aware features
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| soon_release_fs_1step | higher_better | N/A | N/A | N/A |
| soon_release_fs_3step | higher_better | N/A | N/A | N/A |
| avg_remaining_holding_on_path | lower_better | N/A | N/A | N/A |
| max_remaining_holding_on_path | lower_better | N/A | N/A | N/A |
| long_hold_occupied_ratio | lower_better | N/A | N/A | N/A |
| post_lfb_after_near_release | higher_better | N/A | N/A | N/A |

## Correlations with future_blocked_count
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| edge_utilization_mean | lower_better | N/A | N/A | N/A |
| edge_utilization_max | lower_better | N/A | N/A | N/A |
| edge_free_slot_ratio_mean | higher_better | N/A | N/A | N/A |
| edge_free_slot_ratio_min | higher_better | N/A | N/A | N/A |
| edge_largest_free_block_mean | higher_better | N/A | N/A | N/A |
| edge_largest_free_block_min | higher_better | N/A | N/A | N/A |
| min_edge_lfb_margin | higher_better | N/A | N/A | N/A |
| bottleneck_lfb_pressure | lower_better | N/A | N/A | N/A |
| path_pressure_sum | lower_better | N/A | N/A | N/A |
| path_overlap_count | lower_better | N/A | N/A | N/A |
| path_overlap_ratio | lower_better | N/A | N/A | N/A |
| post_allocation_min_lfb | higher_better | N/A | N/A | N/A |
| post_allocation_frag_delta_on_path | lower_better | N/A | N/A | N/A |

## Correlations with future_nsb_count
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| edge_utilization_mean | lower_better | N/A | N/A | N/A |
| edge_utilization_max | lower_better | N/A | N/A | N/A |
| edge_free_slot_ratio_mean | higher_better | N/A | N/A | N/A |
| edge_free_slot_ratio_min | higher_better | N/A | N/A | N/A |
| edge_largest_free_block_mean | higher_better | N/A | N/A | N/A |
| edge_largest_free_block_min | higher_better | N/A | N/A | N/A |
| min_edge_lfb_margin | higher_better | N/A | N/A | N/A |
| bottleneck_lfb_pressure | lower_better | N/A | N/A | N/A |
| path_pressure_sum | lower_better | N/A | N/A | N/A |
| path_overlap_count | lower_better | N/A | N/A | N/A |
| path_overlap_ratio | lower_better | N/A | N/A | N/A |
| post_allocation_min_lfb | higher_better | N/A | N/A | N/A |
| post_allocation_frag_delta_on_path | lower_better | N/A | N/A | N/A |

## Hard-pair analysis

Definition: both actions succeed, current delay diff < 1 ms, block_waste diff < 1, and future_blocked / future_nsb differ or |G_v12| gap > 0.1.

| Metric | Value |
|---|---:|
| Hard-pair count | 0 |
| Total successful pairs | 120 |
| Hard-pair rate | 0.00% |
| Mean return gap | 0.0000 |
| Best hard-pair accuracy | None (edge_utilization_mean) |

### Pairwise prediction accuracy

| Feature | Direction | Accuracy | AUC |
|---|---|---:|---:|
| edge_utilization_mean | lower_better | N/A | N/A |
| edge_utilization_max | lower_better | N/A | N/A |
| edge_free_slot_ratio_mean | higher_better | N/A | N/A |
| edge_free_slot_ratio_min | higher_better | N/A | N/A |
| edge_largest_free_block_mean | higher_better | N/A | N/A |
| edge_largest_free_block_min | higher_better | N/A | N/A |
| min_edge_lfb_margin | higher_better | N/A | N/A |
| bottleneck_lfb_pressure | lower_better | N/A | N/A |
| path_pressure_sum | lower_better | N/A | N/A |
| path_overlap_count | lower_better | N/A | N/A |
| path_overlap_ratio | lower_better | N/A | N/A |
| post_allocation_min_lfb | higher_better | N/A | N/A |
| post_allocation_frag_delta_on_path | lower_better | N/A | N/A |
| soon_release_fs_1step | higher_better | N/A | N/A |
| soon_release_fs_3step | higher_better | N/A | N/A |
| avg_remaining_holding_on_path | lower_better | N/A | N/A |
| max_remaining_holding_on_path | lower_better | N/A | N/A |
| long_hold_occupied_ratio | lower_better | N/A | N/A |
| post_lfb_after_near_release | higher_better | N/A | N/A |
| path_km | lower_better | N/A | N/A |
| required_fs_norm | lower_better | N/A | N/A |
| block_waste | lower_better | N/A | N/A |

## v1.2 ranker vs DeepRMSA selected actions

| Metric | Value |
|---|---:|
| Comparable states | 20 |
| v1.2 better | 0 |
| DeepRMSA better | 0 |
| Same return | 20 |
| Incomparable | 0 |

| Mean difference (v1.2 - DeepRMSA) | Value |
|---|---:|
| path_km | 0.0000 |
| required_fs_norm | 0.0000 |
| bottleneck_lfb_pressure | 0.0000 |
| path_pressure_sum | 0.0000 |
| min_edge_lfb_margin | 0.0000 |
| soon_release_fs_1step | 0.0000 |
| avg_remaining_holding_on_path | 0.0000 |
| G_v12 | 0.0000 |
| future_blocked_count | 0.0000 |
| future_nsb_count | 0.0000 |

## Verdict

**FAIL**

Reasons:
- hard_pair_rate=0.00% < 10% or best accuracy < 58%
- best_bottleneck_spearman=0.000
- best_hard_pair_accuracy=None

Elapsed: 3.0s