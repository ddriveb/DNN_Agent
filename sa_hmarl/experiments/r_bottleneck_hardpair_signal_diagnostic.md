# R-Side Bottleneck / Release / Hard-Pair Signal Diagnostic

- Scenario: `snap24_gnutella_reach`, 24 slots, 4 servers, k=5, max_blocks=10
- Load: arrival_interval=0.09, holding=4.0-14.0, size=5.0-30.0 MB
- Seeds: [42, 123, 456], episodes/seed=5, requests/episode=80
- H-step horizon H=5
- Checkpoints: C=`sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt`, ranker=`sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`

## Aggregate counts

| Metric | Value |
|---|---:|
| Total states | 1200 |
| Multi-action (>=2 legal R) states | 576 (48.00%) |
| R-mask empty states | 610 |
| Legal R count mean/std/median | 4.47 / 5.87 / 0 |
| Total candidates evaluated | 5365 |
| Current success rate | 100.00% |

## Return formulas

**G_v12** = -3·current_block -4·future_blocked -3·future_nsb -2·future_overload -0.03·future_delay_mean -0.05·future_avg_fs

**G_hard** = -4·current_block -5·future_blocked -4·future_nsb -2·future_overload -0.03·future_delay_mean -0.05·future_avg_fs -0.0005·path_km -0.0208·required_fs

## Correlations with G_v12

### Basic features
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| path_km | lower_better | -0.1752 | -0.1248 | N/A |
| required_fs_norm | lower_better | 0.0639 | 0.0787 | N/A |
| block_waste | lower_better | 0.3729 | 0.4148 | N/A |

### Bottleneck-edge features
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| edge_utilization_mean | lower_better | -0.2272 | -0.3033 | -0.2394 |
| edge_utilization_max | lower_better | -0.3728 | -0.4071 | -0.3392 |
| edge_free_slot_ratio_mean | higher_better | 0.2272 | 0.3031 | 0.2394 |
| edge_free_slot_ratio_min | higher_better | 0.3728 | 0.4071 | 0.3392 |
| edge_largest_free_block_mean | higher_better | 0.2313 | 0.3123 | 0.2386 |
| edge_largest_free_block_min | higher_better | 0.3765 | 0.4197 | 0.3427 |
| min_edge_lfb_margin | higher_better | 0.3753 | 0.4146 | 0.3427 |
| bottleneck_lfb_pressure | lower_better | -0.4376 | -0.4161 | -0.4176 |
| path_pressure_sum | lower_better | -0.3309 | -0.3444 | -0.3008 |
| path_overlap_count | lower_better | -0.3506 | -0.3840 | -0.3139 |
| path_overlap_ratio | lower_better | -0.2402 | -0.2008 | -0.2308 |
| post_allocation_min_lfb | higher_better | 0.3650 | 0.3974 | 0.3278 |
| post_allocation_frag_delta_on_path | lower_better | -0.1743 | -0.1912 | -0.1675 |

### Release-aware features
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| soon_release_fs_1step | higher_better | -0.0175 | -0.0572 | -0.0204 |
| soon_release_fs_3step | higher_better | -0.0283 | -0.0681 | -0.0311 |
| avg_remaining_holding_on_path | lower_better | -0.1006 | -0.0552 | -0.0542 |
| max_remaining_holding_on_path | lower_better | -0.1966 | -0.1873 | -0.1520 |
| long_hold_occupied_ratio | lower_better | -0.2420 | -0.2267 | -0.2013 |
| post_lfb_after_near_release | higher_better | 0.1643 | 0.0558 | 0.1865 |

## Correlations with future_blocked_count
| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |
|---|---|---:|---:|---:|
| edge_utilization_mean | lower_better | 0.2272 | 0.3033 | N/A |
| edge_utilization_max | lower_better | 0.3728 | 0.4071 | N/A |
| edge_free_slot_ratio_mean | higher_better | -0.2272 | -0.3031 | N/A |
| edge_free_slot_ratio_min | higher_better | -0.3728 | -0.4071 | N/A |
| edge_largest_free_block_mean | higher_better | -0.2313 | -0.3123 | N/A |
| edge_largest_free_block_min | higher_better | -0.3765 | -0.4197 | N/A |
| min_edge_lfb_margin | higher_better | -0.3753 | -0.4146 | N/A |
| bottleneck_lfb_pressure | lower_better | 0.4376 | 0.4161 | N/A |
| path_pressure_sum | lower_better | 0.3309 | 0.3444 | N/A |
| path_overlap_count | lower_better | 0.3506 | 0.3840 | N/A |
| path_overlap_ratio | lower_better | 0.2402 | 0.2008 | N/A |
| post_allocation_min_lfb | higher_better | -0.3650 | -0.3974 | N/A |
| post_allocation_frag_delta_on_path | lower_better | 0.1743 | 0.1912 | N/A |

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
| Hard-pair count | 71 |
| Total successful pairs | 29996 |
| Hard-pair rate | 0.24% |
| Mean return gap | 4.0000 |
| Best hard-pair accuracy | 0.8591549295774648 (long_hold_occupied_ratio) |

### Pairwise prediction accuracy

| Feature | Direction | Accuracy | AUC |
|---|---|---:|---:|
| edge_utilization_mean | lower_better | 0.4085 | N/A |
| edge_utilization_max | lower_better | 0.3803 | N/A |
| edge_free_slot_ratio_mean | higher_better | 0.4085 | N/A |
| edge_free_slot_ratio_min | higher_better | 0.3803 | N/A |
| edge_largest_free_block_mean | higher_better | 0.4085 | N/A |
| edge_largest_free_block_min | higher_better | 0.3521 | N/A |
| min_edge_lfb_margin | higher_better | 0.3521 | N/A |
| bottleneck_lfb_pressure | lower_better | 0.4225 | N/A |
| path_pressure_sum | lower_better | 0.4085 | N/A |
| path_overlap_count | lower_better | 0.5070 | N/A |
| path_overlap_ratio | lower_better | 0.5070 | N/A |
| post_allocation_min_lfb | higher_better | 0.4366 | N/A |
| post_allocation_frag_delta_on_path | lower_better | 0.4930 | N/A |
| soon_release_fs_1step | higher_better | 0.4085 | N/A |
| soon_release_fs_3step | higher_better | 0.4085 | N/A |
| avg_remaining_holding_on_path | lower_better | 0.4648 | N/A |
| max_remaining_holding_on_path | lower_better | 0.7606 | N/A |
| long_hold_occupied_ratio | lower_better | 0.8592 | N/A |
| post_lfb_after_near_release | higher_better | 0.3521 | N/A |
| path_km | lower_better | 0.5915 | N/A |
| required_fs_norm | lower_better | 0.4789 | N/A |
| block_waste | lower_better | 0.4085 | N/A |

## v1.2 ranker vs DeepRMSA selected actions

| Metric | Value |
|---|---:|
| Comparable states | 590 |
| v1.2 better | 7 |
| DeepRMSA better | 2 |
| Same return | 581 |
| Incomparable | 0 |

| Mean difference (v1.2 - DeepRMSA) | Value |
|---|---:|
| path_km | 62.0678 |
| required_fs_norm | -0.0000 |
| bottleneck_lfb_pressure | -0.1110 |
| path_pressure_sum | -0.1377 |
| min_edge_lfb_margin | 2.3780 |
| soon_release_fs_1step | -0.0017 |
| avg_remaining_holding_on_path | -0.0242 |
| G_v12 | 0.0339 |
| future_blocked_count | -0.0085 |
| future_nsb_count | 0.0000 |

## Verdict

**FAIL**

Reasons:
- FAIL: hard_pair_rate >= 10%
- PASS: bottleneck |Spearman| >= 0.20
- PASS: hard-pair accuracy >= 58%
- PASS: partial corr >= 0.15 (not path_km/required_fs only)

Elapsed: 333.6s