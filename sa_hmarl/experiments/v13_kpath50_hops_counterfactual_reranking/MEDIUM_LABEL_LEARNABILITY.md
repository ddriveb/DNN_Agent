# SA-HMARL v1.3 Medium Dataset Label Learnability Analysis

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- K_C=5, K_path=50, K_prop=30, H=5
- Return reconstruction max diff: 1.907e-06

## Split overview

| Split | Groups | Valid groups | Avg cand | Return mean | Return std | Current block % | Future block % | Future NSB % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7303 | 7303 | 29.01 | -1.6781 | 3.5232 | 0.00% | 5.92% | 0.00% |
| val | 1750 | 1750 | 29.98 | -1.4912 | 3.3315 | 0.00% | 5.29% | 0.00% |
| test | 2079 | 2079 | 28.90 | -2.6595 | 4.4472 | 0.00% | 10.64% | 0.00% |

## Component variance (mean over groups with variance)

| Component | Mean within-group variance | Explained frac of return var |
|---|---:|---:|
| current_block | 0.000000 | 0.00% |
| future_block | 0.159113 | 97.90% |
| future_nsb | 0.000000 | 0.00% |
| future_delay | 0.325604 | 0.73% |
| future_fs | 0.020533 | 0.11% |

## Headroom taxonomy (train)

- Groups with candidate variance: 7302
- Future blocking identical across candidates: 98.53%
- Only delay/FS differ (blocking identical): 83.88%
- Future blocking differs across candidates: 1.47%
- PPO Top-1 outside oracle tie set: 61.15%
- PPO regret mean / P50 / P90: 0.0471 / 0.0102 / 0.0751

## Distribution shift

- val return mean - train: 0.1869
- test return mean - train: -0.9814

## Feature vs return correlations (train)

| Feature | Global Spearman | Avg |Abs| per group |
|---|---:|---:|
| path_length_km | -0.223 | 0.249 |
| hop_count | -0.133 | 0.226 |
| lfb | 0.320 | 0.276 |
| free_ratio | 0.334 | 0.273 |
| frag_index | -0.139 | 0.241 |
| spectral_efficiency | 0.039 | 0.227 |
| reach_km | -0.039 | 0.227 |
| required_fs | 0.034 | 0.242 |
| block_size | 0.213 | 0.275 |
| block_waste | 0.213 | 0.256 |
| path_mod_feasible | 0.000 | 0.000 |
| split_norm | 0.198 | 0.000 |
| server_norm | 0.023 | 0.000 |
| deadline_norm | -0.040 | 0.000 |
| holding_norm | -0.002 | 0.000 |
| intermediate_size_norm | 0.091 | 0.000 |
| server_utilization | -0.379 | 0.000 |
| selected_valid_r_ratio | 0.308 | 0.000 |
| k_c_valid_ratio | 0.246 | 0.000 |
| k_r_total_ratio | 0.293 | 0.000 |
| phi_spec_norm | 0.333 | 0.000 |
| raw_r_valid_ratio | 0.318 | 0.000 |
| path_idx_norm | -0.127 | 0.233 |
| mod_idx_norm | 0.039 | 0.227 |
| block_idx_norm | 0.093 | 0.234 |

## Donor-group future-trace proxy (NOT a true multi-trace stability test)

- This proxy replaces a group's future components with those of another group that happens to have the same candidate count, then realigns by candidate index.
- Different groups usually have different candidate action IDs, so this does **not** fix the state or the action; it is only a coarse sanity check.
- Samples: 136
- Mean Kendall tau-b: 0.018 ± 0.194
- Top-1 agreement: 18.00%

## Interpretation cues

- Future blocking rarely differs across candidates; most candidate-return variance comes from delay/FS rather than future blocking.
- The donor-group proxy above cannot be used to claim that the H=5 label is 'unstable under multiple future traces'. A true multi-trace experiment must fix the state and the action IDs while varying only the future request sequence.
- Per-group correlations between features and returns are weak; the 25-d pre-decision features may not strongly encode the ranking.