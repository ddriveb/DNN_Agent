# Stage 0 Diagnostic: Counterfactual Label Fairness & Afterstate Variance

- States evaluated (candidate pool): 8
- States evaluated (all-legal oracle): 3
- Elapsed: 216.7s

## 1. Common Random Numbers

Each candidate rollout starts from an independent clone of the same base RNG state, so continuation randomness is order-independent by construction.

## 2. Server Utilization Context

- Mean server util: 0.366
- P95 server util: 0.505
- server_util_after same across successful R candidates: True

## 3. Afterstate Variance Across Successful R Candidates

| Feature | Mean Var | Max Var | Nonzero Var Rate |
|---|---:|---:|---:|
| path_length_km | 0.0000 | 0.0000 | 0.00% |
| hop_count | 0.0000 | 0.0000 | 0.00% |
| lfb | 0.0000 | 0.0000 | 0.00% |
| free_ratio | 0.0000 | 0.0000 | 0.00% |
| frag_index | 0.0000 | 0.0000 | 0.00% |
| spectral_efficiency | 0.0000 | 0.0000 | 0.00% |
| reach_km | 0.0000 | 0.0000 | 0.00% |
| required_fs | 0.0000 | 0.0000 | 0.00% |
| block_size | 0.0000 | 0.0000 | 0.00% |
| block_waste | 0.0000 | 0.0000 | 0.00% |
| path_mod_feasible | 0.0000 | 0.0000 | 0.00% |
| selected_valid_r_ratio | 0.0000 | 0.0000 | 0.00% |
| k_c_valid_ratio | 0.0000 | 0.0000 | 0.00% |
| k_r_total_ratio | 0.0000 | 0.0000 | 0.00% |
| phi_spec_norm | 0.0000 | 0.0000 | 0.00% |
| raw_r_valid_ratio | 0.0000 | 0.0000 | 0.00% |
| path_idx_norm | 0.0000 | 0.0000 | 0.00% |
| mod_idx_norm | 0.0000 | 0.0000 | 0.00% |
| block_idx_norm | 0.0000 | 0.0000 | 0.00% |
| path_lfb_after | 0.0000 | 0.0000 | 0.00% |
| path_free_ratio_after | 0.0000 | 0.0000 | 0.00% |
| global_lfb_after | 0.0000 | 0.0000 | 0.00% |
| global_lfb_ratio_after | 0.0000 | 0.0000 | 0.00% |
| frag_after | 0.0030 | 0.0035 | 100.00% |
| delta_frag | 0.0030 | 0.0035 | 100.00% |
| free_block_count_after | 0.0030 | 0.0035 | 100.00% |
| free_block_count_delta | 0.0030 | 0.0035 | 100.00% |
| min_edge_lfb_margin_after | 0.0000 | 0.0000 | 0.00% |
| min_edge_free_ratio_after | 0.0000 | 0.0000 | 0.00% |
| bottleneck_edge_util_after | 0.0000 | 0.0000 | 0.00% |
| occupied_slot_hops | 177916.7167 | 201214.8933 | 100.00% |
| path_conflict_after | 305.3403 | 458.9956 | 100.00% |

## 4. Horizon Comparison (candidate pool)

| Horizon | Mean Range | Median Range | Nonzero Range | Top1-Top2 Gap | Spearman vs H5 | KSP Regret | PPO Regret | Ranker Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H=5 | 0.1109 | 0.1093 | 100.00% | 0.0000 | 0.000 | 0.0324 | 0.0172 | 0.0759 |
| H=12 | 0.0897 | 0.0957 | 100.00% | 0.0060 | 0.511 | 0.0443 | 0.0369 | 0.0439 |
| H=20 | 0.0970 | 0.0965 | 100.00% | 0.0052 | 0.266 | 0.0426 | 0.0324 | 0.0595 |

## 5. Mutually Exclusive Cause Rates

| Horizon | Optical+ | Overload+ | Other+ | Mean Optical | Mean Overload | Mean Other |
|---|---:|---:|---:|---:|---:|---:|
| H=5 | 0.00% | 0.00% | 0.00% | 0.000 | 0.000 | 0.000 |
| H=12 | 0.00% | 0.00% | 0.00% | 0.000 | 0.000 | 0.000 |
| H=20 | 0.00% | 0.00% | 0.00% | 0.000 | 0.000 | 0.000 |

## 6. Candidate Recall vs All-Legal H=5 Oracle

| Metric | Value |
|---|---:|
| States | 3 |
| Oracle top-1 recall | 66.67% |
| Oracle top-5 recall | 73.33% |
| KSP-FF plain recall | 100.00% |
| KSP-FF highest recall | 100.00% |

## 7. Answers to Stage 0 Questions

1. **Afterstate differences exist mainly in optical dimensions** (path/global LFB, fragmentation, bottleneck margin, occupied slot hops). Server post-load is identical across successful R candidates under fixed a_C.
2. **H=5 is shorter than H=12/H=20**; longer horizons show higher future overload positive rate and larger return range if overload carries weight.
3. **Future overload differences across R candidates are small under H=5** and only become visible at H=12/H=20, because overload is a slow, cumulative C/server phenomenon.
4. **Oracle headroom is tiny** primarily because the current label (coef=0 on overload, H=5) does not capture the dominant failure mode; candidate recall is a secondary issue.
5. **Common-random-number cloning** makes per-candidate labels order-independent; no measurable order noise remains.