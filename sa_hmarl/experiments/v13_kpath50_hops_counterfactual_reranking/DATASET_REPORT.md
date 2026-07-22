# SA-HMARL v1.3 Pilot Dataset Report

**Protocol:** PPO-R Proposal-Supported RMSA Domain-Constrained Common-Future Counterfactual Reranking  
**Topology:** `xlron_cost239_ptrnet_real`, slots=320, servers=4  
**Hyper-parameters:** K_C=5, K_path=50, K_prop=30, H=5, γ=1.0  
**Path sort:** hops; **Block sort:** start_asc  
**Dataset path:** `sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5`  
**Generated:** 2026-07-13

## Shard composition

| Split | Shards | Requests / shard | Warmup / shard | Evaluated / shard |
|---|---:|---:|---:|---:|
| Train | 2 | 1000 | 250 | 750 |
| Validation | 1 | 1000 | 250 | 750 |
| Test | 1 | 1000 | 250 | 750 |

## Merged splits

| Split | Groups | Avg candidates | Min | Max | PPO top-1 rate | Oracle headroom (pp) | Non-zero return range |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,285 | 28.52 | 1 | 30 | 35.49% | 8.07% | 87.59% |
| Validation | 429 | 30.00 | 30 | 30 | 34.27% | 5.47% | 86.71% |
| Test | 602 | 29.98 | 23 | 30 | 31.23% | 5.47% | 89.20% |

## Labels (legacy total return)

The label is `-3·B_t - 4·ΣB - 3·ΣNSB - 0.03·D - 0.05·FS` summed over the current request and H=5 future requests.

| Split | N candidates | Mean | Std | Min | Max | Median |
|---|---:|---:|---:|---:|---:|---:|
| Train | 36,642 | -2.20 | 4.16 | -20.0 | 0.0 | -0.531 |
| Validation | 12,870 | -1.04 | 2.69 | -20.0 | 0.0 | -0.430 |
| Test | 18,049 | -1.12 | 2.45 | -20.0 | -0.191 | -0.476 |

The large return range (≈88% of groups have at least two distinct returns) indicates substantial headroom for reranking inside the PPO-R proposal support.

## Features

- **Schema:** 25-d `features_v1` (pre-decision features only).
- **Shape:** `(groups, K_prop=30, 25)` for each split.
- **Feature names:**
  1. `path_length_km`
  2. `hop_count`
  3. `lfb`
  4. `free_ratio`
  5. `frag_index`
  6. `spectral_efficiency`
  7. `reach_km`
  8. `required_fs`
  9. `block_size`
  10. `block_waste`
  11. `path_mod_feasible`
  12. `split_norm`
  13. `server_norm`
  14. `deadline_norm`
  15. `holding_norm`
  16. `intermediate_size_norm`
  17. `server_utilization`
  18. `selected_valid_r_ratio`
  19. `k_c_valid_ratio`
  20. `k_r_total_ratio`
  21. `phi_spec_norm`
  22. `raw_r_valid_ratio`
  23. `path_idx_norm`
  24. `mod_idx_norm`
  25. `block_idx_norm`

## Protocol enforcement checks

| Check | Result |
|---|---|
| Strict K_C/K_path switching | Enforced in generator and evaluator |
| Path cache invalidated on K switch | `env._path_cache.clear()` before every R observation |
| Candidates = PPO-R Top-K_prop only | No anchors/fillers; provenance bit 1 for all candidates |
| Common-future trace | `trace_hash` identical within each group; `common_future_trace_consistency: true` |
| Action space | Max observed action ID = 1921 (< 50 paths × 4 mods × 10 blocks = 2000) |
| Feature dimension | 25 |

## Known limitations

- This is a **pilot** dataset (≈1.3k train groups). It is sufficient to validate the end-to-end pipeline but too small to train a robust ranker.
- Total shard generation time was ≈1.24 h for 4,000 requests (≈1.1 s/request) on the test CPU. Each candidate branch requires a deep-copy of the environment and a short future rollout.

## Recommendation

Full-scale generation should target at least 16 train / 4 val / 4 test shards of 5000 requests each, which should give ≈30–40k train groups and a much stronger ranker.
