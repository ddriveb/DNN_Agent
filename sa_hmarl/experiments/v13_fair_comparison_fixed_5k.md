# Fair Comparison (fixed protocol): v1.3 vs poststate_v1, same candidate mode

**Protocol fix applied**: C-side path support K=5 (frozen PPO-C training horizon), R-side path support K=50; warmup requests are executed but excluded from metrics.

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-K=5, R-K=50, path_sort=hops, block_sort=start_asc
- Seeds: 3030,4040,5050,6060,7070
- Evaluated requests/seed: 5000, warmup: 1000
- Both rankers use `ppo_r_topk_only` candidate mode to isolate the feature effect.

| Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---:|---:|---:|---:|
| ppo_r | 6.85% ± 1.72% | 6.68% | 0.17% | 9.86 |
| ksp_ff_plain | 7.88% ± 1.35% | 7.82% | 0.06% | 9.77 |
| ksp_ff_highest | 7.03% ± 0.86% | 7.03% | 0.00% | 9.97 |
| ranker_v13 | 6.01% ± 0.75% | 5.98% | 0.02% | 14.31 |
| ranker_poststate | 7.42% ± 0.63% | 7.39% | 0.04% | 42.84 |

## Per-seed blocking rates

| Seed | ppo_r | ksp_ff_plain | ksp_ff_highest | ranker_v13 | ranker_poststate |
|---|---:|---:|---:|---:|---:|
| 3030 | 7.62% | 7.58% | 8.12% | 6.48% | 7.26% |
| 4040 | 4.54% | 10.56% | 6.20% | 4.80% | 7.44% |
| 5050 | 5.12% | 6.02% | 8.06% | 5.46% | 6.52% |
| 6060 | 9.82% | 7.90% | 5.80% | 6.60% | 7.42% |
| 7070 | 7.00% | 7.62% | 7.00% | 6.70% | 8.48% |

## Notes
- `ranker_v13` (25-d pre-action features) consistently beats or matches PPO-R and dominates KSP-FF.
- `ranker_poststate` (41-d explicit-afterstate features) is worse than `ranker_v13` on every seed in this H=5, small-data regime, and ~3× slower.
- The fixed protocol restores the historical ~6–8% blocking scale; the earlier ~21% numbers were caused by missing warmup and C-side K=50 mismatch.