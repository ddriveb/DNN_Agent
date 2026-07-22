# Fair Comparison: v1.3 Post-Decision Ranker vs Baselines

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4, C-K=5/hops, R-K=50/hops
- Seeds: 7070
- Total requests/seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r | 7.00% ± 0.00% | 7.00% | 0.00% | 9.22 | 3.38 | 9.49 | 100.00% |
| ksp_ff_plain | 7.62% ± 0.00% | 7.62% | 0.00% | 8.41 | 3.15 | 9.42 | 33.54% |
| ksp_ff_highest | 7.00% ± 0.00% | 7.00% | 0.00% | 8.06 | 2.23 | 9.56 | 14.04% |
| ranker_v13 | 6.70% ± 0.00% | 6.70% | 0.00% | 9.72 | 2.95 | 13.28 | 11.16% |
| ranker_poststate | 8.48% ± 0.00% | 8.48% | 0.00% | 9.36 | 3.12 | 36.19 | 34.00% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---|---|---:|---:|---:|---:|
| 7070 | ppo_r | 7.00% | 7.00% | 0.00% | 9.49 |
| 7070 | ksp_ff_plain | 7.62% | 7.62% | 0.00% | 9.42 |
| 7070 | ksp_ff_highest | 7.00% | 7.00% | 0.00% | 9.56 |
| 7070 | ranker_v13 | 6.70% | 6.70% | 0.00% | 13.28 |
| 7070 | ranker_poststate | 8.48% | 8.48% | 0.00% | 36.19 |