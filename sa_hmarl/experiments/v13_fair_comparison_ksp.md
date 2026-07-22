# Fair Comparison: v1.3 Post-Decision Ranker vs Baselines

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4, K=50, path_sort=hops
- Seeds: 3030,4040,5050,6060,7070
- Requests/seed: 5000, warmup=1000

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ksp_ff_plain | 24.36% ± 5.38% | 24.34% | 0.01% | 10.62 | 3.24 | 24.18 | 43.17% |
| ksp_ff_highest | 26.76% ± 8.67% | 26.76% | 0.00% | 10.73 | 2.28 | 24.02 | 32.21% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---|---|---:|---:|---:|---:|
| 3030 | ksp_ff_plain | 20.42% | 20.42% | 0.00% | 22.23 |
| 4040 | ksp_ff_plain | 35.00% | 34.96% | 0.04% | 27.02 |
| 5050 | ksp_ff_plain | 22.34% | 22.34% | 0.00% | 22.25 |
| 6060 | ksp_ff_plain | 21.32% | 21.30% | 0.02% | 27.66 |
| 7070 | ksp_ff_plain | 22.70% | 22.70% | 0.00% | 21.73 |
| 3030 | ksp_ff_highest | 20.30% | 20.30% | 0.00% | 22.17 |
| 4040 | ksp_ff_highest | 27.80% | 27.80% | 0.00% | 28.30 |
| 5050 | ksp_ff_highest | 22.04% | 22.04% | 0.00% | 22.43 |
| 6060 | ksp_ff_highest | 43.22% | 43.22% | 0.00% | 25.05 |
| 7070 | ksp_ff_highest | 20.46% | 20.46% | 0.00% | 22.15 |