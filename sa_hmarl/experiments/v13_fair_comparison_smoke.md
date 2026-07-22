# Fair Comparison: v1.3 Post-Decision Ranker vs Baselines

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4, K=50, path_sort=hops
- Seeds: 3030
- Requests/seed: 1000, warmup=200

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r | 9.60% ± 0.00% | 9.60% | 0.00% | 8.05 | 2.97 | 27.64 | 100.00% |
| ksp_ff_plain | 20.80% ± 0.00% | 20.80% | 0.00% | 6.85 | 3.19 | 23.88 | 50.50% |
| ksp_ff_highest | 13.20% ± 0.00% | 13.20% | 0.00% | 6.75 | 2.16 | 24.94 | 26.10% |
| ranker_v13 | 14.00% ± 0.00% | 14.00% | 0.00% | 9.37 | 2.79 | 29.13 | 18.10% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---|---|---:|---:|---:|---:|
| 3030 | ppo_r | 9.60% | 9.60% | 0.00% | 27.64 |
| 3030 | ksp_ff_plain | 20.80% | 20.80% | 0.00% | 23.88 |
| 3030 | ksp_ff_highest | 13.20% | 13.20% | 0.00% | 24.94 |
| 3030 | ranker_v13 | 14.00% | 14.00% | 0.00% | 29.13 |