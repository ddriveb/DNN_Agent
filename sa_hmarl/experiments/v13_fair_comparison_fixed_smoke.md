# Fair Comparison: v1.3 Post-Decision Ranker vs Baselines

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4, C-K=5/hops, R-K=50/hops
- Seeds: 3030
- Total requests/seed: 2500 (warmup=500, evaluated=2000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r | 7.55% ± 0.00% | 7.55% | 0.00% | 9.18 | 3.46 | 8.08 | 100.00% |
| ksp_ff_plain | 10.10% ± 0.00% | 10.10% | 0.00% | 8.27 | 3.23 | 7.91 | 33.05% |
| ksp_ff_highest | 9.85% ± 0.00% | 9.85% | 0.00% | 8.20 | 2.23 | 7.75 | 16.35% |
| ranker_v13 | 7.30% ± 0.00% | 7.30% | 0.00% | 10.13 | 2.98 | 11.16 | 13.75% |
| ranker_poststate | 8.25% ± 0.00% | 8.25% | 0.00% | 9.26 | 3.11 | 35.84 | 32.15% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---|---|---:|---:|---:|---:|
| 3030 | ppo_r | 7.55% | 7.55% | 0.00% | 8.08 |
| 3030 | ksp_ff_plain | 10.10% | 10.10% | 0.00% | 7.91 |
| 3030 | ksp_ff_highest | 9.85% | 9.85% | 0.00% | 7.75 |
| 3030 | ranker_v13 | 7.30% | 7.30% | 0.00% | 11.16 |
| 3030 | ranker_poststate | 8.25% | 8.25% | 0.00% | 35.84 |