# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 4001,4002,4003,4004,4005
- Requests per seed: 2500 (warmup=500, evaluated=2000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 5.63% ± 2.94% | 5.51% | 0.12% | 11.18 | 3.45 | 8.37 | 100.00% | 0.00% |
| ksp_ff_plain | 9.57% ± 1.55% | 9.27% | 0.30% | 10.50 | 3.17 | 8.13 | 26.77% | 0.00% |
| ksp_ff_highest | 6.44% ± 1.62% | 6.43% | 0.01% | 10.20 | 2.36 | 8.33 | 9.53% | 0.00% |
| ranker_old_v13 | 4.84% ± 1.40% | 4.78% | 0.06% | 13.05 | 3.17 | 28.36 | 7.26% | 95.16% |
| ranker_full_v13_pilot | 5.40% ± 3.29% | 5.16% | 0.24% | 14.72 | 3.42 | 27.32 | 9.36% | 94.60% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 4001 | ppo_r_top1 | 5.70% | 5.65% | 0.05% | 8.95 | 100.00% |
| 4002 | ppo_r_top1 | 11.30% | 10.75% | 0.55% | 8.50 | 100.00% |
| 4003 | ppo_r_top1 | 3.85% | 3.85% | 0.00% | 9.04 | 100.00% |
| 4004 | ppo_r_top1 | 3.80% | 3.80% | 0.00% | 7.76 | 100.00% |
| 4005 | ppo_r_top1 | 3.50% | 3.50% | 0.00% | 7.60 | 100.00% |
| 4001 | ksp_ff_plain | 12.50% | 12.50% | 0.00% | 8.59 | 13.85% |
| 4002 | ksp_ff_plain | 8.15% | 6.65% | 1.50% | 8.61 | 45.60% |
| 4003 | ksp_ff_plain | 9.55% | 9.55% | 0.00% | 8.83 | 13.05% |
| 4004 | ksp_ff_plain | 8.40% | 8.40% | 0.00% | 7.33 | 29.45% |
| 4005 | ksp_ff_plain | 9.25% | 9.25% | 0.00% | 7.30 | 31.90% |
| 4001 | ksp_ff_highest | 5.40% | 5.35% | 0.05% | 8.95 | 6.40% |
| 4002 | ksp_ff_highest | 7.90% | 7.90% | 0.00% | 8.73 | 7.90% |
| 4003 | ksp_ff_highest | 3.85% | 3.85% | 0.00% | 9.12 | 5.60% |
| 4004 | ksp_ff_highest | 6.90% | 6.90% | 0.00% | 7.43 | 13.35% |
| 4005 | ksp_ff_highest | 8.15% | 8.15% | 0.00% | 7.40 | 14.40% |
| 4001 | ranker_old_v13 | 5.05% | 5.00% | 0.05% | 29.92 | 7.65% |
| 4002 | ranker_old_v13 | 7.45% | 7.25% | 0.20% | 27.03 | 12.85% |
| 4003 | ranker_old_v13 | 3.50% | 3.45% | 0.05% | 30.44 | 6.30% |
| 4004 | ranker_old_v13 | 3.90% | 3.90% | 0.00% | 27.39 | 4.35% |
| 4005 | ranker_old_v13 | 4.30% | 4.30% | 0.00% | 27.04 | 5.15% |
| 4001 | ranker_full_v13_pilot | 4.90% | 4.90% | 0.00% | 28.92 | 7.75% |
| 4002 | ranker_full_v13_pilot | 11.85% | 10.85% | 1.00% | 24.96 | 20.25% |
| 4003 | ranker_full_v13_pilot | 3.05% | 2.90% | 0.15% | 29.50 | 6.60% |
| 4004 | ranker_full_v13_pilot | 3.25% | 3.25% | 0.00% | 26.80 | 5.20% |
| 4005 | ranker_full_v13_pilot | 3.95% | 3.90% | 0.05% | 26.43 | 7.00% |