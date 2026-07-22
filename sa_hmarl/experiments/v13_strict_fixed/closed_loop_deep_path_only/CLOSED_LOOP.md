# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 4001,4002,4003,4004,4005
- Requests per seed: 2500 (warmup=500, evaluated=2000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 5.63% ± 2.94% | 5.51% | 0.12% | 11.18 | 3.45 | 8.56 | 100.00% | 0.00% |
| ksp_ff_plain | 9.57% ± 1.55% | 9.27% | 0.30% | 10.50 | 3.17 | 8.22 | 26.77% | 0.00% |
| ksp_ff_highest | 6.44% ± 1.62% | 6.43% | 0.01% | 10.20 | 2.36 | 8.45 | 9.53% | 0.00% |
| ranker_old_v13 | 4.44% ± 1.32% | 4.36% | 0.08% | 13.04 | 3.33 | 24.39 | 30.83% | 71.29% |
| ranker_full_v13_pilot | 5.82% ± 3.70% | 5.69% | 0.13% | 14.50 | 3.55 | 23.74 | 34.04% | 68.29% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 4001 | ppo_r_top1 | 5.70% | 5.65% | 0.05% | 9.77 | 100.00% |
| 4002 | ppo_r_top1 | 11.30% | 10.75% | 0.55% | 8.80 | 100.00% |
| 4003 | ppo_r_top1 | 3.85% | 3.85% | 0.00% | 8.98 | 100.00% |
| 4004 | ppo_r_top1 | 3.80% | 3.80% | 0.00% | 7.60 | 100.00% |
| 4005 | ppo_r_top1 | 3.50% | 3.50% | 0.00% | 7.64 | 100.00% |
| 4001 | ksp_ff_plain | 12.50% | 12.50% | 0.00% | 9.18 | 13.85% |
| 4002 | ksp_ff_plain | 8.15% | 6.65% | 1.50% | 8.57 | 45.60% |
| 4003 | ksp_ff_plain | 9.55% | 9.55% | 0.00% | 8.77 | 13.05% |
| 4004 | ksp_ff_plain | 8.40% | 8.40% | 0.00% | 7.26 | 29.45% |
| 4005 | ksp_ff_plain | 9.25% | 9.25% | 0.00% | 7.30 | 31.90% |
| 4001 | ksp_ff_highest | 5.40% | 5.35% | 0.05% | 8.96 | 6.40% |
| 4002 | ksp_ff_highest | 7.90% | 7.90% | 0.00% | 8.70 | 7.90% |
| 4003 | ksp_ff_highest | 3.85% | 3.85% | 0.00% | 9.69 | 5.60% |
| 4004 | ksp_ff_highest | 6.90% | 6.90% | 0.00% | 7.40 | 13.35% |
| 4005 | ksp_ff_highest | 8.15% | 8.15% | 0.00% | 7.47 | 14.40% |
| 4001 | ranker_old_v13 | 4.95% | 4.95% | 0.00% | 25.58 | 26.65% |
| 4002 | ranker_old_v13 | 6.80% | 6.40% | 0.40% | 26.00 | 15.70% |
| 4003 | ranker_old_v13 | 3.80% | 3.80% | 0.00% | 26.48 | 29.40% |
| 4004 | ranker_old_v13 | 3.30% | 3.30% | 0.00% | 22.19 | 44.15% |
| 4005 | ranker_old_v13 | 3.35% | 3.35% | 0.00% | 21.73 | 38.25% |
| 4001 | ranker_full_v13_pilot | 5.45% | 5.15% | 0.30% | 26.64 | 26.30% |
| 4002 | ranker_full_v13_pilot | 13.05% | 12.70% | 0.35% | 23.79 | 26.35% |
| 4003 | ranker_full_v13_pilot | 3.05% | 3.05% | 0.00% | 28.51 | 20.35% |
| 4004 | ranker_full_v13_pilot | 3.95% | 3.95% | 0.00% | 19.39 | 52.25% |
| 4005 | ranker_full_v13_pilot | 3.60% | 3.60% | 0.00% | 20.36 | 44.95% |