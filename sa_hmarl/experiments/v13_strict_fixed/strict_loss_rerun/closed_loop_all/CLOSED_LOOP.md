# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 4001,4002,4003,4004,4005
- Requests per seed: 2500 (warmup=500, evaluated=2000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 5.63% ± 2.94% | 5.51% | 0.12% | 11.18 | 3.45 | 8.29 | 100.00% | 0.00% |
| ksp_ff_plain | 9.57% ± 1.55% | 9.27% | 0.30% | 10.50 | 3.17 | 8.02 | 26.77% | 0.00% |
| ksp_ff_highest | 6.44% ± 1.62% | 6.43% | 0.01% | 10.20 | 2.36 | 8.21 | 9.53% | 0.00% |
| ranker_old_v13 | 4.84% ± 1.40% | 4.78% | 0.06% | 13.05 | 3.17 | 27.92 | 7.26% | 95.16% |
| ranker_full_v13_pilot | 4.82% ± 2.56% | 4.74% | 0.08% | 14.27 | 3.23 | 27.71 | 7.49% | 95.18% |
| ranker_full_v13_stratified | 5.05% ± 2.32% | 4.93% | 0.11% | 14.50 | 3.21 | 27.88 | 7.11% | 94.95% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 4001 | ppo_r_top1 | 5.70% | 5.65% | 0.05% | 8.95 | 100.00% |
| 4002 | ppo_r_top1 | 11.30% | 10.75% | 0.55% | 8.43 | 100.00% |
| 4003 | ppo_r_top1 | 3.85% | 3.85% | 0.00% | 8.96 | 100.00% |
| 4004 | ppo_r_top1 | 3.80% | 3.80% | 0.00% | 7.60 | 100.00% |
| 4005 | ppo_r_top1 | 3.50% | 3.50% | 0.00% | 7.54 | 100.00% |
| 4001 | ksp_ff_plain | 12.50% | 12.50% | 0.00% | 8.55 | 13.85% |
| 4002 | ksp_ff_plain | 8.15% | 6.65% | 1.50% | 8.49 | 45.60% |
| 4003 | ksp_ff_plain | 9.55% | 9.55% | 0.00% | 8.57 | 13.05% |
| 4004 | ksp_ff_plain | 8.40% | 8.40% | 0.00% | 7.21 | 29.45% |
| 4005 | ksp_ff_plain | 9.25% | 9.25% | 0.00% | 7.26 | 31.90% |
| 4001 | ksp_ff_highest | 5.40% | 5.35% | 0.05% | 8.83 | 6.40% |
| 4002 | ksp_ff_highest | 7.90% | 7.90% | 0.00% | 8.71 | 7.90% |
| 4003 | ksp_ff_highest | 3.85% | 3.85% | 0.00% | 9.00 | 5.60% |
| 4004 | ksp_ff_highest | 6.90% | 6.90% | 0.00% | 7.25 | 13.35% |
| 4005 | ksp_ff_highest | 8.15% | 8.15% | 0.00% | 7.29 | 14.40% |
| 4001 | ranker_old_v13 | 5.05% | 5.00% | 0.05% | 29.44 | 7.65% |
| 4002 | ranker_old_v13 | 7.45% | 7.25% | 0.20% | 26.48 | 12.85% |
| 4003 | ranker_old_v13 | 3.50% | 3.45% | 0.05% | 30.49 | 6.30% |
| 4004 | ranker_old_v13 | 3.90% | 3.90% | 0.00% | 26.96 | 4.35% |
| 4005 | ranker_old_v13 | 4.30% | 4.30% | 0.00% | 26.23 | 5.15% |
| 4001 | ranker_full_v13_pilot | 4.00% | 4.00% | 0.00% | 29.39 | 5.35% |
| 4002 | ranker_full_v13_pilot | 9.90% | 9.50% | 0.40% | 25.33 | 18.20% |
| 4003 | ranker_full_v13_pilot | 3.15% | 3.15% | 0.00% | 30.13 | 4.50% |
| 4004 | ranker_full_v13_pilot | 3.75% | 3.75% | 0.00% | 26.97 | 5.75% |
| 4005 | ranker_full_v13_pilot | 3.30% | 3.30% | 0.00% | 26.76 | 3.65% |
| 4001 | ranker_full_v13_stratified | 4.55% | 4.55% | 0.00% | 29.61 | 4.85% |
| 4002 | ranker_full_v13_stratified | 9.60% | 9.05% | 0.50% | 26.20 | 15.05% |
| 4003 | ranker_full_v13_stratified | 3.75% | 3.70% | 0.05% | 30.25 | 4.00% |
| 4004 | ranker_full_v13_stratified | 3.20% | 3.20% | 0.00% | 26.77 | 4.60% |
| 4005 | ranker_full_v13_stratified | 4.15% | 4.15% | 0.00% | 26.55 | 7.05% |