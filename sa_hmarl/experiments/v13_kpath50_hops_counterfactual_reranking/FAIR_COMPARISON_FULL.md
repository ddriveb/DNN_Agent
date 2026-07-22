# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 4001,4002
- Requests per seed: 2500 (warmup=500, evaluated=2000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 8.50% ± 2.80% | 8.20% | 0.30% | 13.28 | 3.45 | 9.56 | 100.00% | 0.00% |
| ksp_ff_plain | 10.33% ± 2.17% | 9.57% | 0.75% | 12.04 | 3.16 | 9.52 | 29.73% | 0.00% |
| ksp_ff_highest | 6.65% ± 1.25% | 6.62% | 0.03% | 11.87 | 2.48 | 9.74 | 7.15% | 0.00% |
| ranker_old_v13 | 6.68% ± 0.87% | 6.58% | 0.10% | 13.13 | 3.04 | 19.49 | 10.20% | 93.32% |
| ranker_new_v13 | 8.92% ± 3.82% | 8.62% | 0.30% | 17.96 | 3.45 | 16.20 | 14.00% | 91.07% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 4001 | ppo_r_top1 | 5.70% | 5.65% | 0.05% | 10.01 | 100.00% |
| 4002 | ppo_r_top1 | 11.30% | 10.75% | 0.55% | 9.10 | 100.00% |
| 4001 | ksp_ff_plain | 12.50% | 12.50% | 0.00% | 9.77 | 13.85% |
| 4002 | ksp_ff_plain | 8.15% | 6.65% | 1.50% | 9.27 | 45.60% |
| 4001 | ksp_ff_highest | 5.40% | 5.35% | 0.05% | 9.94 | 6.40% |
| 4002 | ksp_ff_highest | 7.90% | 7.90% | 0.00% | 9.53 | 7.90% |
| 4001 | ranker_old_v13 | 5.80% | 5.80% | 0.00% | 21.12 | 10.65% |
| 4002 | ranker_old_v13 | 7.55% | 7.35% | 0.20% | 17.85 | 9.75% |
| 4001 | ranker_new_v13 | 5.10% | 5.05% | 0.05% | 17.97 | 7.95% |
| 4002 | ranker_new_v13 | 12.75% | 12.20% | 0.55% | 14.42 | 20.05% |