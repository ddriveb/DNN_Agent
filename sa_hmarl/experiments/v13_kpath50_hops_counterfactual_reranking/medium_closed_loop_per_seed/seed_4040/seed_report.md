# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 4040
- Requests per seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 4.54% ± 0.00% | 4.50% | 0.04% | 12.31 | 3.59 | 12.79 | 100.00% | 0.00% |
| ksp_ff_plain | 10.56% ± 0.00% | 10.56% | 0.00% | 11.63 | 3.00 | 12.21 | 11.86% | 0.00% |
| ksp_ff_highest | 6.20% ± 0.00% | 6.20% | 0.00% | 10.96 | 2.36 | 12.72 | 6.84% | 0.00% |
| ranker_new_v13 | 4.68% ± 0.00% | 4.66% | 0.02% | 13.76 | 3.34 | 20.82 | 7.14% | 95.32% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 4040 | ppo_r_top1 | 4.54% | 4.50% | 0.04% | 12.79 | 100.00% |
| 4040 | ksp_ff_plain | 10.56% | 10.56% | 0.00% | 12.21 | 11.86% |
| 4040 | ksp_ff_highest | 6.20% | 6.20% | 0.00% | 12.72 | 6.84% |
| 4040 | ranker_new_v13 | 4.68% | 4.66% | 0.02% | 20.82 | 7.14% |