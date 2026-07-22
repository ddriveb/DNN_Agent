# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 6060
- Requests per seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 9.82% ± 0.00% | 8.82% | 1.00% | 14.03 | 3.30 | 12.31 | 100.00% | 0.00% |
| ksp_ff_plain | 7.90% ± 0.00% | 7.54% | 0.34% | 13.12 | 3.19 | 12.07 | 56.48% | 0.00% |
| ksp_ff_highest | 5.80% ± 0.00% | 5.78% | 0.00% | 12.79 | 2.61 | 12.33 | 6.02% | 0.00% |
| ranker_new_v13 | 8.76% ± 0.00% | 8.50% | 0.24% | 16.99 | 3.17 | 17.62 | 15.56% | 91.24% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 6060 | ppo_r_top1 | 9.82% | 8.82% | 1.00% | 12.31 | 100.00% |
| 6060 | ksp_ff_plain | 7.90% | 7.54% | 0.34% | 12.07 | 56.48% |
| 6060 | ksp_ff_highest | 5.80% | 5.78% | 0.00% | 12.33 | 6.02% |
| 6060 | ranker_new_v13 | 8.76% | 8.50% | 0.24% | 17.62 | 15.56% |