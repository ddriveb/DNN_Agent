# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 7070
- Requests per seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 7.00% ± 0.00% | 7.00% | 0.00% | 9.22 | 3.38 | 10.47 | 100.00% | 0.00% |
| ksp_ff_plain | 7.62% ± 0.00% | 7.62% | 0.00% | 8.41 | 3.15 | 10.28 | 33.54% | 0.00% |
| ksp_ff_highest | 7.00% ± 0.00% | 7.00% | 0.00% | 8.06 | 2.23 | 10.36 | 14.04% | 0.00% |
| ranker_new_v13 | 6.40% ± 0.00% | 6.40% | 0.00% | 9.36 | 2.91 | 18.19 | 9.00% | 93.60% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 7070 | ppo_r_top1 | 7.00% | 7.00% | 0.00% | 10.47 | 100.00% |
| 7070 | ksp_ff_plain | 7.62% | 7.62% | 0.00% | 10.28 | 33.54% |
| 7070 | ksp_ff_highest | 7.00% | 7.00% | 0.00% | 10.36 | 14.04% |
| 7070 | ranker_new_v13 | 6.40% | 6.40% | 0.00% | 18.19 | 9.00% |