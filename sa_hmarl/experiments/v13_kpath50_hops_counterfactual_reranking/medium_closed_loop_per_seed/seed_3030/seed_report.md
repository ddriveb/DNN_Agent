# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 3030
- Requests per seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 7.62% ± 0.00% | 7.62% | 0.00% | 9.36 | 3.44 | 10.39 | 100.00% | 0.00% |
| ksp_ff_plain | 7.58% ± 0.00% | 7.58% | 0.00% | 8.36 | 3.15 | 10.21 | 31.58% | 0.00% |
| ksp_ff_highest | 8.12% ± 0.00% | 8.12% | 0.00% | 8.12 | 2.23 | 10.26 | 15.20% | 0.00% |
| ranker_new_v13 | 7.00% ± 0.00% | 7.00% | 0.00% | 9.28 | 2.87 | 18.06 | 11.22% | 93.00% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 3030 | ppo_r_top1 | 7.62% | 7.62% | 0.00% | 10.39 | 100.00% |
| 3030 | ksp_ff_plain | 7.58% | 7.58% | 0.00% | 10.21 | 31.58% |
| 3030 | ksp_ff_highest | 8.12% | 8.12% | 0.00% | 10.26 | 15.20% |
| 3030 | ranker_new_v13 | 7.00% | 7.00% | 0.00% | 18.06 | 11.22% |