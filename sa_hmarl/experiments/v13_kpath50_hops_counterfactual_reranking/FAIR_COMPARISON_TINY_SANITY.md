# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 4001
- Requests per seed: 600 (warmup=100, evaluated=500)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 10.80% ± 0.00% | 10.80% | 0.00% | 11.32 | 3.28 | 10.51 | 100.00% | 0.00% |
| ksp_ff_plain | 19.60% ± 0.00% | 19.60% | 0.00% | 9.55 | 3.28 | 9.69 | 23.00% | 0.00% |
| ksp_ff_highest | 21.80% ± 0.00% | 21.80% | 0.00% | 9.83 | 2.30 | 9.73 | 27.20% | 0.00% |
| ranker_old_v13 | 8.80% ± 0.00% | 8.80% | 0.00% | 11.77 | 3.12 | 20.73 | 15.80% | 91.20% |
| ranker_new_v13 | 31.00% ± 0.00% | 31.00% | 0.00% | 11.50 | 2.75 | 16.73 | 38.80% | 69.00% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 4001 | ppo_r_top1 | 10.80% | 10.80% | 0.00% | 10.51 | 100.00% |
| 4001 | ksp_ff_plain | 19.60% | 19.60% | 0.00% | 9.69 | 23.00% |
| 4001 | ksp_ff_highest | 21.80% | 21.80% | 0.00% | 9.73 | 27.20% |
| 4001 | ranker_old_v13 | 8.80% | 8.80% | 0.00% | 20.73 | 15.80% |
| 4001 | ranker_new_v13 | 31.00% | 31.00% | 0.00% | 16.73 | 38.80% |