# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 5050
- Requests per seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 5.12% ± 0.00% | 5.12% | 0.00% | 9.43 | 3.43 | 10.68 | 100.00% | 0.00% |
| ksp_ff_plain | 6.02% ± 0.00% | 6.02% | 0.00% | 9.01 | 3.25 | 10.54 | 31.38% | 0.00% |
| ksp_ff_highest | 8.06% ± 0.00% | 8.06% | 0.00% | 9.00 | 2.32 | 10.46 | 14.78% | 0.00% |
| ranker_new_v13 | 5.00% ± 0.00% | 5.00% | 0.00% | 11.00 | 3.03 | 18.40 | 6.24% | 95.00% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 5050 | ppo_r_top1 | 5.12% | 5.12% | 0.00% | 10.68 | 100.00% |
| 5050 | ksp_ff_plain | 6.02% | 6.02% | 0.00% | 10.54 | 31.38% |
| 5050 | ksp_ff_highest | 8.06% | 8.06% | 0.00% | 10.46 | 14.78% |
| 5050 | ranker_new_v13 | 5.00% | 5.00% | 0.00% | 18.40 | 6.24% |