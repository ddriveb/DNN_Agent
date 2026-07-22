# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 5050
- Requests per seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ranker_old_v13 | 5.46% ± 0.00% | 5.44% | 0.02% | 9.27 | 2.85 | 20.50 | 6.66% | 94.54% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 5050 | ranker_old_v13 | 5.46% | 5.44% | 0.02% | 20.50 | 6.66% |