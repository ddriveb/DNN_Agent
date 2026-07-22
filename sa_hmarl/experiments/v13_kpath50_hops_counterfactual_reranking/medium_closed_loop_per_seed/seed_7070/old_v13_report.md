# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4
- C-side: K=5, sort=hops/start_asc
- R-side: K=50, sort=hops/start_asc
- Ranker candidate budget: K_prop=30
- Seeds: 7070
- Requests per seed: 6000 (warmup=1000, evaluated=5000)

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ranker_old_v13 | 6.70% ± 0.00% | 6.70% | 0.00% | 9.72 | 2.95 | 19.56 | 11.16% | 93.30% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |
|---|---|---|---:|---:|---:|---:|
| 7070 | ranker_old_v13 | 6.70% | 6.70% | 0.00% | 19.56 | 11.16% |