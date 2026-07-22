# Fair Comparison: v1.3 Post-Decision Ranker vs Baselines

- Topology: `xlron_cost239_ptrnet_real`, slots=320, servers=4, K=50, path_sort=hops
- Seeds: 3030,4040,5050,6060,7070
- Requests/seed: 5000, warmup=1000

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r | 21.29% ± 9.09% | 21.28% | 0.00% | 11.49 | 3.29 | 27.20 | 100.00% |
| ranker_v13 | 21.69% ± 8.22% | 21.68% | 0.01% | 11.75 | 2.97 | 32.71 | 26.52% |
| ranker_poststate_topk | 22.07% ± 6.29% | 22.06% | 0.01% | 12.20 | 3.29 | 64.71 | 41.34% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---|---|---:|---:|---:|---:|
| 3030 | ppo_r | 13.78% | 13.78% | 0.00% | 25.73 |
| 4040 | ppo_r | 25.96% | 25.96% | 0.00% | 35.48 |
| 5050 | ppo_r | 17.94% | 17.94% | 0.00% | 23.88 |
| 6060 | ppo_r | 36.72% | 36.70% | 0.02% | 27.15 |
| 7070 | ppo_r | 12.04% | 12.04% | 0.00% | 23.77 |
| 3030 | ranker_v13 | 13.00% | 13.00% | 0.00% | 34.18 |
| 4040 | ranker_v13 | 27.92% | 27.92% | 0.00% | 41.79 |
| 5050 | ranker_v13 | 16.48% | 16.48% | 0.00% | 28.10 |
| 6060 | ranker_v13 | 34.66% | 34.60% | 0.04% | 32.51 |
| 7070 | ranker_v13 | 16.38% | 16.38% | 0.00% | 26.95 |
| 3030 | ranker_poststate_topk | 16.62% | 16.62% | 0.00% | 68.16 |
| 4040 | ranker_poststate_topk | 24.50% | 24.48% | 0.02% | 79.46 |
| 5050 | ranker_poststate_topk | 19.06% | 19.06% | 0.00% | 55.61 |
| 6060 | ranker_poststate_topk | 33.30% | 33.26% | 0.04% | 62.81 |
| 7070 | ranker_poststate_topk | 16.86% | 16.86% | 0.00% | 57.50 |