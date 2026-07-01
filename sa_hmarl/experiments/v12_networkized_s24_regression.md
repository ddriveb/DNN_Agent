# Counterfactual R-Side Ranking Closed-Loop Evaluation

| Method | Blocking | Delay mean/P95 | Raw empty | NSB | Overload | Decision mean/P95 | Same as PPO |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r | 71.09% | 8.064/16.128 ms | 71.44% | 60.32% | 10.76% | 6.138/10.125 | 100.00% |
| counterfactual_rank_only | 62.68% | 8.870/18.698 ms | 63.05% | 46.59% | 16.09% | 6.781/12.933 | 68.78% |
| deep_rmsa | 63.51% | 8.875/18.145 ms | 63.81% | 48.34% | 15.17% | 6.304/10.353 | 5.19% |