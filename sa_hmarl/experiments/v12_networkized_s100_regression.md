# Counterfactual R-Side Ranking Closed-Loop Evaluation

| Method | Blocking | Delay mean/P95 | Raw empty | NSB | Overload | Decision mean/P95 | Same as PPO |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r | 3.66% | 8.088/14.672 ms | 3.66% | 3.25% | 0.41% | 9.915/13.558 | 100.00% |
| counterfactual_rank_only | 0.95% | 8.073/15.550 ms | 0.95% | 0.86% | 0.09% | 13.436/23.191 | 11.86% |
| deep_rmsa | 1.29% | 10.183/18.040 ms | 1.29% | 1.21% | 0.07% | 9.796/13.373 | 12.36% |