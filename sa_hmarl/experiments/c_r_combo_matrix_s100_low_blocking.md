# Agent-C / Agent-R Combination Matrix

| Agent-C | Agent-R | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| C-default | counterfactual_rank_only | 1.76% | 1.76% | 1.07% | 0.69% | 8.699/18.212 ms | 11.077/18.682 ms |
| C-default | deep_rmsa | 1.80% | 1.80% | 1.12% | 0.67% | 10.596/20.508 ms | 8.915/12.501 ms |
| C-default | ppo_r | 5.14% | 5.14% | 4.14% | 1.00% | 9.448/19.224 ms | 8.932/12.419 ms |
| C-default | ksp_bf | 7.95% | 7.95% | 6.93% | 1.02% | 9.420/18.190 ms | 8.535/12.203 ms |
| C-delay-aware | counterfactual_rank_only | 0.92% | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms | 12.262/20.956 ms |
| C-delay-aware | deep_rmsa | 1.29% | 1.29% | 1.21% | 0.07% | 10.183/18.040 ms | 9.032/12.436 ms |
| C-delay-aware | ppo_r | 3.66% | 3.66% | 3.25% | 0.41% | 8.088/14.672 ms | 9.041/12.511 ms |
| C-delay-aware | ksp_bf | 7.68% | 7.68% | 6.26% | 1.41% | 8.546/15.536 ms | 8.727/12.112 ms |
| C-r-feasibility | counterfactual_rank_only | 1.16% | 1.18% | 0.54% | 0.62% | 8.819/19.019 ms | 11.267/18.025 ms |
| C-r-feasibility | deep_rmsa | 1.30% | 1.30% | 0.61% | 0.69% | 10.831/21.466 ms | 9.134/12.793 ms |
| C-r-feasibility | ppo_r | 3.60% | 3.61% | 2.12% | 1.47% | 9.500/20.917 ms | 8.941/12.422 ms |
| C-r-feasibility | ksp_bf | 7.95% | 8.01% | 5.61% | 2.34% | 10.048/21.261 ms | 8.565/12.157 ms |