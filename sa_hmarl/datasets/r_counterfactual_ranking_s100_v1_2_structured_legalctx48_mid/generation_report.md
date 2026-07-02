# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `larger_legal_context`
- C candidate config: `{'k_paths': 5, 'path_sort_strategy': 'km', 'block_sort_strategy': 'mixed'}`
- R candidate config: `{'k_paths': 5, 'path_sort_strategy': 'km', 'block_sort_strategy': 'mixed'}`
- Horizon H: 5
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs - 0.05*norm(path_km) - 0.05*norm(required_fs)`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 695 | 0 (0.0%) | 15.94 | 12 | 43 | 81.01% | 20.00% | 0.0029 | 0.0037 |
| val | 225 | 0 (0.0%) | 14.93 | 13 | 33 | 80.44% | 28.89% | 0.0009 | 0.0043 |
| test | 216 | 0 (0.0%) | 9.66 | 4 | 22 | 71.30% | 29.17% | 0.0074 | 0.0130 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 2.005 | 31.40 | 0.547 | 0.169 | 0.216 | 0.026 |
| val | 1.993 | 33.02 | 0.521 | 0.152 | 0.153 | 0.000 |
| test | 2.061 | 15.39 | 0.315 | 0.105 | 0.105 | 0.000 |

### Phi-after viability profile

| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |
|---|---:|---:|---:|---:|
| train | 1.1136 | 0.2538 | 95.61% | -0.152 |
| val | 1.1084 | 0.2603 | 95.47% | 0.143 |
| test | 0.9828 | 0.3919 | 87.10% | 0.155 |

**Verdict: STOP_AND_AUDIT_DATASET**