# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `larger_legal_context`
- C candidate config: `{'k_paths': 5, 'path_sort_strategy': 'km', 'block_sort_strategy': 'mixed'}`
- R candidate config: `{'k_paths': 5, 'path_sort_strategy': 'km', 'block_sort_strategy': 'mixed'}`
- Horizon H: 3
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs - 0.05*norm(path_km) - 0.05*norm(required_fs)`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 60 | 0 (0.0%) | 11.67 | 15 | 16 | 85.00% | 5.00% | 0.0000 | 0.0000 |
| val | 20 | 0 (0.0%) | 15.00 | 15 | 15 | 100.00% | 0.00% | 0.0000 | 0.0000 |
| test | 20 | 0 (0.0%) | 4.00 | 4 | 4 | 60.00% | 20.00% | 0.0000 | 0.0000 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 2.114 | 89.21 | 0.972 | 0.171 | 0.257 | 0.086 |
| val | 2.000 | 93.66 | 0.975 | 0.200 | 0.200 | 0.000 |
| test | 2.500 | 80.65 | 0.970 | 0.000 | 0.000 | 0.000 |

### Phi-after viability profile

| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |
|---|---:|---:|---:|---:|
| train | 1.1323 | 0.2599 | 95.00% | 0.092 |
| val | 1.1337 | 0.2601 | 95.00% | 0.000 |
| test | 1.1152 | 0.2558 | 95.00% | 0.000 |

**Verdict: STOP_AND_AUDIT_DATASET**