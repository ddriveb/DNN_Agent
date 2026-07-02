# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `larger_legal_context`
- C candidate config: `{'k_paths': 5, 'path_sort_strategy': 'km', 'block_sort_strategy': 'mixed'}`
- R candidate config: `{'k_paths': 50, 'path_sort_strategy': 'hops', 'block_sort_strategy': 'start_asc'}`
- Horizon H: 3
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs - 0.05*norm(path_km) - 0.05*norm(required_fs)`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 60 | 0 (0.0%) | 12.00 | 16 | 16 | 85.00% | 35.00% | 0.0000 | 0.0000 |
| val | 20 | 0 (0.0%) | 16.00 | 16 | 16 | 100.00% | 90.00% | 0.0000 | 0.0000 |
| test | 20 | 0 (0.0%) | 4.00 | 4 | 4 | 60.00% | 20.00% | 0.0000 | 0.0000 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 1.562 | 87.91 | 0.961 | 0.110 | 0.188 | 0.002 |
| val | 1.425 | 92.60 | 0.970 | 0.139 | 0.142 | 0.000 |
| test | 2.500 | 80.65 | 0.970 | 0.000 | 0.500 | 0.000 |

### Phi-after viability profile

| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |
|---|---:|---:|---:|---:|
| train | 1.1333 | 0.2601 | 95.00% | -0.417 |
| val | 1.1356 | 0.2605 | 95.00% | -0.695 |
| test | 1.1152 | 0.2558 | 95.00% | 0.000 |

**Verdict: STOP_AND_AUDIT_DATASET**