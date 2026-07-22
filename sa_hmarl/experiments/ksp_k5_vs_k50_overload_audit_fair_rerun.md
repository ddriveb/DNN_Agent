# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.0625`
- `holding_min`: `20.0`
- `holding_max`: `30.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_plain_k5_hops | ppo_c | ksp_ff_plain_k5_hops | 7.46% | 9.69% | 0.00% | 7.46% | 0.00% | 0.00% | 8.989/16.470 ms | 5.943/8.438 ms |
| ppo_c+ksp_ff_plain_k50_hops | ppo_c | ksp_ff_plain_k50_hops | 7.92% | 10.34% | 0.00% | 7.92% | 0.00% | 0.00% | 8.964/16.562 ms | 8.360/12.321 ms |
| ppo_c+ksp_ff_highest_mod_k5_hops | ppo_c | ksp_ff_highest_mod_k5_hops | 9.93% | 12.97% | 0.00% | 9.93% | 0.00% | 0.00% | 9.086/17.171 ms | 5.512/8.015 ms |
| ppo_c+ksp_ff_highest_mod_k50_hops | ppo_c | ksp_ff_highest_mod_k50_hops | 9.93% | 12.97% | 0.00% | 9.93% | 0.00% | 0.00% | 9.086/17.171 ms | 8.311/12.363 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_plain_k5_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 11303 | 28.26% |
| 1 | 9517 | 23.79% |
| 2 | 9518 | 23.79% |
| 3 | 9662 | 24.16% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.943 | 0.993 | 1.000 | 87.85% | 125.062 |
| 1 | 0.943 | 0.964 | 0.995 | 95.00% | 125.062 |
| 2 | 0.934 | 0.963 | 0.995 | 87.54% | 125.062 |
| 3 | 0.938 | 0.963 | 0.995 | 90.42% | 125.062 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 2983
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 2983 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 2983 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ksp_ff_plain_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 11399 | 28.50% |
| 1 | 9484 | 23.71% |
| 2 | 9504 | 23.76% |
| 3 | 9613 | 24.03% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.943 | 0.993 | 1.000 | 87.72% | 125.062 |
| 1 | 0.943 | 0.964 | 0.995 | 94.33% | 125.062 |
| 2 | 0.934 | 0.963 | 0.995 | 86.68% | 125.062 |
| 3 | 0.940 | 0.964 | 0.995 | 91.88% | 125.062 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 3167
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 3167 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 3167 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ksp_ff_highest_mod_k5_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 12006 | 30.01% |
| 1 | 9185 | 22.96% |
| 2 | 9334 | 23.34% |
| 3 | 9475 | 23.69% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.952 | 0.994 | 1.000 | 94.01% | 125.062 |
| 1 | 0.941 | 0.966 | 0.998 | 91.78% | 125.062 |
| 2 | 0.935 | 0.964 | 0.998 | 87.88% | 125.062 |
| 3 | 0.938 | 0.964 | 0.999 | 89.55% | 125.062 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 3972
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 3972 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 3972 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ksp_ff_highest_mod_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 12006 | 30.01% |
| 1 | 9185 | 22.96% |
| 2 | 9334 | 23.34% |
| 3 | 9475 | 23.69% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.952 | 0.994 | 1.000 | 94.01% | 125.062 |
| 1 | 0.941 | 0.966 | 0.998 | 91.78% | 125.062 |
| 2 | 0.935 | 0.964 | 0.998 | 87.88% | 125.062 |
| 3 | 0.938 | 0.964 | 0.999 | 89.55% | 125.062 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 3972
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 3972 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 3972 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |


## Reproducibility Metadata

- Git commit: `6f6337f44c036518886ce8e1146b07b6b6f6d1af`
- Git dirty: `True`
- Python: `3.14.4`
- Torch: `2.11.0+cu130`
- Numpy: `2.4.4`
- Evaluator SHA-256: `be7a31f358dfd4eb6d2972710ba5ecbb1ec53bb24227cecc52776142caec2a86`
- Agent-C checkpoint SHA-256: `a5c9eb39bc33cf3b77a06d80e9c45315880dcb963ad9c3c95663eadaded9f2c8`
- Agent-R checkpoint SHA-256: `e32e908196ac0c39898aab6e98516842c53f817150eeaba97e4ac172a61f8b94`
- Ranking checkpoint SHA-256: `9e9f06c766bdd8ea6c37400a5fed6cc3ed3c7593804183a58be13a65dfaaeb21`