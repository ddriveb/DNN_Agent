# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_jpn48`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.15`
- `holding_min`: `4.0`
- `holding_max`: `10.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 5.04% | 6.56% | 1.94% | 2.81% | 0.27% | 0.01% | 15.695/28.622 ms | 23.720/37.569 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 3.61% | 3.87% | 1.85% | 1.31% | 0.42% | 0.03% | 16.352/29.654 ms | 34.242/48.203 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 1499 | 18.74% |
| 1 | 1982 | 24.77% |
| 2 | 2086 | 26.07% |
| 3 | 2433 | 30.41% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.390 | 0.962 | 0.993 | 15.54% | 3.300 |
| 1 | 0.553 | 0.951 | 0.996 | 14.54% | 4.050 |
| 2 | 0.606 | 0.961 | 0.997 | 16.56% | 2.550 |
| 3 | 0.649 | 0.961 | 0.991 | 17.52% | 2.700 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 225
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 225 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 225 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 1165 | 14.56% |
| 1 | 1751 | 21.89% |
| 2 | 2168 | 27.10% |
| 3 | 2916 | 36.45% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.357 | 0.928 | 0.993 | 7.56% | 6.900 |
| 1 | 0.543 | 0.936 | 0.992 | 10.74% | 4.350 |
| 2 | 0.608 | 0.956 | 0.994 | 17.05% | 4.200 |
| 3 | 0.707 | 0.972 | 0.996 | 29.71% | 3.150 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 105
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 105 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 105 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
