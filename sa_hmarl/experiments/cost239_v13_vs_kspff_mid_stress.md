# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.08`
- `holding_min`: `4.0`
- `holding_max`: `8.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050`
- `episodes`: `10`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 6.87% | 7.17% | 0.00% | 6.87% | 0.00% | 0.00% | 10.543/20.102 ms | 9.378/13.173 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 8.46% | 8.75% | 0.00% | 8.46% | 0.00% | 0.00% | 11.559/21.346 ms | 16.329/24.428 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 601 | 25.04% |
| 1 | 611 | 25.46% |
| 2 | 610 | 25.42% |
| 3 | 578 | 24.08% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.517 | 0.989 | 0.995 | 31.79% | 2.240 |
| 1 | 0.693 | 0.979 | 0.992 | 42.54% | 1.040 |
| 2 | 0.664 | 0.982 | 0.996 | 43.04% | 3.360 |
| 3 | 0.600 | 0.974 | 0.992 | 32.46% | 0.960 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 165
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 165 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 165 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 654 | 27.25% |
| 1 | 601 | 25.04% |
| 2 | 640 | 26.67% |
| 3 | 505 | 21.04% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.550 | 0.983 | 0.997 | 33.92% | 3.600 |
| 1 | 0.693 | 0.980 | 0.994 | 41.04% | 1.040 |
| 2 | 0.701 | 0.987 | 0.998 | 50.33% | 3.840 |
| 3 | 0.593 | 0.986 | 0.994 | 34.38% | 0.960 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 203
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 203 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 203 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
