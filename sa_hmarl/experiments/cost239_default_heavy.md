# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.07`
- `holding_min`: `4.0`
- `holding_max`: `6.0`
- `size_min_mb`: `15.0`
- `size_max_mb`: `50.0`
- `seeds`: `3030,4040,5050`
- `episodes`: `10`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 1.13% | 1.21% | 0.00% | 1.13% | 0.00% | 0.00% | 11.053/18.408 ms | 40.775/63.343 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.83% | 0.92% | 0.04% | 0.79% | 0.00% | 0.00% | 9.659/16.383 ms | 17.663/27.465 ms |

## Server-Level Diagnostics

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 695 | 28.96% |
| 1 | 496 | 20.67% |
| 2 | 601 | 25.04% |
| 3 | 608 | 25.33% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.600 | 0.972 | 0.996 | 29.46% | 0.630 |
| 1 | 0.376 | 0.966 | 0.989 | 12.62% | 2.800 |
| 2 | 0.532 | 0.977 | 0.990 | 30.17% | 3.920 |
| 3 | 0.574 | 0.973 | 0.984 | 28.83% | 1.260 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 27
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 27 | 66.67% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 27 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 679 | 28.29% |
| 1 | 418 | 17.42% |
| 2 | 668 | 27.83% |
| 3 | 635 | 26.46% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.621 | 0.967 | 0.984 | 27.88% | 0.630 |
| 1 | 0.328 | 0.959 | 0.984 | 11.25% | 1.330 |
| 2 | 0.518 | 0.977 | 0.992 | 29.92% | 0.980 |
| 3 | 0.584 | 0.980 | 0.984 | 28.96% | 1.260 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 19
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 19 | 66.67% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 19 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
