# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.1`
- `holding_min`: `4.0`
- `holding_max`: `8.0`
- `size_min_mb`: `10.0`
- `size_max_mb`: `40.0`
- `seeds`: `3030,4040,5050`
- `episodes`: `10`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.190/16.304 ms | 23.021/33.136 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 9.606/16.033 ms | 13.214/18.091 ms |

## Server-Level Diagnostics

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 691 | 28.79% |
| 1 | 504 | 21.00% |
| 2 | 498 | 20.75% |
| 3 | 707 | 29.46% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.408 | 0.914 | 0.979 | 10.21% | 0.900 |
| 1 | 0.331 | 0.854 | 0.936 | 7.42% | 1.000 |
| 2 | 0.327 | 0.974 | 0.996 | 15.58% | 1.300 |
| 3 | 0.398 | 0.952 | 0.978 | 11.42% | 0.700 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 0
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 0 | 0.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 717 | 29.88% |
| 1 | 569 | 23.71% |
| 2 | 476 | 19.83% |
| 3 | 638 | 26.58% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.416 | 0.965 | 0.978 | 11.92% | 0.900 |
| 1 | 0.359 | 0.952 | 0.992 | 8.33% | 1.000 |
| 2 | 0.348 | 0.974 | 0.996 | 17.46% | 1.300 |
| 3 | 0.343 | 0.893 | 0.964 | 7.12% | 0.700 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 0
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 0 | 0.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
