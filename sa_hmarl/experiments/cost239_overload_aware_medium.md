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
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.25% | 0.25% | 0.00% | 0.25% | 0.00% | 0.00% | 12.135/20.585 ms | 33.624/52.803 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.21% | 0.25% | 0.00% | 0.21% | 0.00% | 0.00% | 11.225/20.418 ms | 24.664/38.968 ms |

## Server-Level Diagnostics

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 621 | 25.88% |
| 1 | 543 | 22.62% |
| 2 | 561 | 23.38% |
| 3 | 675 | 28.12% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.584 | 0.986 | 0.997 | 31.00% | 1.800 |
| 1 | 0.500 | 0.975 | 0.987 | 22.54% | 1.300 |
| 2 | 0.558 | 0.974 | 0.996 | 23.00% | 3.300 |
| 3 | 0.656 | 0.987 | 0.997 | 35.83% | 3.000 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 6
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 6 | 66.67% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 6 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 622 | 25.92% |
| 1 | 495 | 20.62% |
| 2 | 537 | 22.38% |
| 3 | 746 | 31.08% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.621 | 0.990 | 0.998 | 36.29% | 1.600 |
| 1 | 0.473 | 0.970 | 0.998 | 22.13% | 1.500 |
| 2 | 0.567 | 0.986 | 0.993 | 33.46% | 3.200 |
| 3 | 0.659 | 0.979 | 0.996 | 38.75% | 2.800 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 5
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 5 | 33.33% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 5 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
