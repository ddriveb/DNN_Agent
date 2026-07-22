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
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 6.92% | 7.21% | 0.00% | 6.92% | 0.00% | 0.00% | 11.870/21.140 ms | 39.130/64.347 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 7.46% | 7.71% | 0.00% | 7.46% | 0.00% | 0.00% | 10.946/20.206 ms | 15.089/28.030 ms |

## Server-Level Diagnostics

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 695 | 28.96% |
| 1 | 590 | 24.58% |
| 2 | 571 | 23.79% |
| 3 | 544 | 22.67% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.652 | 0.991 | 0.995 | 47.58% | 1.960 |
| 1 | 0.575 | 0.982 | 0.992 | 39.46% | 3.150 |
| 2 | 0.662 | 0.982 | 0.996 | 40.71% | 2.030 |
| 3 | 0.701 | 0.990 | 0.990 | 49.46% | 1.540 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 166
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 166 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 166 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 698 | 29.08% |
| 1 | 614 | 25.58% |
| 2 | 513 | 21.38% |
| 3 | 575 | 23.96% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.681 | 0.989 | 0.996 | 48.79% | 0.910 |
| 1 | 0.586 | 0.989 | 0.992 | 41.12% | 1.330 |
| 2 | 0.641 | 0.994 | 0.999 | 42.67% | 2.030 |
| 3 | 0.708 | 0.992 | 0.995 | 51.71% | 2.100 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 179
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 179 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 179 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
