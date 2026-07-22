# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_german17`
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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 1.09% | 1.62% | 0.07% | 1.01% | 0.00% | 0.00% | 13.157/23.037 ms | 11.433/14.796 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.57% | 0.60% | 0.06% | 0.51% | 0.00% | 0.00% | 13.557/23.813 ms | 20.989/29.010 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 2138 | 26.73% |
| 1 | 1722 | 21.52% |
| 2 | 2171 | 27.14% |
| 3 | 1969 | 24.61% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.562 | 0.972 | 0.996 | 25.05% | 2.850 |
| 1 | 0.544 | 0.959 | 0.997 | 13.66% | 6.750 |
| 2 | 0.606 | 0.966 | 0.997 | 19.10% | 4.800 |
| 3 | 0.603 | 0.968 | 0.998 | 19.10% | 5.100 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 81
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 81 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 81 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 2267 | 28.34% |
| 1 | 1671 | 20.89% |
| 2 | 1908 | 23.85% |
| 3 | 2154 | 26.93% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.581 | 0.958 | 0.995 | 16.43% | 2.250 |
| 1 | 0.550 | 0.956 | 0.997 | 13.57% | 6.900 |
| 2 | 0.583 | 0.951 | 0.994 | 12.61% | 6.450 |
| 3 | 0.602 | 0.958 | 0.995 | 16.41% | 7.350 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 41
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 41 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 41 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
