# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.472/19.558 ms | 8.730/12.090 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.21% | 0.21% | 0.00% | 0.21% | 0.00% | 0.00% | 11.123/20.717 ms | 16.856/24.845 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 1791 | 22.39% |
| 1 | 2336 | 29.20% |
| 2 | 2093 | 26.16% |
| 3 | 1780 | 22.25% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.473 | 0.970 | 0.997 | 17.93% | 2.250 |
| 1 | 0.574 | 0.969 | 0.996 | 23.40% | 1.950 |
| 2 | 0.546 | 0.965 | 0.995 | 18.70% | 1.950 |
| 3 | 0.473 | 0.944 | 0.995 | 14.03% | 1.800 |

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

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 1708 | 21.35% |
| 1 | 2261 | 28.26% |
| 2 | 2340 | 29.25% |
| 3 | 1691 | 21.14% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.482 | 0.968 | 0.996 | 17.72% | 1.500 |
| 1 | 0.574 | 0.971 | 0.997 | 25.21% | 1.950 |
| 2 | 0.591 | 0.972 | 0.993 | 22.66% | 2.250 |
| 3 | 0.480 | 0.956 | 0.996 | 13.85% | 1.800 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 17
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 17 | 80.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 17 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
