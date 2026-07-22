# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_nsfnet_deeprmsa`
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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 3.26% | 3.28% | 0.83% | 2.23% | 0.05% | 0.16% | 19.406/33.423 ms | 7.931/11.099 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 3.00% | 3.06% | 0.74% | 2.05% | 0.12% | 0.09% | 19.629/33.901 ms | 10.935/15.679 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 1813 | 22.66% |
| 1 | 1857 | 23.21% |
| 2 | 2079 | 25.99% |
| 3 | 2251 | 28.14% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.611 | 0.973 | 0.997 | 25.96% | 1.800 |
| 1 | 0.689 | 0.982 | 0.998 | 38.64% | 1.500 |
| 2 | 0.661 | 0.978 | 0.997 | 32.09% | 2.700 |
| 3 | 0.676 | 0.982 | 0.999 | 36.72% | 1.800 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 178
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 178 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 178 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 1815 | 22.69% |
| 1 | 1916 | 23.95% |
| 2 | 1991 | 24.89% |
| 3 | 2278 | 28.48% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.612 | 0.971 | 0.997 | 24.25% | 1.950 |
| 1 | 0.701 | 0.983 | 0.997 | 40.21% | 1.500 |
| 2 | 0.639 | 0.977 | 0.994 | 31.18% | 5.400 |
| 3 | 0.680 | 0.982 | 0.998 | 41.72% | 1.800 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 164
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 164 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 164 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
