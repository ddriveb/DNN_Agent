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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 13.75% | 14.17% | 0.00% | 13.75% | 0.00% | 0.00% | 10.795/20.789 ms | 8.248/11.974 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 12.42% | 12.79% | 0.00% | 12.42% | 0.00% | 0.00% | 11.114/21.841 ms | 14.828/22.836 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 740 | 30.83% |
| 1 | 579 | 24.12% |
| 2 | 581 | 24.21% |
| 3 | 500 | 20.83% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.548 | 0.994 | 0.996 | 38.04% | 2.870 |
| 1 | 0.672 | 0.986 | 0.997 | 44.75% | 0.910 |
| 2 | 0.637 | 0.983 | 0.992 | 45.50% | 1.890 |
| 3 | 0.622 | 0.982 | 0.996 | 39.79% | 0.980 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 330
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 330 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 330 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 735 | 30.62% |
| 1 | 535 | 22.29% |
| 2 | 620 | 25.83% |
| 3 | 510 | 21.25% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.540 | 0.993 | 0.996 | 35.62% | 3.360 |
| 1 | 0.641 | 0.976 | 0.991 | 35.17% | 0.910 |
| 2 | 0.678 | 0.981 | 0.997 | 48.54% | 2.660 |
| 3 | 0.607 | 0.986 | 0.994 | 38.33% | 0.980 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 298
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 298 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 298 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
