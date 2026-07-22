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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 2.00% | 2.21% | 0.00% | 2.00% | 0.00% | 0.00% | 10.939/21.611 ms | 8.681/12.072 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 4.29% | 4.42% | 0.00% | 4.29% | 0.00% | 0.00% | 11.657/22.366 ms | 16.059/23.368 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 469 | 19.54% |
| 1 | 665 | 27.71% |
| 2 | 686 | 28.58% |
| 3 | 580 | 24.17% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.506 | 0.985 | 0.996 | 27.75% | 4.100 |
| 1 | 0.648 | 0.983 | 0.995 | 36.17% | 1.400 |
| 2 | 0.638 | 0.983 | 0.997 | 39.21% | 1.400 |
| 3 | 0.596 | 0.983 | 0.995 | 27.46% | 1.200 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 48
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 48 | 66.67% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 48 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 535 | 22.29% |
| 1 | 625 | 26.04% |
| 2 | 689 | 28.71% |
| 3 | 551 | 22.96% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.521 | 0.988 | 0.997 | 29.67% | 4.900 |
| 1 | 0.628 | 0.976 | 0.991 | 32.38% | 1.400 |
| 2 | 0.686 | 0.983 | 0.996 | 46.54% | 1.400 |
| 3 | 0.592 | 0.984 | 0.991 | 31.21% | 1.200 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 103
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 103 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 103 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
