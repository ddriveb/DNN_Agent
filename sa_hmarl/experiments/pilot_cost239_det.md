# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.0625`
- `holding_min`: `20.0`
- `holding_max`: `30.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 69.51% | 71.92% | 0.00% | 69.51% | 0.00% | 0.00% | 16.774/27.879 ms | 4.924/8.265 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 69.64% | 72.08% | 0.00% | 69.64% | 0.00% | 0.00% | 17.412/28.197 ms | 7.745/18.022 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 6209 | 77.61% |
| 1 | 601 | 7.51% |
| 2 | 611 | 7.64% |
| 3 | 579 | 7.24% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.976 | 0.998 | 1.000 | 99.99% | 125.062 |
| 1 | 0.969 | 0.996 | 0.999 | 99.69% | 125.062 |
| 2 | 0.968 | 0.995 | 1.000 | 99.69% | 125.062 |
| 3 | 0.968 | 0.995 | 1.000 | 99.88% | 125.062 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 5561
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 5561 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 5561 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 6231 | 77.89% |
| 1 | 578 | 7.22% |
| 2 | 593 | 7.41% |
| 3 | 598 | 7.47% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.976 | 0.998 | 1.000 | 99.86% | 125.062 |
| 1 | 0.968 | 0.994 | 1.000 | 99.60% | 125.062 |
| 2 | 0.968 | 0.993 | 1.000 | 99.72% | 125.062 |
| 3 | 0.968 | 0.992 | 1.000 | 99.79% | 125.062 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 5571
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 5571 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 5571 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
