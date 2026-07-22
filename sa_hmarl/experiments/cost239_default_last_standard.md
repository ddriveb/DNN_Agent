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
- `seeds`: `3030,4040,5050`
- `episodes`: `10`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 9.029/15.321 ms | 22.039/31.757 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 7.957/13.304 ms | 13.517/19.035 ms |

## Server-Level Diagnostics

### ppo_c+v12_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 344 | 14.33% |
| 1 | 465 | 19.38% |
| 2 | 895 | 37.29% |
| 3 | 696 | 29.00% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.190 | 0.710 | 0.905 | 1.17% | 3.300 |
| 1 | 0.271 | 0.934 | 0.989 | 7.29% | 5.700 |
| 2 | 0.470 | 0.862 | 0.923 | 1.83% | 3.600 |
| 3 | 0.375 | 0.850 | 0.942 | 1.46% | 4.050 |

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
| 0 | 328 | 13.67% |
| 1 | 401 | 16.71% |
| 2 | 992 | 41.33% |
| 3 | 679 | 28.29% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.197 | 0.742 | 0.937 | 1.50% | 3.300 |
| 1 | 0.236 | 0.958 | 0.996 | 9.08% | 5.700 |
| 2 | 0.511 | 0.874 | 0.939 | 2.58% | 5.700 |
| 3 | 0.356 | 0.841 | 0.946 | 1.88% | 4.050 |

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
