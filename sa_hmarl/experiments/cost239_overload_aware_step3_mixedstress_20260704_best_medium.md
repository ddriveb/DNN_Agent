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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 8.321/14.189 ms | 47.009/73.830 ms |

## Server-Level Diagnostics

### ppo_c+ksp_ff_k50_hops

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 761 | 31.71% |
| 1 | 250 | 10.42% |
| 2 | 799 | 33.29% |
| 3 | 590 | 24.58% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.430 | 0.954 | 0.991 | 9.29% | 2.400 |
| 1 | 0.151 | 0.835 | 0.925 | 6.42% | 2.700 |
| 2 | 0.413 | 0.963 | 0.990 | 15.50% | 2.300 |
| 3 | 0.315 | 0.937 | 0.987 | 7.92% | 2.700 |

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
