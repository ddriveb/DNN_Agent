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
| ppo_c+v12 | ppo_c | v12 | 13.92% | 14.29% | 0.00% | 13.92% | 0.00% | 0.00% | 11.393/21.266 ms | 12.474/20.344 ms |
| ppo_c+ppo_r | ppo_c | ppo_r | 15.17% | 15.71% | 0.00% | 15.17% | 0.00% | 0.00% | 14.015/23.909 ms | 10.366/17.069 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 13.75% | 14.17% | 0.00% | 13.75% | 0.00% | 0.00% | 10.795/20.789 ms | 12.161/18.651 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ppo_r | +1.25 pp | +0.00 pp | +1.25 pp | +2.623 ms |
| ppo_c+ksp_ff_k50_hops | -0.17 pp | +0.00 pp | -0.17 pp | -0.598 ms |

## Server-Level Diagnostics

### ppo_c+v12

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 766 | 31.92% |
| 1 | 595 | 24.79% |
| 2 | 577 | 24.04% |
| 3 | 462 | 19.25% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.541 | 0.987 | 0.996 | 32.83% | 3.360 |
| 1 | 0.693 | 0.983 | 0.993 | 46.79% | 0.910 |
| 2 | 0.675 | 0.991 | 0.996 | 50.08% | 2.940 |
| 3 | 0.585 | 0.990 | 0.994 | 35.92% | 0.980 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 334
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 334 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 334 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

### ppo_c+ppo_r

**Server selection distribution**
| Server | Selected count | Selected % |
|---:|---:|---:|
| 0 | 740 | 30.83% |
| 1 | 565 | 23.54% |
| 2 | 671 | 27.96% |
| 3 | 424 | 17.67% |

**Server utilization**
| Server | Mean | P95 | Max | High-util fraction | First high-util time |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.530 | 0.997 | 0.998 | 37.21% | 3.780 |
| 1 | 0.669 | 0.985 | 0.997 | 44.29% | 0.910 |
| 2 | 0.740 | 0.984 | 0.994 | 54.29% | 2.100 |
| 3 | 0.578 | 0.988 | 0.994 | 39.12% | 0.980 |

**Server queue delay (ms)**
| Server | Mean | P95 | Max |
|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 |
| 3 | 0.000 | 0.000 | 0.000 |

**Overload distribution**
Total overloads: 364
| Server | Overloads | % of overloads |
|---:|---:|---:|
| 0 | 364 | 100.00% |
| 1 | 0 | 0.00% |
| 2 | 0 | 0.00% |
| 3 | 0 | 0.00% |

**Overload by split × server**
| Split | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 0 | 364 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |

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

## Diagnostic Findings

1. **Overload is extremely concentrated**: 100% of `server_overload` events occur on **Server 0**, and all within **Split 0**. Servers 1–3 never trigger an overload under any of the three R backends.
2. **Server utilization is balanced in mean but Server 0 is the bottleneck**: mean utilization across servers is 0.53–0.74, yet Server 0 is the one whose capacity is exhausted first and is responsible for every overload.
3. **R-side differences barely move the needle**: `ppo_c+ksp_ff_k50_hops` only shaves 0.17 pp blocking vs `ppo_c+v12`; `ppo_c+ppo_r` increases it by 1.25 pp. The common denominator is Server 0 / Split 0 being overloaded.
4. **C-side server allocation is the bottleneck**: because overload is tied to a single (split, server) pair, improving path selection (R-side) cannot fundamentally fix the issue. The C policy needs to either (a) avoid mapping so many Split-0 requests to Server 0, (b) be allowed/encouraged to use other servers for Split 0, or (c) see server-utilization/pressure features that expose the impending overload.
5. **Queue delay EMA is zero**: the environment does not currently update `_queue_delay_ema` during `step()`, so queue pressure is not a live signal. Any C-side fix should rely on live utilization / capacity features rather than queue delay.
