# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `snap24_gnutella_reach`
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
| ppo_c+v12 | ppo_c | v12 | 0.92% | 0.92% | 0.84% | 0.09% | 0.00% | 0.00% | 8.377/16.304 ms | 18.687/32.045 ms |
| ppo_c+ksp_ff | ppo_c | ksp_ff | 7.38% | 7.38% | 5.94% | 1.44% | 0.00% | 0.00% | 8.201/16.280 ms | 13.732/19.752 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 1.05% | 1.05% | 0.94% | 0.11% | 0.00% | 0.00% | 7.919/15.229 ms | 15.742/23.413 ms |
| ppo_c+deep_rmsa | ppo_c | deep_rmsa | 1.29% | 1.29% | 1.21% | 0.07% | 0.00% | 0.00% | 10.183/18.040 ms | 13.665/19.001 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff | +6.45 pp | +5.10 pp | +1.35 pp | -0.175 ms |
| ppo_c+ksp_ff_k50_hops | +0.12 pp | +0.10 pp | +0.02 pp | -0.457 ms |
| ppo_c+deep_rmsa | +0.36 pp | +0.38 pp | -0.01 pp | +1.806 ms |