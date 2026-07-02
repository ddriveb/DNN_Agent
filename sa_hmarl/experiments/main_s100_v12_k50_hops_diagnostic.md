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

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 0.92% | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms | 12.187/20.987 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.94% | 0.94% | 0.81% | 0.12% | 8.596/16.921 ms | 29.653/82.529 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 1.05% | 1.05% | 0.94% | 0.11% | 7.919/15.229 ms | 10.197/14.736 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | +0.01 pp | -0.03 pp | +0.04 pp | +0.219 ms |
| ppo_c+ksp_ff_k50_hops | +0.12 pp | +0.10 pp | +0.02 pp | -0.457 ms |