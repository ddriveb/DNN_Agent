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
- `ksp_ff_k50_hops_k_paths`: `10`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 0.92% | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms | 8.877/17.190 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.94% | 0.94% | 0.79% | 0.15% | 8.584/16.849 ms | 11.374/25.165 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 1.05% | 1.05% | 0.94% | 0.11% | 7.920/15.229 ms | 5.414/8.446 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | +0.01 pp | -0.05 pp | +0.06 pp | +0.207 ms |
| ppo_c+ksp_ff_k50_hops | +0.12 pp | +0.10 pp | +0.02 pp | -0.457 ms |