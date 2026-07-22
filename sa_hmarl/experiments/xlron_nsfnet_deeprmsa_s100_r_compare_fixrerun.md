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
| ppo_c+v12 | ppo_c | v12 | 3.09% | 3.15% | 0.76% | 2.18% | 0.14% | 0.01% | 19.526/33.812 ms | 43.643/68.318 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 3.26% | 3.28% | 0.83% | 2.23% | 0.05% | 0.16% | 19.406/33.423 ms | 34.125/52.329 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | +0.17 pp | +0.06 pp | +0.05 pp | -0.120 ms |