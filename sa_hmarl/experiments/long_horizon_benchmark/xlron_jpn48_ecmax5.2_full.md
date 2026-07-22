# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_jpn48`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.1`
- `holding_min`: `20.0`
- `holding_max`: `30.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 9.19% | 10.85% | 0.20% | 8.66% | 0.32% | 0.02% | 16.118/28.327 ms | 17.676/24.918 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 8.03% | 9.29% | 0.29% | 7.23% | 0.50% | 0.01% | 16.866/29.153 ms | 26.285/39.019 ms |