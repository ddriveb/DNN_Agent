# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_nsfnet_deeprmsa`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.07692307692307693`
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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 15.29% | 18.16% | 0.10% | 14.97% | 0.15% | 0.07% | 17.542/29.014 ms | 6.772/9.581 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 14.92% | 17.71% | 0.20% | 14.46% | 0.20% | 0.06% | 17.183/28.907 ms | 11.296/18.607 ms |