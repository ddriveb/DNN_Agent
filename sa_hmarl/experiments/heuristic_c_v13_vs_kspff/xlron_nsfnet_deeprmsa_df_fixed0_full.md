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
| df_fixed0_c+ksp_ff_k50_hops | df_fixed0_c | ksp_ff_k50_hops | 72.88% | 65.86% | 0.01% | 72.55% | 0.27% | 0.07% | 26.251/38.916 ms | 6.306/9.561 ms |
| df_fixed0_c+v12_k50_hops | df_fixed0_c | v12_k50_hops | 72.80% | 65.82% | 0.03% | 72.39% | 0.33% | 0.06% | 26.358/38.764 ms | 7.976/14.729 ms |