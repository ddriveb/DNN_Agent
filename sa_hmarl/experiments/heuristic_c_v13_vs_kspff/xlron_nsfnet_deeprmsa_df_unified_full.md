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
| df_c+ksp_ff_k50_hops | df_c | ksp_ff_k50_hops | 20.33% | 23.23% | 0.07% | 20.10% | 0.13% | 0.03% | 18.190/29.187 ms | 7.668/10.202 ms |
| df_c+v12_k50_hops | df_c | v12_k50_hops | 20.26% | 23.01% | 0.18% | 19.75% | 0.25% | 0.07% | 18.044/29.132 ms | 15.590/22.821 ms |