# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.0625`
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
| df_fixed0_c+ksp_ff_k50_hops | df_fixed0_c | ksp_ff_k50_hops | 73.97% | 76.60% | 0.00% | 73.97% | 0.00% | 0.00% | 18.205/27.159 ms | 8.686/14.421 ms |
| df_fixed0_c+v12_k50_hops | df_fixed0_c | v12_k50_hops | 73.97% | 76.60% | 0.00% | 73.97% | 0.00% | 0.00% | 19.503/28.759 ms | 11.484/24.462 ms |