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
- `seeds`: `3030`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iwd_c+ksp_ff_k50_hops | iwd_c | ksp_ff_k50_hops | 58.49% | 63.82% | 0.00% | 58.49% | 0.00% | 0.00% | 13.932/26.252 ms | 5.190/7.851 ms |
| iwd_c+v12_k50_hops | iwd_c | v12_k50_hops | 58.49% | 63.82% | 0.00% | 58.49% | 0.00% | 0.00% | 14.790/27.177 ms | 14.751/43.970 ms |