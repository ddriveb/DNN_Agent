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
- `requests_per_episode`: `200`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_fixed0_c+ksp_ff_k50_hops | df_fixed0_c | ksp_ff_k50_hops | 39.33% | 40.00% | 0.00% | 39.33% | 0.00% | 0.00% | 11.964/24.489 ms | 8.083/11.184 ms |
| df_fixed0_c+v12_k50_hops | df_fixed0_c | v12_k50_hops | 39.33% | 40.00% | 0.00% | 39.33% | 0.00% | 0.00% | 14.404/27.020 ms | 13.073/19.735 ms |
| rf_fixed0_c+ksp_ff_k50_hops | rf_fixed0_c | ksp_ff_k50_hops | 36.00% | 40.00% | 0.00% | 36.00% | 0.00% | 0.00% | 13.081/25.255 ms | 8.183/11.323 ms |
| rf_fixed0_c+v12_k50_hops | rf_fixed0_c | v12_k50_hops | 36.00% | 40.00% | 0.00% | 36.00% | 0.00% | 0.00% | 14.602/28.131 ms | 13.045/20.012 ms |