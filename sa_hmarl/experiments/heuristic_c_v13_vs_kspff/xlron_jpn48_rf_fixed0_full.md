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
| rf_fixed0_c+ksp_ff_k50_hops | rf_fixed0_c | ksp_ff_k50_hops | 68.05% | 51.41% | 0.00% | 67.78% | 0.27% | 0.00% | 23.893/37.568 ms | 12.981/19.971 ms |
| rf_fixed0_c+v12_k50_hops | rf_fixed0_c | v12_k50_hops | 68.42% | 52.14% | 0.05% | 68.11% | 0.26% | 0.00% | 25.566/38.346 ms | 15.919/31.579 ms |