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
- `seeds`: `3030`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_c+ksp_ff_k50_hops | df_c | ksp_ff_k50_hops | 14.01% | 15.88% | 0.12% | 13.61% | 0.27% | 0.00% | 16.571/28.838 ms | 11.826/15.535 ms |
| df_c+v12_k50_hops | df_c | v12_k50_hops | 13.71% | 15.44% | 0.20% | 13.03% | 0.49% | 0.00% | 18.094/29.997 ms | 140.626/403.286 ms |