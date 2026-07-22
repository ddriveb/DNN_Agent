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
| df_c+ksp_ff_k50_hops | df_c | ksp_ff_k50_hops | 11.28% | 12.84% | 0.15% | 10.80% | 0.33% | 0.00% | 16.353/28.448 ms | 20.227/27.300 ms |
| df_c+v12_k50_hops | df_c | v12_k50_hops | 11.06% | 12.41% | 0.18% | 10.42% | 0.47% | 0.00% | 17.346/29.288 ms | 37.315/54.161 ms |