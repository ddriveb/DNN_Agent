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
| rf_c+ksp_ff_k50_hops | rf_c | ksp_ff_k50_hops | 12.84% | 14.58% | 0.18% | 12.40% | 0.26% | 0.00% | 16.572/29.049 ms | 15.487/20.956 ms |
| rf_c+v12_k50_hops | rf_c | v12_k50_hops | 11.67% | 13.08% | 0.21% | 11.02% | 0.44% | 0.00% | 17.375/29.717 ms | 25.892/36.434 ms |