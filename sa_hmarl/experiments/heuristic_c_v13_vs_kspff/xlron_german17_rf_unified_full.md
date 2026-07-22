# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_german17`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.07142857142857142`
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
| rf_c+ksp_ff_k50_hops | rf_c | ksp_ff_k50_hops | 13.32% | 15.67% | 0.01% | 13.31% | 0.01% | 0.00% | 11.310/19.033 ms | 10.977/14.681 ms |
| rf_c+v12_k50_hops | rf_c | v12_k50_hops | 12.78% | 15.05% | 0.04% | 12.73% | 0.01% | 0.00% | 11.887/20.257 ms | 23.604/37.333 ms |