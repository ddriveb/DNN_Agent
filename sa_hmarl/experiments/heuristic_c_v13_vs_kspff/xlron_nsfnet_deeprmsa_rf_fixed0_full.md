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
| rf_fixed0_c+ksp_ff_k50_hops | rf_fixed0_c | ksp_ff_k50_hops | 72.42% | 64.91% | 0.01% | 72.02% | 0.32% | 0.07% | 26.351/38.697 ms | 5.935/9.051 ms |
| rf_fixed0_c+v12_k50_hops | rf_fixed0_c | v12_k50_hops | 72.68% | 65.65% | 0.01% | 72.29% | 0.32% | 0.06% | 26.422/39.027 ms | 7.854/14.444 ms |