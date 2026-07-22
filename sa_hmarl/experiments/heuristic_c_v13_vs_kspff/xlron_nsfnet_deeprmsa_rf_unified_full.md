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
| rf_c+ksp_ff_k50_hops | rf_c | ksp_ff_k50_hops | 20.53% | 23.40% | 0.10% | 20.25% | 0.15% | 0.03% | 18.854/29.591 ms | 7.917/10.349 ms |
| rf_c+v12_k50_hops | rf_c | v12_k50_hops | 21.10% | 23.86% | 0.18% | 20.58% | 0.27% | 0.07% | 18.720/29.760 ms | 12.616/19.542 ms |