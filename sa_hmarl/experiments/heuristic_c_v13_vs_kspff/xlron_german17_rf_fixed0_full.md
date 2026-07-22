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
| rf_fixed0_c+ksp_ff_k50_hops | rf_fixed0_c | ksp_ff_k50_hops | 72.21% | 71.03% | 0.00% | 72.19% | 0.02% | 0.00% | 19.455/29.321 ms | 9.461/15.501 ms |
| rf_fixed0_c+v12_k50_hops | rf_fixed0_c | v12_k50_hops | 72.30% | 71.12% | 0.00% | 72.28% | 0.01% | 0.00% | 20.925/31.673 ms | 11.889/28.031 ms |