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
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_c+ksp_ff_k50_hops | df_c | ksp_ff_k50_hops | 14.84% | 17.50% | 0.00% | 14.84% | 0.00% | 0.00% | 9.064/15.613 ms | 10.078/13.657 ms |
| df_c+v12_k50_hops | df_c | v12_k50_hops | 14.84% | 17.50% | 0.00% | 14.84% | 0.00% | 0.00% | 9.865/17.204 ms | 31.032/43.981 ms |