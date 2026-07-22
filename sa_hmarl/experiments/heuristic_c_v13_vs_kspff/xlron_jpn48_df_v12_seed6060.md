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
- `seeds`: `6060`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| df_c+v12_k50_hops | df_c | v12_k50_hops | 9.90% | 11.28% | 0.22% | 9.25% | 0.41% | 0.01% | 17.888/29.492 ms | 178.722/500.450 ms |