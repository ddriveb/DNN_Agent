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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 7.60% | 9.48% | 0.01% | 7.58% | 0.01% | 0.00% | 10.557/18.471 ms | 10.766/15.524 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 7.63% | 9.55% | 0.05% | 7.58% | 0.01% | 0.00% | 11.028/19.469 ms | 22.414/38.170 ms |