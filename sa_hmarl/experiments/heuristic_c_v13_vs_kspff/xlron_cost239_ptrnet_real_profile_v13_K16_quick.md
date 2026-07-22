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
- `seeds`: `3030`
- `episodes`: `1`
- `requests_per_episode`: `1000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 13.88% | 15.88% | 0.00% | 13.88% | 0.00% | 0.00% | 10.145/18.091 ms | 8.564/12.073 ms |

## Ranker Profile Breakdown (v1.3)

| Method | build_action_features | legal extract | candidate select | feature batch | normalize | ranker forward | total ranker |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | 0.552 | 0.010 | 1.487 | 0.183 | 0.005 | 0.073 | 2.309 |