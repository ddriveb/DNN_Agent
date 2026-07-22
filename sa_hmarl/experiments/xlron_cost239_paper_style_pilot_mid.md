# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.06`
- `holding_min`: `8.0`
- `holding_max`: `12.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030`
- `episodes`: `5`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 25.25% | 26.00% | 0.00% | 25.25% | 0.00% | 0.00% | 11.470/18.791 ms | 7.653/12.044 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 24.25% | 24.75% | 0.00% | 24.25% | 0.00% | 0.00% | 11.623/21.639 ms | 14.435/23.224 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 23.25% | 23.50% | 0.00% | 23.25% | 0.00% | 0.00% | 10.170/18.264 ms | 8.076/11.797 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | -1.00 pp | +0.00 pp | -1.00 pp | +0.152 ms |
| ppo_c+ksp_ff_k50_hops | -2.00 pp | +0.00 pp | -2.00 pp | -1.300 ms |