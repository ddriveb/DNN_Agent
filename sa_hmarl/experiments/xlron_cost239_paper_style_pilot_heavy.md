# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.07`
- `holding_min`: `4.0`
- `holding_max`: `6.0`
- `size_min_mb`: `15.0`
- `size_max_mb`: `50.0`
- `seeds`: `3030`
- `episodes`: `5`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 8.50% | 8.50% | 0.00% | 8.50% | 0.00% | 0.00% | 11.275/21.841 ms | 8.229/12.386 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 8.75% | 8.75% | 0.00% | 8.75% | 0.00% | 0.00% | 11.709/21.845 ms | 16.951/29.953 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 12.75% | 13.25% | 0.00% | 12.75% | 0.00% | 0.00% | 10.682/21.159 ms | 8.049/11.922 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | +0.25 pp | +0.00 pp | +0.25 pp | +0.433 ms |
| ppo_c+ksp_ff_k50_hops | +4.25 pp | +0.00 pp | +4.25 pp | -0.593 ms |