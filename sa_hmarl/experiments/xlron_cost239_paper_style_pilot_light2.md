# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.09`
- `holding_min`: `4.0`
- `holding_max`: `8.0`
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
| ppo_c+v12 | ppo_c | v12 | 1.25% | 1.25% | 0.00% | 1.25% | 0.00% | 0.00% | 11.172/19.615 ms | 8.803/12.405 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.25% | 0.50% | 0.00% | 0.25% | 0.00% | 0.00% | 11.602/21.226 ms | 18.220/34.428 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 1.25% | 1.25% | 0.00% | 1.25% | 0.00% | 0.00% | 10.098/19.408 ms | 8.534/11.742 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | -1.00 pp | +0.00 pp | -1.00 pp | +0.429 ms |
| ppo_c+ksp_ff_k50_hops | +0.00 pp | +0.00 pp | +0.00 pp | -1.074 ms |