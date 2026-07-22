# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.08`
- `holding_min`: `4.0`
- `holding_max`: `8.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 8.21% | 8.50% | 0.00% | 8.21% | 0.00% | 0.00% | 11.597/21.051 ms | 27.236/49.830 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 7.41% | 7.71% | 0.00% | 7.41% | 0.00% | 0.00% | 11.918/21.776 ms | 37.315/66.773 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 6.75% | 6.94% | 0.00% | 6.75% | 0.00% | 0.00% | 10.564/19.773 ms | 22.736/37.820 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | -0.80 pp | +0.00 pp | -0.80 pp | +0.322 ms |
| ppo_c+ksp_ff_k50_hops | -1.46 pp | +0.00 pp | -1.46 pp | -1.033 ms |