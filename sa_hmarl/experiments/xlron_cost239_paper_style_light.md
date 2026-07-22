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
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 4.19% | 4.34% | 0.00% | 4.19% | 0.00% | 0.00% | 11.449/20.572 ms | 26.445/45.427 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 3.72% | 3.89% | 0.00% | 3.72% | 0.00% | 0.00% | 11.750/21.477 ms | 40.056/72.303 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 3.12% | 3.23% | 0.00% | 3.12% | 0.00% | 0.00% | 10.542/19.565 ms | 22.585/37.705 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | -0.46 pp | +0.00 pp | -0.46 pp | +0.301 ms |
| ppo_c+ksp_ff_k50_hops | -1.06 pp | +0.00 pp | -1.06 pp | -0.907 ms |