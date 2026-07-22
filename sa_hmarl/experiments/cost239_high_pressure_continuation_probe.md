# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.15`
- `holding_min`: `4.0`
- `holding_max`: `10.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050`
- `episodes`: `10`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 0.12% | 0.12% | 0.00% | 0.12% | 0.00% | 0.00% | 11.185/20.145 ms | 15.549/23.270 ms |
| ppo_c+ppo_r | ppo_c | ppo_r | 0.42% | 0.42% | 0.00% | 0.42% | 0.00% | 0.00% | 13.382/22.591 ms | 10.627/13.318 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.667/20.214 ms | 12.269/15.489 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ppo_r | +0.29 pp | +0.00 pp | +0.29 pp | +2.197 ms |
| ppo_c+ksp_ff_k50_hops | -0.12 pp | +0.00 pp | -0.12 pp | -0.518 ms |