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
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ppo_r | ppo_c | ppo_r | 0.15% | 0.15% | 0.00% | 0.15% | 0.00% | 0.00% | 12.850/21.317 ms | 8.359/11.903 ms |
| ppo_c+v12 | ppo_c | v12 | 0.20% | 0.20% | 0.00% | 0.20% | 0.00% | 0.00% | 11.060/20.027 ms | 10.085/15.746 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.472/19.558 ms | 8.764/12.046 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ppo_r | -0.05 pp | +0.00 pp | -0.05 pp | +1.790 ms |
| ppo_c+ksp_ff_k50_hops | -0.20 pp | +0.00 pp | -0.20 pp | -0.588 ms |