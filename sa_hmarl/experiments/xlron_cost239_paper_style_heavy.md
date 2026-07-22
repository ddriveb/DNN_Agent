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
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 13.40% | 13.80% | 0.14% | 13.26% | 0.00% | 0.00% | 11.494/21.352 ms | 27.286/50.220 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 13.39% | 13.84% | 0.00% | 13.39% | 0.00% | 0.00% | 11.723/22.112 ms | 35.081/65.762 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 12.35% | 12.73% | 0.00% | 12.35% | 0.00% | 0.00% | 10.779/20.617 ms | 22.457/38.047 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | -0.01 pp | -0.14 pp | +0.13 pp | +0.229 ms |
| ppo_c+ksp_ff_k50_hops | -1.05 pp | -0.14 pp | -0.91 pp | -0.715 ms |