# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.1`
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
| ppo_c+v12 | ppo_c | v12 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 11.133/18.904 ms | 9.133/12.728 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 0.75% | 0.75% | 0.00% | 0.75% | 0.00% | 0.00% | 11.370/21.529 ms | 19.528/37.767 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.117/19.100 ms | 8.634/11.875 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | +0.75 pp | +0.00 pp | +0.75 pp | +0.237 ms |
| ppo_c+ksp_ff_k50_hops | +0.00 pp | +0.00 pp | +0.00 pp | -1.016 ms |