# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.08`
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
| ppo_c+v12 | ppo_c | v12 | 3.75% | 3.75% | 0.00% | 3.75% | 0.00% | 0.00% | 11.150/21.758 ms | 8.697/12.759 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 6.00% | 6.00% | 0.00% | 6.00% | 0.00% | 0.00% | 11.677/22.106 ms | 17.262/30.503 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 6.25% | 6.50% | 0.00% | 6.25% | 0.00% | 0.00% | 10.442/20.191 ms | 9.158/13.696 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | +2.25 pp | +0.00 pp | +2.25 pp | +0.527 ms |
| ppo_c+ksp_ff_k50_hops | +2.50 pp | +0.00 pp | +2.50 pp | -0.708 ms |