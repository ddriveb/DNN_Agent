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
- `seeds`: `3030,4040,5050`
- `episodes`: `10`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 13.92% | 14.29% | 0.00% | 13.92% | 0.00% | 0.00% | 11.393/21.266 ms | 10.489/15.611 ms |
| ppo_c+ppo_r | ppo_c | ppo_r | 15.17% | 15.71% | 0.00% | 15.17% | 0.00% | 0.00% | 14.015/23.909 ms | 9.182/13.507 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 13.75% | 14.17% | 0.00% | 13.75% | 0.00% | 0.00% | 10.795/20.789 ms | 10.215/14.471 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ppo_r | +1.25 pp | +0.00 pp | +1.25 pp | +2.623 ms |
| ppo_c+ksp_ff_k50_hops | -0.17 pp | +0.00 pp | -0.17 pp | -0.598 ms |