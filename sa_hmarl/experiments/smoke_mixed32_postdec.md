# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `snap24_gnutella_reach`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.15`
- `holding_min`: `4.0`
- `holding_max`: `10.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030`
- `episodes`: `1`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 6.764/11.416 ms | 12.822/14.418 ms |
| ppo_c+v12 | ppo_c | v12 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 7.463/12.126 ms | 14.425/17.963 ms |
| ppo_c+ppo_r_proposer_postdec | ppo_c | ppo_r_proposer_postdec | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 8.942/16.995 ms | 23.202/28.722 ms |
| ppo_c+mixed32_postdec | ppo_c | mixed32_postdec | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 8.541/18.390 ms | 21.007/28.287 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | +0.00 pp | +0.00 pp | +0.00 pp | -0.699 ms |
| ppo_c+ppo_r_proposer_postdec | +0.00 pp | +0.00 pp | +0.00 pp | +1.479 ms |
| ppo_c+mixed32_postdec | +0.00 pp | +0.00 pp | +0.00 pp | +1.079 ms |