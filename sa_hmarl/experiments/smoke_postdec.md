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
| ppo_c+v12 | ppo_c | v12 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 7.463/12.126 ms | 15.059/18.228 ms |
| ppo_c+ppo_r_proposer_ksp_ff | ppo_c | ppo_r_proposer_ksp_ff | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 8.375/14.921 ms | 13.861/16.670 ms |
| ppo_c+ppo_r_proposer_postdec | ppo_c | ppo_r_proposer_postdec | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 8.380/16.393 ms | 17.809/20.115 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ppo_r_proposer_ksp_ff | +0.00 pp | +0.00 pp | +0.00 pp | +0.912 ms |
| ppo_c+ppo_r_proposer_postdec | +0.00 pp | +0.00 pp | +0.00 pp | +0.918 ms |