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
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 0.92% | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms | 13.431/22.852 ms |
| ppo_c+deep_rmsa | ppo_c | deep_rmsa | 1.29% | 1.29% | 1.21% | 0.07% | 10.183/18.040 ms | 9.974/13.589 ms |
| ppo_c+ppo_r | ppo_c | ppo_r | 3.66% | 3.66% | 3.25% | 0.41% | 8.088/14.672 ms | 10.090/13.916 ms |
| ppo_c+ksp_bf | ppo_c | ksp_bf | 7.68% | 7.68% | 6.26% | 1.41% | 8.546/15.536 ms | 9.608/13.157 ms |
| ppo_c+ksp_ff | ppo_c | ksp_ff | 7.38% | 7.38% | 5.94% | 1.44% | 8.201/16.280 ms | 9.452/12.922 ms |
| greedy_c+v12 | greedy_c | v12 | 1.74% | 1.74% | 1.74% | 0.00% | 6.240/10.197 ms | 11.722/18.135 ms |
| df_c+v12 | df_c | v12 | 1.18% | 1.18% | 0.61% | 0.56% | 6.076/11.389 ms | 10.954/15.452 ms |
| rf_c+v12 | rf_c | v12 | 2.00% | 2.00% | 2.00% | 0.00% | 7.536/13.190 ms | 12.104/19.345 ms |
| wo_c+v12 | wo_c | v12 | 2.20% | 2.20% | 2.20% | 0.00% | 8.195/14.123 ms | 12.383/19.611 ms |
| iwd_c+v12 | iwd_c | v12 | 1.18% | 1.18% | 0.60% | 0.57% | 7.796/15.394 ms | 13.467/20.675 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+deep_rmsa | +0.36 pp | +0.38 pp | -0.01 pp | +1.806 ms |
| ppo_c+ppo_r | +2.74 pp | +2.41 pp | +0.33 pp | -0.289 ms |
| ppo_c+ksp_bf | +6.75 pp | +5.42 pp | +1.32 pp | +0.169 ms |
| ppo_c+ksp_ff | +6.45 pp | +5.10 pp | +1.35 pp | -0.175 ms |
| greedy_c+v12 | +0.81 pp | +0.90 pp | -0.09 pp | -2.136 ms |
| df_c+v12 | +0.25 pp | -0.23 pp | +0.47 pp | -2.300 ms |
| rf_c+v12 | +1.08 pp | +1.16 pp | -0.09 pp | -0.841 ms |
| wo_c+v12 | +1.27 pp | +1.36 pp | -0.09 pp | -0.181 ms |
| iwd_c+v12 | +0.25 pp | -0.24 pp | +0.49 pp | -0.581 ms |