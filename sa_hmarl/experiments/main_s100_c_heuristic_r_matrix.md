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
| df_c+v12 | df_c | v12 | 1.18% | 1.18% | 0.61% | 0.56% | 6.076/11.389 ms | 11.104/15.864 ms |
| rf_c+v12 | rf_c | v12 | 2.00% | 2.00% | 2.00% | 0.00% | 7.536/13.190 ms | 12.080/19.371 ms |
| wo_c+v12 | wo_c | v12 | 2.20% | 2.20% | 2.20% | 0.00% | 8.195/14.123 ms | 11.990/18.936 ms |
| iwd_c+v12 | iwd_c | v12 | 1.18% | 1.18% | 0.60% | 0.57% | 7.796/15.394 ms | 13.157/19.819 ms |
| df_c+deep_rmsa | df_c | deep_rmsa | 1.44% | 1.44% | 0.78% | 0.66% | 8.728/14.371 ms | 9.247/12.576 ms |
| rf_c+deep_rmsa | rf_c | deep_rmsa | 3.55% | 3.55% | 3.55% | 0.00% | 9.452/15.850 ms | 9.476/12.499 ms |
| wo_c+deep_rmsa | wo_c | deep_rmsa | 3.79% | 3.79% | 3.79% | 0.00% | 9.992/16.756 ms | 9.584/12.612 ms |
| iwd_c+deep_rmsa | iwd_c | deep_rmsa | 1.62% | 1.62% | 0.95% | 0.68% | 10.354/18.613 ms | 10.959/14.205 ms |
| df_c+ksp_ff | df_c | ksp_ff | 9.08% | 9.08% | 8.99% | 0.09% | 6.088/11.824 ms | 8.785/11.906 ms |
| rf_c+ksp_ff | rf_c | ksp_ff | 9.75% | 9.75% | 9.75% | 0.00% | 7.095/12.226 ms | 9.258/12.587 ms |
| wo_c+ksp_ff | wo_c | ksp_ff | 9.92% | 9.92% | 9.92% | 0.00% | 7.745/13.085 ms | 9.197/12.479 ms |
| iwd_c+ksp_ff | iwd_c | ksp_ff | 8.60% | 8.60% | 5.66% | 2.94% | 7.638/15.753 ms | 10.288/13.677 ms |