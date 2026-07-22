# SA-HMARL v1.3 Medium Closed-Loop Fair Comparison

- Topology: COST239, slots=320, servers=4
- C-side: K_C=5, sort=hops/start_asc
- R-side: K_path=50, sort=hops/start_asc
- K_prop=30, candidate_mode=ppo_r_topk_only
- Seeds: [3030, 4040, 5050, 6060, 7070]
- Requests per seed: 6000 (warmup=1000, evaluated=5000)
- Strict v1.3 ranker: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt`

## Summary

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Delay P95 | Avg FS | Decision ms | Decision P95 | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 6.82% ± 1.88% | 6.61% | 0.21% | 10.87 | 13.68 | 3.43 | 11.33 | 12.69 | 100.00% | 0.00% |
| ksp_ff_plain | 7.94% ± 1.47% | 7.86% | 0.07% | 10.11 | 12.82 | 3.15 | 11.06 | 12.18 | 32.97% | 0.00% |
| ksp_ff_highest | 7.04% ± 0.94% | 7.03% | 0.00% | 9.79 | 12.43 | 2.35 | 11.23 | 12.64 | 11.38% | 0.00% |
| ranker_old_v13 | 6.01% ± 0.75% | 5.98% | 0.02% | 11.07 | 13.75 | 2.95 | 21.41 | 24.09 | 9.69% | 93.99% |
| ranker_new_v13 | 6.37% ± 1.47% | 6.31% | 0.05% | 12.08 | 16.34 | 3.06 | 18.62 | 20.34 | 9.83% | 93.63% |

## Per-seed blocking

| Seed | ppo_r_top1 | ksp_ff_plain | ksp_ff_highest | ranker_old_v13 | ranker_new_v13 |
|---|---:|---:|---:|---:|---:|
| 3030 | 7.62% | 7.58% | 8.12% | 6.48% | 7.00% |
| 4040 | 4.54% | 10.56% | 6.20% | 4.80% | 4.68% |
| 5050 | 5.12% | 6.02% | 8.06% | 5.46% | 5.00% |
| 6060 | 9.82% | 7.90% | 5.80% | 6.60% | 8.76% |
| 7070 | 7.00% | 7.62% | 7.00% | 6.70% | 6.40% |

## strict v1.3 vs PPO-R Top-1

- Mean blocking difference (new - ppo): -0.0045
- Wins / ties / losses: 4 / 0 / 1
- Paired bootstrap 95% CI for (PPO - new): [0.0010, 0.0079]
- Wilcoxon signed-rank p-value: 0.1875

## strict v1.3 vs KSP-FF plain

- Mean blocking difference (new - plain): -0.0157
- Paired bootstrap 95% CI for (plain - new): [-0.0016, 0.0385]
- Wilcoxon signed-rank p-value: 0.1875

## strict v1.3 vs KSP-FF highest-mod

- Mean blocking difference (new - highest): -0.0067
- Paired bootstrap 95% CI for (highest - new): [-0.0135, 0.0214]
- Wilcoxon signed-rank p-value: 0.4375