# Strict Loss-Matched Closed-Loop Comparison (COST239)

Traffic seeds: 4001–4005. Warmup=500, evaluated requests=2000 per seed. All ranker methods use the same PPO-R Top-30 candidate pool and the same PPO-C/PPO-R checkpoints.

## Overall blocking rate

| Gate | Method | Blocking % | Overload % | NSB % | Delay ms | FS | PPO agree |
|---|---|---:|---:|---:|---:|---:|---:|
| all | PPO-R Top-1 | 5.63% ± 2.94% | 5.51% | 0.12% | 11.18 | 3.45 | 100.00% |
| all | KSP-FF plain | 9.57% ± 1.55% | 9.27% | 0.30% | 10.50 | 3.17 | 26.77% |
| all | KSP-FF highest-mod | 6.44% ± 1.62% | 6.43% | 0.01% | 10.20 | 2.36 | 9.53% |
| all | Old v1.3 | 4.84% ± 1.40% | 4.78% | 0.06% | 13.05 | 3.17 | 7.26% |
| all | Full-state uniform strict | 4.82% ± 2.56% | 4.74% | 0.08% | 14.27 | 3.23 | 7.49% |
| all | Full-state stratified strict | 5.05% ± 2.32% | 4.93% | 0.11% | 14.50 | 3.21 | 7.11% |
| deep_path_only | PPO-R Top-1 | 5.63% ± 2.94% | 5.51% | 0.12% | 11.18 | 3.45 | 100.00% |
| deep_path_only | KSP-FF plain | 9.57% ± 1.55% | 9.27% | 0.30% | 10.50 | 3.17 | 26.77% |
| deep_path_only | KSP-FF highest-mod | 6.44% ± 1.62% | 6.43% | 0.01% | 10.20 | 2.36 | 9.53% |
| deep_path_only | Old v1.3 | 4.44% ± 1.32% | 4.36% | 0.08% | 13.04 | 3.33 | 30.83% |
| deep_path_only | Full-state uniform strict | 5.32% ± 2.68% | 5.23% | 0.09% | 14.02 | 3.38 | 36.21% |
| deep_path_only | Full-state stratified strict | 5.21% ± 2.45% | 5.14% | 0.06% | 14.32 | 3.37 | 35.77% |

## Per-seed blocking rates

| Gate | Method | s1 | s2 | s3 | s4 | s5 | Mean | Std |
|---|---||---:|---:|---:|---:|---:|---:|---:|
| all | PPO-R Top-1 | 5.70% | 11.30% | 3.85% | 3.80% | 3.50% | 5.63% | 2.94% |
| all | KSP-FF plain | 12.50% | 8.15% | 9.55% | 8.40% | 9.25% | 9.57% | 1.55% |
| all | KSP-FF highest-mod | 5.40% | 7.90% | 3.85% | 6.90% | 8.15% | 6.44% | 1.62% |
| all | Old v1.3 | 5.05% | 7.45% | 3.50% | 3.90% | 4.30% | 4.84% | 1.40% |
| all | Full-state uniform strict | 4.00% | 9.90% | 3.15% | 3.75% | 3.30% | 4.82% | 2.56% |
| all | Full-state stratified strict | 4.55% | 9.60% | 3.75% | 3.20% | 4.15% | 5.05% | 2.32% |
| deep_path_only | PPO-R Top-1 | 5.70% | 11.30% | 3.85% | 3.80% | 3.50% | 5.63% | 2.94% |
| deep_path_only | KSP-FF plain | 12.50% | 8.15% | 9.55% | 8.40% | 9.25% | 9.57% | 1.55% |
| deep_path_only | KSP-FF highest-mod | 5.40% | 7.90% | 3.85% | 6.90% | 8.15% | 6.44% | 1.62% |
| deep_path_only | Old v1.3 | 4.95% | 6.80% | 3.80% | 3.30% | 3.35% | 4.44% | 1.32% |
| deep_path_only | Full-state uniform strict | 4.75% | 10.60% | 4.15% | 3.60% | 3.50% | 5.32% | 2.68% |
| deep_path_only | Full-state stratified strict | 4.70% | 10.05% | 4.00% | 3.60% | 3.70% | 5.21% | 2.45% |

## E-gate subgroup counts

| Gate | Method | E=0 states | E=1 states | Ranker changes E=0 | Ranker changes E=1 |
|---|---|---:|---:|---:|---:|
| all | PPO-R Top-1 | 0.0 | 0.0 | 0 | 0 |
| all | KSP-FF plain | 0.0 | 0.0 | 0 | 0 |
| all | KSP-FF highest-mod | 0.0 | 0.0 | 0 | 0 |
| all | Old v1.3 | 513.6 | 1389.6 | 0 | 0 |
| all | Full-state uniform strict | 635.2 | 1268.4 | 0 | 0 |
| all | Full-state stratified strict | 660.0 | 1239.0 | 0 | 0 |
| deep_path_only | PPO-R Top-1 | 0.0 | 0.0 | 0 | 0 |
| deep_path_only | KSP-FF plain | 0.0 | 0.0 | 0 | 0 |
| deep_path_only | KSP-FF highest-mod | 0.0 | 0.0 | 0 | 0 |
| deep_path_only | Old v1.3 | 485.4 | 1425.8 | 0 | 0 |
| deep_path_only | Full-state uniform strict | 593.6 | 1300.0 | 0 | 0 |
| deep_path_only | Full-state stratified strict | 601.8 | 1294.0 | 0 | 0 |

## Paired comparisons

| Comparison | Mean Δ blocking | 95% CI | Wilcoxon p | Interpretation |
|---|---:|---:|---:|---|
| old_gated_vs_old_all | -0.4000% | [-0.7600%, -0.0300%] | 0.188 | Gated old improves over all-gate; direction consistent with E=0 OOD but n=5 underpowered. |
| strict_uniform_all_vs_ppo_r | -0.8100% | [-1.3800%, -0.2400%] | 0.062 | Strict full-state improves over PPO-R; nearly significant with 5 seeds. |
| strict_uniform_all_vs_old_gated | 0.3800% | [-0.6500%, 1.7600%] | 1.000 | Strict full-state is NOT better than old gated (mean +0.38 pp). |
| strict_uniform_gated_vs_strict_uniform_all | 0.5000% | [0.1000%, 0.8400%] | 0.125 | Gating strict full-state hurts (+0.50 pp), suggesting it learned useful E=0 behavior. |
| strict_stratified_vs_strict_uniform | 0.2300% | [-0.2700%, 0.6900%] | 0.375 | Stratified sampling does not improve over uniform. |
| strict_uniform_vs_ksp_ff_plain | -4.7500% | [-7.2400%, -1.4200%] | 0.125 | Strict full-state much better than KSP-FF plain. |
| strict_uniform_vs_ksp_ff_highest | -1.6200% | [-3.6800%, 0.4300%] | 0.312 | Strict full-state directionally better than KSP-FF highest but CI crosses 0. |