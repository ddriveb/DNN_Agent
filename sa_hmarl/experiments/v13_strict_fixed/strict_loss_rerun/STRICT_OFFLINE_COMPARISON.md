# Strict Loss-Matched Offline Comparison (COST239)

Training protocol: reg_weight=1.0, lambda_pair=0.0, lambda_hard=0.0. Models selected by validation regret. Test metrics are out-of-sample.

## Overall test metrics (mean ± std over 3 seeds)

| Method | Top-1 | Tie Top-1 | Spearman | Kendall | Pairwise | Model regret | PPO regret | Regret red | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-state uniform strict | 10.69% ± 0.066 | 50.75% ± 0.007 | -0.008 ± 0.055 | -0.006 ± 0.044 | 0.513 ± 0.056 | 0.0292 ± 0.0014 | 0.0319 ± 0.0000 | 8.55% ± 0.044 | 11.03% ± 0.052 |
| Full-state stratified strict | 10.11% ± 0.018 | 51.15% ± 0.006 | -0.013 ± 0.053 | -0.011 ± 0.042 | 0.505 ± 0.057 | 0.0290 ± 0.0010 | 0.0319 ± 0.0000 | 9.11% ± 0.031 | 9.89% ± 0.011 |
| Old v1.3 checkpoint (OOD on full test) | 7.76% ± 0.000 | 50.52% ± 0.000 | -0.029 ± 0.000 | -0.023 ± 0.000 | 0.488 ± 0.000 | 0.0302 ± 0.0000 | 0.0319 ± 0.0000 | 5.42% ± 0.000 | 10.17% ± 0.000 |

## Per-seed raw values

| Method | Seed | Top-1 | Regret red | Model regret | PPO regret |
|---|---|---:|---:|---:|---:|
| Full-state uniform strict | 42 | 9.83% | 2.37% | 0.0312 | 0.0319 |
| Full-state uniform strict | 43 | 19.14% | 12.26% | 0.0280 | 0.0319 |
| Full-state uniform strict | 44 | 3.10% | 11.03% | 0.0284 | 0.0319 |
| Full-state stratified strict | 42 | 12.59% | 4.76% | 0.0304 | 0.0319 |
| Full-state stratified strict | 43 | 9.48% | 11.08% | 0.0284 | 0.0319 |
| Full-state stratified strict | 44 | 8.28% | 11.49% | 0.0283 | 0.0319 |
| Old v1.3 checkpoint | 42 | 7.76% | 5.42% | 0.0302 | 0.0319 |

## E-gate subgroups (mean ± std)

| Method | Gate | Groups | Top-1 | Regret red | Model regret | PPO regret |
|---|---|---:|---:|---:|---:|---:|
| Full-state uniform strict | E=0 | 120.0 ± 0.0 | 28.61% ± 40.46% | 0.00% ± 0.00% | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Full-state uniform strict | E=1 | 460.0 ± 0.0 | 6.01% ± 4.60% | 8.55% ± 4.40% | 0.0368 ± 0.0018 | 0.0403 ± 0.0000 |
| Full-state stratified strict | E=0 | 120.0 ± 0.0 | 29.72% ± 21.35% | 0.00% ± 0.00% | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Full-state stratified strict | E=1 | 460.0 ± 0.0 | 5.00% ± 3.89% | 9.11% ± 3.08% | 0.0366 ± 0.0012 | 0.0403 ± 0.0000 |
| Old v1.3 checkpoint | E=0 | 120.0 ± 0.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Old v1.3 checkpoint | E=1 | 460.0 ± 0.0 | 9.78% ± 0.00% | 5.42% ± 0.00% | 0.0381 ± 0.0000 | 0.0403 ± 0.0000 |

## Depth-stratum metrics (seed 42 only)

| Method | Stratum | Groups | Top-1 | Regret red | Model regret |
|---|---|---:|---:|---:|---:|
| Full-state uniform strict | 0 | 120 | 0.00% | 0.00% | 0.0000 |
| Full-state uniform strict | 1 | 9 | 0.00% | -15.14% | 0.0409 |
| Full-state uniform strict | 2 | 244 | 16.80% | -2.51% | 0.0312 |
| Full-state uniform strict | 3 | 207 | 7.73% | 6.26% | 0.0488 |
| Full-state stratified strict | 0 | 120 | 49.17% | 0.00% | 0.0000 |
| Full-state stratified strict | 1 | 9 | 0.00% | -10.58% | 0.0393 |
| Full-state stratified strict | 2 | 244 | 3.28% | -1.43% | 0.0309 |
| Full-state stratified strict | 3 | 207 | 2.90% | 9.49% | 0.0471 |
| Old v1.3 checkpoint | 0 | 120 | 0.00% | 0.00% | 0.0000 |
| Old v1.3 checkpoint | 1 | 9 | 0.00% | -15.14% | 0.0409 |
| Old v1.3 checkpoint | 2 | 244 | 11.07% | 1.03% | 0.0301 |
| Old v1.3 checkpoint | 3 | 207 | 8.70% | 9.07% | 0.0473 |