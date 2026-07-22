# Offline Comparison: Full-State v1.3 Pilot (COST239)

Metrics computed on the held-out test split. Lower model regret is better.

| Method | Seed | Test groups | Model regret | PPO regret | Regret reduction | Top-1 | NDCG@3 | PPO agree |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_state_uniform_strict | 42 | 580 | 0.0312 | 0.0319 | 2.37% | 9.83% | 0.678 | 10.34% |
| full_state_uniform_strict | 43 | 580 | 0.0280 | 0.0319 | 12.26% | 19.14% | 0.717 | 17.76% |
| full_state_uniform_strict | 44 | 580 | 0.0284 | 0.0319 | 11.03% | 3.10% | 0.718 | 5.00% |
| full_state_stratified_depth_strict | 42 | 580 | 0.0304 | 0.0319 | 4.76% | 12.59% | 0.680 | 11.21% |
| full_state_stratified_depth_strict | 43 | 580 | 0.0284 | 0.0319 | 11.08% | 9.48% | 0.716 | 8.45% |
| full_state_stratified_depth_strict | 44 | 580 | 0.0283 | 0.0319 | 11.49% | 8.28% | 0.714 | 10.00% |
| current_v13_old_checkpoint | 42 | 580 | 0.0302 | 0.0319 | 5.42% | 7.76% | 0.689 | 10.17% |

## E-gate subgroup (offline)

| Method | Seed | E | Groups | Model regret | PPO regret | Top-1 | NDCG@3 | PPO agree |
|---|---|---|---|---:|---:|---:|---:|---:|
| full_state_uniform_strict | 42 | E0 | 120 | 0.0000 | 0.0000 | 0.00% | 1.000 | 0.00% |
| full_state_uniform_strict | 42 | E1 | 460 | 0.0393 | 0.0403 | 12.39% | 0.594 | 13.04% |
| full_state_uniform_strict | 43 | E0 | 120 | 0.0000 | 0.0000 | 85.83% | 1.000 | 85.83% |
| full_state_uniform_strict | 43 | E1 | 460 | 0.0353 | 0.0403 | 1.74% | 0.643 | 0.00% |
| full_state_uniform_strict | 44 | E0 | 120 | 0.0000 | 0.0000 | 0.00% | 1.000 | 0.00% |
| full_state_uniform_strict | 44 | E1 | 460 | 0.0358 | 0.0403 | 3.91% | 0.645 | 6.30% |
| full_state_stratified_depth_strict | 42 | E0 | 120 | 0.0000 | 0.0000 | 49.17% | 1.000 | 49.17% |
| full_state_stratified_depth_strict | 42 | E1 | 460 | 0.0384 | 0.0403 | 3.04% | 0.597 | 1.30% |
| full_state_stratified_depth_strict | 43 | E0 | 120 | 0.0000 | 0.0000 | 40.00% | 1.000 | 40.00% |
| full_state_stratified_depth_strict | 43 | E1 | 460 | 0.0358 | 0.0403 | 1.52% | 0.642 | 0.22% |
| full_state_stratified_depth_strict | 44 | E0 | 120 | 0.0000 | 0.0000 | 0.00% | 1.000 | 0.00% |
| full_state_stratified_depth_strict | 44 | E1 | 460 | 0.0357 | 0.0403 | 10.43% | 0.640 | 12.61% |
| current_v13_old_checkpoint | 42 | E0 | 120 | 0.0000 | 0.0000 | 0.00% | 1.000 | 0.00% |
| current_v13_old_checkpoint | 42 | E1 | 460 | 0.0381 | 0.0403 | 9.78% | 0.608 | 12.83% |