# Offline Comparison: Full-State v1.3 Pilot (COST239)

Metrics computed on the held-out test split. Lower model regret is better.

| Method | Seed | Test groups | Model regret | PPO regret | Regret reduction | Top-1 | NDCG@3 | PPO agree |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_state_uniform | 42 | 580 | 0.0309 | 0.0319 | 3.41% | 1.03% | 0.685 | 0.69% |
| full_state_uniform | 43 | 580 | 0.0293 | 0.0319 | 8.22% | 18.62% | 0.695 | 17.93% |
| full_state_uniform | 44 | 580 | 0.0294 | 0.0319 | 7.93% | 7.93% | 0.692 | 7.24% |
| full_state_stratified_depth | 42 | 580 | 0.0306 | 0.0319 | 4.10% | 0.86% | 0.682 | 0.52% |
| full_state_stratified_depth | 43 | 580 | 0.0306 | 0.0319 | 4.30% | 25.00% | 0.685 | 27.07% |
| full_state_stratified_depth | 44 | 580 | 0.0281 | 0.0319 | 11.97% | 7.41% | 0.689 | 6.21% |
| current_v13_old_checkpoint | 42 | 580 | 0.0302 | 0.0319 | 5.42% | 7.76% | 0.689 | 10.17% |

## E-gate subgroup (offline)

| Method | Seed | E | Groups | Model regret | PPO regret | Top-1 | NDCG@3 | PPO agree |
|---|---|---|---|---:|---:|---:|---:|---:|
| full_state_uniform | 42 | E0 | 120 | 0.0000 | 0.0000 | 2.50% | 1.000 | 2.50% |
| full_state_uniform | 42 | E1 | 460 | 0.0389 | 0.0403 | 0.65% | 0.603 | 0.22% |
| full_state_uniform | 43 | E0 | 120 | 0.0000 | 0.0000 | 84.17% | 1.000 | 84.17% |
| full_state_uniform | 43 | E1 | 460 | 0.0370 | 0.0403 | 1.52% | 0.616 | 0.65% |
| full_state_uniform | 44 | E0 | 120 | 0.0000 | 0.0000 | 35.00% | 1.000 | 35.00% |
| full_state_uniform | 44 | E1 | 460 | 0.0371 | 0.0403 | 0.87% | 0.612 | 0.00% |
| full_state_stratified_depth | 42 | E0 | 120 | 0.0000 | 0.0000 | 1.67% | 1.000 | 1.67% |
| full_state_stratified_depth | 42 | E1 | 460 | 0.0386 | 0.0403 | 0.65% | 0.599 | 0.22% |
| full_state_stratified_depth | 43 | E0 | 120 | 0.0000 | 0.0000 | 90.83% | 1.000 | 90.83% |
| full_state_stratified_depth | 43 | E1 | 460 | 0.0385 | 0.0403 | 7.83% | 0.602 | 10.43% |
| full_state_stratified_depth | 44 | E0 | 120 | 0.0000 | 0.0000 | 30.00% | 1.000 | 30.00% |
| full_state_stratified_depth | 44 | E1 | 460 | 0.0355 | 0.0403 | 1.52% | 0.608 | 0.00% |
| current_v13_old_checkpoint | 42 | E0 | 120 | 0.0000 | 0.0000 | 0.00% | 1.000 | 0.00% |
| current_v13_old_checkpoint | 42 | E1 | 460 | 0.0381 | 0.0403 | 9.78% | 0.608 | 12.83% |