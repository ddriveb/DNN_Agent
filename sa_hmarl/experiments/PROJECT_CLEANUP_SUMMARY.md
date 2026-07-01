# Project Cleanup Summary

## Kept Mainline Artifacts

- PPO-side checkpoints: selected Agent-C checkpoints, `agent_r_mixed.pt`, and DeepRMSA baselines for S24/S80/S100.
- Final R-ranker checkpoints: `r_counterfactual_ranking_v1_2_mixed_low`, `r_counterfactual_ranking_s100_v1_2_mixed_low`, plus two v1.2 variants for reference.
- Reproducibility datasets: v1.2 mixed-low, S100 v1.2 mixed-low, and selected v1.2 variants.
- High-signal experiment reports: final v1.2 recommendation, formal validation, S100/topology verification, cumulative curves, and key negative diagnostics.

## Removed

- Early DQN/checkpoint sweeps.
- Smoke checkpoints and smoke datasets.
- v1.1/v1.3 failed sweep checkpoints and datasets, except final reports.
- Old post-decision/TD/smoke datasets.
- `sa_hmarl/archive` quick-smoke artifacts.
- `sa_hmarl/.pytest_cache`.

## Size After Cleanup

- `sa_hmarl/checkpoints`: about 12 MB.
- `sa_hmarl/experiments`: about 14 MB.
- `sa_hmarl/datasets`: about 3.9 MB.

## Notes

- Source code was intentionally not removed. In particular, `c_agent.py` is DQN-based but still contains feature-construction utilities that are reused by PPO/evaluation code paths.
- Root-level `backups/` and `experiments/` were not removed because they are whole project directories and may contain code or reports outside the SA-HMARL artifact cleanup scope.
