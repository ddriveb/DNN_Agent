# Implementation Re-Audit

## Changes made in this rerun

### 1. `run_v13_strict_fixed_pilot.py`
- Added explicit Strict loss parameters for all Strict training runs:
  - `--reg_weight 1.0`
  - `--lambda_pair 0.0`
  - `--lambda_hard 0.0`
- Renamed training methods to `full_state_uniform_strict` and `full_state_stratified_depth_strict` to avoid confusion with previous non-strict checkpoints.
- Updated closed-loop ranker specs to include the new Strict uniform and stratified checkpoints.

### 2. Training script (`train_r_counterfactual_ranking.py`)
- No changes were required; the script already supports `--reg_weight`, `--lambda_pair`, and `--lambda_hard`.
- The default values remain non-strict (`reg_weight=0.1`, `lambda_pair=0.5`, `lambda_hard=1.0`), which is why the launcher must explicitly pass Strict values.

### 3. Online evaluator (`eval_r_counterfactual_ranking_cost239_kpath50_hops_fair.py`)
- No changes. `--ranker_gate {all,deep_path_only}` and E-gate subgroup counters were already implemented.

## What was verified
- `pilot_dataset_cost239/metadata.json` confirms `group_filter=all`, 1753 groups, 229 E=0, 1524 E=1.
- Train/val/test shards are disjoint by seed (`train_s9101_e0`, `val_s9201_e0`, `test_s9301_e0`).
- All models use the same 25-d feature definition, candidate pool (`ppo_r_topk_only`), and action IDs.
- Closed-loop uses identical PPO-C/PPO-R checkpoints, warmup=500, evaluated requests=2000, and the same traffic seeds for every method.
- Checkpoint selection uses validation regret only; test and closed-loop results were not used for model selection.

## Hard-negative bug
Documented separately in `HARD_NEGATIVE_AUDIT.md`. It does not affect Strict results because `lambda_hard=0.0`.
