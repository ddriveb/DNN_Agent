# Strict v1.3 Training Configuration

This document records the exact training configuration used for the Strict loss-matched full-state rerun.

## Common protocol
- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- `group_filter`: `all`
- `K_C`: 5, `K_path`: 50, `K_prop`: 30, `H`: 5, `gamma`: 1.0
- Candidate mode: `ppo_r_topk_only`
- Feature dim: 25 (pre-decision state-action features)
- Model: MLP with hidden_dims=(128, 64), dropout=0.0
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-5
- Max grad norm: 1.0
- Epochs: 80 (with early stopping, patience=20)
- Selection metric: validation regret (lower is better)
- Seeds: 42, 43, 44

## Strict loss parameters
All Strict models were trained with exactly:

```bash
--reg_weight 1.0
--lambda_pair 0.0
--lambda_hard 0.0
```

This matches the original Strict v1.3 protocol: listwise KL over teacher returns plus regression on normalized returns, with no pairwise margin loss and no hard-negative upweighting.

## Sampling modes
- `full_state_uniform_strict`: `--sampling_mode uniform`
- `full_state_stratified_depth_strict`: `--sampling_mode stratified_depth`

## Non-strict baseline (for reference)
The previous `full_state_uniform` and `full_state_stratified_depth` runs used the training-script defaults:
- `reg_weight=0.1`
- `lambda_pair=0.5`
- `lambda_hard=1.0`

Those results are reported in the parent directory and are superseded by this Strict rerun for the purpose of judging full-state training.
