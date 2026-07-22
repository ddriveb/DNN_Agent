# Hard-Negative Implementation Audit

## Location
`sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py`, training loop around line 582:

```python
if args.lambda_hard > 0.0:
    hard_weights = _hard_negative_weights(scores.detach(), r, eff_m, args.alpha_hard)
    loss_hard_rank = _ranking_loss(scores, r, eff_m, args.tau_label, args.tau_model)
    loss = loss + args.lambda_hard * (hard_weights * loss_hard_rank).mean()
```

## Issue
`_ranking_loss` returns a **scalar** that is the mean KL over the batch:

```python
return (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1).mean()
```

Therefore `hard_weights` (shape `[B]`) is multiplied by a scalar and then `.mean()` is taken. The result is equivalent to:

```
loss_hard_rank * mean(hard_weights)
```

This globally scales the already-averaged ranking loss by the average hard-negative weight. It does **not** reweight the per-group ranking loss as intended.

## Impact on Strict v1.3
The Strict v1.3 protocol requires `lambda_hard = 0.0`. Consequently this bug has **no effect** on the Strict loss-matched experiments reported in this directory. The Strict results remain valid.

## Correct implementation (for reference only)
To actually upweight hard groups, `_ranking_loss` would need to return per-group losses:

```python
per_group = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)  # [B]
return per_group
```

Then the training loop would compute:

```python
per_group_rank_loss = _ranking_loss(scores, r, eff_m, args.tau_label, args.tau_model)
loss = loss + args.lambda_hard * (hard_weights * per_group_rank_loss).mean()
```

If such a fix is applied in the future, a unit test must verify:
- Two identical batches except for one hard group receive different gradients when `lambda_hard > 0`.
- `lambda_hard = 0` produces numerically identical results to the current code.

## Decision
No code change is made in this Strict rerun. `lambda_hard` is explicitly set to `0.0` in the training launcher, so the bug is irrelevant to the reported results.
