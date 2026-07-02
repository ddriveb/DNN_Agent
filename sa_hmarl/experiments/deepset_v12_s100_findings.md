# DeepSet v1.2 S100 Findings

This diagnostic tests whether the final v1.2 candidate-level MLP can be
upgraded to a permutation-invariant DeepSet listwise evaluator.

## Setup

- Topology: `snap24_gnutella_reach`
- Slots: `100`
- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_mixed_low`
- DeepSet checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_deepset/ranking_model.pt`
- Closed-loop output: `sa_hmarl/experiments/main_s100_deepset_v12.md`
- Model changes after first diagnostic:
  - MLP-compatible local residual head.
  - Context branch initialized with zero scale.
  - Best checkpoint selected by validation KL rather than top-1.
  - MLP loader/training path remains backward compatible.

## Offline Ranking Metrics

| Model | Selection | Best val top-1 | Test top-1 | Test top-3 | Test Spearman | Test KL |
|---|---:|---:|---:|---:|---:|
| DeepSet v1.2 | val KL | 29.38% | 26.34% | 69.39% | 0.792 | 0.0297 |

## Closed-Loop S100 Result

| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|---:|
| MLP v1.2 original | 0.92% | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms | 13.431/22.852 ms |
| DeepSet v1.2 | 0.95% | 0.95% | 0.85% | 0.10% | 8.145/16.011 ms | 12.070/20.142 ms |
| DeepSet v1.2 legalctx48 | 0.95% | 0.95% | 0.86% | 0.09% | 8.155/15.954 ms | 12.647/21.103 ms |
| DeepSet legalctx48, context disabled | 0.96% | 0.96% | 0.88% | 0.09% | 8.120/15.910 ms | 16.766/27.740 ms |
| SetTransformer v1.2 legalctx48 | 0.97% | 0.97% | 0.87% | 0.10% | 7.982/15.390 ms | 8.417/14.798 ms |
| SetTransformer structured mid | 1.05% | 1.05% | 0.94% | 0.11% | 8.036/15.456 ms | 8.792/15.971 ms |

## Larger Legal-Context Dataset Diagnostic

The first DeepSet run exposed a train/inference mismatch: training groups used
the historical `candidate_mode=v1` subset, while online inference scored all
legal R actions.  To reduce this mismatch, a DeepSet-specific dataset was
generated with:

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_legalctx48`
- Candidate mode: `larger_legal_context`
- Max candidates per group: `48`
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_deepset_legalctx48/ranking_model.pt`
- Closed-loop output: `sa_hmarl/experiments/main_s100_deepset_legalctx48_v12.md`

All-legal generation was attempted first, but it was too slow for the full S100
dataset because each candidate requires an H-step counterfactual rollout.  The
`larger_legal_context` mode keeps the historical v1 candidate set and fills the
remaining context with high-logit legal actions up to `max_candidates=48`.

Dataset candidate coverage:

| Split | Groups | Avg candidate mask | Max mask | Avg legal actions | Max legal actions | Mask equals legal |
|---|---:|---:|---:|---:|---:|---:|
| train | 2307 | 17.20 | 48 | 18.29 | 92 | 92.28% |
| val | 776 | 23.40 | 48 | 25.68 | 93 | 87.89% |
| test | 748 | 13.70 | 48 | 14.24 | 89 | 95.59% |

Offline DeepSet legalctx48 ranking metrics:

| Selection | Best checkpoint val top-1 | Test top-1 | Test top-3 | Test Spearman | Test KL | Test model regret | Test PPO regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| val KL | 11.73% | 20.32% | 56.68% | 0.794 | 0.0277 | 0.0487 | 0.1360 |

Offline SetTransformer legalctx48 ranking metrics:

| Selection | Best checkpoint val top-1 | Test top-1 | Test top-3 | Test Spearman | Test KL | Test model regret | Test PPO regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| val KL | 16.49% | 24.47% | 61.10% | 0.786 | 0.0267 | 0.0484 | 0.1360 |

## Takeaway

DeepSet v1.2 is essentially tied with the original MLP v1.2 in S100 closed-loop
blocking (`+0.03 pp`) while giving the method a cleaner architectural story:

- AC-MPC: amortized counterfactual MPC for R-side RMSA.
- LPD: listwise planner distillation from H-step counterfactual returns.
- DeepSet evaluator: permutation-invariant candidate-list conditioning.

The current DeepSet runs do not improve blocking over the MLP, but the gap is
small enough that the architecture is viable as a paper-facing upgrade.

The first suspected limitation was candidate-set mismatch:

- Offline dataset groups are `candidate_mode=v1`, i.e. a top-K / heuristic subset
  of legal actions.
- Online DeepSet inference receives all legal R actions.
- This mismatch does not affect an independent MLP scorer much, but it changes
  the global set context seen by DeepSet.

The `legalctx48` dataset reduced this mismatch, but closed-loop blocking stayed
at `0.95%`.  That suggests the current bottleneck is no longer only context
coverage.  More likely causes:

- The H-step labels have low oracle headroom, so a stronger evaluator has little
  extra signal to exploit.
- KL-based listwise training may preserve the label distribution but not optimize
  the final top action enough for blocking.
- The DeepSet context scale starts as a residual upgrade over MLP, so it is
  stable but may under-use candidate interactions.
- Online C-side decisions may dominate the residual R-side improvements in this
  S100 setting.

## Mask Strictness and Context-Ablation Check

To test whether legal R candidates contain hidden current-step conflicts, a
mask strictness diagnostic copied the environment state and executed sampled
`agent_r_mask=True` actions:

| Checked legal actions | Successful steps | Failure rate | Failure reasons |
|---:|---:|---:|---|
| 14,173 | 14,173 | 0.00% | none |

This supports the interpretation that the R mask is already strict: legal
actions are executable one-at-a-time in the current state.  Therefore there is
no evidence that the legal candidate pool contains hidden physical conflicts
that a DeepSet or SetTransformer must resolve online.

The DeepSet context branch was also ablated by forcing `context_scale=0` in the
trained `legalctx48` checkpoint:

| Model | Blocking | NSB | Interpretation |
|---|---:|---:|---|
| DeepSet legalctx48 | 0.95% | 0.86% | Local MLP + set context |
| DeepSet legalctx48, context disabled | 0.96% | 0.88% | Local branch only |

The context branch helps slightly (`0.01 pp` blocking, `0.02 pp` NSB), but the
gain is too small to justify claiming that set-level candidate interaction is
the main source of improvement.  In the current R-side formulation, the original
MLP remains the strongest closed-loop model because the candidate's own features
already encode most of the relevant post-action resource impact, while the mask
removes infeasible current-step conflicts.

The first low-cost SetTransformer test also supports this interpretation.  It
improved some offline ranking metrics over DeepSet (`Test KL 0.0267` vs
`0.0277`, `Test top-1 24.47%` vs `20.32%`) but closed-loop blocking worsened to
`0.97%`.  With the current 25-dimensional summary features, self-attention can
model interactions between candidate summaries, but it cannot directly observe
link overlap or slot-interval overlap.  Therefore the attention layer does not
yet have the structural information needed to reliably model spectrum
contention.

## Structured SetTransformer Pilot

A second-stage pilot added structural candidate features:

- `block_start_norm`, `block_end_norm`, `block_center_norm`
- topology-specific path edge bitmap (`40` edge indicators on `snap24_gnutella_reach`)
- total feature dimension: `68`

Files:

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_structured_legalctx48_mid`
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_settransformer_structured_mid/ranking_model.pt`
- Closed-loop output: `sa_hmarl/experiments/main_s100_settransformer_structured_mid_v12.md`

This pilot verified that structured checkpoints can train and run online with
checkpoint-specific feature schemas.  However, the medium-size dataset was too
small and sparse for the edge bitmap schema:

| Metric | Value |
|---|---:|
| Test top-1 | 7.41% |
| Test top-3 | 60.19% |
| Test Spearman | 0.735 |
| Test KL | 0.0420 |
| Test model regret | 0.0971 |
| Closed-loop blocking | 1.05% |

One concrete issue was feature normalization: edge bitmap columns that are all
zero in the train split can explode on validation/test if normalized with a
`1e-6` std floor.  Training now supports `--feature_std_floor`; the structured
pilot used `0.05`, which stabilized scores but did not recover ranking quality.

Takeaway: the structured-feature path is technically working, but it needs a
larger/better-balanced dataset before it can fairly test the SetTransformer
hypothesis.  The current mid-size pilot should be treated as a negative
engineering diagnostic, not a final architecture result.

If DeepSet is kept, the next useful experiments are:

- Train with `selection_metric=regret` or a top-action/listwise hybrid loss.
- Increase label contrast by using a stronger planner return or larger H only on
  ambiguous/high-impact groups.
- Sweep context scale initialization and aggregation (`mean`, `max`,
  `mean+max`).
- Or make online DeepSet inference use exactly the same `larger_legal_context`
  candidate construction instead of all legal actions.
- If SetTransformer is kept, add structured overlap features first: path edge
  bitmap/embedding, block start/end slot, and pairwise link/slot-overlap bias.
- For structured features, use a non-tiny normalization floor for sparse binary
  columns and prefer a full/legalctx-scale dataset over the mid-size pilot.
