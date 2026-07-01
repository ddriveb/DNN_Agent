# v1.2 Networkization Plan

## 1. Goal

Convert the current v1.2 counterfactual R-side ranker from an experiment-level
implementation into a clean network module:

> Planner-Distilled Amortized Counterfactual MPC for R-side RMSA.

The purpose is not to replace the proven v1.2 mechanism, but to make it look and
behave like a first-class model component:

- explicit model class;
- explicit input/output contract;
- explicit training objective;
- explicit online inference API;
- clear comparison against PPO-R and DeepRMSA.

Method framing:

```text
offline H-step counterfactual rollout = finite-horizon planner
candidate-level neural scorer = distilled/amortized planner
online argmax over legal R actions = cheap deployment policy
```

The ranker imitates the counterfactual planner's within-state action ranking.
It does not behaviorally clone PPO-R and it does not perform TD bootstrapping.

The final method should still use the same online information as v1.2:

```text
current state + current legal R candidate set only
```

No online future requests, no online rollout, and no oracle information should be
used during evaluation.

## 2. Current Baseline To Preserve

Current best R-side method:

```text
PPO-C fixed split/server selector
    -> enumerate legal R actions
    -> build per-candidate v1.2 features
    -> planner-distilled ranker scores each candidate
    -> choose argmax score
```

Key files:

- `sa_hmarl/sa_hmarl/agents/post_decision_value.py`
- `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py`
- `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py`
- `sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_closed_loop.py`

Key checkpoint:

- `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`

S100 checkpoint:

- `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_mixed_low/ranking_model.pt`

This baseline must remain reproducible after the refactor.

## 3. Proposed Model Definition

### 3.1 Name

Use a name that describes the method directly:

```text
CounterfactualActionValueRanker
```

or shorter:

```text
CounterfactualRRanker
```

Recommended paper name:

```text
Planner-Distilled Amortized Counterfactual MPC
```

Chinese explanation:

```text
规划器蒸馏式摊销反事实 MPC
```

### 3.2 Input

For each legal R candidate action `a`, the network receives:

```text
phi(s, a)
```

where `s` is the current post-C-decision RMSA state, and `a` is one legal
`(path, modulation, spectrum-block)` candidate.

Feature groups:

- path/mod/block physical features;
- C-decision context features;
- current R feasibility field features;
- normalized action identity features.

Current feature schema is defined by `FEATURE_NAMES` in:

```text
sa_hmarl/sa_hmarl/evaluation/generate_r_post_decision_dataset.py
```

### 3.3 Output

For each candidate:

```text
Q_cf(s, a) in R
```

This is a scalar distilled planner score. It is not a TD bootstrap value and it
is not a PPO-R action imitation logit. It is trained to preserve the ranking
induced by H-step counterfactual planner labels.

Online action selection:

```text
a* = argmax_a Q_cf(s, a),  a in legal_R(s)
```

## 4. Training Target

For each sampled decision state, generate a candidate group:

```text
G = {a_1, a_2, ..., a_K}
```

For every candidate action, create a branch environment, execute that action,
roll out `H` future requests, and compute:

```text
y_i = G_H(s, a_i)
```

Current v1.2 return:

```text
y_i =
  - current_block
  - future_block
  - future_NSB
  - delay_cost
  - avg_FS_cost
  - server_overload
  - path_length_penalty
  - required_FS_penalty
```

The important v1.2 regularizers are:

- `path_length_penalty`;
- `required_FS_penalty`.

They encode the resource-conservation prior that produced the stable gain over
DeepRMSA.

## 5. Loss Function

Keep the current listwise ranking objective as the main loss:

```text
q = softmax(y / tau_label)
p = softmax(Q_cf(s, a) / tau_model)
L_rank = KL(q || p)
```

Optionally keep the auxiliary regression loss:

```text
L_reg = SmoothL1(Q_cf, normalized_y)
```

Final loss:

```text
L = L_rank + beta * L_reg
```

This is preferable to TD-style training because the main performance difference
comes from fine-grained candidate ordering within the same state, not from coarse
state-value estimation.

## 6. Code Refactor Plan

### Phase 1: Model Module

Create or update:

```text
sa_hmarl/sa_hmarl/agents/counterfactual_r_ranker.py
```

Move or wrap the current `PostDecisionValueNetwork` into:

```python
class CounterfactualActionValueRanker(nn.Module):
    def forward(self, candidate_features):
        ...
```

Requirements:

- accept both `[N, F]` and `[B, K, F]` tensors;
- return `[N]` or `[B, K]` scalar scores;
- store no environment-specific logic inside the network;
- keep checkpoint compatibility with the existing v1.2 model where possible.

Do not delete `post_decision_value.py` immediately. Keep it as a compatibility
shim until all scripts are migrated.

### Phase 2: Feature Builder API

Create a small feature-builder wrapper:

```text
sa_hmarl/sa_hmarl/evaluation/r_ranker_features.py
```

Expose:

```python
build_r_ranker_feature(env, req, obs_c, obs_r, r_features, r_action_idx, split_id, server_id)
```

This should call the existing `_r_feature_vector` first, then gradually become
the canonical API.

Purpose:

- avoid importing dataset-generation scripts during online evaluation;
- make inference code easier to read;
- define one feature schema for training and evaluation.

### Phase 3: Inference Wrapper

Create:

```text
sa_hmarl/sa_hmarl/agents/r_ranker_policy.py
```

Expose:

```python
class CounterfactualRRankerPolicy:
    def select_action(env, req, obs_c, obs_r, split_id, server_id) -> Optional[int]:
        ...
```

Responsibilities:

- build all legal R candidates;
- build and normalize candidate features;
- run the ranker once;
- return the best flat R action index;
- handle empty-mask and single-action cases.

This wrapper should replace `_select_rank_only_r_action` inside
`eval_r_counterfactual_ranking_closed_loop.py`.

### Phase 4: Training Script Rename/Wrapper

Keep existing training script working:

```text
train_r_counterfactual_ranking.py
```

Add a clearer alias script:

```text
train_counterfactual_r_ranker.py
```

The alias can call the same training function. The goal is readability, not a
new training method.

### Phase 5: Evaluation Cleanup

Update closed-loop evaluation so the main branch reads like:

```python
if mode == "counterfactual_rank_only":
    r_idx = ranker_policy.select_action(...)
```

Move failed/diagnostic modes such as Lyapunov reranking to separate diagnostic
scripts or keep them below the main path with clear comments.

## 7. Experiments To Run After Refactor

### Smoke Test

Purpose: verify compatibility.

Scenario:

- S80 or small S24 smoke;
- 1 seed;
- 1-2 episodes;
- compare old `_select_rank_only_r_action` and new policy wrapper.

Pass condition:

```text
same selected R action rate = 100%
same blocking/delay metrics
```

### Regression Test: Main S24 Result

Scenario:

- 24-slot `default3`;
- same checkpoint;
- same seeds as final validation where possible.

Expected:

```text
v1.2 remains around 53.92% blocking
DeepRMSA remains around 54.94% blocking
gap remains about 1 pp
```

### Regression Test: S100 Supplement

Scenario:

- 100-slot standard low-blocking;
- same S100 checkpoint and DeepRMSA S100 baseline.

Expected:

```text
v1.2 remains below DeepRMSA by about 0.3-0.4 pp on the original S100 setting
```

### Topology Sanity

Reuse:

```text
sa_hmarl/experiments/topology_sweep_s100_v12_deep_verify/
```

Expected:

- easy topologies may have 0% vs 0%;
- discriminative topology should preserve small v1.2 advantage;
- delay should remain lower for v1.2 in most cases.

## 8. What Not To Do

Do not immediately replace v1.2 with:

- TD-style value learning;
- online rollout;
- Lyapunov dynamic reranking;
- slot-position features only;
- sliding-window batching.

These have already been diagnosed as low-gain or negative under the current
action space.

Do not claim global optimality. The correct claim is:

```text
The ranker provides denser candidate-level supervision than trajectory-only RL
and improves R-side action selection under matched online information.
```

## 9. Risks And Mitigation

### Risk 1: Refactor changes behavior

Mitigation:

- first implement wrapper around existing functions;
- run action-level equivalence smoke test;
- only then clean imports.

### Risk 2: Feature schema drift

Mitigation:

- keep `FEATURE_NAMES` as the single source of truth;
- store `feature_names`, `feature_mean`, and `feature_std` in every checkpoint;
- assert schema equality at load time.

### Risk 3: Overclaiming in paper

Mitigation:

Use precise wording:

```text
counterfactual action-value ranking
```

not:

```text
globally optimal value estimation
```

### Risk 4: Confusion with PPO-R

Mitigation:

Make the final architecture explicit:

```text
PPO-C + Counterfactual R-Ranker
```

PPO-R is used for:

- baseline comparison;
- data-generation rollout policy;
- candidate diversity during offline dataset generation.

It is not the final R-side decision module in v1.2.

## 10. Deliverables

1. `counterfactual_r_ranker.py`
2. `r_ranker_features.py`
3. `r_ranker_policy.py`
4. compatibility update to `eval_r_counterfactual_ranking_closed_loop.py`
5. smoke equivalence report
6. S24 regression report
7. S100 regression report
8. short architecture note for paper/PPT

## 11. Decision Gate

Networkization is accepted only if:

- old and new inference select identical actions in smoke/equivalence tests;
- final S24/S100 metrics remain statistically consistent with existing reports;
- code readability improves without changing the method definition.

If metrics change, stop and debug feature normalization, mask handling, and
checkpoint schema before running any new experiments.
