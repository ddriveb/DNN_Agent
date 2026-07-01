# v1.2 Method Framing: Planner-Distilled Amortized Counterfactual MPC

## Recommended Name

Paper-facing English name:

```text
Planner-Distilled Amortized Counterfactual MPC for R-side RMSA
```

Short English name:

```text
Planner-Distilled R-Ranker
```

Chinese name:

```text
面向 R 端 RMSA 的规划器蒸馏式摊销反事实 MPC
```

Alternative Chinese name:

```text
后决策频谱可行性引导的模仿学习式摊销 MPC
```

Use "spectrum fragmentation" or "spectrum continuity degradation" in formal
writing. Avoid using "spectrum collapse" as the main academic term.

## Core Idea

The final v1.2 R-side method is not a TD value learner and not behavioral
cloning of PPO-R. It is planner imitation:

```text
offline:
    enumerate legal RMSA candidates
    evaluate each candidate with H-step counterfactual rollout
    obtain a finite-horizon planner return/ranking

training:
    distill the planner ranking into a candidate-level neural scorer

online:
    enumerate current legal RMSA candidates
    score them with one network forward pass
    select argmax score
```

This is an amortized planner because the expensive planner is used only during
dataset generation. Online inference uses the distilled network instead of
performing rollout.

## Mathematical Form

Given the C-side decision `a_C = (split, server)`, the R-side legal candidate
set is:

```text
A_R(s, a_C) = {a_R^1, a_R^2, ..., a_R^K}
```

The offline planner evaluates each candidate:

```text
G_H(s, a_C, a_R^k)
```

where `G_H` is computed by executing candidate `a_R^k` in a copied environment
and rolling out `H` future requests.

The neural scorer is trained to approximate the planner ranking:

```text
f_theta(phi(s, a_C, a_R^k)) ~= G_H(s, a_C, a_R^k)
```

The deployed online policy is:

```text
a_R* = argmax_{a_R in A_R(s, a_C)} f_theta(phi(s, a_C, a_R))
```

The listwise training objective is:

```text
q = softmax(G_H / tau_label)
p = softmax(f_theta / tau_model)
L_rank = KL(q || p)
```

## What PPO-R Does Here

PPO-R is not the teacher that v1.2 imitates. PPO-R provides:

- the R-side action representation and feature builder;
- top-K proposals and rollout behavior during offline planner evaluation;
- a baseline policy for comparison.

The teacher is the H-step counterfactual planner, not PPO-R's selected action.

## Code Map

- `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py`
  generates planner-distillation groups with H-step counterfactual rollout.
- `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py`
  trains the listwise scorer to imitate planner rankings.
- `sa_hmarl/sa_hmarl/agents/counterfactual_r_ranker.py`
  defines `CounterfactualActionValueRanker`, the distilled planner scorer.
- `sa_hmarl/sa_hmarl/evaluation/r_ranker_features.py`
  builds `phi(s, a_C, a_R)` candidate features.
- `sa_hmarl/sa_hmarl/agents/r_ranker_policy.py`
  wraps the scorer into an online argmax policy over legal R candidates.
- `sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_closed_loop.py`
  evaluates the deployed planner-distilled R policy in closed loop.

## Safe Paper Claim

Use:

```text
Compared with DeepRMSA, which learns RMSA actions from single-trajectory online
returns, v1.2 explicitly constructs counterfactual finite-horizon planning
targets for feasible RMSA candidates. The resulting planner is distilled into a
lightweight candidate-level neural scorer, enabling online decisions with
planner-like long-term awareness but without online rollout overhead.
```

Avoid:

```text
v1.2 is a strict residual MPC controller.
```

The residual formulation would require an explicit `base_score + residual`
implementation. The current implementation is cleaner: direct planner
distillation.
