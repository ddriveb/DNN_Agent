# Paper Interpretation: What the v1.3 Post-Decision Upgrade Changes

This note translates the engineering changes into the fixed-C conditional R-side framework used in the paper and reports the first experimental comparison.

## Theoretical boundary

We keep:
- Frozen PPO-C (`a_C`) — the compute-offloading split/server decision is not retrained.
- Conditional R-side ranking: after `a_C` is fixed, we rank only R actions (`a_R`) that are feasible for the chosen `(split, server)`.
- No claim of global optimum. The ranker is a planner-distilled *amortized afterstate critic*, not a gradient-based discrete RMSA policy.

## From pre-action scorer to afterstate ranker

The v1.3 ranker computed `phi(s, a_R)` using pre-action spectrum statistics. The v1.3+ upgrade explicitly includes the afterstate `s'` induced by `a_R`:

```
score(a_R) = f_theta( phi(s, a_R, s') )
```

where `s'` is captured analytically by temporarily marking the slots that `a_R` would occupy and recomputing spectrum statistics. This is an *afterstate value function* for the R-side sub-problem conditioned on `a_C`.

## Opportunity-cost semantics

The training label is the H-step counterfactual return of committing to `a_R` and continuing with frozen PPO-C + PPO-R. Because the future continuation is shared across candidates via cloned RNG, differences in the label are due to differences in:

1. the immediate success/failure of `a_R`,
2. the future optical feasibility of the afterstate `s'`,
3. server-overload dynamics that accumulate over the horizon.

This is an explicit opportunity-cost signal rather than a one-step reward.

## Mutually-exclusive failure accounting

The new label decomposes future failures into `optical`, `overload`, and `other`. This prevents the old formula from double-counting NSB as both a `blocked` event and an `NSB` event, and it lets us assign a separate (small) weight to overload only when it actually varies across candidates.

## Candidate pool as a design choice

The `poststate_anchor48` pool is a lightweight diversified search over the legal R actions. It is not an optimization oracle; it simply ensures that the ranker sees enough optical afterstates at training and inference time. The PPO-R top-K proposals remain the backbone; the anchors add robustness for path/mod/block combinations that PPO-R may underweight.

## Experimental signature (fixed evaluator protocol)

The first evaluator had two protocol bugs (no warmup; C-side K=50 instead of K=5). After fixing them, the comparison was rerun on COST239 (5 seeds, 5000 evaluated requests/seed, 1000 warmup, C-K=5, R-K=50 hops). To isolate the feature effect, both rankers used the same `ppo_r_topk_only` candidate mode.

Aggregate blocking rates:

| Mode | Blocking % |
|---|---:|
| ranker_v13 | 6.01% ± 0.75% |
| ppo_r | 6.85% ± 1.72% |
| ksp_ff_highest | 7.03% ± 0.86% |
| ksp_ff_plain | 7.88% ± 1.35% |
| ranker_poststate | 7.42% ± 0.63% |

Key observations:
- The fixed protocol restores the historical ~6–8% scale; v1.3 is close to the previously reported 6.29%.
- Failures are dominated by `server_overload`; NSB is small.
- `ranker_v13` is the strongest method, beating PPO-R and KSP-FF.
- `ranker_poststate` is worse than `ranker_v13` on every seed and ~3× slower, even with identical candidates.

Interpretation: under the fixed-C conditional boundary, the current 41-d explicit-afterstate feature upgrade does not improve ranking and adds latency. The bottleneck is likely the training signal (H=5, `return_mode=v1`, 240 groups), not the idea of afterstate features per se. A fair conclusion is that this particular poststate checkpoint did not help; it does not prove that afterstate information is useless.

## Phrasing for the paper

Safe phrasing:
- "planner-distilled afterstate ranker"
- "conditional R-side reranking under a fixed compute policy"
- "amortized finite-horizon opportunity-cost scoring"

Avoid:
- "near-optimal"
- "gradient-based discrete RMSA policy"
- "global planner"
- claims that the ranker re-optimizes `a_C`
