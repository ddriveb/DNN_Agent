# E=0 / E=1 Causality Limits

## What we know for certain
- The old v1.3 checkpoint was trained on a dataset where every group satisfies `max_path_idx >= 5` (E=1). Therefore, by construction, E=0 states are **out-of-distribution** for that model. This is a distributional fact, not an empirical claim.
- The new `group_filter=all` dataset contains E=0 groups (~13% in this pilot), so a model trained on it is no longer OOD on E=0.

## What the closed-loop results suggest
- `old_v13` with `ranker_gate=deep_path_only` (4.44%) achieved lower aggregate blocking than `old_v13` with `ranker_gate=all` (4.84%).
- The paired mean difference is −0.40 pp with a bootstrap 95% CI of [−0.76, −0.03] pp.
- However, the Wilcoxon signed-rank p-value is 0.188 and n=5.

## What we cannot claim
1. **We cannot claim a proven causal effect of E=0 on blocking.** The aggregate blocking difference is driven by the full trajectory, not by a counterfactual comparison at each E=0 state. Gating changes the entire future state distribution.
2. **We cannot interpret E=0/E=1 subgroup blocking rates as local effects.** In this pilot, blocking occurs almost exclusively when no legal candidate exists; ranker-invoked states (E=0 or E=1) show near-zero immediate blocking. The gate affects future states through trajectory divergence.
3. **We cannot say the full-state model is “proved safe” on E=0.** We can only say it was trained on E=0 data and that its gated deployment hurts relative to its all-gate deployment, which is consistent with it having learned non-trivial E=0 behavior.

## Recommended wording
Use:
> "The results are consistent with the hypothesis that the E=1-only ranker suffers OOD degradation on E=0 states, but the evidence from 5 shared seeds is directional and not statistically significant."

Do not use:
> "E=0 causes blocking to increase" or "gating strictly improves blocking."
