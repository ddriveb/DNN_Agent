# Paired Closed-Loop Statistics (COST239)

All comparisons use the same 5 traffic seeds (4001–4005). Difference = method_A blocking rate − method_B blocking rate. Negative = A is better.

| Comparison | A | B | Mean Δ | 95% CI | Wilcoxon p | Permutation p | #better/#worse/#equal |
|---|---|---|---:|---:|---:|---:|---:|
| old_gated_vs_old_all | old_v13_gated | old_v13_all | -0.4000% | [-0.7600%, -0.0300%] | 0.1875 | 0.6084 | 4/1/0 |
| strict_uniform_all_vs_ppo_r | full_uniform_all | ppo_r_top1 | -0.8100% | [-1.3800%, -0.2400%] | 0.0625 | 0.4902 | 5/0/0 |
| strict_uniform_all_vs_old_gated | full_uniform_all | old_v13_gated | 0.3800% | [-0.6500%, 1.7600%] | 1.0000 | 0.8528 | 3/2/0 |
| strict_uniform_gated_vs_strict_uniform_all | full_uniform_gated | full_uniform_all | 0.5000% | [0.1000%, 0.8400%] | 0.1250 | 0.5222 | 1/4/0 |
| strict_stratified_vs_strict_uniform | full_stratified_all | full_uniform_all | 0.2300% | [-0.2700%, 0.6900%] | 0.3750 | 0.6954 | 2/3/0 |
| strict_uniform_vs_ksp_ff_plain | full_uniform_all | ksp_ff_plain | -4.7500% | [-7.2400%, -1.4200%] | 0.1250 | 0.0366 | 4/1/0 |
| strict_uniform_vs_ksp_ff_highest | full_uniform_all | ksp_ff_highest | -1.6200% | [-3.6800%, 0.4300%] | 0.3125 | 0.2988 | 4/1/0 |

## Interpretation notes

- `old_gated_vs_old_all`: CI 跨 0，Wilcoxon p>0.05 ⇒ gated 的方向性改善不能被 5 seeds 证明为显著，但与 E=0 OOD harm 假设一致。
- `strict_uniform_all_vs_ppo_r`: 需要 CI 和 p 值判断 full-state 是否稳定优于 PPO-R。
- `strict_uniform_all_vs_old_gated`: 判断 full-state 是否能替代旧 gated 基准。
- `strict_uniform_gated_vs_strict_uniform_all`: 若为正，说明对 full-state 模型加 gate 反而有害（因为它已经学会了 E=0）。