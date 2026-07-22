# FINAL STRICT DECISION: SA-HMARL v1.3 Full-State Fix

## 1. Full-state strict 是否优于 PPO-R？
**是。** Strict full-state uniform all-gate mean blocking = 4.82% vs PPO-R = 5.63%. Paired mean difference = -0.8100%, bootstrap 95% CI = [-1.3800%, -0.2400%], Wilcoxon p = 0.062. All 5 seeds show improvement.

## 2. 是否优于 old gated v1.3？
**否。** Strict full-state uniform all-gate = 4.82% vs old gated = 4.44%. Paired mean difference = 0.3800%, CI = [-0.6500%, 1.7600%], Wilcoxon p = 1.000. Full-state is not yet competitive with the gated baseline.

## 3. stratified_depth 是否有效？
**否。** Strict stratified all-gate = 5.05% vs strict uniform all-gate = 4.82%. Paired mean difference = 0.2300%, CI crosses 0. Stratified sampling does not help on this small dataset.

## 4. E=0 OOD harm 是已证明、方向性支持，还是不支持？
**directionally_supported_not_proven。** Old gated vs old all: mean diff = -0.4000%, CI = [-0.7600%, -0.0300%], Wilcoxon p = 0.188. The direction is consistent with E=0 OOD harm, but n=5 is underpowered and the CI from a different resample may cross 0. No local causal claim can be made.

## 5. 是否允许进入 v1.35 diagnostic pilot？
**是（Category B）。** Category B: Strict full-state improves over PPO-R but does not beat old gated. Allow a v1.35 diagnostic pilot limited to ≤1000 groups, with explicit difficulty gate as the safety baseline.

## 6. 推荐 full-state 还是 explicit gate？
**explicit_gate_main_full_state_diagnostic。** The explicit difficulty gate (ranker_gate=deep_path_only) is the only deployment strategy with consistent improvement over PPO-R in this pilot. Full-state training is promising but must remain diagnostic until it consistently beats the gated baseline on ≥3 training seeds and ≥5 evaluation seeds.

## 7. 下一步唯一推荐实验是什么？
Scale the full-state training dataset to ≤1000 groups under the Strict loss protocol, and ablate shallow-state architecture/regularization. Evaluate against the same old gated baseline on the same 5 traffic seeds before any broader deployment. Do not start Japan/German experiments.

## Caveats

- Pilot dataset is small (1.7k groups); confidence intervals are wide.
- Old checkpoint is OOD on the full-state test set; it is included as a reference, not as a fair full-state model.
- E=0/E=1 online blocking rates are near zero because blocking occurs on no-candidate states.
- Wilcoxon p-values with n=5 are underpowered; bootstrap CIs should be interpreted with caution.