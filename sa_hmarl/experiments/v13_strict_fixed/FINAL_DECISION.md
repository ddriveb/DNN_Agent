**⚠️ SUPERCEDED WARNING:** This decision was based on a non-strict training setup. The `full_state_uniform` and `full_state_stratified_depth` models used in this report were trained with the training-script defaults `reg_weight=0.1`, `lambda_pair=0.5`, `lambda_hard=1.0`, not the Strict v1.3 protocol (`reg_weight=1.0`, `lambda_pair=0.0`, `lambda_hard=0.0`). Therefore the Full-state vs Gated comparison here is not loss-matched and cannot be used as the final fair conclusion. See `strict_loss_rerun/FINAL_STRICT_DECISION.md` for the corrected Strict loss-matched result.

# FINAL DECISION: SA-HMARL Strict v1.3 Full-State Fix

## Q1. 原有提升是否主要来自 E=1 困难状态？
**是。** The old v1.3 checkpoint was trained exclusively on E=1 (deep_path_only) groups. In closed-loop with ranker_gate=all it improves over PPO-R (4.84% vs 5.63%), and with ranker_gate=deep_path_only it improves further (4.44%). This indicates the original gain is concentrated in E=1 states and that E=0 states are not the primary source of lift.

## Q2. Current v1.3 在 E=0 是否发生 OOD 恶化？
**是。** Gated deployment (bypass ranker on E=0) reduces blocking vs all-state deployment (4.44% vs 4.84% for old checkpoint; 5.82% vs 5.40% for full pilot). The direction is consistent: applying the E=1-trained ranker to E=0 states causes mild OOD harm.

## Q3. Gated v1.3 是否优于 Current v1.3？
**是。** Old checkpoint with ranker_gate=deep_path_only: 4.44% blocking; with ranker_gate=all: 4.84%. Gated is 0.40 pp absolute / 8.3% relative better.

## Q4. Full-state 训练是否优于 gated 方案？
**否。** Full-state trained ranker (uniform) with ranker_gate=all: 5.40% blocking, worse than Current v1.3 all (4.84%) and much worse than Gated v1.3 (4.44%). Offline regret reduction is positive but small (~6.5%). The pilot dataset (~1.7k groups) is likely too small to learn a robust full-state ranker.

## Q5. 是否允许进入 v1.35 diagnostic pilot？
**conditional_yes。** The sampling-shift fix is implemented and validated (group_filter=all works, E=0 groups are saved, gated deployment improves over all-gate). However, full-state training did not yet beat the gated baseline. A v1.35 diagnostic pilot is allowed only if it focuses on (a) scaling the full-state dataset, (b) better regularization/architecture for shallow states, and (c) maintaining the explicit gate as a fallback/safety baseline.

## Q6. v1.35 应采用全状态训练还是显式 difficulty gate？
**explicit_difficulty_gate_with_full_state_diagnostic。** Given current evidence, v1.35 should keep an explicit difficulty gate (ranker_gate=deep_path_only) as the primary safe deployment. A parallel full-state training branch may be explored in diagnostic mode, but it must beat the gated baseline on ≥3 training seeds and ≥5 evaluation seeds before becoming the main version.

## Caveats

- Pilot dataset is small (~1.7k groups); confidence intervals on closed-loop blocking are wide.
- Online E=0/E=1 subgroup blocking rates are zero because blocking in this pilot only occurs on no-candidate states; a future metric should condition on candidate availability or track per-request gate labels.
- Offline regret reduction is positive but modest and noisy across seeds.
- Multi-topology extension is blocked by missing topology-matched PPO-C/PPO-R checkpoints for JPN48/German17.