# SA-HMARL Strict v1.3 Loss-Matched Rerun — Status

## Completed
- [x] Re-audited generator, evaluator, training script, and existing reports.
- [x] Fixed launcher to explicitly pass Strict loss parameters (reg_weight=1.0, lambda_pair=0.0, lambda_hard=0.0).
- [x] Trained full_state_uniform_strict and full_state_stratified_depth_strict on seeds 42/43/44.
- [x] Computed offline metrics with per-seed, E-gate, and depth-stratum breakdowns.
- [x] Ran closed-loop comparison for 9 method×gate combinations on 5 shared seeds.
- [x] Computed paired statistics (mean diff, bootstrap CI, Wilcoxon, permutation, seed counts).
- [x] Wrote corrected reports with careful causal wording.
- [x] Added superseded warning to previous FINAL_DECISION.md.

## Key results
- Strict full-state uniform all-gate blocking: 4.82% (vs PPO-R 5.63%).
- Old v1.3 gated blocking: 4.44% (best baseline).
- Strict full-state does not beat old gated (mean +0.38 pp).
- Gating strict full-state hurts (+0.50 pp), suggesting it learned E=0 behavior.
- Stratified sampling does not improve over uniform.

## Decision
Category B: Allow limited v1.35 diagnostic pilot (≤1000 groups) with explicit gate as safety baseline.

## Blockers unchanged
- Multi-topology extension still blocked by missing topology-matched PPO-C/PPO-R checkpoints for JPN48/German17.