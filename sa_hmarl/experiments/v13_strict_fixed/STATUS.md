# SA-HMARL Strict v1.3 Fix — Status

## Completed
- [x] Generator fix: removed implicit path_idx>=5 filter, added `--group_filter all/deep_path_only`.
- [x] Online evaluator fix: added `--ranker_gate all/deep_path_only` and E-gate subgroup tracking.
- [x] Training script fix: added `--sampling_mode uniform/balanced/stratified_depth`.
- [x] Unit tests for filter/stratum/gate behaviors.
- [x] COST239 full-state pilot dataset (~1.7k groups).
- [x] Offline comparison (3 seeds × 2 sampling modes + old checkpoint).
- [x] Closed-loop pilot (5 seeds × 2 gates × 5 methods).
- [x] Multi-topology blocker assessment.

## Key results
- Offline full-state uniform regret reduction: 3.41% (seed 42), 8.22% (seed 43), 7.93% (seed 44).
- Closed-loop PPO-R blocking: 5.63%.
- Closed-loop Current v1.3 (all) blocking: 4.84%.
- Closed-loop Gated v1.3 blocking: 4.44%.
- Closed-loop Full v1.3 pilot (all) blocking: 5.40%.

## Decision
Proceed with v1.35 diagnostic pilot under explicit difficulty gate; full-state training remains experimental until it consistently beats the gated baseline.

## Blockers
- Multi-topology (JPN48, German17) requires topology-matched PPO-C and PPO-R checkpoints. Do not run fair multi-topology experiments until these are available.