# Final Multi-Topology Decision (Verified)

**Classification:** E
**Reason:** no consistent cross-topology advantage

## Decision criteria
- Stable (>=4/5 seeds same direction) vs KSP-FF: 0/8
- Stable vs PPO-R Top-1: 0/8
- Holm-significant vs KSP-FF: 0/8
- Holm-significant vs PPO-R Top-1: 0/8

## Answers
1. Strict v1.3 vs KSP-FF: see PAIRED_STATISTICS.
2. Strict v1.3 vs PPO-R Top-1: see PAIRED_STATISTICS.
3. Consistency across C-sides: compare PPO-C and DF_C tables.
4. Topology-specific results: see MULTITOPOLOGY_RESULTS.
5. Failure component drivers: see FAILURE_DECOMPOSITION.
6. E-stratum contribution: per-row e_stratum_counts in raw results.
7. Native vs transfer: see CHECKPOINT_PROVENANCE.
8. Checkpoint-deployment mismatch: PPO-C transfer for German17/NSFNET/JPN48; Strict ranker zero-shot.
9. v1.35 diagnostic pilot: no
10. v1.35 full launch: no