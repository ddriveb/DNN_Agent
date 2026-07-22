# Final Multi-Topology Decision (Verified)

**Classification:** B
**Reason:** directionally stable gains but not all Holm-significant

## Decision criteria
- Stable (>=4/5 seeds same direction) vs KSP-FF: 4/8
- Stable vs PPO-R Top-1: 3/8
- Holm-significant vs KSP-FF: 0/8
- Holm-significant vs PPO-R Top-1: 0/8

## Required answers
1. Request-sync bug fully fixed: yes (validated by request_id_mismatch_count=0, final_event_queue_length=0).
2. Any invalid_path/modulation_reach/post-action NSB: no (validated by action_observation_consistency_error_count=0).
3. Current 120-cell results valid: yes, if all micro/smoke/full hard gates passed.
4. Strict better than formal KSP-FF K=50 hops: see PAIRED_STATISTICS (delta=base-strict).
5. Strict better than PPO-R Top-1: see PAIRED_STATISTICS.
6. PPO-C vs DF_C consistency: compare per-C-side tables in MULTITOPOLOGY_RESULTS.
7. COST239 vs zero-shot topologies: COST239 source-matched; German17/NSFNET/JPN48 Strict zero-shot and PPO-C transfer.
8. E=0/E=1 benefit distribution: see E0_E1_REPORT.
9. v1.35 diagnostic pilot: yes
10. v1.35 full launch: no