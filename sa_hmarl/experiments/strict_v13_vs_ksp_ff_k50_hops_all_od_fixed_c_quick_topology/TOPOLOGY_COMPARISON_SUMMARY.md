# Topology Comparison Summary: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / all-OD)

## Summary Table

| topology | seeds | Strict blocking | KSP-FF K=50 hops blocking | gap pp | strict_win | ksp_win | conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| xlron_cost239_ptrnet_real | 20 | 5.6342% | 5.6342% | 0.00 | 0 | 0 | identical |
| xlron_jpn48 | 5 | 5.6133% | 5.5933% | 0.02 | 25 | 31 | near-identical |
| xlron_german17 | 5 | 5.4767% | 5.4767% | 0.00 | 1 | 1 | identical |

## Decision

**Case A: Blocking is identical or near-identical across all tested topologies.**

In fixed-C / all-OD pure RMSA isolation, Strict v1.3 and KSP-FF K=50 hops show no meaningful blocking difference on COST239, JPN48, or German17. The small gaps (0.00-0.02 pp) are within seed noise.

### Key observations

1. **Action preferences differ but outcomes do not.** Strict v1.3 consistently chooses longer paths (higher path_idx, more hops, longer km) and higher-start blocks, while KSP-FF K=50 hops always chooses the first hops-ordered path and lowest-start First-Fit block. Yet blocking rates are identical.

2. **Blocking is dominated by r_no_valid_action.** In all three topologies, >99% of blocked requests fail because the R-side legal mask is empty (spectrum exhausted), not because the chosen action was suboptimal.

3. **Candidate-pool coverage varies by topology.** KSP-FF K=50 hops action is inside Strict v1.3's PPO-R Top-30 in 32.71% (COST239), 14.65% (JPN48), and 28.67% (German17) of requests. JPN48's larger path diversity makes PPO-R less likely to rank the KSP action in its Top-30.

### Conclusion

The pure R-side RMSA blocking advantage previously attributed to Strict v1.3 in PPO-C closed-loop settings does **not** come from single-step RMSA action quality. In isolated fixed-C / all-OD settings, the two policies are indistinguishable by blocking rate.

The next step should therefore focus on **trajectory-level coupled effects**: how R-side actions change post-decision optical/server state, which then influences future C-side decisions and future blocking. This is exactly the v2.0 sliding-window afterstate dataset generation and DF_C closed-loop coupling diagnosis.

## Next Steps

1. **Proceed with v2.0 Phase-1**: sliding-window afterstate dataset generation with H={5,20,50} labels.
2. **Primary source distribution**: DF_C closed-loop (mode=B), because fixed-C / all-OD has proven pure R blocking is identical.
3. **Secondary check**: run the same afterstate dataset generation on fixed-C / all-OD (mode=A) to confirm that even with trajectory labels, fixed-C cannot distinguish the two policies.

## Output Locations

- COST239: `sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_diagnosis/`
- JPN48: `sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_fixed_c_quick_topology/xlron_jpn48/`
- German17: `sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_fixed_c_quick_topology/xlron_german17/`
