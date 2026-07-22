# FINAL DECISION: Strict v1.3 vs KSP-FF K=50 hops under PPO-C and DF_C

## Primary results

- PPO-C + Strict v1.3 vs PPO-C + KSP-FF K=50 hops: **Significant win**
- DF_C + Strict v1.3 vs DF_C + KSP-FF K=50 hops: **Directional win**
- C-side interaction mean: 2.4050pp
- Lowest-blocking C×R combination: **DF_C + PPO-R Top-1** (6.1642%)

## Answers to the required questions

1. **PPO-C significantly better?** Yes.
2. **DF_C significantly better?** Directional only.
3. **Advantage cross-C stable?** Yes.
4. **DF_C causes distribution shift?** Yes.
5. **Strict stable vs PPO-R?** No.
6. **Lowest blocking combination:** DF_C + PPO-R Top-1.
7. **v1.35 diagnostic pilot allowed?** Yes.
8. **Strongest paper conclusion:** Strict v1.3 is directionally better than KSP-FF under both C strategies, but significance is conditional on C strategy.
9. **Next recommended experiment:** If cross-C advantage holds: run v1.35 diagnostic pilot with ≤1000 groups under Strict protocol and explicit gate safety baseline. If advantage is PPO-C-specific: first diagnose C-policy distribution shift and train a COST239-native PPO-R checkpoint before any broader claim.

## Caveats

- PPO-R checkpoint `agent_r_mixed.pt` was trained on NSFNET/extended/32 slots; PPO-C checkpoint was trained on COST239/default/100 slots. Both are deployed here on COST239/default/320 slots with longer episodes. Results are therefore existing-system evaluations, not fully native comparisons.
- DF_C + Strict v1.3 is a cross-C-policy transfer test because the ranker was trained on data generated under PPO-C.
- Causal explanations for any blocking differences (fragmentation, load balancing, future-resource protection) remain hypotheses without trajectory-divergence or action-level counterfactual diagnostics.