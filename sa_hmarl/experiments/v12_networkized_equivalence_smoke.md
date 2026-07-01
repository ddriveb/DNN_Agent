# v1.2 Networkized Equivalence Smoke

## Purpose

Verify that the new `CounterfactualRRankerPolicy` wrapper preserves the original
v1.2 action-selection behavior.

The test compares:

- old path: `_select_rank_only_r_action(...)`
- new path: `CounterfactualRRankerPolicy.select_action(...)`

Both use the same v1.2 checkpoint:

```text
sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt
```

## Scenario

- Topology: `snap24_gnutella_reach`
- Slots: 80
- Seed: 3030
- Requests checked: 80
- C policy: `agent_c_delayaware_v2_snap24_best.pt`
- R feature source: `agent_r_mixed.pt`

## Result

| Metric | Value |
|---|---:|
| Checked decisions | 80 |
| Mismatches | 0 |
| Agreement rate | 100.00% |

## Verdict

PASS.

The networkized wrapper is behavior-preserving for this smoke trace.  It can now
replace the old inline v1.2 selection function in closed-loop evaluation.
