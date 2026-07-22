# Current Complete Experiment Record

> Scope note: this file belongs to the historical `independent PPO-R` research
> branch, not the current `PPO-C + v1.2 static counterfactual ranker` mainline.
> Canonical naming is documented in `CANONICAL_NAMES.md`.

> Updated: 2026-06-16. This record consolidates the latest evidence after the
> R-side same-state diagnostic, exhaustive Oracle-R analysis, activation
> ablation, conservative frozen-R fine-tuning, and typed mean-field Agent-C
> mainline validation.

## 0. Final Current Position

The main scientific method should be described as:

```text
Typed-Mean-Field Agent-C + independently trained PPO-R
```

BC-PPO-R should be treated as a teacher-transfer / sanity-check baseline, not
as the primary contribution.  Some earlier tables use BC-PPO-R because it was
the stable comparison backend at the time, but the current evidence shows that
independent PPO-R reaches the same R-side blocking capability.

Core claim:

```text
PPO-R independently learns a DeepRMSA-level RMSA executor and reaches the
R-side blocking lower bound under fixed Agent-C decisions.  Typed-Mean-Field
Agent-C improves the upstream split/server decision by exposing active-request
population pressure, lowering blocking without increasing average delay.
```

## 1. Main Evidence Summary

| Question | Current Answer | Key Evidence |
|---|---|---|
| What is the main method? | Typed-Mean-Field Agent-C + independent PPO-R | Mean-field mainline validation improves blocking by 2.40pp on average |
| Is R still improvable for blocking? | No, not under current action space | Exhaustive Oracle-R gain = 0.00 pp |
| Why do PPO-R and DeepRMSA perform similarly? | R is feasibility-dominated | When valid R actions exist, PPO-R succeeds; when none exist, no R policy can help |
| Is BC-PPO-R the main method? | No | It is a teacher-transfer baseline and equivalence check |
| Where is the remaining bottleneck? | C-side split/server decisions | Oracle-R gain is zero; typed mean-field improves C choices |
| Did typed mean-field help? | Yes | PPO-C + independent PPO-R mainline blocking drops 0.3865 -> 0.3625 |
| Did activation changes help? | No | Tanh, GELU, LeakyReLU, SiLU are within noise |
| Did joint/frozen-R fine-tuning help? | No | Frozen-R C pressure shaping regressed by 0.42 pp |

## 2. R-Side Formal Comparison

Setting: `snap24_gnutella_reach`, fixed Agent-C checkpoint,
5 seeds x 20 episodes x 60 requests.

| R Backend | Blocking | Success | AvgReward | Delay | AvgFS | Waste | PathKm | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PPO-R | 0.224 | 0.776 | +0.074 | 7.9ms | 2.09 | 0.559 | 425.8 | independent learned R; best/tied |
| Dual-Critic v2 A R | 0.224 | 0.776 | +0.074 | 7.9ms | 2.09 | 0.559 | 425.8 | no gain over PPO-R |
| DeepRMSA | 0.225 | 0.775 | +0.075 | 8.5ms | 2.02 | 0.550 | 495.2 | teacher-level baseline, essentially tied |
| KSP-BF | 0.264 | 0.736 | +0.041 | 7.7ms | 2.47 | 0.529 | 422.2 | weaker heuristic R |

Failure reasons:

| R Backend | no_suitable_block | server_overload |
|---|---:|---:|
| PPO-R | 93.4% | 6.6% |
| DeepRMSA | 90.7% | 9.3% |
| KSP-BF | 96.7% | 3.3% |

Interpretation: PPO-R independently reaches DeepRMSA-level blocking performance
without teacher distillation.  KSP-BF remains a weak RMSA baseline.

Source: `fixed_c_r_compare_snap24.md`.

## 3. PPO-R vs BC-PPO-R Same-State Diagnostic

Setting: same pre-step environment state, deep-copy comparison,
`snap24_gnutella_reach`, S20/k=5/mixed, 2000 requests.

| Outcome | Count | Rate |
|---|---:|---:|
| Both ok | 1438 | 71.90% |
| Both fail | 561 | 28.05% |
| BC win / PPO fail | 0 | 0.00% |
| PPO win / BC fail | 1 | 0.05% |

Conclusion:

```text
There are zero states where BC-PPO-R succeeds and independent PPO-R fails.
Targeted DAgger has no correction samples.
```

This is the strongest evidence that BC-PPO-R should not be treated as the main
method.  It is a useful teacher-transfer baseline, while the independent PPO-R
already has the same per-state success/failure decision quality.

Source: `targeted_dagger_r_summary.json`.

## 4. Exhaustive Oracle-R Analysis

Setting: fixed Agent-C action, exhaustive enumeration of all valid R actions
per request, S20/k=5/max_blocks=10/mixed, 5 seeds x 5 episodes x 80 requests =
2000 requests.

| Metric | Value | Meaning |
|---|---:|---|
| PPO-R blocking | 0.2950 | PPO-R blocking in this diagnostic setting |
| Oracle blocking | 0.2950 | Exhaustive R oracle blocking |
| Oracle gain | 0.00 pp | zero R-side blocking headroom |
| PPO fail -> Oracle success | 0.0000 | PPO-R never fails when a valid alternative succeeds |
| PPO success -> Oracle better | 0.1890 | some FS/delay efficiency headroom, not blocking headroom |
| Mean valid R actions/request | 3.7 | out of max 200 candidate actions |
| Failure reason | 100% no_valid_r | all blocking is caused by no valid R action |

Important nuance: the latest report used the BC-PPO-R checkpoint as the frozen
R backend, but the same-state diagnostic above shows independent PPO-R and
BC-PPO-R are equivalent for success/failure decisions.  Therefore the Oracle-R
ceiling result supports the same R-side conclusion for the independent PPO-R
main method.

Final R-side verdict:

```text
R-side blocking optimization is done under the current action space.
Remaining blocking comes from C selecting split/server pairs with no feasible
R action.
```

Source: `oracle_r_exhaustive.md`.

## 5. Teacher Transfer / BC-PPO-R Role

Setting: `snap24_gnutella_reach`, 5 seeds x 20 episodes.

| Method | Blocking | Success | AvgReward | Delay | AvgFS | Waste | PathKm | SrvOver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy-C + DeepRMSA | 0.235 | 0.765 | +0.148 | 7.2ms | 2.14 | 0.523 | 640.1 | 0.004 |
| Agent-C + DeepRMSA | 0.216 | 0.784 | +0.087 | 8.4ms | 2.02 | 0.554 | 492.5 | 0.067 |
| Greedy-C + BC-PPO-R | 0.234 | 0.766 | +0.150 | 7.2ms | 2.14 | 0.520 | 627.9 | 0.004 |
| Agent-C + BC-PPO-R | 0.215 | 0.784 | +0.091 | 8.4ms | 2.02 | 0.549 | 488.7 | 0.067 |

Interpretation:

- BC-PPO-R reproduces DeepRMSA behavior almost exactly.
- Agent-C improves over Greedy-C under both DeepRMSA and BC-PPO-R.
- BC-PPO-R is evidence that the PPO-R architecture can reproduce teacher-level
  RMSA behavior, but the main method should remain independent PPO-R.

Source: `r_teacher_transfer_snap24.md`.

## 6. Typed Mean-Field Agent-C Mainline Validation

Setting: `snap24_gnutella_reach`, S20_R80, `k_paths=5`,
`max_blocks=10`, `block_sort=mixed`, `split_profile=default3`.
Training uses PPO Agent-C with frozen independently trained PPO-R
(`agent_r_mixed.pt`).  Evaluation uses 5 seeds x 10 episodes x 80 requests.

| Train Seed | C Feature Mode | Blocking | Delay | AvgFS | noC% | avgRacts |
|---:|---|---:|---:|---:|---:|---:|
| 42 | default | 0.3883 | 8.1ms | 2.63 | 44.75% | 3.5 |
| 42 | typed_mean_field | 0.3585 | 9.7ms | 2.32 | 42.30% | 4.1 |
| 123 | default | 0.3847 | 10.0ms | 2.48 | 44.73% | 3.0 |
| 123 | typed_mean_field | 0.3665 | 8.3ms | 2.43 | 43.50% | 3.1 |
| Average | default | 0.3865 | 9.1ms | 2.56 | 44.74% | 3.3 |
| Average | typed_mean_field | 0.3625 | 9.0ms | 2.38 | 42.90% | 3.6 |

Interpretation:

- Typed mean-field reduces blocking by 2.40pp on average.
- Both training seeds improve: seed 42 by 2.98pp, seed 123 by 1.82pp.
- noC% falls by 1.84pp on average.
- AvgFS falls from 2.56 to 2.38, indicating better spectrum efficiency.
- Average delay is essentially unchanged: 9.1ms -> 9.0ms.
- Typed mean-field Agent-C beats all evaluated heuristic C baselines in both
  trained checkpoint evaluations.

Conclusion:

```text
typed_mean_field is the first C-side structural change that gives a stable
positive mainline gain with independently trained PPO-R.
```

Source: `typed_mean_field_mainline_eval.md`.

## 7. Main System / Scenario Results

The most paper-facing visible scenario remains:

```text
snap24_gnutella_reach
S20_R80
k_paths = 5
max_blocks = 10
block_sort = mixed
```

Slots-load sweep, 5 seeds x 5 episodes:

| Scenario | AgentBlk | noC% | avgCa | avgRa | IWDblk | DFblk | WOblk | RFblk | BestHeu | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| S20_R80 | 0.2305 | 0.27 | 10.0 | 5.3 | 0.2610 | 0.2790 | 0.2875 | 0.2915 | IWD | -0.0305 |
| S24_R80 | 0.1940 | 0.24 | 10.7 | 6.5 | 0.2130 | 0.2190 | 0.2385 | 0.2400 | IWD | -0.0190 |
| S28_R80 | 0.1650 | 0.21 | 10.7 | 8.0 | 0.1820 | 0.1835 | 0.1975 | 0.2035 | IWD | -0.0170 |
| S32_R80 | 0.1445 | 0.19 | 11.1 | 9.0 | 0.1575 | 0.1630 | 0.1775 | 0.1750 | IWD | -0.0130 |

Interpretation:

- S20_R80 gives the clearest visible Agent-C advantage over heuristics.
- Higher slots lower blocking, but also reduce method separation.
- These rows were produced with the stable BC-PPO-R-style backend; the R-side
  equivalence diagnostics support using the same narrative for independent
  PPO-R after same-script confirmation if required.

Source: `slots_load_sweep.md`.

## 8. Activation Ablation

Setting: PPO Agent-C activation only, frozen R, S20_R80/k=5/max_blocks=10/mixed.

| Activation | Blocking | Delay | AvgFS | noC% | avgRacts | Delta vs Tanh |
|---|---:|---:|---:|---:|---:|---:|
| Tanh baseline | 0.246 | 8.0ms | 2.14 | 30.5% | 3.8 | 0 |
| LeakyReLU | 0.242 | 8.2ms | 2.11 | 29.7% | 3.7 | -0.4pp |
| GELU | 0.249 | 8.1ms | 2.14 | 30.7% | 3.6 | +0.3pp |
| SiLU | 0.260 | 8.8ms | 2.15 | 31.8% | 3.5 | +1.4pp |

Conclusion:

```text
Activation ablation has no meaningful gain.
Retain tanh as the default PPO actor activation.
```

Source: `activation_ablation_agent_c.md`.

## 9. Conservative Frozen-R Fine-Tuning

This was originally described as joint fine-tuning, but the final corrected
interpretation is:

```text
Frozen-R conservative Agent-C fine-tuning with C-side pressure shaping.
```

Formal 5-seed cross-evaluation:

| Configuration | Blocking | Delay | FS | noC | avgRa | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Orig C + Orig R | 0.2760 | 8.4 | 2.09 | 0.34 | 5.3 | baseline |
| FT C + Orig R | 0.2802 | 8.5 | 2.12 | 0.35 | 5.0 | -0.42pp regression |
| FT C + FT R | 0.2802 | 8.5 | 2.12 | 0.35 | 5.0 | same as FT C; R unchanged |
| Orig C + FT R | 0.2760 | 8.4 | 2.09 | 0.34 | 5.3 | proves R was frozen |
| IWD-C + BC-PPO-R | 0.2975 | 9.9 | 2.25 | 0.37 | 4.2 | heuristic baseline |
| DF-C + BC-PPO-R | 0.3118 | 8.0 | 2.44 | 0.38 | 3.5 | heuristic baseline |

Conclusion:

```text
Conservative C-side pressure shaping does not improve blocking.
Do not continue minor coefficient tuning or joint/frozen-R fine-tuning sweeps.
```

Source: `conservative_joint_finetune_eval.md`.

## 10. Why Some Earlier Blocking Rates Look Very Different

Do not directly compare numbers across different experiment conditions.

| Observed value | Reason |
|---|---|
| 0.000 to 0.03 blocking | usually tiny diagnostic runs, seed bias, or too-easy settings |
| metro24 near-zero blocking | topology is too easy / short-link dominated |
| k=7 near-zero in early R action-space sweep | 2 seeds x 2 episodes; later 5-seed formal showed this was seed-biased |
| S32_R80 = 0.1445 | more slots, easier spectrum availability, smaller method gap |
| S16/K5 around 0.39 | harder resource setting, high no-valid-C rate |
| 0.215 vs 0.2305 vs 0.246 vs 0.276 | different scripts, slots, split profiles, checkpoints, episode counts, and eval lengths |

Rule:

```text
Only compare methods within the same script, same topology, same slots/load,
same seeds, same request count, same k_paths/max_blocks/block_sort, and same
C/R checkpoint family.
```

## 11. Final Method Ranking

Paper-facing ranking by contribution:

1. **Typed-Mean-Field Agent-C + independently trained PPO-R**: current main
   method and strongest C-side structural improvement.
2. Enhanced Agent-C + independently trained PPO-R: previous main method and
   strong non-mean-field baseline.
3. Enhanced Agent-C + BC-PPO-R: teacher-transfer / equivalence baseline.
4. Enhanced Agent-C + DeepRMSA: external teacher-style strong baseline.
5. IWD-C + PPO-R or BC-PPO-R: strongest traditional C baseline.
6. DF-C + PPO-R or BC-PPO-R: often lower delay, higher blocking.
7. WO-C / RF-C / Greedy-C: weaker C baselines.
8. KSP-BF / SP-HM-FF: weak heuristic R backends.
9. Oracle-R / DAgger / activation / frozen-R fine-tuning / frag-aware:
   diagnostic or negative ablations, not final methods.

## 12. Paper Narrative

Recommended wording:

```text
We propose a hierarchical learned C+R framework in which Typed-Mean-Field
Agent-C performs population-aware split/server selection and an independently
trained PPO-R performs low-level RMSA execution.  PPO-R matches DeepRMSA-level
RMSA performance without teacher distillation, and exhaustive Oracle-R analysis
shows that PPO-R reaches the R-side blocking lower bound under fixed C
decisions.  Typed-Mean-Field Agent-C improves the remaining C-side bottleneck
by exposing active-request population pressure, reducing blocking without
increasing average delay.
```

Main contribution split:

- **R side:** independent PPO-R learns a DeepRMSA-level RMSA executor and is
  oracle-optimal for blocking under fixed C decisions.
- **C side:** Typed-Mean-Field Agent-C learns cross-layer split/server decisions
  with active-population pressure and beats traditional C heuristics in the hard
  snap24 scenarios.
- **Limit:** residual failures are dominated by no-valid-R / no-suitable-block
  states induced upstream by C-side feasibility constraints.

## 13. Stop / Continue Decisions

Stop:

- R fine-tuning for blocking.
- Targeted DAgger for R.
- Activation-function sweeps.
- Minor joint/frozen-R coefficient tuning.
- Frag-aware / future-feasibility / Oracle-R rerank attempts.

Continue only if needed:

- Expand typed_mean_field validation to 5 training seeds and 20 evaluation
  episodes per seed.
- Run typed-mean-field ablations: global vs 3-type vs 4/9-type; srv-only vs
  dem-only vs rel-only vs combined statistics.
- Same-script formal confirmation of `Typed-Mean-Field Agent-C + independent
  PPO-R` on the final paper-facing table.

## 14. One-Line Final Verdict

```text
R side is solved for blocking; independent PPO-R is the main R contribution.
Typed-Mean-Field Agent-C is the current best C-side improvement direction.
```
