# Oracle Ladder Phase 1 — Final Report (S100 low-blocking)

> Date: 2026-06-28  
> Scenario: `snap24_gnutella_reach`, 100 slots, 4 servers, k=5, max_blocks=10, `default3`, arrival interval 0.15, size_max 30 MB.  
> Eval: 3 seeds × 5 episodes × 80 requests = 1,200 requests, H=5.

## Files produced

- Diagnostic script: `sa_hmarl/sa_hmarl/evaluation/diagnose_oracle_ladder_v12.py`
- Results JSON: `sa_hmarl/experiments/oracle_ladder_v12_s100.json`
- Results Markdown: `sa_hmarl/experiments/oracle_ladder_v12_s100.md`
- Smoke outputs: `sa_hmarl/experiments/oracle_ladder_v12_s100_smoke.json/.md`

## What the oracles do

1. **baseline_v12**: delay-aware PPO-C + v1.2 ranker.
2. **r_oracle_in_mask**: same C, but for the selected split/server enumerate every raw-mask-legal R action and pick the H-step-return-maximizing one.
3. **r_expanded_block_oracle**: same C, but temporarily raise `max_blocks` to 32 to get an enlarged R candidate set, then pick by H-step return.
4. **joint_c_topk_r_oracle**: enumerate all raw-mask-legal C candidates, score their R actions with the v1.2 ranker, keep top-3 R per C, and pick the best (C, R) pair by H-step return.

No new models were trained; all checkpoints were used read-only.

## Results

| Mode | Current blocking rate | Future block rate | Headroom (pp) |
|---|---:|---:|---:|
| baseline_v12 | 0.667% | 7.50% | — |
| r_oracle_in_mask | 0.667% | 0.00% | 0.000 |
| r_expanded_block_oracle | 0.667% | 0.00% | 0.000 |
| joint_c_topk_r_oracle | 0.667% | 0.00% | 0.000 |

## Coverage and behavior change

| Metric | Value |
|---|---:|
| Evaluated blocked states | 8 |
| Mean legal R actions (in-mask) | 0.00 |
| Mean expanded candidates | 0.00 |
| Mean joint (C, top-3 R) pairs | 0.00 |
| R-oracle changed action rate | 0.00% |
| Expanded-oracle changed action rate | 0.00% |
| Joint-oracle changed C rate | 0.00% |
| Joint-oracle changed R rate | 0.00% |

## Failure reason distribution

| Reason | Count | Share |
|---|---:|---:|
| no_valid_c_action | 8 | 100.00% |

All 8 blocked requests had an empty raw C-mask: no `(split, server)` pair had any feasible R action.

## Verdict

**STOP_ACTION_SPACE_EXPANSION**

All oracle headrooms are 0.00 pp. The residual blocking in the S100 low-blocking scenario is a **hard C-side feasibility floor**, not an R-ranker or C/R joint-ranking error.

## Answers to the four questions

1. **Where does the largest pp opportunity come from?**  
   **Nowhere in the current action space.** There is no remaining recoverable blocking via better R selection, expanded R candidates, or joint C-R reranking.

2. **Is it worth expanding the R block candidate set?**  
   **No.** The blocked states have zero legal R actions under the selected C, and the expanded candidate set also has zero candidates because the raw C-mask is empty.

3. **Is it worth doing a C-R joint ranker?**  
   **No.** The joint oracle evaluated zero pairs in the blocked states (no valid C candidates to enumerate).

4. **Is it worth training a separate C-ranker?**  
   **No.** The failure mode is not "PPO-C picked a bad C while another C would succeed"; it is "no C had any feasible R action". A C-ranker cannot create feasibility out of exhausted resources.

## Interpretation and next steps

- The v1.2 + delay-aware PPO-C combination already sits at the feasibility boundary of the S100 low-blocking configuration.
- Further blocking reduction would require changing the resource envelope or admission policy, not the action-selection policy.
- Candidate next directions (outside the current action-space expansion scope):
  - Increase `num_slots` or `num_servers`.
  - Lower the load (`arrival_interval`, `size_max`, `holding_max`).
  - Add admission control / request rejection when no feasible C exists.
  - Shift focus back to the hard 24-slot scenario and investigate capacity-aware C-side risk masks if a larger action envelope is introduced.
