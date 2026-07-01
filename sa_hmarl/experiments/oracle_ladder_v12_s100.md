# Oracle Ladder Phase 1 — v1.2 Residual Headroom (S100)

- Total states: 1200
- Baseline v1.2 blocked: 8 (0.667%)
- Oracle evaluated on blocked states: 8

## Blocking and future block rates

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

## Failure reason distribution (baseline blocked states)

| Reason | Count | Share |
|---|---:|---:|
| no_valid_c_action | 8 | 100.00% |

## Verdict: STOP_ACTION_SPACE_EXPANSION
- Direction: none