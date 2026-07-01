# R Slot-Position Fragmentation Oracle Findings

## Question

Can we further widen the gap over DeepRMSA by teaching the v1.2 R-ranker to choose better spectrum positions, not just shorter paths and fewer FS?

The tested hypothesis:

> For the same C decision and the same `(path, modulation, required_fs)`, different legal spectrum blocks may have different long-term fragmentation impact.

If this signal is common and predictive, a v1.4 fragmentation-aware slot-position ranker would be worth training.

## Diagnostic Design

For each request:

1. Freeze Agent-C and select `(split, server)`.
2. Build the R action mask.
3. Group legal R actions by `(path_idx, mod_idx, required_fs)`.
4. Keep only groups with at least two different legal blocks.
5. For each block action, simulate current allocation and roll out `H=5` future requests.
6. Measure whether different block choices produce different future blocking / NSB outcomes.

This isolates spectrum position from path length, modulation, and FS demand.

## Important Environment Detail

The current R action does **not** choose an arbitrary start slot inside a large free block.

It chooses a candidate free block, and the environment allocates at `block.start_slot`. Therefore, the current action space mostly supports left-edge placement within each free block. It does not directly support center placement or arbitrary "cut a large block into two" placement.

This matters because the proposed fragmentation-position mechanism has less room than it would in a richer slot-start action space.

## S80 Medium-Low Blocking Scenario

- Slots: 80
- Workload: `default3`, arrival interval `0.15s`, holding `4-10s`, size `5-40MB`
- Seeds: `3030,4040,5050`
- Episodes: `3`

| Metric | Value |
|---|---:|
| Total states | 720 |
| Multi-action states | 666 / 720 = 92.50% |
| States with same path/mod/fs and multiple blocks | 266 / 720 = 36.94% |
| Evaluated position groups | 2004 |
| Evaluated actions | 5775 |
| Nonzero return range group rate | 1.75% |
| Future-block difference group rate | 1.75% |
| Future-NSB difference group rate | 0.00% |
| Mean return range | 0.0140 |
| P95 return range | 0.0000 |
| v1.2 best rate when selected action in group | 30.36% |
| DeepRMSA best rate when selected action in group | 37.56% |

Feature correlations with H-step return:

| Feature | Spearman |
|---|---:|
| `lfb_after` | 0.263 |
| `block_size` | 0.195 |
| `right_free_norm` | 0.189 |
| `block_consumed_ratio` | -0.187 |
| `free_block_delta` | 0.162 |
| `block_idx` | 0.107 |
| `frag_proxy_after` | 0.085 |
| `start_norm` | 0.061 |
| `lfb_drop` | 0.044 |

Interpretation:

- Slot-position features have some signal.
- However, groups where position actually changes future blocking are too rare.
- Most same path/mod/fs block choices produce identical H-step outcomes.

## S24 High-Pressure Scenario

- Slots: 24
- Workload: `default3`, arrival interval `0.09s`, holding `4-14s`, size `5-30MB`
- Seeds: `3030,4040`
- Episodes: `2`

| Metric | Value |
|---|---:|
| Total states | 320 |
| Multi-action states | 135 / 320 = 42.19% |
| States with same path/mod/fs and multiple blocks | 0 |
| Evaluated position groups | 0 |

Interpretation:

- Under high pressure, there is almost no same-path same-mod same-FS block-position choice left.
- R-side position optimization cannot help when the action space has already collapsed to zero or one block per path/mod.

## Verdict

**STOP_POSITION_FEATURES.**

The fragmentation-aware position idea is physically sensible, but the current action abstraction and workloads do not expose enough usable headroom:

1. In S80, position alternatives exist but rarely change future blocking.
2. In S24 high pressure, position alternatives largely disappear.
3. The environment allocates at the left edge of candidate blocks, so true arbitrary slot-start placement is not available.

## Recommendation

Do not train v1.4 by only adding `left_free/right_free/lfb_after` features to the current R action space.

If we want to pursue this mechanism, the action space must be expanded first:

- allow multiple placement offsets inside a large free block, such as left-fit, right-fit, center-fit;
- or expose explicit start-slot candidates instead of only block candidates;
- then rerun this oracle diagnostic.

Without that action-space expansion, v1.2 remains the strongest R-side method.
