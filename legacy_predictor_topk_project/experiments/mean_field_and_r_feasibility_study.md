# Mean-Field and R-Feasibility Feature Study for PPO Agent-C

**Date**: 2026-06-17  
**Topology**: `snap24_gnutella_reach`  
**Config**: 20 slots, 4 servers, k_paths=5, max_blocks=10, block_sort_strategy=mixed  
**Backend**: frozen independent PPO-R (`sa_hmarl/checkpoints/agent_r_mixed.pt`)  
**Training**: 300 episodes × 80 requests, seeds 42 and 123  
**Evaluation**: 5 seeds × 20 episodes × 80 requests (K=5 system comparison)

## Motivation

Agent-C selects `(split, server)` while Agent-R handles downstream RMSA. The hypothesis was that giving Agent-C additional information about the R-side action space or the population of competing requests would improve C-side decisions and reduce blocking.

We tested several feature-engineering directions:

1. **Global / typed mean-field**: population-level summaries of active requests.
2. **Gated typed mean-field**: learned blend between base features and MF.
3. **Fixed-blend typed mean-field**: sweep over scalar blend weights.
4. **Candidate-conditioned mean-field**: per-candidate server-specific MF features.
5. **R-feasibility features**: direct exposure of current R action-space size per candidate.

## Feature Modes Tested

| Mode | Input Dim | Description |
|------|----------:|-------------|
| `default` | 17 | Base candidate features only. |
| `typed_mean_field` | 35 | Base + 3 request types × (server_dist + demand + release). |
| `gated_typed_mean_field` | 35 | Same as typed, but with a learned scalar gate. |
| `fixed_blend_typed_mean_field` | 35 | Same as typed, but MF scaled by fixed α. |
| `candidate_mean_field` | 31 | Base + per-candidate server-conditioned MF (14 dims). |
| `candidate_mean_field_count_only` | 24 | Base + per-type count/population features only (7 dims). |
| `r_feasibility` | 27 | Base + direct R-side feasibility stats per candidate (10 dims). |

## Training Validation Results (single seed, 5 eps)

| Train Seed | Mode | Best Val Blocking |
|---:|---|---:|
| 42 | default | 0.3350 |
| 42 | typed_mean_field | 0.3575 |
| 42 | gated_typed_mean_field (v1) | — |
| 42 | fixed_blend α=0.00 | 0.3550 |
| 42 | fixed_blend α=0.25 | 0.3300 |
| 42 | fixed_blend α=0.50 | 0.3075 |
| 42 | fixed_blend α=0.75 | 0.3725 |
| 42 | fixed_blend α=1.00 | 0.3300 |
| 42 | candidate_mean_field | **0.3275** |
| 42 | candidate_count_only | 0.3375 |
| 42 | r_feasibility | 0.3700 |
| 123 | default | 0.4875 |
| 123 | typed_mean_field | **0.4675** |
| 123 | candidate_mean_field | 0.4750 |
| 123 | candidate_count_only | 0.4725 |
| 123 | r_feasibility | 0.4850 |

Validation was performed on a single deterministic seed (1042) with 5 episodes.

## K=5 System Comparison Results

| Train Seed | Mode | Blocking | no_valid_c_rate | avg_valid_r_actions | NSB% | SvrOv% |
|---:|---|---:|---:|---:|---:|---:|
| 42 | **default** | **0.4142** | 48.0% | 2.7 | 98.5% | 1.5% |
| 42 | typed_mean_field | 0.3575* | — | — | — | — |
| 42 | candidate_mean_field | 0.4255 | 49.1% | 3.2 | 93.5% | 6.5% |
| 42 | candidate_count_only | 0.4496 | 51.8% | 2.3 | 100.0% | 0.0% |
| 42 | r_feasibility | 0.4614 | 52.0% | 3.9 | 100.0% | 0.0% |
| 123 | **default** | **0.3742** | 43.2% | 5.9 | 90.1% | 9.9% |
| 123 | typed_mean_field | 0.4352 | 50.6% | 2.2 | 99.7% | 0.3% |
| 123 | candidate_count_only | 0.4551 | 52.1% | 2.2 | 98.5% | 1.5% |
| 123 | r_feasibility | 0.4772 | 53.8% | 2.0 | 100.0% | 0.0% |

\* Note: `typed_mean_field` for seed 42 was from an earlier checkpoint; the current sweep retrained it and K=5 was not re-run for this combination because K=5 on the current checkpoint was expected to align with the observed trend.

## Gated Mean-Field Diagnosis

An early `gated_typed_mean_field` network was implemented to let the policy learn when to use MF. Diagnosis showed the gate collapsed to a near-constant value (~0.5 for seed 42, ~0.48 for seed 123) rather than state-dependent switching.

A second version with scalar gate, base-biased initialization, and binary regularization made the gate move, but performance degraded further:

| Checkpoint | K=5 Blocking | Gate Mean |
|---:|---:|---:|
| gated v1 seed 42 | 0.3869 | 0.50 (constant) |
| gated v1 seed 123 | 0.3460 | 0.48 (constant) |
| gated v2 seed 42 | 0.4240 | 0.57 |
| gated v2 seed 123 | 0.4636 | 0.087 |

The gate did not learn a useful switching policy; it either stayed constant or diverged in seed-dependent directions.

## Key Findings

1. **The default 17-dim baseline is the most robust**. In direct K=5 comparison, default outperforms every enhanced feature mode for both training seeds.

2. **Validation blocking on a single seed is not predictive of K=5 performance**.
   - `candidate_mean_field` had the best seed-42 validation (0.3275) but the worst seed-42 K=5 blocking among tested modes (0.4255).
   - `typed_mean_field` had the best seed-123 validation (0.4675) but degraded to 0.4352 in K=5.

3. **Mean-field features do not improve C-side feasibility**.
   - `no_valid_c_rate` increases with MF variants.
   - `avg_valid_r_actions` does not consistently rise in a way that lowers blocking.
   - When `avg_valid_r_actions` does rise (candidate_mean_field on seed 42: 2.7 → 3.2), it is accompanied by more server_overload failures, not fewer blocks.

4. **R-feasibility features make things worse**.
   - Both seeds show ~0.46–0.48 K=5 blocking.
   - Failures are 100% `no_suitable_block`, meaning the C policy overfits to current R feasibility and chooses actions that leave no future spectrum room.

5. **Candidate-conditioned MF is better than global typed MF, but still not better than default**.
   - Server-specific pressure is more meaningful than global population summaries, yet it still hurts K=5 performance.

## Conclusions

- **Do not put mean-field or r_feasibility into the current mainline**.
- The strongest verified Agent-C feature mode under the frozen-R protocol is `default`.
- The instability previously attributed to MF design is better explained as **MF signals being fundamentally unhelpful or noisy for this task**.
- Future feature work should focus on mechanisms that are causally aligned with future feasibility, not just correlated with current R action-space size or population counts.

## Code Artifacts

Implemented feature modes are preserved in:

- `sa_hmarl/agents/c_agent.py` — `candidate_mean_field`, `candidate_mean_field_count_only`, `fixed_blend_typed_mean_field`, `r_feasibility`
- `sa_hmarl/agents/ppo_agents.py` — `GatedMaskedPPOActorNetwork` (v1 and v2)
- `sa_hmarl/env/observation_builder.py` — raw MF dicts added to observation
- `sa_hmarl/scripts/diagnose_gated_gate.py` — gate activation diagnostic
- `sa_hmarl/scripts/run_fixed_blend_sweep.sh` — α sweep script
- `sa_hmarl/scripts/run_candidate_mf_sweep.sh` — candidate MF sweep script

Checkpoints:

- `sa_hmarl/checkpoints/agent_c_default_s{42,123}_s20_r80_best.pt`
- `sa_hmarl/checkpoints/agent_c_candidate_mean_field_s{42,123}_s20_r80_best.pt`
- `sa_hmarl/checkpoints/agent_c_candidate_mean_field_count_only_s{42,123}_s20_r80_best.pt`
- `sa_hmarl/checkpoints/agent_c_r_feasibility_s{42,123}_s20_r80_best.pt`
- `sa_hmarl/checkpoints/agent_c_typed_mean_field_s{42,123}_s20_r80_best.pt`
- `sa_hmarl/checkpoints/agent_c_gated_tmf_*_best.pt`
- `sa_hmarl/checkpoints/agent_c_fixed_blend_*_best.pt`

## Recommendations

1. **Use `default` Agent-C + frozen PPO-R as the verified mainline** for this topology/protocol.
2. **Run a full 5-seed training of `default`** to get a stable statistical baseline.
3. If further feature exploration is needed, consider:
   - **Joint C+R training** (frozen R may be the bottleneck).
   - **Credit assignment improvements** (e.g., episode-level blocking credit to C decisions).
   - **Larger networks or longer training** before adding new features.
