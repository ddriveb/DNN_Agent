# Multi-Topology Extension Plan / Blocker Assessment

## Status
**NOT STARTED** — extension is gated until the COST239 pilot passes and topology-matched checkpoints exist.

## Required topologies
1. COST239 (`xlron_cost239_ptrnet_real`)
2. Japan standard topology (`xlron_jpn48`)
3. German/Germany standard topology (`xlron_german17`)

## Required per-topology assets
For a fair comparison, each topology must use:
- Topology-matched environment (already available).
- Topology-matched PPO-C checkpoint.
- Topology-matched PPO-R checkpoint.
- (Optional) DF-C / heuristic C baseline implementation.

## Current asset inventory

| Topology | Env | PPO-C checkpoint | PPO-R checkpoint | DF-C/heuristic C |
|---|---|---|---|---|
| COST239 | ✅ | ✅ `agent_c_cost239_r_feasibility_safe_last.pt` | ⚠️ shared `agent_r_mixed.pt` | ✅ via `eval_strict_v13_multitopology_cside_fair.py` |
| JPN48 | ✅ | ❌ only generic `agent_c_delayaware_v2_snap24_best.pt` | ❌ only shared `agent_r_mixed.pt` | ✅ |
| German17 | ✅ | ❌ only generic `agent_c_delayaware_v2_snap24_best.pt` | ❌ only shared `agent_r_mixed.pt` | ✅ |

## Blockers
1. **No topology-matched PPO-R checkpoints for JPN48 / German17.** The repository only contains `agent_r_mixed.pt` (trained on `nsfnet`, 32 slots) and `agent_r_multitopo.pt`. Using `agent_r_mixed.pt` directly on JPN48/German17 and calling it a fair comparison would violate the strict protocol.
2. **No topology-matched PPO-C checkpoints for JPN48 / German17.** The available `agent_c_delayaware_v2_snap24_best.pt` is a generic snap24 checkpoint, not trained on JPN48 or German17 traffic/link structures.

## Next steps (gated)
Before running multi-topology experiments:
1. Train dedicated PPO-C and PPO-R checkpoints for JPN48 and German17 under the same v1.3 protocol (K_C=5, K_path=50, frozen continuation).
2. Re-train or verify `agent_r_multitopo.pt` as a legitimate topology-matched PPO-R alternative (only if it was trained with all three topologies represented).
3. Re-run the COST239 pilot gate: Full-state or Gated offline regret must be stable better than PPO-R across ≥3 seeds.
4. Run short pilots on JPN48 and German17 with the same C-side baselines (topology-matched PPO-C, DF-C/heuristic C) and R-side methods (PPO-R, Current v1.3, Gated v1.3, Full v1.3 pilot).
5. Each `topology × C-policy × method` uses ≥5 common evaluation seeds and identical request traces.

## Decision
**Multi-topology full experiments are blocked.** The COST239 pilot can proceed, but Japan/German results must not be reported as fair comparisons until topology-matched checkpoints are available.
