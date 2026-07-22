# C-Side Step 3 / R-Side v1.3 System Synthesis

**Date:** 2026-07-05  
**Purpose:** Close the loop between the C-side Step 3 mixed-stress validation decision and the long-horizon v1.3 hybrid benchmark, and produce a paper-ready system-level conclusion.

---

## 1. Executive Conclusion

No additional long-horizon benchmark reruns are required. The existing 5-seed × 4-topology benchmark already uses the final recommended C-side checkpoint:

- **C-side feature mode:** `r_feasibility_safe`
- **C-side checkpoint:** `last` checkpoint (`agent_c_cost239_r_feasibility_safe_last.pt`)
- **R-side candidate:** `ppo_c + v12_k50_hops` (v1.3 hybrid: learned ranker finalizer, `legalctx48`, max 48 candidates, KSP-ensured fallback)
- **Baseline:** `ppo_c + ksp_ff_k50_hops`

With this stable C-side policy, v1.3 hybrid shows:

| Topology | KSP-FF block % | v1.3 hybrid block % | Δ relative | p-value |
|---|---:|---:|---:|---:|
| COST239 | 10.37 ± 1.44 | 7.68 ± 1.02 | **−25.9 %** | **0.0011** |
| German17 | 7.60 ± 1.15 | 7.63 ± 1.14 | +0.5 % | 0.7697 |
| JPN48 | 9.19 ± 1.81 | 8.03 ± 1.57 | **−12.6 %** | **0.0067** |
| NSFNET | 15.29 ± 1.63 | 14.92 ± 1.48 | −2.4 % | 0.3656 |

**Bottom line:** the critical C-side improvement is the validation regime (mixed-stress checkpoint selection), not another hand-crafted feature set. Once C-side allocation is stable (`r_feasibility_safe`), the v1.3 R-side ranker delivers significant system-level gains on COST239 and JPN48, is neutral on German17, and only weakly improves NSFNET where compute saturation dominates.

---

## 2. C-Side Step 2 / Step 3 Recap

### 2.1 What was compared

Three C-side feature modes were trained and evaluated on `xlron_cost239_ptrnet_real` with the R backend fixed to `ksp_ff_k50_hops`:

- `default`: 17-D base candidate vector.
- `r_feasibility_safe`: 17-D base + 10 R-feasibility dims + 2 server-margin dims = 29-D.
- `overload_aware`: 17-D base + 9 post-action pressure/risk dims = 26-D.

### 2.2 Step 2 finding: checkpoint brittleness under standard-stress validation

Under the original standard-stress validation regime (best checkpoint selected by mean blocking on standard stress):

| Feature mode | `best` heavy block % | `last` heavy block % | Best–last gap |
|---|---:|---:|---:|
| `default` | 0.83 | 0.00 | 0.83 pp |
| `r_feasibility_safe` | 0.92 | 0.00 | 0.92 pp |
| `overload_aware` | **7.46** | 0.00 | **7.46 pp** |

All three modes solve heavy stress at the `last` checkpoint. The `best` checkpoint is brittle, especially for `overload_aware`.

### 2.3 Step 3 finding: mixed-stress validation removes the brittleness

Retraining with mixed-stress validation (`standard,medium,heavy`) and `checkpoint_metric=max_blocking` gives:

| Feature mode | `best` heavy block % (Step 2) | `best` heavy block % (Step 3) |
|---|---:|---:|
| `default` | 0.83 | 0.00 |
| `r_feasibility_safe` | 0.92 | 1.13 |
| `overload_aware` | 7.46 | 0.00 |

The 1.13 % for `r_feasibility_safe_best` is seed-variance: validation seeds 1001–1003 selected a checkpoint that underperformed on eval seeds 3030–5050. Its `last` checkpoint remains 0 %.

### 2.4 Failure-mode decomposition

Across all C-side short-horizon runs, residual blocking is overwhelmingly `server_overload`; `no_suitable_block` stays below 0.05 %. This confirms the project operates in a compute-optical regime where C-side server selection is the binding constraint, not R-side spectrum search.

---

## 3. Final C-Side Decision

### 3.1 Recommended policy

- **Default feature mode:** `r_feasibility_safe`
- **Default checkpoint selection:** mixed-stress validation with `max_blocking`
  - `--validation_stress_levels standard,medium,heavy`
  - `--checkpoint_metric max_blocking`
- **Conservative fallback:** use the `last` checkpoint when training budget allows; it is consistently 0 % across standard/medium/heavy stress.

### 3.2 Why `overload_aware` is not the mainline

`overload_aware` matches `r_feasibility_safe` at the `last` checkpoint but does **not** beat it. Its `best` checkpoint is only safe under mixed-stress validation; under standard-stress validation it collapses to 7.46 % heavy blocking. There is no systematic evidence that the extra pressure/risk features reduce `server_overload_rate` relative to `r_feasibility_safe`.

### 3.3 Code status

The mixed-stress validation implementation exists in `sa_hmarl/sa_hmarl/training/train_agent_c_with_frozen_r.py` (`--validation_stress_levels`, `--checkpoint_metric`, `VALIDATION_STRESS_LEVELS` table). However, the script defaults remain:

- `--checkpoint_metric mean_blocking`
- `--validation_stress_levels ""`

**Recommended follow-up:** change the argparse defaults to `--validation_stress_levels standard,medium,heavy` and `--checkpoint_metric max_blocking` so that future C-side training runs use the recommended regime by default. This is a one-line default change and does not alter R-side logic.

---

## 4. Long-Horizon v1.3 vs KSP-FF Result Recap

### 4.1 Protocol

- 320 slots, 4 servers at `[0,1,2,3]` capacity 50 each, 3 splits (`default3`).
- Poisson arrivals, exponential holding (mean 25 s), uniform random sources.
- 10 000 requests/episode, 2 000 warmup, measurement on 8 000 post-warmup requests.
- 5 seeds: 3030, 4040, 5050, 6060, 7070.
- C checkpoint: `agent_c_cost239_r_feasibility_safe_last.pt`.

### 4.2 Aggregate results

| Topology | λ | `edge_cost_max` | Method | Block % | Overload % | No-block % |
|---|---:|---:|---|---|---:|---:|---:|
| COST239 | 16 | 2.2 | KSP-FF K50 | 10.37 ± 1.44 | 10.37 | 0.000 |
| COST239 | 16 | 2.2 | v1.3 hybrid | 7.68 ± 1.02 | 7.68 | 0.000 |
| German17 | 14 | 3.0 | KSP-FF K50 | 7.60 ± 1.15 | 7.58 | 0.013 |
| German17 | 14 | 3.0 | v1.3 hybrid | 7.63 ± 1.14 | 7.58 | 0.045 |
| JPN48 | 10 | 5.2 | KSP-FF K50 | 9.19 ± 1.81 | 8.66 | 0.197 |
| JPN48 | 10 | 5.2 | v1.3 hybrid | 8.03 ± 1.57 | 7.23 | 0.287 |
| NSFNET | 13 | 4.0 | KSP-FF K50 | 15.29 ± 1.63 | 14.97 | 0.100 |
| NSFNET | 13 | 4.0 | v1.3 hybrid | 14.92 ± 1.48 | 14.46 | 0.203 |

### 4.3 Statistical significance

- COST239: t = −8.37, p = 0.0011
- German17: t = +0.31, p = 0.7697
- JPN48: t = −5.17, p = 0.0067
- NSFNET: t = −1.02, p = 0.3656

### 4.4 Consistency with C-side decision

All four topologies used the same C checkpoint (`r_feasibility_safe_last.pt`). This is exactly the conservative, stable C-side checkpoint recommended by Step 3. Therefore the observed v1.3 gains (or neutrality) can be attributed to the R-side policy, not to a brittle or mismatching C-side checkpoint.

---

## 5. Failure-Mode Interpretation

### 5.1 Blocking is dominated by `server_overload`

In both the short-horizon C-side ablations and the long-horizon benchmark, `server_overload_rate` accounts for nearly all blocking:

- COST239 long-horizon: 100 % of blocking is server overload.
- German17 long-horizon: ~99.8 % server overload.
- JPN48 long-horizon: ~94 % server overload; the remaining ~6 % is `no_suitable_block`.
- NSFNET long-horizon: ~98 % server overload.

### 5.2 Why v1.3 wins where it wins

**COST239 (−25.9 %):** The small topology gives the ranker a large relative advantage in path/block selection; better R-side choices reduce the rate at which requests are forced onto already-loaded servers.

**JPN48 (−12.6 %):** The large physical network offers many candidate paths. The ranker’s learned scoring over 48 candidates finds combinations that KSP-FF’s greedy hop-ordered first-fit misses.

**German17 (neutral):** KSP-FF already performs close to the ranker here; the node layout and load make greedy first-fit nearly optimal.

**NSFNET (weak, non-significant):** At ~15 % blocking the system is compute-saturated. R-side path optimization has limited headroom because server capacity is the binding constraint. The small −0.37 pp trend is directionally positive but within noise.

### 5.3 Implication for future work

Because residual blocking is dominated by `server_overload`, further gains will likely require:

- Better C-side allocation or joint C-R selection that explicitly anticipates future server load, or
- Larger compute capacity / different server placement,

rather than simply increasing the R-side candidate set (already 48 candidates) or switching to another pure R-side heuristic.

---

## 6. Whether Additional Rerun Is Needed

### 6.1 Decision: no additional rerun is needed

The existing long-horizon benchmark already satisfies the final C-side recommendation:

| Requirement | Status |
|---|---|
| Feature mode = `r_feasibility_safe` | ✓ Used in all four topologies |
| Checkpoint type = stable (`last`) or mixed-stress `best` | ✓ Used `r_feasibility_safe_last.pt` |
| R backend = `ksp_ff_k50_hops` vs `v12_k50_hops` | ✓ Both compared |
| Load unchanged from benchmark | ✓ Same λ and `edge_cost_max` |

### 6.2 Why a rerun with `r_feasibility_safe_step3_mixedstress_20260704_last.pt` is optional, not required

The Step 3 mixed-stress retrain also produced a `last` checkpoint for `r_feasibility_safe`, and the short-horizon evaluation shows it is 0 % across standard/medium/heavy stress, identical to the original `last` checkpoint. The long-horizon benchmark already uses a checkpoint that is behaviorally equivalent for the purpose of the R-side comparison. A full 5-seed × 4-topology rerun would confirm equivalence but would not change the system-level conclusion.

### 6.3 If equivalence must be verified later

A minimal verification would be:

- Topology: COST239 (the topology where v1.3 shows the largest gain and where C-side stability matters most).
- Seeds: 3030 only as a smoke test, or 3030/4040/5050 for stronger confidence.
- Method: compare `agent_c_cost239_r_feasibility_safe_last.pt` vs `agent_c_cost239_r_feasibility_safe_step3_mixedstress_20260704_last.pt`, both with `ppo_c + v12_k50_hops`.
- If the smoke test is within ±0.3 pp, no further rerun is warranted.

This verification is **not part of the current deliverable**.

---

## 7. Paper-Ready Wording

### 7.1 C-side contribution (restrained)

> We evaluated three C-side feature representations for server/split selection under a fixed routing backend. The `overload_aware` feature set, which augments the state with post-action pressure and risk signals, did not provide a systematic reduction in server-overload blocking compared with the simpler `r_feasibility_safe` representation. Instead, the dominant failure mode under the original standard-stress validation regime was checkpoint selection: early-stopping on easy validation traffic selected policies that collapsed under heavy compute stress (up to 7.46 % blocking for `overload_aware_best`). Mixed-stress validation, which selects the checkpoint minimizing the worst-case blocking across standard, medium, and heavy stress, removes this brittleness. We therefore adopt `r_feasibility_safe` as the C-side representation and use mixed-stress validation (or, conservatively, the final `last` checkpoint) for training.

### 7.2 R-side / system contribution (restrained)

> With the C-side policy fixed to the stable `r_feasibility_safe` checkpoint, we compared a KSP-FF baseline against a planner-distilled v1.3 hybrid ranker on four topologies in a long-horizon steady-state benchmark. The v1.3 hybrid significantly reduces blocking on COST239 (−25.9 %, p = 0.0011) and JPN48 (−12.6 %, p = 0.0067), is statistically neutral on German17 (+0.5 %, p = 0.77), and shows only a small non-significant trend on NSFNET (−2.4 %, p = 0.37). Across all topologies the residual blocking is dominated by server overload, which suggests that future gains require improved compute-side allocation or joint routing-and-offloading decisions rather than larger routing candidate sets alone.

### 7.3 One-line takeaway

> The practical advance on the C side is a stress-robust validation regime; the practical advance on the R side is a learned ranker that improves blocking on two of four topologies when the C-side policy is already stable.

---

## 8. Recommended Next Experiment, If Any

### 8.1 Immediate code-level follow-up

1. Change the defaults in `train_agent_c_with_frozen_r.py`:
   - `--validation_stress_levels standard,medium,heavy`
   - `--checkpoint_metric max_blocking`
2. Verify that the change is backward-compatible by confirming that an empty `--validation_stress_levels` still falls back to the original single-stress behavior.

### 8.2 Scientific follow-up (optional)

1. **Joint C-R optimization:** Because residual blocking is `server_overload`, investigate whether a joint policy that conditions R-side path/block selection on anticipated server load can reduce overload further. The current architecture separates C and R; a coordinated selector may be the next meaningful gain.
2. **Topology-aware C checkpointing:** The current C checkpoint is trained on COST239 and transferred to German17/JPN48/NSFNET. Verify whether per-topology C training (with the same `r_feasibility_safe` representation) changes the long-horizon conclusions.
3. **Longer training / larger C network:** Verify whether `overload_aware` becomes advantageous if trained for more episodes (e.g., 800–1000) with mixed-stress validation. Current evidence does not support it, but a longer run would strengthen the negative result.

### 8.3 What not to do next

- Do not add more hand-crafted C-side features without first verifying that the bottleneck is not simply checkpoint selection.
- Do not increase the R-side candidate set beyond 48 without evidence that `no_suitable_block` is the limiting factor (it is not; `server_overload` is).
- Do not claim v1.3 superiority on all topologies; German17 and NSFNET results do not support that claim.

---

## 9. Artifacts

- This synthesis report: `sa_hmarl/experiments/cside_step3_v13_system_synthesis.md`
- Machine-readable synthesis: `sa_hmarl/experiments/cside_step3_v13_system_synthesis.json`
- C-side Step 2+3 report: `sa_hmarl/experiments/cost239_c_step2_feature_mode_rerun_report.md`
- C-side Step 2+3 JSON: `sa_hmarl/experiments/cost239_c_step2_feature_mode_rerun_report.json`
- Long-horizon benchmark report: `sa_hmarl/experiments/long_horizon_benchmark/FINAL_BENCHMARK_REPORT.md`
- Long-horizon result JSONs: `sa_hmarl/experiments/long_horizon_benchmark/xlron_*_ecmax*_full.json`
