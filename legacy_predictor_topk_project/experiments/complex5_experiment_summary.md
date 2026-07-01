# Complex5 Split Profile Experiment — Summary

## Objective
Implement and evaluate a 5-split DNN complexity profile (`complex5`) to amplify Agent-C's decision space and verify spectrum-aware advantage over rule-based baselines (WO, DF, RF, Greedy, IWD).

## Changes Made

### 1. `sa_hmarl/training/utils.py`
- Added `SPLIT_PROFILES` dictionary defining `default3` (backward compatible) and `complex5` profiles.
- `generate_requests()` now accepts `split_profile` parameter.
- `complex5` design:
  - split0: size=0.25×base, edge_ratio=0.75-0.95 (small data, high edge compute)
  - split1: size=0.45×base, edge_ratio=0.55-0.75
  - split2: size=0.70×base, edge_ratio=0.40-0.60
  - split3: size=1.00×base, edge_ratio=0.25-0.45
  - split4: size=1.35×base, edge_ratio=0.05-0.25 (large data, low edge compute)

### 2. Training & Evaluation Scripts
- `train_agent_c_with_frozen_r.py`: added `--split_profile` CLI arg, passed to `generate_requests`.
- `eval_delayaware_compare.py`: added `--split_profile` CLI arg.
- `eval_offloading_baselines.py`: added `--split_profile` CLI arg.
- `train_joint_alternating.py`: `_build_eval_set()` now passes `split_profile` via `getattr(args, "split_profile", "default3")`.
- **New:** `eval_complex5_offloading.py` — dedicated unified evaluation script for complex5 experiments.

### 3. Training Runs
Two 2000-episode training runs on `snap24_gnutella_reach` (24 slots, 4 servers), warm-started from existing checkpoints:

| Variant | Warm-Start | Objective | Best Val Blocking | Best Episode |
|---------|-----------|-----------|-------------------|--------------|
| Complex5-Default | Original Agent-C (`agent_c_frozen_r_snap24_reach_best.pt`) | default | 0.197 | ep 200 |
| Complex5-DelayAware | Delay-Aware v2 (`agent_c_delayaware_v2_snap24_best.pt`) | blocking_delay (coef=0.5) | 0.000 | ep 100 |

Training completed on GPU (~15-20 min each).

## Evaluation Results

### 24 Slots (Original Capacity)
**Protocol:** 5 seeds × 20 episodes × 60 requests = 6,000 requests per method

| Method | Blocking | Delay (ms) | ObjScore | Split Distribution |
|--------|----------|------------|----------|-------------------|
| Complex5-Default | 0.000±0.000 | 7.7±0.6 | 0.0594±0.0045 | split0=58.6%, split4=20.5% |
| Complex5-DelayAware | 0.000±0.000 | 5.6±0.4 | 0.0434±0.0033 | split0=31.4%, split4=37.6% |
| **DF** | **0.000±0.000** | **5.2±0.4** | **0.0402±0.0032** | split0=29.2%, split4=68.0% |
| Greedy | 0.000±0.000 | 5.5±0.4 | 0.0426±0.0033 | split0=29.9%, split4=68.8% |
| RF | 0.000±0.000 | 6.4±0.5 | 0.0493±0.0035 | split0=30.2%, split4=65.9% |
| IWD | 0.000±0.000 | 6.3±0.5 | 0.0487±0.0042 | split0=36.9%, split4=26.6% |
| WO | 0.000±0.000 | 6.7±0.5 | 0.0518±0.0038 | split0=30.6%, split4=65.4% |

**Key Finding:** All methods achieve **0% blocking**. The expanded action space does not create spectrum scarcity. DF and Greedy outperform both Agent-C variants on delay.

### 24 Slots, 120 Requests/episode (Higher Load)
Same 5 seeds, doubled load.

| Method | Blocking | Delay (ms) | ObjScore |
|--------|----------|------------|----------|
| Complex5-Default | 0.000±0.000 | 8.1±0.7 | 0.0625±0.0055 |
| Complex5-DelayAware | 0.000±0.000 | 5.7±0.4 | 0.0440±0.0028 |
| DF | 0.000±0.000 | 5.4±0.4 | 0.0417±0.0032 |
| Greedy | 0.000±0.000 | 5.5±0.3 | 0.0419±0.0023 |

**Still 0% blocking** even with 2× load. Baselines remain competitive.

### 12 Slots (Reduced Capacity)
Halved spectrum to force scarcity.

| Method | Blocking | Delay (ms) | ObjScore | Split Distribution |
|--------|----------|------------|----------|-------------------|
| Complex5-Default | 0.000±0.000 | 4.8±0.5 | 0.0370±0.0040 | split0=80.0%, split4=10.6% |
| Complex5-DelayAware | 0.000±0.000 | 3.5±0.3 | 0.0268±0.0026 | split0=53.7%, split4=31.4% |
| DF | 0.000±0.000 | 3.5±0.4 | 0.0269±0.0030 | split0=53.0%, split4=44.2% |
| Greedy | 0.000±0.000 | 3.6±0.4 | 0.0277±0.0029 | split0=52.9%, split4=44.9% |

**Still 0% blocking.** All methods adapt by shifting toward split0 (smaller data). Agent-C delay-aware matches DF on objective score but does not clearly dominate.

### Cross-Evaluation: Original 3-Split Agent-C on Complex5 Traffic
Evaluated the original `agent_c_frozen_r_snap24_reach_best.pt` (trained on 3 splits) on complex5 requests (24 slots, 5 eps × 60 req):

| Agent | Blocking | Delay | Splits |
|-------|----------|-------|--------|
| Original-C (3-split trained) | 0.000 | **4.6 ms** | split0=44.3%, split4=37.0% |
| Complex5-C (5-split trained) | 0.000 | 5.7 ms | split0=55.0%, split4=31.0% |

**The original 3-split Agent-C generalizes to 5-split traffic and achieves LOWER delay than the complex5-trained Agent-C.**

## Root Cause Analysis: Why No Spectrum-Aware Advantage?

1. **Low-Edge-Compute Splits Are "Too Good"**
   - Complex5 split4 has edge_ratio=0.05-0.25, which is even lower than original split2's 0.1-0.30.
   - This makes server overload extremely rare, even for simple baselines.

2. **Action Mask Filters Out Infeasible Options**
   - The `agent_c_mask` already removes split/server combinations that would fail spectrum allocation.
   - Baselines operate only on valid actions, so they naturally avoid blocking.

3. **Network Has Sufficient Headroom**
   - Even with 12 slots and 120 requests, the low-data splits (split0: 0.25× base_size) provide enough "escape hatches" to prevent blocking.

4. **Monotonic Trade-off Is Too Clean**
   - The size↔edge_ratio relationship is perfectly monotonic, making the optimal choice obvious: pick the lowest-edge-compute split that still fits in spectrum.
   - Greedy baseline (min edge_compute_ms) essentially implements this rule.

## Comparison: 3-Split vs 5-Split

| Metric | 3-Split (Original) | 5-Split (Complex5) |
|--------|-------------------|-------------------|
| Blocking | ~0.215 | 0.000 |
| Best Baseline | DF-C (Obj=0.275) | DF (Obj=0.040) |
| Agent-C Advantage | Yes (beats all baselines) | No (baselines match/beats) |

## Recommendations for Future Work

To create a scenario where the expanded action space genuinely helps:

1. **Increase Compute Demands**
   - Raise `edge_cost_max` from 15.0 to 25.0+ so that even split4 causes occasional server overload.
   - OR reduce server capacities.

2. **Tighten the Trade-off**
   - Make intermediate splits (split1-split3) more attractive by giving them better delay/FS ratios.
   - Current profile makes extremes (split0, split4) too dominant.

3. **Add Non-Monotonic Options**
   - Introduce splits where size and edge_ratio don't move in perfect lockstep.
   - Example: a split with medium data but very high edge compute (bad at both) — forces the agent to learn which "medium" option is actually best.

4. **Increase Data Size Range**
   - Make split4 size = 2.0× base or higher to force genuine spectrum pressure.
   - Currently 1.35× is not large enough to exhaust 12-24 slots.

5. **Joint C+R Fine-Tuning**
   - The frozen R backend may be suboptimal for complex5 splits.
   - Fine-tuning R jointly with C could unlock better spectrum utilization.

## Checkpoints Generated
- `sa_hmarl/checkpoints/agent_c_complex5_default_best.pt`
- `sa_hmarl/checkpoints/agent_c_complex5_delayaware_best.pt`

## Files Modified/Created
- Modified: `sa_hmarl/training/utils.py`, `train_agent_c_with_frozen_r.py`, `eval_delayaware_compare.py`, `eval_offloading_baselines.py`, `train_joint_alternating.py`
- Created: `sa_hmarl/evaluation/eval_complex5_offloading.py`
- Results: `experiments/complex5_offloading_results*.json/md`, `experiments/complex5_experiment_summary.md`
