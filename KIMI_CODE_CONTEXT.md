# Kimi Code Context

## 2026-07-20 Multi-topology heuristic comparison (current)

- Task: Strict v1.3 vs KSP-FF/FF-KSP on NSFNET/USNET/JPN48 fixed-C/all-OD pure RMSA 5-seed pilot.
- Script: `sa_hmarl/sa_hmarl/evaluation/generate_multitopology_strict_v13_vs_heuristics.py`
- Output dir: `sa_hmarl/experiments/strict_v13_vs_topology_best_heuristics_fixed_c_all_od/`
- Seeds: 6101,6102,6103,6104,6105
- Topology order: xlron_nsfnet_deeprmsa, xlron_usnet_gcnrmsa, xlron_jpn48
- Background task: bash-9y80d4ci, started 2026-07-20 13:42Z
- Log: `.../run_full.log`
- Status: 15 tasks queued (14 remaining after smoke seed 6101 NSFNET already done)


## Current Task

COST239 fixed-C / all-OD pure RMSA three-method fair comparison (5-seed pilot).

- **Output directory target**: `sa_hmarl/experiments/strict_v13_vs_ksp_ff_vs_deeprmsa_cost239_fixed_c_all_od/`
- **Seeds**: 5001, 5002, 5003, 5004, 5005 (evaluation pilot only; do not scale to 20 seeds without explicit confirmation)

## Protocol Lock

This is **fixed-C / all-OD pure RMSA**. PPO-C and DF_C are not invoked. Use fixed `split_id=0` and deterministic server mapping by `dst_node` (server_node_ids = [0,1,2,3]).

## Source-of-Truth Configuration

The previous diagnosis that produced **5.6342% blocking** is in:
- `sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_diagnosis/DIAGNOSIS.json`
- `sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_diagnosis/DIAGNOSIS.md`
- `sa_hmarl/sa_hmarl/evaluation/diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od.py`

Required config to reuse:
- topology: `xlron_cost239_ptrnet_real`
- num_slots: 320
- num_servers: 4
- server_node_ids: [0, 1, 2, 3]
- fixed_split_id: 0
- k_paths_r: 50
- path_sort_strategy_r: `hops`
- block_sort_strategy_r: `start_asc`
- max_blocks: 10
- modulation_profile: `default`
- arrival_interval: 0.3
- holding_min: 20.0, holding_max: 30.0
- deadline_min: 30.0, deadline_max: 100.0
- size_min_mb: 5.0, size_max_mb: 30.0
- edge_cost_min: 0.1, edge_cost_max: 2.2
- split_profile: `default3`
- num_splits: 3
- warmup_requests: 500
- requests_per_episode: 6000
- poisson_arrivals: false
- exponential_holding: false
- traffic: uniform all-OD (src from all nodes, dst from server nodes)

Use `_generate_all_od_requests` from `diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od.py`.

## Methods to Compare

1. **Strict v1.3**
   - PPO-R checkpoint: `sa_hmarl/checkpoints/agent_r_mixed.pt`
   - Ranker checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt`
   - Candidate pool: PPO-R legal Top-30 only
   - No KSP anchor / filler / diversity / all-legal candidate

2. **KSP-FF K=50 hops**
   - Use `ksp_ff_highest_mod_action` from `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py`
   - K_path=50, hops order, tie-break by km, start_asc, distance-adaptive highest modulation
   - Do NOT use `ksp_ff_action`

3. **Topology-matched adapted DeepRMSA K=50 hops**
   - Main engineering task. Create a new adapted agent class (suggested: `sa_hmarl/sa_hmarl/agents/deep_rmsa_adapted_k50_agent.py`) and training script (suggested: `sa_hmarl/sa_hmarl/training/train_deep_rmsa_adapted_k50.py`).
   - Must use DeepRMSA's 5-layer 128-unit ELU MLP backbone and A2C algorithm.
   - Output head must cover the unified 2000-action space: 50 paths × 4 modulations × 10 blocks.
   - Flat action encoding: `a = p * (|M| * B) + m * B + b`.
   - Apply the same physical R mask as PPO-R (do not select illegal actions at evaluation).
   - Train from scratch on the SA-HMARL COST239 fixed-C/all-OD environment.
   - Do not load existing snap24/NSFNET/Germany/Japan checkpoints.
   - Checkpoint selected by validation blocking rate.

## Key Files Already Read

- `docs/STRICT_V13_KSP50_ALL_OD_PROMPT_LOCK.md`
- `sa_hmarl/sa_hmarl/evaluation/diagnose_strict_v13_vs_ksp_ff_k50_hops_all_od.py`
- `sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_diagnosis/DIAGNOSIS.json`
- `sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_diagnosis/DIAGNOSIS.md`
- `sa_hmarl/sa_hmarl/agents/deep_rmsa_agent.py`
- `sa_hmarl/sa_hmarl/agents/deep_rmsa_source_semantic_agent.py`
- `sa_hmarl/sa_hmarl/training/train_deep_rmsa.py`
- `sa_hmarl/sa_hmarl/training/train_deeprmsa_source_semantic_k50.py`
- `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py`

## Recent v2.0 Phase-1 Fix (for reference)

- Fixed `generate_v20_sliding_window_afterstate_dataset.py` to use atomic tmp+rename and stratified sampling.
- Rewrote `diagnose_v20_sliding_window_afterstate_dataset.py` to report risk variance, not just block_count variance.
- Found that under `fixed_c_ppo_r_top1` future policy, group-level block_count variance is 0 (current action choice does not affect future block_count within H=5/20/50). Future blocking labels exist, but candidates are only distinguishable by resource usage, not blocking risk.
- This finding is separate from the current three-method comparison task.

## Status

- The user manually interrupted the delegated subagent that was about to implement the three-method comparison.
- The task is ready to continue; no code has been written yet for the adapted DeepRMSA or the three-way comparison.
- First step should be generating `PROTOCOL_AUDIT.md` and `EXPERIMENT_MANIFEST.json`, then running KSP-FF parity to verify ~5.6342% blocking before training DeepRMSA.

## Required Outputs for Current Task

- `PROTOCOL_AUDIT.md`
- `EXPERIMENT_MANIFEST.json`
- `FAIRNESS_AUDIT.md`
- `EXPERIMENT_MATRIX.md`
- `RESULTS.json`
- `per_seed_summary.csv`
- `paired_trace.jsonl.gz`
- `FINAL_REPORT.md`
- `RUN_CONFIG.json`
- `RUN_LOG.md`

## Notes

- Do not overwrite existing experimental result directories.
- Do not automatically scale to 20 seeds.
- If a checkpoint mismatch is detected, fail closed rather than silently loading/cropping/padding.
- Use atomic tmp + rename for all output files.
\n--- Multi-topology 5-seed pilot start: 2026-07-20T13:41:47+08:00 ---\n

## Multi-topology 5-seed pilot completed

- Background task: bash-9y80d4ci
- Wall time: 562.1 s
- Tasks: 15 (3 topologies × 5 seeds)
- All done markers written; no errors in run log.

### Aggregate blocking (5 seeds, 6000 eval requests each)

| Topology | Strict v1.3 | Main heuristic | Strict-main pp |
|---|---:|---:|---:|
| NSFNET (14n/22l) | 5.3767% | KSP-FF 5.3600% | +0.02 |
| USNET (24n/43l) | 5.4433% | FF-KSP 5.4400% | +0.00 |
| JPN48 (48n/82l) | 5.6133% | FF-KSP 5.5933% | +0.02 |

Interpretation: Strict v1.3 frozen cross-topology transfer is statistically tied with the paper-recommended strong heuristics; the heuristics are marginally lower by ~0.02 pp. No topology shows a clear Strict advantage in this pilot.

### Output files

All required files in `sa_hmarl/experiments/strict_v13_vs_topology_best_heuristics_fixed_c_all_od/`:
- PROTOCOL_LOCK.md, TOPOLOGY_PROVENANCE.md/json, FF_KSP_IMPLEMENTATION_AUDIT.md, PATH_ORDERING_AUDIT.md, CHECKPOINT_TRANSFER_AUDIT.md
- EXPERIMENT_MANIFEST.json, per_seed_summary.csv, paired_action_trace.jsonl.gz
- NSFNET_RESULTS.md/json, USNET_RESULTS.md/json, JPN48_RESULTS.md/json, TOPOLOGY_COMPARISON_SUMMARY.md, NEXT_STEP_DECISION.md


## Phase-R0 Expanded-Path Rescue Ceiling (current)

- Implementation complete.
- Files changed:
  - `sa_hmarl/sa_hmarl/network/ksp.py` (lexicographic path tie-break)
  - `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py` (`expanded_ksp_ff_rescue_action`)
  - `sa_hmarl/sa_hmarl/evaluation/generate_r_mask_empty_rescue_phase_r0.py` (new script)
  - `sa_hmarl/tests/test_expanded_path_rescue.py` (unit + integration tests)
- Unit tests: 12 passed, 0 failed.
- Smoke test started: task `bash-thcr10t1`, `--smoke_only`.
- Output dir: `sa_hmarl/experiments/r_mask_empty_expanded_path_rescue_phase_r0/`
- Note: corrected USNET/JPN48 seeds to 6101–6105 (audit manifest), not 7101/8101.


## Phase-R0 smoke completed

- Smoke task `bash-thcr10t1` failed due to missing output directory; retry `bash-75spuaqm` succeeded.
- Smoke results (seed 5001 for COST239, 6101 for others, 2000 eval requests each):
  - K50-empty events: 228 (COST239), 245 (NSFNET), 258 (USNET), 236 (JPN48)
  - **K500 rescue successes: 0 across all four topologies**
  - Baseline vs rescue blocking rates identical (no change)
- Full 5-seed pilot started: task `bash-gi14mju4`, `--skip_smoke --max_workers 8`.


## Phase-R0 full pilot completed

- Full task: bash-gi14mju4, wall time 1938.0 s, 20 tasks (4 topologies × 5 seeds).
- K50-empty baseline events: 14191.
- K100 available: 0; K200 available: 0; K500 available: 0.
- Closed-loop rescue successes: 0 across all topologies/methods.
- Blocking unchanged by K500 rescue in every topology/seed.
- Decision: STOP expanding static KSP path pool; bottleneck is spectrum/contention, not missing shortest paths.
