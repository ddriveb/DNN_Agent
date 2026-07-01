# Project Status — DNN Agent

> **Last updated**: 2026-06-30  
> **Current phase**: SA-HMARL planner-distilled R-ranking finalized. The S100 normal-load comparison is now the recommended main experiment; the S24 high-blocking setting is retained as a stress validation. Scenario-specific retraining and Lyapunov inference-time rerank do not add value.  

---

## Final SA-HMARL Recommendation (2026-06-30)

After screening the 100-slot normal-load scenario, the 24-slot stress scenario,
scenario-specific retraining (Stage 3), and Lyapunov inference-time rerank
sweeps, the **v1.2 planner-distilled R-ranker** remains the best practical
method.

### Main result (normal 100-slot)
- **Topology / config**: `snap24_gnutella_reach`, 100 slots, 4 servers, k=5, max_blocks=10, `default3` split profile, arrival interval 0.15, size 5-30 MB, holding 4-10 s.
- **PPO-C + v1.2 planner-distilled R-ranker**: **0.925%** blocking
- **PPO-C + DeepRMSA S100**: **1.288%** blocking
- **PPO-C + PPO-R**: **3.663%** blocking
- **PPO-C + KSP-BF / KSP-FF**: **7.675% / 7.375%** blocking
- **Interpretation**: The main result now uses a low-blocking, normal-capacity system setting. v1.2 still improves over DeepRMSA while avoiding the unrealistically high blocking rate of the S24 stress setting.
- **Checkpoint**: `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`
- **Report**: `sa_hmarl/experiments/main_s100_system_comparison.md`

### Stress validation (hard 24-slot)
- **Topology / config**: `snap24_gnutella_reach`, 24 slots, 4 servers, k=5, max_blocks=10, `default3` split profile, arrival interval 0.09, size_max 30 MB, holding_max 14 s.
- **v1.2**: **53.92 ± 5.90%** blocking
- **DeepRMSA**: **54.94 ± 5.82%** blocking
- **Gain**: **+1.01 pp** (p = 0.002, paired one-sided t-test)
- **Side effects**: delay -0.05 ms, server overload +0.60 pp
- **Interpretation**: S24 should be described as a stress-test validation, not the main normal-system operating point.

### What did not work
- **Scenario-specific retraining (Stage 3)**: Adding a future server-overload penalty to the return produced identical ranking labels because H=5 rollouts contain too few overload events.
- **Lyapunov rerank**: Best formal gain only +0.06 pp; <2% of actions changed; `H_spec` saturated at the clip limit. Keep only as a theoretical interpretation, not a practical method.
- **v1.3 spectrum-viability label**: Adding a post-decision `Phi_after` bonus based on next-request feasible counts degraded v1.2 blocking by +0.26 to +0.37 pp. The bonus has weak negative correlation with the true H-step return (Spearman ≈ -0.06), so it perturbs the ranker away from blocking-minimizing actions.

### Paper implication
Use the 100-slot `default3` normal-load scenario as the primary result. Use the
24-slot result only as a stress-test validation showing that the v1.2 advantage
also appears under highly constrained spectrum resources.

## Completed Milestones

### ✅ Phase A — Predictor-Guided Rule Agent (MVP)
- **Location**: `experiments/predictor_mvp/`, `experiments/agent_mvp/`
- **Key files**: `predictor.py`, `rule_agent.py`, `env_wrapper.py`, `eval_comprehensive.py`
- **Results**: RuleAgent outperforms YinLike by 14-21% relative reduction in blocking
- **Artifacts**: `pretrained_nsfnet_v2b.pt`, `pretrained_mixed_v2b.pt`

### ✅ Phase B — Imitation Learning
- **Location**: `experiments/agent_mvp/`
- **Key files**: `train_imitation.py`, `enhance_state_and_retrain.py`, `eval_imitation.py`
- **Results**: 30D enhanced state → 74.6% accuracy; top-3 hit rate 96.3%
- **Artifacts**: `imitation_agent_enhanced.pt`

### ✅ Phase C — Predictor-Guided Top-K Selector
- **Location**: `experiments/agent_mvp/`, `src/agents/`
- **Key files**: `topk_selector_agent.py`, `eval_topk_selector.py`
- **Results**: TopK2-30D achieves 15.51% (NSFNET std), 27.49% (high load), 26.85% (USNET)
- **Artifacts**: copied to `src/checkpoints/`

### ✅ Phase D — Direct Offline DQN (Proved Infeasible)
- **Location**: `experiments/agent_mvp/`
- **Key files**: `train_offline_dqn.py`, `eval_dqn.py`
- **Results**: 38-55% blocking — worse than raw imitation. Confirmed RL must not replace selector.
- **Artifacts**: `offline_dqn.pt` (kept as negative result)

### ✅ Phase E — CorrectionNet v1 (Residual RL)
- **Location**: `experiments/agent_mvp/`, `src/agents/`
- **Key files**: `correction_net.py`, `train_correction_net.py`, `eval_correction_net_aligned.py`
- **Results**:
  - NSFNET standard: **13.04±1.58%**
  - NSFNET high load: **23.50±1.64%**
  - USNET cross-topo: **21.77±0.89%**
- **Key property**: λ=0 is *exactly* equivalent to TopK2-30D (0 mismatches / 2000 requests)
- **Artifacts**: `correction_net.pt` (5K parameters)

### 🔄 Phase F — Yin 2024 Protocol Simulation (In Progress)
- **Location**: `experiments/yin2024_sim/`
- **Key files**:
  - `yin2024_network.py` — Net-1/2/3 topologies, spectrum init, Yin2024MECServer
  - `yin2024_requests.py` — Dynamic DNN model builder from paper params
  - `yin2024_env.py` — Environment factory (compatible with existing agents)
  - `yin2024_eval.py` — Baseline evaluation script
  - `yin_baselines.py` — WO, DF, RF, IWD-Approx
- **Results (Net-1, frag=0.2, 5 seeds)**:

| N_req | WO | DF | RF | IWD-Approx | YinLike |
|------:|---:|---:|---:|-----------:|--------:|
| 15 | 0.0% | 44.0% | 56.0% | 38.7% | 0.0% |
| 30 | 0.0% | 58.0% | 66.0% | 47.3% | 4.7% |
| 60 | 0.0% | 60.3% | 68.0% | 56.7% | 14.0% |

- **Open questions**: YinLike blocking is very low (0-14%) — may be correct for lightly loaded small network, but needs validation on Net-2/3.

### ✅ Phase H — Mean-Field and R-Feasibility Study for PPO Agent-C (Updated)
- **Location**: `experiments/mean_field_and_r_feasibility_study.md`, `sa_hmarl/experiments/rfeas_round*_summary.md`, `sa_hmarl/agents/c_agent.py`, `sa_hmarl/agents/ppo_agents.py`
- **What was tested**:
  - `typed_mean_field`, `gated_typed_mean_field`, `fixed_blend_typed_mean_field`
  - `candidate_mean_field`, `candidate_mean_field_count_only`
  - `r_feasibility`
  - `r_feasibility_safe` (Round 2, r_feasibility + 2-dim server margin)
- **Protocol**: `snap24_gnutella_reach`, 20 slots, 4 servers, k_paths=5, max_blocks=10, block_sort=mixed, split_profile=default3, frozen PPO-R backend, 300 training episodes, K=5 system comparison (5 eval seeds × 20 episodes × 80 requests).
- **Consistency audit**: `sa_hmarl/sa_hmarl/evaluation/diagnose_r_feasibility_consistency.py` audited 14,400 C candidates across seeds 42/123/456. Existing `r_feasibility` feature count matches real `Agent-R mask.sum()` exactly: **exact_match_rate=100%, max_abs_error=0**.
- **Round 1 — default vs r_feasibility** (3 training seeds):
  - default: blocking=0.3869, no_valid_c_rate=0.4524, avg_valid_r_actions=3.310, server_overload_ratio=0.0359, avg_delay_ms=9.81
  - r_feasibility: blocking=0.3620, no_valid_c_rate=0.4239, avg_valid_r_actions=4.446, server_overload_ratio=0.0802, avg_delay_ms=9.08
  - diff: blocking=-2.48pp, no_valid_c_rate=-2.85pp, avg_valid_r_actions=+1.136, server_overload_ratio=+4.43pp, avg_delay_ms=-0.72ms
  - `r_feasibility` clearly beats `default` on blocking and R-side feasibility, but raises server_overload from 3.5% to 8.0%.
- **Round 2 — r_feasibility_safe** (r_feasibility + 2-dim server margin):
  - r_feasibility_safe: blocking=0.3760, no_valid_c_rate=0.4425, avg_valid_r_actions=3.303, server_overload_ratio=0.0190, avg_delay_ms=8.70
  - vs default: blocking=-1.09pp, server_overload_ratio=-1.69pp, avg_delay_ms=-1.11ms
  - vs r_feasibility: blocking=+1.40pp, server_overload_ratio=-6.12pp, avg_delay_ms=-0.38ms
  - Server margin successfully suppresses server_overload from 8.0% to 1.9%, but also gives back most of the blocking gain.
- **Interpretation**:
  - The earlier negative `r_feasibility` result (blocking ~0.46-0.48) appears setup-dependent or seed-sensitive. With a clean multi-seed protocol and verified feature consistency, `r_feasibility` is a genuine improvement over default.
  - The remaining challenge is balancing spectrum feasibility against compute-side safety: the current 2-dim server margin is too conservative.
- **Artifacts**: Full summary in `experiments/mean_field_and_r_feasibility_study.md`; all feature-mode implementations preserved in code.

### ✅ Phase I — Counterfactual Ranking & C-Side Survivability Diagnostic (SA-HMARL)
- **Location**: `sa_hmarl/`
- **Key files**:
  - `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py` — listwise R-ranker training
  - `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py` — counterfactual dataset builder
  - `sa_hmarl/sa_hmarl/evaluation/diagnose_c_downstream_survivability.py` — C-side downstream survivability probe diagnostic
  - `sa_hmarl/tests/test_c_downstream_survivability.py` — unit tests for the diagnostic
- **What was tested**:
  - Trained a frozen v1 R-ranker to rerank `(path, modulation, block)` actions for a frozen PPO-C agent.
  - Swept return weights (v1.1) and C-side look-ahead reranking (`top_k_c`, λ, β) to close the gap to a frozen DeepRMSA teacher.
  - Ran a survivability probe diagnostic: for each real request, enumerate all legal C candidates, roll forward with probe requests, and measure downstream survivability `Phi`.
- **Protocol**: `snap24_gnutella_reach`, 24 slots, 4 servers, k_paths=5, max_blocks=10, block_sort=mixed, split_profile=default3, arrival_interval=0.15.
- **Key results**:
  - v1 R-ranker (no teacher): **33.45%** blocking
  - Frozen PPO-R baseline: **38.50%** blocking
  - Frozen DeepRMSA teacher: **32.81%** blocking
  - Gap to DeepRMSA: **0.64 pp**
  - C-side look-ahead rerank sweep: FAIL — best config (top_k_c=3, λ=0.5, β=0) gave **33.71%**, no reduction in zero-legal rate.
  - R-ranker v1.1 return-weight sweep: FAIL — six weight configurations all landed in **33.39%–33.45%**.
  - R-ranker hard-case mining: FAIL — zero blocked-by-ranker cases where DeepRMSA succeeds at the same state.
  - C-side downstream survivability diagnostic: **FAIL** — mean Phi gap = 0.0008, Phi range > 0.1 on only 1.08% of requests, PPO-C selects best-Phi 98.79% of the time, and oracle-Phi closed-loop gain is only 0.42 pp (54.17% → 53.75%).
- **Interpretation (pre-trajectory)**:
  - The 0.64 pp gap to DeepRMSA is **not** caused by instantaneous R-action selection mistakes, C-side return-weight tuning, or C-side downstream survivability.
  - PPO-C already picks the candidate with the best downstream survivability; the bottleneck lies elsewhere (likely trajectory-level compounding / long-horizon credit assignment, not a single-step correction).
- **Phase I.1 — Trajectory divergence diagnostic**:
  - Ran the same request trace through `PPO-C + v1 R-ranker` and `PPO-C + DeepRMSA`, aligning per-request decisions.
  - Protocol: 3 seeds × 10 episodes × 80 requests = 2400 requests.
  - Key results:
    - v1 R-ranker blocking: **37.50%**
    - DeepRMSA blocking: **37.17%**
    - First divergence occurs at request index **0 in 30/30 episodes** (median 0, max 1).
    - First divergence type: **R-action in all 30 episodes**; C actions never diverge first.
    - Time from first divergence to first block: median **12 requests**, mean **~24–25 requests** for both methods.
    - Resource delta at first divergence (v1 − DeepRMSA, both succeed):
      - path_km: **+104 km**
      - num_slots (FS): **+0.43**
      - block_waste: **−0.018**
      - delay_ms: **+1.05 ms**
      - immediate fragmentation / LFB / free-block-count differences are negligible.
  - Interpretation: the gap is a **systematic immediate R-action preference difference**, not a single hard-case. DeepRMSA prefers shorter / lower-FS paths and the benefit compounds over ~12 requests.
- **Phase I.2 — Lightweight inference-time resource-penalty sweep**:
  - Added a soft penalty `score' = score_ranker − λ_path·norm(path_km) − λ_fs·norm(required_fs)` at inference time, **without retraining**.
  - Protocol: 5 seeds × 20 episodes × 80 requests = 8000 requests (same as v1 eval).
  - Key results:

    | Config | λ_path | λ_fs | Blocking | Avg path km | Avg FS | Overload |
    |---|---:|---:|---:|---:|---:|---:|
    | baseline | 0.00 | 0.00 | 33.45% | 596.0 | 2.20 | 0.40% |
    | path_only_mid | 0.10 | 0.00 | 33.14% | 554.3 | 2.19 | 0.49% |
    | fs_only_low | 0.00 | 0.05 | 32.45% | 594.3 | 2.03 | 0.41% |
    | **mixed_mid** | **0.10** | **0.05** | **32.40%** | **546.3** | **2.03** | **0.55%** |

  - **Verdict: PASS**. `mixed_mid` beats the DeepRMSA teacher (32.81%) and the v1 baseline (33.45%) with only a small inference-time preference correction. Delay and overload do not materially worsen.
  - Interpretation: the counterfactual R-ranker benefits from explicitly valuing short paths and low FS usage.
- **Phase I.3 — v1.2 R-ranker (internalized resource preference)**:
  - Added the same path/FS penalty to the **training label** (`generate_r_counterfactual_ranking_dataset.py`) and retrained three models.
  - Datasets: train 10 eps, val 5 eps, test 5 eps; same feature schema as v1.
  - Closed-loop results (5 seeds × 20 eps × 80 req):

    | Method | Blocking | Avg path km | Avg FS |
    |---|---:|---:|---:|
    | v1 R-ranker | 33.45% | 596.0 | 2.20 |
    | Resource-regularized v1 (inference penalty) | 32.40% | 546.3 | 2.03 |
    | **v1.2_fs** | **32.40%** | **590.4** | **2.03** |
    | **v1.2_mixed_low** | **32.42%** | **548.4** | **2.03** |
    | v1.2_mixed_mid | 32.55% | 543.1 | 2.03 |
    | Frozen DeepRMSA | 32.81% | — | — |

  - **Verdict: PASS** for `v1.2_fs` and `v1.2_mixed_low`. The preference can be internalized; no inference-time penalty is needed.
  - **Final recommendation**: use **v1.2_mixed_low** as the final SA-HMARL R-ranker (or `v1.2_fs` if absolute blocking is the only metric).
- **Artifacts**:
  - `sa_hmarl/experiments/c_downstream_survivability_smoke.json/.md`
  - `sa_hmarl/experiments/c_downstream_survivability_medium.json/.md`
  - `sa_hmarl/experiments/c_downstream_survivability_diagnostic.json/.md`
  - `sa_hmarl/experiments/trajectory_divergence_diagnostic.json/.md`
  - `sa_hmarl/experiments/r_ranker_resource_penalty_sweep.json/.md`
  - `sa_hmarl/experiments/r_ranker_resource_regularized_main_result.md`
  - `sa_hmarl/experiments/r_ranker_v1_2_final_report.md`
  - `sa_hmarl/datasets/r_counterfactual_ranking_v1_2_*/`
  - `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_*/`

### ✅ Phase G — Fragmentation-Aware Mainline Evaluation
- **Location**: `experiments/agent_mvp/`
- **Key files**:
  - `eval_fragmentation_mainline.py` — tunes explicit fragmentation weights and runs multi-scenario comparison
  - `results/fragmentation_mainline_eval.json` — raw runs + summaries
  - `results/fragmentation_mainline_summary.md` — compact paper-style table
- **What was added**:
  - Candidate-level mapper preview for `(split, server)` actions
  - Explicit penalties for path fragmentation increase and largest-free-block loss
  - Environment-level spectrum metrics: average fragmentation, largest free block, utilization, free-block count

- **Main findings**:
  - `TopK2+Feasibility` slightly improves cross-topology blocking vs `TopK2-30D` on USNET: **27.30% → 25.91%**
  - Explicit fragmentation-aware scoring reduces fragmentation consistently:
    - NSFNET standard: **0.3414 → 0.3078**
    - NSFNET high load: **0.4122 → 0.3480**
    - USNET cross-topology: **0.3350 → 0.2834**
  - However, explicit fragmentation penalties **increase blocking** in all three scenarios:
    - NSFNET standard: **16.00% → 17.59%**
    - NSFNET high load: **27.44% → 33.18%**
    - USNET cross-topology: **27.30% → 32.72%**
  - `CorrectionNet λ=0.2` remains the best acceptance-oriented method:
    - NSFNET standard: **14.35%**
    - NSFNET high load: **23.08%**
    - USNET cross-topology: **21.01%**
- **Interpretation**:
  - The new mainline is **partially validated**:
    - The mechanism is physically meaningful and measurably protects spectrum structure
    - But naive explicit fragmentation aversion is too conservative and hurts immediate acceptance
  - Stronger immediate gains come from **mapper-feasibility awareness** and **learned long-term correction**, not from raw fragmentation penalties alone

---

## Code Organization

### `src/` — Paper-ready source
Cleaned-up subset for reproduction and extension.

```
src/
├── run_paper_eval.py          # Reproduces all paper tables
├── compile_paper_results.py   # Compiles from existing JSON results
├── checkpoints/               # Model checkpoints (symlinks to agent_mvp/)
├── core/                      # Env, network, encoder, predictor, mapper
├── agents/                    # Imitation, TopKSelector, CorrectionNet, CorrectionAgent
├── baselines/                 # Heuristic baselines
├── utils/                     # State builder, fixed trace
└── results/                   # Paper tables (JSON + markdown)
```

### `experiments/` — Research log

```
experiments/
├── predictor_mvp/             # Phase A: predictor training & calibration
│   ├── predictor.py
│   ├── encoder.py
│   ├── train.py
│   ├── main_cross_topology.py
│   └── pretrained_nsfnet_v2b.pt
│
├── agent_mvp/                 # Phase B-E: all agent experiments
│   ├── baselines.py           # Random, ShortestPath, YinLike, LoadBalanced
│   ├── yin_baselines.py       # WO, DF, RF, IWD-Approx (Yin-protocol style)
│   ├── rule_agent.py          # Phase A: predictor-guided rule agent
│   ├── topk_selector_agent.py # Phase C
│   ├── correction_net.py      # Phase E
│   ├── correction_agent.py    # Phase E: full agent
│   ├── train_*.py             # Training scripts
│   ├── eval_*.py              # Evaluation scripts (~20 files)
│   ├── results/               # All JSON results
│   ├── logs/                  # Training logs
│   ├── checkpoints/           # Model files
│   └── traces/                # Fixed request traces
│
└── yin2024_sim/               # Phase F: Yin 2024 protocol simulation
    ├── yin2024_network.py
    ├── yin2024_requests.py
    ├── yin2024_env.py
    ├── yin2024_eval.py
    └── results/
```

### `configs/` — Configuration files
- `env_config.yaml` — Original environment parameters
- `agent_config.yaml` — Agent hyperparameters
- `training_config.yaml` — Training hyperparameters
- `yin2024_protocol.yaml` — Yin 2024 protocol parameters

### `docs/` — Documentation
- `PROJECT_PLAN.md` — Original project plan (v1.0, 2026-04-24)
- `ARCHITECTURE.md` — System architecture (placeholder)
- `EXPERIMENTS.md` — Experiment results record (updated to Phase E)
- `PAPER_DRAFT.md` — Paper draft v0.1 (method + experiments)

### `snapshots/` — Historical checkpoints
7 snapshots documenting key milestones (phase AB complete, phase CAB progress, TopK2 success, final report, DQN failure, CorrectionNet v1 success, CorrectionNet aligned).

---

## Key Results Summary

### Main Results (NSFNET + USNET, 5 seeds × 2000 requests)

| Method | NSFNET std | NSFNET high | USNET |
|--------|-----------|-------------|-------|
| Random | 49.39% | 52.60% | 49.20% |
| ShortestPath | 61.72% | 71.97% | 76.23% |
| YinLike | 19.48% | 33.08% | 32.53% |
| Imitation-30D | 20.53% | 32.92% | 28.34% |
| TopK2-30D | 15.51% | 27.49% | 26.85% |
| **CorrectionNet λ=0.2** | **13.04%** | **23.50%** | **21.77%** |

### Ablation (NSFNET standard)

| Component | Blocking |
|-----------|----------|
| YinLike | 19.48% |
| + Neural proposal | 20.53% |
| + Predictor reranking | 15.51% |
| + Residual correction | **13.04%** |
| RL replaces selector | 38.23% |

### SA-HMARL Counterfactual Ranking (snap24, K5/M10, 5 eval seeds × 20 episodes × 80 requests)

| Method | Blocking |
|--------|----------|
| Frozen PPO-R | 38.50% |
| v1 R-ranker (no teacher) | 33.45% |
| Frozen DeepRMSA | 32.81% |

### Yin 2024 Protocol (Net-1, N=60, frag=0.2)

| Method | Blocking |
|--------|----------|
| WO | 0% |
| YinLike | 14.0% |
| IWD-Approx | 56.7% |
| DF | 60.3% |
| RF | 68.0% |

---

## Known Issues & Technical Debt

### High Priority
1. **YinLike blocking suspiciously low on Net-1** — RF (same server selection) blocks at 68%, but YinLike at 14%. Root cause: `greedy_partition` in RF selects split with *minimum delay* (often split=2, bw=2), while YinLike forces split=0 (bw=8). The split choice dominates. Need to verify this is expected or adjust `greedy_partition` penalties.
2. **Net-2/Net-3 not evaluated** — Only Net-1 has baseline results.
3. **frag=0.5 not evaluated** — Only frag=0.2 tested.

### Medium Priority
4. **TopK2/CorrectionNet not tested on Yin2024 topologies** — Zero-shot cross-topology test pending.
5. **ARCHITECTURE.md is empty** — Needs module interface documentation.
6. **Limited unit tests** — `sa_hmarl/tests/test_c_downstream_survivability.py` has 8 passing tests; root `tests/` still empty.
7. **Code duplication** between `experiments/agent_mvp/` and `src/` — `src/` is a cleaned subset but some files have diverged.

### Low Priority
8. **Snapshot cleanup** — 7 snapshots taking space; some may be redundant.
9. ** notebooks/ is empty** — Could add visualization notebooks.
10. ** requirements.txt may be outdated** — Needs audit.

---

## Reproducibility Checklist

| Task | Command | Status |
|------|---------|--------|
| Reproduce Table 1 (main results) | `cd src && python compile_paper_results.py` | ✅ |
| Reproduce Table 1 from scratch | `cd src && python run_paper_eval.py` | ✅ (slow, ~30 min) |
| Reproduce Yin 2024 Net-1 baselines | `cd experiments/yin2024_sim && python yin2024_eval.py` | ✅ |
| Run TopK2 on Yin2024 topologies | Not yet implemented | 🔄 |
| Reproduce C-side survivability diagnostic | `cd /mnt/d/project/DNN_Agent && PYTHONPATH=sa_hmarl .venv/bin/python -m pytest sa_hmarl/tests/test_c_downstream_survivability.py -q` | ✅ |
