# Project TODO — Prioritized Action Plan

> **Last updated**: 2026-06-28  
> **Current focus**: SA-HMARL ranking experiments complete. Accept v1.2 static ranker as the final method. Remaining work is paper drafting/figure generation.

---

## 🔴 P0 — Must Do (Blocking next milestone)

### P0.1 Complete Yin 2024 Protocol Simulation
**Goal**: Run all baselines on Net-2/Net-3 + frag=0.5 to get complete comparison curves.

| Subtask | Details | Est. Time |
|---------|---------|-----------|
| P0.1.1 Run Net-2 baselines | `yin2024_eval.py`: add `topologies=["net2"]` | 30 min |
| P0.1.2 Run Net-3 baselines | `yin2024_eval.py`: add `topologies=["net3"]` | 30 min |
| P0.1.3 Run frag=0.5 | `frag_levels=[0.2, 0.5]` | 1 hr |
| P0.1.4 Plot Fig. 8 curves | Blocking vs N_req for all methods × all topologies × frag | 2 hrs |

**Key question to answer**: Does the relative ordering WO < YinLike < IWD-Approx < DF < RF hold across all topologies?

### P0.2 Zero-Shot TopK2/CorrectionNet on Yin2024 Topologies
**Goal**: Test whether our trained models (on NSFNET/USNET) generalize to Yin2024 topologies without retraining.

| Subtask | Details | Est. Time |
|---------|---------|-----------|
| P0.2.1 Adapt TopK2 for Yin2024 env | Handle `num_servers` mismatch (5-7 servers vs 5 in training) | 2 hrs |
| P0.2.2 Adapt CorrectionNet for Yin2024 env | Same server mismatch issue | 1 hr |
| P0.2.3 Run zero-shot eval | Net-1/2/3, N=15-60 | 2 hrs |
| P0.2.4 Analyze results | Compare vs baselines; if gap is large, plan fine-tuning | 1 hr |

**Expected outcome**: Either (a) zero-shot works → strong generalization story, or (b) gap is large → need fine-tuning data collection.

### P0.3 SA-HMARL Counterfactual Ranking & C-Side Bottleneck Diagnosis
**Goal**: Determine whether the 0.64 pp gap between the v1 R-ranker (33.45%) and the frozen DeepRMSA teacher (32.81%) is due to R-action selection, C-side return weights, or C-side downstream survivability.

| Subtask | Details | Status |
|---------|---------|--------|
| P0.3.1 Train v1 R-ranker | `train_r_counterfactual_ranking.py` with frozen PPO-C + PPO-R | ✅ Done |
| P0.3.2 Return-weight sweep | v1.1 sweep: no_fs, weak_fs, stronger_block, stronger_nsb | ✅ Done — all FAIL |
| P0.3.3 C-side look-ahead rerank sweep | `run_c_lookahead_rerank_sweep.py`: top_k_c, λ, β | ✅ Done — FAIL |
| P0.3.4 Hard-case mining | Find states where ranker blocks but DeepRMSA succeeds | ✅ Done — 0 hard cases |
| P0.3.5 C-side downstream survivability diagnostic | 3 seeds × 5 eps × 80 req × 16 probes | ✅ Done — **FAIL** |
| P0.3.6 Trajectory-level divergence diagnosis | Same trace → v1 vs DeepRMSA → align per request → find first divergence and time-to-block | ✅ Done — R-action at request 0, long-term effect |
| P0.3.7 Run inference-time resource-penalty sweep | λ_path / λ_fs on frozen v1 ranker, no retraining | ✅ Done — mixed_mid 32.40% beats DeepRMSA |
| P0.3.8 Train v1.2 R-ranker | Bake path/FS preference into return/loss (resource penalty in dataset label) | ✅ Done |
| P0.3.9 Evaluate v1.2 vs resource-regularized v1 | 5-seed × 20-episode eval; keep the better as final method | ✅ Done |

**Final method**: **v1.2_mixed_low** (or v1.2_fs if optimizing only blocking). The resource preference has been internalized; no inference-time penalty needed.

**Key findings**:
- v1 R-ranker beats frozen PPO-R by **5.05 pp**, but still trails DeepRMSA by **~0.3–0.6 pp** depending on seed set.
- Return-weight tuning, C-side reranking, and downstream survivability do **not** explain the gap.
- PPO-C already selects the candidate with the best downstream survivability 98.79% of the time.
- Oracle-Phi closed-loop gain is only **0.42 pp**, so a C-ranker is unlikely to help.
- **Trajectory diagnosis**: first divergence is an **R-action choice at request index 0 in 30/30 episodes**. DeepRMSA systematically picks shorter paths (−104 km) and slightly fewer FS (−0.43), while v1 ranker takes longer paths. The blocking impact appears with a median delay of **12 requests**.

**Decision**: The inference-time penalty sweep **passed**. `mixed_mid` (λ_path=0.10, λ_fs=0.05) reaches **32.40%** blocking, beating both v1 baseline (33.45%) and DeepRMSA (32.81%).

**Next step — train v1.2**:
- Option A (recommended): modify the counterfactual ranking label to include a soft penalty for path_km / required_fs, then retrain the listwise ranker. This bakes the preference into the model so inference remains a single forward pass.
- Option B: keep the inference-time penalty as a runtime heuristic and declare v1 + penalty as the final method. Simpler, but slightly slower and adds two hyperparameters.
- Not recommended: DeepRMSA imitation or joint online fine-tuning.

**Acceptance criteria for v1.2**:
- Blocking ≤ 32.5% on the standard 5-seed × 20-episode eval.
- Avg path km and avg FS both decrease vs v1 baseline.
- NSB and overload do not materially worsen.

---

## 🟡 P1 — Should Do (Paper quality)

### P1.1 Paper Draft v0.2 — Add Yin 2024 Protocol Section
**Goal**: Integrate Phase F results into `docs/PAPER_DRAFT.md`.

| Subtask | Details | Est. Time |
|---------|---------|-----------|
| P1.1.1 Write Section 5.7 | Yin 2024 protocol experiment design & setup | 2 hrs |
| P1.1.2 Add Fig. 8 reproduction | Blocking probability vs number of requests | 2 hrs |
| P1.1.3 Add comparison table | Our method vs Yin 2024 baselines on their protocol | 1 hr |
| P1.1.4 Update Abstract & Conclusion | Mention cross-protocol generalization | 1 hr |

### P1.2 Generate All Paper Figures
**Goal**: Script-based figure generation for the paper.

| Figure | Data Source | Status |
|--------|-------------|--------|
| Fig 1: Architecture diagram | Draw in draw.io / TikZ | ❌ Not started |
| Fig 2: Training curves | `correction_net_history.json` | ❌ |
| Fig 3: Load sweep | `paper_table_3_load_sweep.json` | ❌ |
| Fig 4: Ablation bar chart | `paper_table_2_ablation.json` | ❌ |
| Fig 5: t-SNE of z vectors | Need to extract z from episodes | ❌ |
| Fig 6: Correction score distribution | `paper_table_4_interpretability.json` | ❌ |
| Fig 7: Interpretability heatmap | `correction_net_interpretability.json` | ❌ |
| Fig 8: Yin 2024 blocking curves | `experiments/yin2024_sim/results/` | 🔄 Partial |

**Deliverable**: `notebooks/generate_paper_figures.py` or `.ipynb`

### P1.3 Code Cleanup & Deduplication
**Goal**: `src/` should be the single source of truth for paper reproduction.

| Subtask | Action | Est. Time |
|---------|--------|-----------|
| P1.3.1 Audit `src/` vs `experiments/agent_mvp/` | Find diverged files; decide which is canonical | 2 hrs |
| P1.3.2 Unify imports | All `src/` files should use relative imports; remove `sys.path` hacks where possible | 2 hrs |
| P1.3.3 Delete dead code | Remove failed experiment scripts (unless historically important) | 1 hr |
| P1.3.4 Add `__init__.py` where missing | Make `src/` a proper package | 30 min |

### P1.4 Unit Tests
**Goal**: Core components have basic smoke tests.

| Test | Coverage | Est. Time |
|------|----------|-----------|
| `test_environment.py` | Env reset, step, metrics | 2 hrs |
| `test_encoder.py` | Encoder output shape, deterministic | 1 hr |
| `test_agent.py` | Agent decide() returns valid actions | 2 hrs |
| `test_mapper.py` | Spectrum allocation constraints | 2 hrs |
| `test_yin2024_sim.py` | Yin2024 env factory, request generation | 1 hr |

---

## 🟢 P2 — Nice to Have (Polish)

### P2.1 ARCHITECTURE.md
Write proper module interface documentation. Currently empty.

### P2.2 README.md Polish
- Add quick start with actual commands
- Add badge for Python version
- Add citation placeholder

### P2.3 Requirements Audit
```bash
pip freeze > requirements_new.txt
diff requirements.txt requirements_new.txt
```
Clean up unused dependencies.

### P2.4 Snapshot Cleanup
Keep only 3 most important snapshots:
- `20260509_081732_phaseAB_complete` — baseline
- `20260509_092715_topk_selector_success` — TopK2 milestone
- `20260509_181635_correctionnet_v1_aligned` — final aligned results

Archive or delete the rest.

### P2.5 Jupyter Notebooks
- `notebooks/visualize_topology.ipynb` — Plot NSFNET, USNET, Net-1/2/3
- `notebooks/analyze_results.ipynb` — Interactive result exploration
- `notebooks/correction_net_analysis.ipynb` — Interpretability deep dive

### P2.6 Adaptive λ for CorrectionNet
Current λ=0.2 is fixed. Experiment with:
- λ scheduled by correction confidence (`|correction_score|`)
- λ scheduled by state characteristics (high fragmentation → higher λ)

---

## 📅 Suggested Timeline

| Week | Focus | Deliverables |
|------|-------|-------------|
| **Week 1** | P0.1 + P0.2 | Net-2/3 results, frag=0.5 results, zero-shot TopK2/CorrectionNet results |
| **Week 2** | P1.1 + P1.2 | Paper draft v0.2 with Yin 2024 section; all figures generated |
| **Week 3** | P1.3 + P1.4 | Clean `src/`, basic unit tests, reproducibility verified |
| **Week 4** | P2.* | Polish docs, notebooks, requirements, snapshot cleanup |

---

## Quick Reference: One-Line Commands

```bash
# Reproduce paper tables from existing results
cd src && python compile_paper_results.py

# Reproduce paper tables from scratch (slow)
cd src && python run_paper_eval.py

# Run Yin 2024 Net-1 baselines
cd experiments/yin2024_sim && python yin2024_eval.py

# Run ablation study
cd experiments/agent_mvp && python run_ablation.py

# Run load sweep
cd experiments/agent_mvp && python run_load_sweep.py

# Train CorrectionNet (if needed)
cd experiments/agent_mvp && python train_correction_net.py

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```
