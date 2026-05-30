# Project TODO — Prioritized Action Plan

> **Last updated**: 2026-05-09  
> **Current focus**: Complete Phase F (Yin 2024 protocol sim), then paper polish

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
