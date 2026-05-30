# Project Status — DNN Agent

> **Last updated**: 2026-05-12  
> **Current phase**: Phase F (Yin 2024 protocol sim) in progress; explicit fragmentation-aware mainline evaluated  

---

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
6. **No unit tests** — `tests/` directory is empty.
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
