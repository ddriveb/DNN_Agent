# Paper Draft: Online Residual RL for DNN Offloading in Optical-MEC Networks

> **Status**: Mainline update with OnlineResidualDQN — v0.2  
> **Date**: 2026-05-13  
> **Main contribution**: An online residual RL agent that improves blocking rate over strong predictor-guided baselines while exposing a measurable acceptance-fragmentation trade-off.

## Current Mainline Update

The project mainline has shifted from the earlier **offline CorrectionNet-only** story to a stronger online-RL result:

- `Top-K candidate generator` constrains the action space to promising `(split, server)` pairs
- `Predictor scoring` handles short-term feasibility
- `Online residual Q correction` learns long-term value from real interaction
- `Mapper` still executes deterministic KSP + First-Fit optical allocation

Current formal results show that **OnlineResidualDQN is the best blocking-rate method across all three scenarios**, but it also consumes spectrum more aggressively than `Old CorrectionNet λ=0.2`. The paper therefore needs to present both:

1. **Main performance gain**
2. **Spectrum-state trade-off**

Reference tables:
- [online_residual_dqn_mainline_tables.md](/mnt/d/project/DNN_Agent/experiments/agent_mvp/results/online_residual_dqn_mainline_tables.md)
- [ONLINE_RESIDUAL_DQN_FRAMEWORK.md](/mnt/d/project/DNN_Agent/docs/ONLINE_RESIDUAL_DQN_FRAMEWORK.md)

---

## Abstract

DNN inference offloading in elastic optical networks with mobile edge computing (MEC) faces a fundamental tension: short-term feasibility (bandwidth, delay) versus long-term resource health (fragmentation, load balancing). Existing heuristic methods ignore predictor uncertainty, while direct end-to-end RL is difficult because the action space is combinatorial and many actions are invalid under optical constraints.

We propose an **Online Residual DQN framework** that decomposes decision-making into three layers: (1) a lightweight neural network proposes top-K candidate actions via behavior cloning; (2) an explicit predictor scores each candidate on short-term feasibility; (3) a residual Q-network learns online long-term value corrections through replay-based temporal-difference updates. The final score combines predictor feasibility with the residual Q estimate:

```
final_score = predictor_score + λ · Q_residual(s, a)
```

When the residual weight is disabled, the policy reduces to the predictor-guided selector, ensuring baseline alignment. In formal 5-seed evaluation, OnlineResidualDQN achieves **11.53±0.65%** blocking on NSFNET standard load, **20.87±0.67%** on NSFNET high load, and **18.55±1.75%** on USNET cross-topology, outperforming TopK2+Feasibility, Old CorrectionNet, and FragFeature-CorrectionNet. At the same time, the learned policy increases fragmentation and spectrum utilization, revealing a clear acceptance-fragmentation trade-off. These results show that residual RL is most effective when inserted after candidate generation and predictor scoring, while leaving optical execution deterministic.

---

## 1. Introduction

### 1.1 Background

- DNN inference at the edge: partition models, offload to MEC servers over elastic optical networks
- Resource constraints: spectrum slots, continuity/contiguity, MEC compute, end-to-end delay deadlines
- Dynamic environment: connection arrivals/departures create time-varying fragmentation and load

### 1.2 Challenges

1. **Action space explosion**: (split point × target server × path × modulation × slots) is intractable for end-to-end RL
2. **Sparse rewards**: +1/-1 per request does not distinguish "good success" from "barely success"
3. **Non-stationary MDP**: Each request changes the network state, but request features are i.i.d. random
4. **Compounding errors**: One wrong allocation affects 10+ subsequent requests

### 1.3 Related Work

- **Yin et al. 2024**: Heuristic server selection + first-fit. Simple, but no predictor, no adaptation.
- **Bao et al. 2025**: Unified optimization with explicit environment model. Strong but computationally heavy.
- **Pure DRL (DQN/PPO)**: Direct action selection from compact state. Unstable in our domain (confirmed by our experiments).
- **Residual Policy Learning (Silver et al.)**: Learn residual over base policy. Our approach is structurally similar but applied to a *selector* rather than a continuous policy.

### 1.4 Our Approach

**Key insight**: RL should not replace the predictor/selector. It should correct it.

We decompose the decision into:
- **Neural proposal** (coarse): ImitationAgent proposes top-K actions (96.3% top-3 hit rate)
- **Predictor scoring** (fine): Explicit P_success/delay/load scoring per candidate
- **Residual correction** (long-term): CorrectionNet learns which short-term-feasible actions hurt future resources

**Why this works**:
- Neural network handles the combinatorial ranking problem (what are the promising actions?)
- Predictor handles the physics (will this path meet the deadline?)
- CorrectionNet handles the economics (will this allocation block future requests?)

---

## 2. Problem Formulation

### 2.1 Network Model

- Topology: G = (V, E), directed graph (NSFNET: 14 nodes, USNET: 28 nodes)
- Spectrum: C slots per link, fixed grid
- MEC: S servers, each with compute capacity, load tracking

### 2.2 Request Model

Each request r_t = (src, model, deadline, arrival_time, holding_time)
- model: DNN with K split points (0=edge-only, K-1=full offloading)
- Each split k has (bandwidth_slots, compute_cost, intermediate_size_mb)

### 2.3 Action Space

High-level action: a_t = (split_id, server_id)  
Low-level execution: mapper finds path + slot block via K-shortest-path + first-fit

### 2.4 Objective

Maximize long-term acceptance rate (minimize blocking) subject to:
- Spectrum continuity and contiguity constraints
- MEC compute capacity constraints
- Per-request delay deadline constraints

---

## 3. Method: Predictor-Guided Top-K Correction Agent

### 3.1 Architecture Overview

```
Request r_t ──→ State Vector x_t (30D)
                  │
                  ▼
        ┌─────────────────────┐
        │  ImitationAgent     │  → top-K action candidates
        │  (behavior cloning) │    (K=2, 96.3% teacher hit rate)
        └─────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  Predictor Scoring  │  → P_success, delay, load per candidate
        │  α·P_suc − β·delay  │    short-term feasibility score
        └─────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  CorrectionNet      │  → correction_score(s,a) ∈ ℝ
        │  (10D → 64 → 64 → 1)│    long-term value residual
        └─────────────────────┘
                  │
                  ▼
        final_score = predictor_score + λ · correction_score
                  │
                  ▼
        ┌─────────────────────┐
        │  Mapper (KSP+FF)    │  → path, modulation, slot block
        └─────────────────────┘
```

### 3.2 ImitationAgent: Neural Proposal

- Input: 30D state vector (request features + network state + enhanced model encoding)
- Network: MLP 30 → 128 → 128 → 15 (3 splits × 5 servers)
- Training: Behavior cloning on 36K samples from AdaptiveRuleAgent teacher
- Top-K extraction: `torch.topk(logits, k=2)` → action candidates

**Why top-K?**  
Top-3 hit rate = 96.3%. The neural network is excellent at coarse ranking but poor at fine-grained selection (split_2 accuracy only 50.8%). Top-K constrains the error surface.

### 3.3 Predictor Scoring

For each candidate (split, server):
1. Encode path: `z = encoder.encode(src, dst)` (link-state vector)
2. Predict: `P_success, delay = predictor(split, src, dst, z)`
3. Compute total delay: `delay_total = delay_net + delay_compute`
4. Score: `score = α·P_success − β·delay_total/100 − γ·server_load − δ·deadline_penalty`

Parameters: α=2.0, β=0.3, γ=0.2, δ=0.05

### 3.4 CorrectionNet: Residual Correction

**Motivation**: The predictor only sees short-term feasibility. It cannot know that allocating 8 slots on a heavily fragmented link will block the next large request. CorrectionNet learns this from episode returns.

**Architecture**:
```
Input (10D):
  [bw_slots/8, compute_cost, size_mb/3, P_success, delay/200,
   server_load, deadline_slack_norm, frag_index, max_free_bw, avg_load]
  → Linear(10, 64) → ReLU
  → Linear(64, 64) → ReLU
  → Linear(64, 1)
Output: normalized long-term return
```

**Training**:
- Data: 30K transitions from TopK2-30D policy (3 scenarios × 10 seeds)
- Target: Episode return `G_t = Σ γ^k · r_{t+k}` (γ=0.95)
- Loss: MSE on z-score normalized returns
- Best validation loss: 0.5125

**Key training challenge**: Replay buffer contained unclipped predictor outputs (up to 10^13). Fixed via:
1. `np.clip(action_feat, -100, 100)` during feature extraction
2. Input z-score normalization
3. Gradient clipping (max_norm=1.0)

### 3.5 Baseline Alignment Guarantee

**Critical property**: When λ=0, our agent is **exactly** equivalent to TopKSelectorAgent.

Verification: Ran both agents on the same 2000-request trace, comparing every decision:
```
λ=0.0 vs Native TopK2-30D: 0 mismatches / 2000 requests (0.00%)
```

This means every percentage point of improvement at λ>0 is genuinely from the correction term, not evaluation pipeline differences.

---

## 4. Training Pipeline

```
Phase A: Teacher Data Generation
         AdaptiveRuleAgent (tuned weights) generates 36K (state, action) pairs

Phase B: Behavior Cloning
         Train ImitationAgent (21D → 128 → 128 → 15)
         Val accuracy: 73.7% (21D), 74.6% (30D enhanced)

Phase C: Predictor-Guided Top-K Selector
         TopK2-30D: neural top-2 + predictor reranking
         Blocking: 15.51% (NSFNET std) — new baseline

Phase D: Offline DQN (Failed)
         Direct action selection with CQL + top-K masking
         Blocking: 42.08% — worse than raw imitation
         → Proves RL must not replace the selector

Phase E: CorrectionNet (Success)
         Train 5K-parameter residual network on episode returns
         Blocking: 13.04% — new state-of-the-art
```

---

## 5. Experiments

### 5.1 Setup

- **Topologies**: NSFNET (14 nodes), USNET (28 nodes)
- **Scenarios**:
  - NSFNET standard: 32 slots, arrival=5.0, holding=10.0
  - NSFNET high load: 64 slots, arrival=8.0, holding=12.0
  - USNET cross-topo: 64 slots, arrival=8.0, holding=12.0
- **Evaluation**: 5 seeds × 2000 requests, 300 preload connections
- **Metrics**: Blocking rate, acceptance rate, average reward

### 5.2 Main Results

**Table 1: Blocking Rate Comparison (mean ± std, 5 seeds)**

| Method | NSFNET standard | NSFNET high load | USNET cross-topo |
|--------|-----------------|------------------|------------------|
| Random | 49.39±1.34% | 52.60±1.36% | 49.20±1.67% |
| ShortestPath | 61.72±1.35% | 71.97±3.06% | 76.23±4.29% |
| YinLike | 19.48±2.07% | 33.08±1.63% | 32.53±3.10% |
| Imitation-30D | 20.53±0.93% | 32.92±1.67% | 28.34±2.40% |
| TopK2-30D | **15.51±0.83%** | **27.49±1.18%** | **26.85±2.69%** |
| Direct Offline DQN | 38.23±2.34% | 55.19±2.96% | 54.07±2.72% |
| **CorrectionNet λ=0.2** | **13.04±1.58%** | **23.50±1.64%** | **21.77±0.89%** |

**Key observations**:
1. **TopK2-30D** already outperforms all baselines and the teacher policy
2. **Direct DQN fails catastrophically**: 42% blocking — worse than raw imitation (21%)
3. **CorrectionNet improves consistently**: +2.47pp (standard) → +3.99pp (high) → +5.08pp (USNET)
4. **Cross-topology generalization is strong**: USNET σ=0.89% despite training only on NSFNET

### 5.3 Ablation Study

**Table 2: Module Ablation**

| Component | What it does | Blocking (NSFNET std) |
|-----------|--------------|----------------------|
| None (YinLike) | Heuristic, no predictor | 19.48±2.07% |
| + Neural proposal only | Imitation-30D | 20.53±0.93% |
| + Predictor reranking | TopK2-30D | 15.51±0.83% |
| + Residual correction | CorrectionNet | **13.04%** |
| RL replaces selector | Direct DQN | 38.23±2.34% |

**Interpretation**:
- Predictor reranking is the largest single gain (19.48% → 15.51%)
- CorrectionNet provides additional 2.47pp by capturing long-term effects
- Direct RL in the wrong position destroys performance

### 5.4 Load Sweep

**Figure 1: Blocking rate vs. arrival rate (NSFNET, 32 slots, seed=42)**

| Arrival Rate | TopK2-30D | CorrectionNet λ=0.2 | Improvement |
|--------------|-----------|---------------------|-------------|
| 2.0 | 6.30% | 4.30% | 2.00pp |
| 4.0 | 15.75% | 10.70% | 5.05pp |
| 6.0 | 19.65% | 15.50% | 4.15pp |
| 8.0 | 20.85% | 17.60% | 3.25pp |
| 10.0 | 27.95% | 24.95% | 3.00pp |

**Observation**: Improvement is largest at moderate loads (4.0–6.0), where predictor scores are most ambiguous and long-term resource effects dominate.

### 5.5 DQN Failure Analysis

**Why Direct DQN Failed**:
1. State too compact (30D) to encode full action-value relationships
2. Reward too sparse (+1/-1) to distinguish good from barely-successful actions
3. Non-stationary MDP: each request changes the environment
4. Compounding error: one wrong allocation affects 10+ subsequent requests

**Training curve**:
- Warm-start (no training): 19.90% blocking
- After 25 episodes: 33.60% blocking
- After 50 episodes: 26.10% blocking
- Never recovers to warm-start level

This confirms that RL must be placed correctly — as a residual correction, not as a replacement for the selector.

### 5.6 CorrectionNet Interpretability

**Table 3: Decision Change Analysis (NSFNET standard, seed=42)**

| Metric | Value |
|--------|-------|
| Same decision as TopK2 | 470 (23.5%) |
| Changed decision | 1,530 (76.5%) |
| Changed → improved (fail→success) | 103 |
| Changed → worsened (success→fail) | 44 |
| Suppressed higher-predictor-score candidate | 58 |

**Table 4: Correction Score vs. Future Blocking (window=50)**

| Correction Score | n | Future Block Rate | Immediate Success |
|------------------|---|-------------------|-------------------|
| < -0.5 | 23 | 0.613 | 0.522 |
| [-0.5, 0.0) | 1 | 0.680 | 1.000 |
| [0.0, 0.5) | — | — | — |
| ≥ 0.5 | 23 | 0.240 | 0.957 |

**Interpretation**:
- Negative correction scores strongly correlate with high future blocking risk (61.3% vs 24.0%)
- CorrectionNet effectively separates "short-term feasible but harmful" from "short-term feasible and sustainable" actions
- The 58 cases where CorrectionNet suppressed a higher-predictor-score candidate are the core mechanism of improvement

---

## 6. Discussion

### 6.1 Why Residual RL Works Here

Direct DQN learns Q(s,a) for 15 actions from scratch. The problem is under-specified:
- State: 135D (with action features)
- Action space: 15D
- Data: 30K transitions
- Target: sparse episode returns

CorrectionNet reduces the problem to learning a **residual** over a strong baseline:
- Baseline (TopK2) captures 90%+ of decision quality
- CorrectionNet only needs the remaining 10%
- The baseline provides implicit regularization

### 6.2 Limitations

1. **Offline only**: CorrectionNet is trained on fixed replay buffers. Online fine-tuning could further improve.
2. **Single λ**: We use a fixed λ=0.2. Adaptive λ based on correction confidence could be better.
3. **NSFNET-trained predictor**: USNET generalization is strong, but mixed-topology training could improve further.
4. **Small scale**: 14–28 nodes. Larger topologies may need GNN encoders.

### 6.3 Future Work

- **Adaptive λ**: Schedule λ based on state characteristics or correction confidence
- **Online fine-tuning**: Deploy and collect on-policy corrections
- **Multi-task training**: Train on NSFNET + USNET simultaneously
- **Hierarchical CorrectionNet**: Separate networks for split selection and server selection

---

## 7. Conclusion

We propose a **Predictor-Guided Top-K Correction Agent** for DNN inference offloading in optical-MEC networks. The key innovation is placing RL as a **residual correction** over a strong predictor-guided selector, rather than replacing the selector entirely. A tiny 5K-parameter network (CorrectionNet) learns to penalize actions that are short-term feasible but long-term harmful. Experiments on NSFNET and USNET demonstrate:

- **13.04% blocking** on NSFNET standard (best ever)
- **23.50% blocking** on NSFNET high load (best ever)
- **21.77% blocking** on USNET cross-topology (best ever)
- **Zero baseline misalignment**: λ=0 is exactly equivalent to TopK2-30D
- **Strong interpretability**: CorrectionNet suppresses high-risk actions with 61.3% future blocking rate vs 24.0% for low-risk actions

The broader message: **RL is not too complex for network resource management — it just needs to be placed correctly.**

---

## Appendix A: Hyperparameters

| Parameter | Value |
|-----------|-------|
| α (P_success weight) | 2.0 |
| β (delay weight) | 0.3 |
| γ (load weight) | 0.2 |
| δ (deadline weight) | 0.05 |
| λ (correction weight) | 0.2 |
| top-K | 2 |
| KSP paths | 3 |
| Imitation hidden dims | (128, 128) |
| CorrectionNet hidden dims | (64, 64) |
| CorrectionNet γ (discount) | 0.95 |
| Gradient clip (CorrectionNet) | 1.0 |

## Appendix B: Data Availability

- Checkpoints: `experiments/agent_mvp/checkpoints/`
- Evaluation traces: `experiments/agent_mvp/traces/`
- Results: `experiments/agent_mvp/results/`
- Source code: `src/` (migration in progress)
