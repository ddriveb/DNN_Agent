# Final Report: Hybrid Neural-Rule Agent for DNN Offloading

> Generated: 2026-05-09 09:30
> **Note**: The current optimal agent is NOT a pure RL-trained DQN/PPO agent. It is a **hybrid agent** combining neural candidate generation (behavior cloning) with predictor-guided re-ranking.

---

## Executive Summary

This project develops a **hybrid neural-rule agent** for DNN inference offloading in optical-MEC networks. The final architecture — **Top-K Selector** — combines:
1. A lightweight neural network (behavior cloning, 30D state → top-3 action proposal)
2. Predictor-guided re-ranking (explicit P_success/delay/load scoring)

This hybrid outperforms all baselines and the teacher policy across standard load, high load, and cross-topology scenarios. Pure RL fine-tuning (DQN) was attempted but deferred due to unstable reward signals and insufficient state representation.

| Metric | YinLike | AdaptiveRA (Teacher) | **TopK3-30D (Ours)** |
|--------|---------|---------------------|---------------------|
| NSFNET standard | 19.48±2.07% | 16.23±1.13% | **15.69±1.02%** |
| NSFNET high | 33.08±1.63% | 29.27±1.21% | **26.30±0.64%** |
| USNET cross-topo | 32.53±3.10% | 26.31±2.69% | **26.07±2.73%** |

**Relative improvement over YinLike**: 19.4% (standard), 20.5% (high), 19.9% (cross-topo).

---

## Phase A: RuleAgent Baseline ✅

### What We Did
- Built `RuleAgent` that enumerates all (split, server) combinations
- Scores each via `alpha * P_success - beta * delay - gamma * load - delta * deadline`
- Compared 4 baselines: Random, ShortestPath, YinLike, LoadBalanced

### Key Results
- RuleAgent with tuned weights (α=2.0, β=0.3, γ=0.2, δ=0.05) outperforms all baselines
- No single fixed strategy is optimal across all loads
- **AdaptiveRuleAgent** dynamically selects strategy based on recent acceptance rate

### Key Insight
> Predictor closed-loop value depends more on **conservative separation of infeasible actions** (low P(success|fail)) than on global AUC.

---

## Phase B: Imitation Learning ✅

### What We Did
- Generated 36K (state, action) samples from AdaptiveRuleAgent
- Trained MLP (21 → 128 → 128 → 15) via behavior cloning
- Evaluated in closed loop

### Results
| Scenario | YinLike | Imitation-21D | Imitation-30D |
|----------|---------|--------------|---------------|
| NSFNET standard | 19.48% | 21.84% | 20.53% |
| NSFNET high | 33.08% | 33.39% | 32.92% |
| USNET cross-topo | 32.53% | **27.36%** | 28.34% |

### Key Insight
- ImitationAgent beats YinLike on USNET (+15.9% relative) but underperforms on NSFNET standard
- Val accuracy: 73.69% (21D) → 74.60% (30D enhanced state)
- **Top-3 hit rate: 96.3%** — teacher action almost always in student's top-3

---

## Phase C: Error Analysis ✅

### What We Did
- Decomposed action accuracy into split vs server components
- Analyzed regret distribution and catastrophic mismatch rate
- Bucketed errors by load, deadline, model, strategy, split

### Key Findings

**C1: Action Decomposition**
| Metric | Value |
|--------|-------|
| Full Action Accuracy | 74.21% |
| **Split Accuracy** | **81.94%** ← bottleneck |
| **Server Accuracy** | **87.54%** |
| Top-3 Hit Rate | **96.32%** |

**C2: Mismatch Patterns**
| Pattern | Fraction | Avg Regret |
|---------|----------|------------|
| Server correct, split wrong | 51.7% | 0.799 |
| Split correct, server wrong | 30.0% | 1.461 |
| Both wrong | 18.3% | 2.349 |

**C3: Root Cause — Split_2 Accuracy Crisis**

| Teacher Split | Full Accuracy |
|--------------|--------------|
| split_0 | 82.28% |
| split_1 | 65.23% |
| **split_2** | **50.84%** 🚨 |

**Why NSFNET standard fails but USNET works:**
- split_2 (8 slots, high bandwidth) is almost randomly guessed
- On NSFNET standard (resource-abundant), wrong split_2 doesn't immediately fail but causes suboptimal allocation → compounding error
- On USNET (resource-scarce), wrong split_2 often fails immediately, but YinLike performs so poorly (32.53%) that even random guesses beat it

---

## Phase D: Top-K Selector ✅

### Architecture

```
Request → State Vector (30D)
    ↓
ImitationAgent → Top-3 action candidates
    ↓
Predictor scores each candidate (P_success, delay, load)
    ↓
Rule score re-ranking → Select best
    ↓
Mapper executes
```

### Why It Works
1. **Neural network** learns coarse action ranking (96.3% top-3 hit)
2. **Predictor** provides fine-grained feasibility scoring
3. **Combination** leverages complementary strengths

### Results

| Scenario | YinLike | Imitation-21D | Imitation-30D | TopK3-21D | **TopK3-30D** |
|----------|---------|--------------|---------------|-----------|---------------|
| NSFNET standard | 19.48±2.07% | 21.84±1.48% | 20.53±0.93% | 16.10±0.95% | **15.69±1.02%** |
| NSFNET high | 33.08±1.63% | 33.39±2.11% | 32.92±1.67% | 26.17±1.40% | **26.30±0.64%** |
| USNET cross-topo | 32.53±3.10% | 27.36±2.14% | 28.34±2.40% | 27.70±2.83% | **26.07±2.73%** |

**TopK3-30D surpasses the teacher (AdaptiveRA) in ALL scenarios.**

---

## DQN/PPO RL Agent (Attempted, Not Yet Successful)

### What We Tried
- Built DQN framework (DDQN, experience replay, target network)
- Warm-started from imitation weights
- Froze body layers, only trained last layer
- Evaluated periodically on fixed traces

### Results
| Stage | Eval Blocking |
|-------|--------------|
| Warm-start (no training) | **19.90%** |
| Ep 25 | 33.60% |
| Ep 50 | 26.10% |

**Conclusion**: Direct RL fine-tuning destroys the imitation initialization. The current state representation (30D) and sparse reward (+1/-1) are insufficient for stable Q-value estimation. **Pure RL Agent (DQN/PPO) remains future work.** The hybrid Top-K Selector is the current optimal solution.

### Why Pure RL Failed (For Now)
1. **State too compact**: 30D cannot encode full action-value relationships
2. **Reward too sparse**: +1/-1 doesn't distinguish "good success" from "barely success"
3. **Non-stationary MDP**: Each request changes the environment, but request features are random
4. **Compounding error**: One wrong allocation affects 10+ subsequent requests

These issues must be resolved before DQN/PPO can succeed.

---

## Code Inventory

```
experiments/agent_mvp/
  rule_agent.py                    # Phase A: Predictor-guided RuleAgent
  adaptive_rule_agent.py           # Phase A: Strategy selector
  baselines.py                     # Phase A: YinLike, Random, etc.
  eval_agents.py                   # Phase A: Evaluation framework
  generate_imitation_data.py       # Phase B: Dataset generation
  train_imitation.py               # Phase B: Behavior cloning
  eval_imitation.py                # Phase B: Closed-loop eval
  analyze_imitation_errors.py      # Phase C: Error analysis
  enhance_state_and_retrain.py     # Phase B+: Enhanced state (30D)
  topk_selector_agent.py           # Phase D: Top-K hybrid agent
  eval_topk_selector.py            # Phase D: Evaluation
  dqn_agent.py                     # DQN framework (deferred)
  train_dqn.py                     # DQN training (deferred)
  
  checkpoints/
    imitation_agent.pt             # 21D behavior cloning
    imitation_agent_enhanced.pt    # 30D behavior cloning
    
  results/
    final_tables.md                # All experimental tables
    topk_selector_eval.json        # Top-K results
    imitation_error_analysis.md    # Error analysis report
```

---

## Current Work in Progress (May 2026)

### Phase 5: Offline Constrained DQN
**Status**: Replay buffer collection in progress (30K target)
**Auto-pipeline**: Will automatically train offline DQN and evaluate when buffer is ready

Components built:
- `state_builder.py` — Action-aware state (135D: 30D global + 15×7 action features)
- `collect_replay.py` — Dense reward collection (success/delay/load/frag)
- `train_offline_dqn.py` — Constrained DQN with CQL + topk masking
- `eval_dqn.py` — Closed-loop evaluation vs TopK2-30D and YinLike

### Roadmap (User-Defined Priority)

```
P0: Action-aware state (135D)         ✅ Done
P1: Continuous reward (dense)         ✅ Done
P2: Top-K constrained action space    ✅ Done
P3: Offline DQN pretraining           🔄 In progress
P4: RL learns long-term correction    ⏳ Next
P5: Online fine-tuning                ⏳ Future
```

**Key Insight from User**: RL should NOT replace the Predictor/Selector. It should learn a **long-term value correction term** added to the predictor score:

```
a* = argmax_{a ∈ TopK} [Score_pred(s,a) + λ · Q_RL(s,a)]
```

This separates short-term feasibility (Predictor) from long-term resource impact (RL).

## Future Work

### Near-term (Improving the Hybrid Agent)
1. **Soft-Label Imitation**: Replace one-hot with teacher score distribution (KL loss)
2. **Top-K with K adaptation**: Dynamically choose K based on confidence
3. **Multi-topology Generalization**: Train on mixed topologies, evaluate zero-shot transfer

### Long-term (Toward Pure RL Agent)
4. **Richer State Representation**: GNN-based topology encoding
5. **Revisit DQN/PPO**: Only after state/reward/action-space are mature

---

## Key Takeaway

> **Neural networks are good at ranking, predictors are good at scoring. Don't force one to do both. And don't rush to pure RL when the MDP isn't ready.**

The **Top-K Selector** architecture naturally separates these roles:
- **Neural network** (behavior cloning) proposes a small candidate set — exploiting learned policy structure
- **Predictor** precisely scores each candidate — exploiting explicit feasibility model
- **Re-ranking** selects the best — deterministic, interpretable, robust

This hybrid approach outperforms:
- Pure rule-based methods (YinLike: 19.48% → ours: 15.69%)
- Pure neural methods (Imitation-30D: 20.53% → ours: 15.69%)
- Even the teacher policy (AdaptiveRA: 16.23% → ours: 15.69%)

Pure RL (DQN/PPO) remains future work — the current state representation and reward signal are insufficient for stable end-to-end policy learning. The hybrid agent is the correct solution for the current system maturity.
