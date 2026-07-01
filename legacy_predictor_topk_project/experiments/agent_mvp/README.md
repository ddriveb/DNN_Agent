# Phase A: Predictor-guided Rule Agent (MVP)

This is the **Phase A** implementation of the agent development roadmap: a rule-based heuristic agent that uses the pretrained predictor to guide high-level action selection.

## Philosophy

Before jumping to RL (DQN/PPO), we first validate that the predictor + selector + mapper loop has value. The Rule Agent:

1. Enumerates all `(split_point, target_server)` combinations
2. Queries the predictor for each combo to get `P_success` and `delay_pred`
3. Scores each combo with a weighted heuristic
4. Picks the highest-scoring action

If this simple approach outperforms baselines, it proves the framework direction is correct.

---

## File Structure

```
experiments/agent_mvp/
├── README.md                    # This file
├── dnn_models.py                # DNN model registry
├── mec_servers.py               # MEC server cluster
├── traffic_generator.py         # Request generator
├── rule_agent.py                # Base Rule Agent
├── pruned_rule_agent.py         # Rule Agent with action pruning
├── adaptive_rule_agent.py       # Load-aware adaptive strategy selector
├── baselines.py                 # Random, ShortestPath, YinLike, LoadBalanced
├── env_wrapper.py               # Gym-like env wrapper
├── fixed_trace.py               # Fixed request trace generator/loader
├── eval_fixed.py                # Fixed-trace evaluation
├── eval_comprehensive.py        # All baselines × predictors × scenarios
├── eval_robustness.py           # Multi-seed + load sweep
├── eval_adaptive.py             # AdaptiveRuleAgent evaluation
├── eval_pruning.py              # Action pruning impact
├── search_rule_weights.py       # Random search for optimal score weights
├── compare_predictors.py        # Compare predictor models in closed loop
├── analyze_predictors.py        # Predictor calibration/discrimination analysis
├── summarize_results.py         # Result summarizer
└── results/                     # JSON/CSV outputs
```

---

## Quick Start

### Run the comprehensive evaluation

```bash
cd experiments/agent_mvp
python eval_comprehensive.py --seed 42
```

### Run multi-seed robustness + load sweep

```bash
python eval_robustness.py
```

### Run AdaptiveRuleAgent evaluation

```bash
python eval_adaptive.py
```

### Analyze predictor behavior

```bash
python analyze_predictors.py
```

---

## Key Results

### 1. Multi-Seed Robustness (5 seeds, 2000 requests each)

#### NSFNET Standard Load (32 slots, arr=5.0, HT=10.0)

| Agent | Blocking% mean±std | vs YinLike |
|-------|-------------------|------------|
| YinLike | 19.48±2.07% | baseline |
| RA-default+mixed | 16.87±1.03% | +2.61% |
| **RA-default+nsfnet** | **15.70±1.49%** | **+3.78%** |
| RA-pruned+mixed | 16.36±0.76% | +3.12% |
| RA-pruned+nsfnet | 15.83±0.88% | +3.65% |
| RA-tuned+mixed | 17.17±1.39% | +2.31% |
| RA-tuned+nsfnet | 16.49±1.21% | +2.99% |
| **AdaptiveRA** | **16.23±1.13%** | **+3.25%** |

#### NSFNET High Load (64 slots, arr=8.0, HT=12.0)

| Agent | Blocking% mean±std | vs YinLike |
|-------|-------------------|------------|
| YinLike | 33.08±1.63% | baseline |
| RA-default+mixed | 30.31±1.53% | +2.77% |
| **RA-default+nsfnet** | **28.40±1.35%** | **+4.68%** |
| RA-pruned+mixed | 31.34±1.68% | +1.74% |
| RA-pruned+nsfnet | 28.75±1.38% | +4.33% |
| RA-tuned+mixed | 29.95±1.38% | +3.13% |
| **RA-tuned+nsfnet** | **28.15±2.11%** | **+4.93%** |
| AdaptiveRA | 29.27±1.21% | +3.81% |

#### USNET Cross-Topology (64 slots, arr=8.0, HT=12.0)

| Agent | Blocking% mean±std | vs YinLike |
|-------|-------------------|------------|
| YinLike | 32.53±3.10% | baseline |
| RA-default+mixed | 26.83±2.52% | +5.70% |
| **RA-default+nsfnet** | **25.86±3.52%** | **+6.67%** |
| RA-pruned+mixed | 26.15±3.05% | +6.38% |
| RA-pruned+nsfnet | 25.45±2.78% | +7.08% |
| RA-tuned+mixed | 26.39±2.28% | +6.14% |
| RA-tuned+nsfnet | 26.50±3.62% | +6.03% |
| AdaptiveRA | 26.31±2.69% | +6.22% |

### 2. Load Sweep (NSFNET, 32 slots, seed=42)

| ArrRate | YinLike | RA-default+nsfnet | RA-pruned+mixed | AdaptiveRA |
|--------:|--------:|------------------:|----------------:|-----------:|
| 2.0 | 10.65% | **6.60%** | 7.80% | ~7.2% |
| 4.0 | 15.55% | **13.10%** | 15.00% | ~14.0% |
| 6.0 | 20.05% | 20.35% | **18.95%** | ~19.7% |
| 8.0 | 22.25% | **19.80%** | 22.95% | ~21.4% |
| 10.0 | 32.00% | 31.65% | **31.25%** | ~31.4% |

### 3. Predictor Analysis (Why nsfnet-specific beats mixed on USNET?)

| Metric | mixed-v2b (NSFNET) | nsfnet-v2b (NSFNET) | mixed-v2b (USNET) | nsfnet-v2b (USNET) |
|--------|-------------------|--------------------|-------------------|--------------------|
| AUC | 0.9896 | **0.9928** | **0.9355** | 0.8978 |
| Brier | 0.0367 | **0.0282** | **0.0853** | 0.1721 |
| ECE | 0.0304 | **0.0147** | **0.0651** | 0.1679 |
| Prob Gap | 0.7372 | **0.8434** | 0.6107 | **0.6504** |

**Key insight**: nsfnet-v2b has worse AUC/Brier/ECE on USNET, but its **P(success|fail) = 0.131** vs mixed's **0.267**. It is more "decisive" in assigning low probabilities to failures. This decisiveness is more valuable in closed-loop decision-making than overall calibration.

---

## Core Findings

### Finding 1: Predictor-guided RuleAgent significantly outperforms baselines

- **USNET cross-topology**: YinLike 32.53% → RA-default+nsfnet 25.86%, **relative reduction 20.5%**
- **NSFNET high load**: YinLike 33.08% → RA-tuned+nsfnet 28.15%, **relative reduction 14.9%**
- **NSFNET standard**: YinLike 19.48% → RA-default+nsfnet 15.70%, **relative reduction 19.4%**

### Finding 2: Optimal strategy is load-dependent

| Load Level | Best Fixed Strategy | Why |
|-----------|---------------------|-----|
| Low (arr ≤ 4) | RA-default + nsfnet | Sharp predictor, avoid over-pruning |
| Medium (arr ≈ 6) | RA-pruned + mixed | Balance between confidence and action space |
| High (arr ≥ 8) | RA-default + nsfnet | Don't prune, need all options |

### Finding 3: Action pruning is conditionally effective

- **Helps** in standard/medium load (excludes clearly bad actions)
- **Hurts** in high load (may prune the only feasible option)
- **Mis-prune rate = 0%** with conservative thresholds

### Finding 4: AUC ≠ closed-loop performance

nsfnet-v2b has **lower AUC** (0.8978) than mixed (0.9355) on USNET, but achieves **lower blocking rate** (20.05% vs 24.50%).

Reason: nsfnet-v2b assigns **lower P_success to failure cases** (0.131 vs 0.267), making the Agent more confident in rejecting bad actions. This "discrimination decisiveness" matters more than overall calibration in the decision loop.

---

## Design Decisions

1. **Fixed request traces**: All agents evaluated on identical request sequences for fair comparison.
2. **Shared env state**: Agents must share `env.mec` and `env.encoder` with the environment.
3. **MEC compute tracking**: Compute resources allocated on success, released with lightpath expiration.
4. **Reward v1**: `+1.0 - delay_norm` on success, `-2.0` on failure.

---

## Next Steps

| Phase | Task |
|-------|------|
| **B** | Imitation learning: train MLP to mimic AdaptiveRuleAgent per scenario |
| **C** | DQN: replace rule-based scoring with learned Q-values |

The `env_wrapper.py` provides a `step()` interface compatible with RL libraries.
