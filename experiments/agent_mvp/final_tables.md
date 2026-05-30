# Phase A + B Final Experimental Tables

All results are from fixed-trace evaluation with 2000 requests per seed, averaged over 5 seeds (42, 123, 456, 789, 2024).

---

## Table 1: Multi-Seed Robustness (Blocking Rate %, mean ± std)

| Agent | NSFNET Standard (32s, arr=5) | NSFNET High Load (64s, arr=8) | USNET Cross-Topo (64s, arr=8) |
|-------|------------------------------|-------------------------------|-------------------------------|
| Random | 48.55±1.23% | 54.65±1.45% | 48.25±2.10% |
| ShortestPath | 62.60±1.88% | 72.00±1.92% | 74.60±2.35% |
| **YinLike** | **19.48±2.07%** | **33.08±1.63%** | **32.53±3.10%** |
| LoadBalanced | 17.90±2.05% | 31.35±1.63% | 30.90±3.08% |
| RA-default+mixed | 16.87±1.03% | 30.31±1.53% | 26.83±2.52% |
| RA-default+nsfnet | **15.70±1.49%** | **28.40±1.35%** | **25.86±3.52%** |
| RA-tuned+mixed | 17.17±1.39% | 29.95±1.38% | 26.39±2.28% |
| RA-tuned+nsfnet | 16.49±1.21% | 28.15±2.11% | 26.50±3.62% |
| RA-pruned+mixed | 16.36±0.76% | 31.34±1.68% | 26.15±3.05% |
| RA-pruned+nsfnet | 15.83±0.88% | 28.75±1.38% | 25.45±2.78% |
| **AdaptiveRA** | **16.23±1.13%** | **29.27±1.21%** | **26.31±2.69%** |
| ImitationAgent (Phase B) | 21.84±1.48% | 33.39±2.11% | **27.36±2.14%** |
| **TopK3-30D (Phase B+)** | **15.69±1.02%** | **26.30±0.64%** | **26.07±2.73%** |

**Key finding**: RuleAgent variants consistently outperform baselines. AdaptiveRA approaches scene-oracle-best without requiring prior knowledge of the scenario. Imitation Agent (behavior cloning, val_acc=73.7%) already beats YinLike on USNET (+15.9% relative reduction), confirming the teacher policy is learnable.

---

## Table 2: Load Sweep on NSFNET (32 slots, seed=42)

| ArrRate | YinLike | RA-default+mixed | RA-default+nsfnet | RA-tuned+mixed | RA-tuned+nsfnet | RA-pruned+mixed | RA-pruned+nsfnet | AdaptiveRA |
|--------:|--------:|-----------------:|--------------------:|---------------:|----------------:|----------------:|-----------------:|-----------:|
| 2.0 | 10.65% | 8.35% | **6.60%** | 7.00% | 7.50% | 7.80% | 7.45% | ~7.2% |
| 4.0 | 15.55% | 16.10% | **13.10%** | 15.30% | 15.15% | 15.00% | 13.60% | ~14.0% |
| 6.0 | 20.05% | 18.75% | 20.35% | 20.50% | 18.55% | **18.95%** | 18.00% | ~19.7% |
| 8.0 | 22.25% | 23.40% | **19.80%** | 23.20% | 22.40% | 22.95% | 23.25% | ~21.4% |
| 10.0 | 32.00% | 33.00% | 31.65% | 32.50% | 32.75% | **31.25%** | 32.00% | ~31.4% |

**Key finding**: No single fixed strategy is optimal across all loads. Low load favors sharp predictors without pruning; medium/high load benefits from pruning or tuned weights.

---

## Table 3: Predictor Behavior Analysis (5000 test samples)

| Predictor | Topology | AUC | Brier | ECE | P(success\|label=1) | P(success\|label=0) | Gap |
|-----------|----------|-----|-------|-----|--------------------:|--------------------:|----:|
| mixed-v2b | NSFNET | 0.9896 | 0.0367 | 0.0304 | 0.9682 | 0.2310 | 0.7372 |
| nsfnet-v2b | NSFNET | **0.9928** | **0.0282** | **0.0147** | **0.9747** | **0.1313** | **0.8434** |
| mixed-v2b | USNET | **0.9355** | **0.0853** | **0.0651** | **0.8778** | 0.2671 | 0.6107 |
| nsfnet-v2b | USNET | 0.8978 | 0.1721 | 0.1679 | 0.7812 | **0.1308** | **0.6504** |

**Key finding**: nsfnet-v2b has worse AUC/ECE on USNET, but its P(success|fail) is much lower (0.131 vs 0.267). This conservative separation of infeasible actions is more valuable for closed-loop decision-making than global ranking accuracy.

---

## Table 4: AdaptiveRuleAgent Strategy Distribution (seed=42)

| Scenario | default+nsfnet | pruned+mixed | Load Level |
|----------|---------------:|-------------:|------------|
| NSFNET standard | 56.4% | 43.6% | medium |
| NSFNET high load | 52.9% | 47.1% | high (but estimated medium) |
| USNET cross-topo | 74.6% | 25.4% | medium |

**Key finding**: AdaptiveRA dynamically switches between strategies based on recent acceptance rate, spending more time on the sharper predictor in uncertain scenarios.

---

## Table 5: Imitation Agent Closed-Loop (5 seeds, 2000 requests)

| Scenario | YinLike | ImitationAgent | vs YinLike |
|----------|---------|---------------|------------|
| NSFNET standard | 19.48±2.07% | **21.84±1.48%** | -12.1% |
| NSFNET high load | 33.08±1.63% | **33.39±2.11%** | -0.9% |
| USNET cross-topo | 32.53±3.10% | **27.36±2.14%** | **+15.9%** |

**Key finding**: Imitation Agent (behavior cloning from AdaptiveRA, val_acc=73.7%) outperforms YinLike on USNET (+15.9% relative reduction), but underperforms on NSFNET standard load. This reflects the compounding error of 26.3% action mismatch in the closed loop. Nevertheless, it demonstrates that a lightweight MLP can learn a nontrivial approximation of the teacher policy, providing a viable warm-start for subsequent RL fine-tuning.

---

## Table 6: Top-K Selector Results (5 seeds, 2000 requests)

| Scenario | YinLike | AdaptiveRA | Imitation-21D | Imitation-30D | **TopK3-21D** | **TopK3-30D** |
|----------|---------|-----------|---------------|---------------|---------------|---------------|
| NSFNET standard | 19.48±2.07% | 16.23±1.13% | 21.84±1.48% | 20.53±0.93% | **16.10±0.95%** | **15.69±1.02%** |
| NSFNET high | 33.08±1.63% | 29.27±1.21% | 33.39±2.11% | 32.92±1.67% | **26.17±1.40%** | **26.30±0.64%** |
| USNET cross-topo | 32.53±3.10% | 26.31±2.69% | 27.36±2.14% | 28.34±2.40% | 27.70±2.83% | **26.07±2.73%** |

**Key finding**: Top-K Selector (neural top-3 proposal + predictor re-ranking) outperforms the teacher AdaptiveRA in **all three scenarios**. On NSFNET standard, it achieves **15.69%** blocking vs AdaptiveRA 16.23% and YinLike 19.48%. This validates the hybrid architecture: the neural network learns coarse action ranking (96.3% top-3 hit rate), while the predictor provides fine-grained feasibility scoring for final selection.

---

## Stage Conclusion

> Predictor-guided RuleAgent achieves 15%–20% relative blocking reduction across standard load, high load, and cross-topology scenarios. No single fixed strategy is universally optimal; AdaptiveRuleAgent dynamically selects between predictor models and score policies based on recent network state, approaching scene-oracle-best without prior scenario knowledge. Crucially, we observe that a predictor's closed-loop value depends not only on AUC but also on its conservative separation of infeasible actions — nsfnet-v2b outperforms the higher-AUC mixed model on USNET because it assigns lower success probabilities to failed actions, enabling more confident rejection of poor candidates.
>
> Furthermore, a lightweight MLP trained via behavior cloning on AdaptiveRA decisions (36K samples, 21-dim state, 15 actions) achieves 73.7% action-matching accuracy on validation. While top-1 imitation underperforms the teacher on NSFNET standard load due to compounding error, **error analysis reveals a 96.3% top-3 hit rate**. Exploiting this insight, a **Top-K Selector** architecture — neural top-3 proposal followed by predictor-guided re-ranking — surpasses the teacher AdaptiveRA in all scenarios, achieving **15.69%** blocking on NSFNET standard, **26.30%** on high load, and **26.07%** on USNET cross-topology. This hybrid approach leverages the complementary strengths of neural policy approximation and explicit predictor-based scoring, establishing a strong baseline for future end-to-end RL fine-tuning.
