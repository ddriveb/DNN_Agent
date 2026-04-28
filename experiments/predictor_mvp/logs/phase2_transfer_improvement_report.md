# Phase 2: Fine-Tuning & Mixed Training — Results Report
**Date:** 2026-04-27  
**Objective:** Close the large-topology transfer gap (NSFNET→Random100) via fine-tuning and mixed training.

---

## 1. Experiment A: Fine-Tuning on Target Domain (Random100)

**Setup:** Pre-train on NSFNET (15K, 30 epochs), then fine-tune on Random100 with varying data budgets.

| Strategy | FT Samples | FT Epochs | AUC | ECE (Platt) | vs Zero-shot |
|----------|-----------|-----------|-----|-------------|--------------|
| Intra-topology (upper bound) | — | — | **0.9294** | 0.040 | — |
| Zero-shot (NSFNET→Random100) | — | — | 0.7970 | 0.207 | baseline |
| Fine-tune 500 | 500 | 5 | 0.7845 | 0.186 | **−0.012** |
| Fine-tune 2K | 2,000 | 5 | 0.7749 | 0.029 | **−0.022** |
| Fine-tune 5K | 5,000 | 10 | 0.8074 | 0.086 | **+0.010** |

### Key Finding
**Fine-tuning is surprisingly ineffective.** Even with 5K target samples (10 epochs), AUC only improves by +0.01 over zero-shot. Smaller budgets (500–2K) actually **hurt** AUC, suggesting catastrophic forgetting or overfitting.

The transfer gap to intra-topology upper bound remains **>0.12 AUC** even after 5K fine-tuning.

---

## 2. Experiment B: Mixed-Topology Pre-Training

**Setup:** Train on mixed data: NSFNET (5K) + USNET (5K) + Random50 (5K) = 15K total. Test zero-shot on Random100.

| Strategy | AUC | ECE | ECE (+Platt) |
|----------|-----|-----|--------------|
| NSFNET-only → Zero-shot | 0.7970 | 0.207 | 0.033* |
| Mixed → Zero-shot | **0.8238** | 0.146 | 0.033 |
| Mixed → FT 2K + Platt | 0.8100 | — | 0.036 |

\* Platt from cross-calibration experiment (2K calib samples)

### Key Finding
**Mixed training provides a modest but stable improvement.** Zero-shot AUC improves from 0.797 → 0.824 (+0.027). Unlike fine-tuning, this gain does not require any target-domain data.

Interestingly, fine-tuning the mixed model on 2K Random100 samples **degrades** performance (0.824 → 0.810), reinforcing the pattern that small-sample fine-tune is harmful.

---

## 3. Head-to-Head Comparison (All Strategies on Random100)

| Rank | Strategy | AUC | ECE (best) | Needs Target Data? |
|------|----------|-----|------------|-------------------|
| 1 | Intra-topology (native train) | 0.929 | 0.040 | 10K samples |
| 2 | Mixed pre-train → Zero-shot | 0.824 | 0.033 | None |
| 3 | NSFNET + Platt Scaling | 0.797 | 0.032 | 2K calib |
| 4 | Fine-tune 5K@10ep | 0.807 | 0.086 | 5K samples |
| 5 | Mixed → Fine-tune 2K | 0.810 | 0.036 | 2K samples |
| 6 | Fine-tune 2K@5ep | 0.775 | 0.029 | 2K samples |
| 7 | Fine-tune 500@5ep | 0.785 | 0.186 | 500 samples |

---

## 4. Critical Insights

### 4.1 Fine-tuning does NOT solve the transfer gap
- 500–2K sample fine-tune **degrades** AUC compared to zero-shot.
- 5K samples barely help (+0.01 AUC).
- The gap to intra-topology upper bound (>0.12 AUC) persists even after fine-tuning.

**Hypothesis:** The predictor's learned representations are not sufficiently topology-agnostic. Fine-tuning on small target samples shifts the decision boundary but does not learn new topology-invariant features. The model "forgets" useful pre-trained features faster than it learns new ones.

### 4.2 Mixed training is the best no-target-data strategy
- +0.027 AUC over NSFNET-only with zero additional target data.
- Calibration is still good (ECE 0.033 with Platt).
- **Recommendation:** For deployment across diverse topologies, pre-train on a mixture of small, medium, and large graphs rather than a single source topology.

### 4.3 Platt Scaling remains the most efficient fix
- Zero-shot + Platt achieves AUC=0.797, ECE=0.032 with only **2K calibration samples**.
- Fine-tuning 5K samples achieves AUC=0.807, ECE=0.086 — slightly better AUC but worse ECE, and requires 2.5× more data.
- **Trade-off:** If the goal is calibrated probability estimates (e.g., for admission control), Platt on zero-shot is superior. If the goal is raw discriminative accuracy, mixed training is better.

---

## 5. Recommendations

### For small topologies (≤50 nodes)
✅ **Current solution is sufficient.** v2b zero-shot + Platt Scaling achieves AUC > 0.85 with excellent calibration. No fine-tuning needed.

### For large topologies (80–100 nodes)
The transfer gap is real and not easily closed. Three paths:

1. **Best AUC, no target training data:** Mixed pre-training (AUC=0.824)
2. **Best calibration, minimal target data:** Zero-shot + Platt (AUC=0.797, ECE=0.032, needs 2K calib)
3. **Best overall if target data available:** Train natively on target topology (AUC=0.929, needs 10K)

**Fine-tuning is NOT recommended** unless you have >10K target samples and can afford many epochs.

### For future work
To truly close the gap, consider:
- **Topology-aware architectures:** Graph Neural Networks (GNN) that consume the network graph directly
- **Meta-learning:** MAML-style adaptation that learns how to adapt quickly to new topologies
- **Larger mixed pre-training:** Include more diverse topologies (mesh, ring, Waxman with varying α/β)
