# Phase 1: v2b Predictor Validation Report
**Objective:** Ensure v2b predictor robustness and generalization across all topologies (14–100 nodes).  
**Date:** 2026-04-27

---

## 1. Intra-Topology Performance (Train & Test on Same Topology)

| Topology | Nodes | v0 AUC | v2b AUC | Δ AUC | v0 ECE | v2b ECE |
|----------|-------|--------|---------|-------|--------|---------|
| NSFNET | 14 | 0.9020 | **0.9944** | +0.092 | 0.022 | 0.012 |
| USNET | 28 | 0.8952 | **0.9528** | +0.058 | 0.013 | 0.025 |
| COST266 | 28 | 0.9092 | **0.9528** | +0.044 | 0.017 | 0.029 |
| Random50 | 50 | 0.9244 | **0.9545** | +0.030 | 0.030 | 0.018 |
| Random80 | 80 | 0.9378 | **0.9580** | +0.020 | 0.033 | 0.020 |
| Random100 | 100 | 0.8844 | **0.9294** | +0.045 | 0.071 | 0.040 |

**Conclusion:** v2b consistently dominates v0 across all scales. Even on 100-node random topology, v2b achieves AUC=0.929 when trained natively.

---

## 2. Cross-Topology Zero-Shot Transfer (NSFNET→X)

| Target | Nodes | v0 Cross | v2b Cross | Δ AUC | v0 ECE | v2b ECE |
|--------|-------|----------|-----------|-------|--------|---------|
| USNET | 28 | 0.824 | **0.913** | +0.089 | 0.084 | 0.154 |
| COST266 | 28 | 0.812 | **0.910** | +0.098 | 0.066 | 0.128 |
| Germany50 | 50 | 0.806 | **0.864** | +0.058 | 0.113 | 0.272 |
| Random50 | 50 | 0.806 | **0.855** | +0.049 | 0.103 | 0.171 |
| Random80 | 80 | 0.785 | **0.815** | +0.030 | 0.099 | 0.261 |
| Random100 | 100 | 0.768 | **0.797** | +0.029 | 0.155 | 0.207 |

**Conclusion:** v2b retains superior discriminative power in zero-shot transfer. However, calibration (ECE) degrades severely for v2b in cross-topology settings (up to 0.27 ECE).

---

## 3. Transfer Gap Analysis (Intra − Cross)

| Target | Nodes | v0 Gap | v2b Gap | v2b Gap / v0 Gap |
|--------|-------|--------|---------|------------------|
| USNET | 28 | 0.071 | **0.040** | 56% ✓ |
| COST266 | 28 | 0.098 | **0.043** | 44% ✓ |
| Random50 | 50 | 0.119 | **0.100** | 84% ✓ |
| Random80 | 80 | 0.153 | **0.143** | 94% ✓ |
| Random100 | 100 | 0.116 | **0.132** | 114% ✗ |

**Key Finding:** v2b's transfer gap advantage shrinks with topology scale. At 100 nodes, the gap is comparable to v0. Zero-shot from 14-node NSFNET to 100-node random is fundamentally difficult for both encoders.

---

## 4. Calibration Recovery (Platt Scaling, 2K samples)

| Target | v0 Raw ECE | v0 Platt | v2b Raw ECE | v2b Platt |
|--------|-----------|----------|-------------|-----------|
| USNET | 0.084 | **0.028** | 0.154 | **0.026** |
| COST266 | 0.066 | **0.030** | 0.128 | **0.029** |
| Germany50 | 0.113 | **0.041** | 0.272 | **0.037** |
| Random50 | 0.103 | **0.033** | 0.171 | **0.039** |
| Random80 | 0.099 | **0.013** | 0.261 | **0.038** |
| Random100 | 0.155 | **0.038** | 0.207 | **0.032** |

**Conclusion:** Platt Scaling with just 2,000 target-domain calibration samples effectively recovers calibration for BOTH encoders to ~0.03 ECE, neutralizing v2b's calibration disadvantage.

---

## 5. Robustness (Multi-Seed Stability)

### NSFNET — 5 seeds
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| AUC | 0.9916 | 0.0018 | 0.9884 | 0.9941 |
| Acc | 0.9554 | 0.0069 | 0.9463 | 0.9675 |
| ECE | 0.0156 | 0.0039 | 0.0106 | 0.0219 |

### Random100 — 5 seeds (different Waxman graphs + load)
| Seed | AUC | Acc | ECE | SR |
|------|-----|-----|-----|-----|
| 42 | 0.9375 | 0.8500 | 0.0431 | 0.616 |
| 123 | 0.9569 | 0.9575 | 0.0129 | 0.899 |
| 456 | 0.9498 | 0.8825 | 0.0445 | 0.722 |
| 789 | 0.9182 | 0.8425 | 0.0356 | 0.624 |
| 2024 | 0.9302 | 0.8581 | 0.0294 | 0.613 |
| **Mean±Std** | **0.9385±0.014** | **0.878±0.042** | **0.033±0.011** | — |

> Note: Variance across seeds reflects both (a) different Waxman graph topology and (b) different traffic load. AUC std=0.014 on 100-node random graphs is acceptable. The strong outlier at seed 123 (SR=0.899) generated an unusually sparse/connected graph with low blocking. ECE is well-controlled (mean 0.033) when trained natively.

---

## 6. Overall Assessment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| **Discriminative power (AUC)** | ✅ Excellent | v2b AUC >0.92 intra, >0.79 cross (100 nodes) |
| **Calibration (ECE)** | ⚠️ Needs post-hoc fix in transfer | Raw ECE up to 0.27 in cross; Platt fixes to ~0.03 |
| **Scalability** | ✅ Confirmed | Stable AUC 0.93–0.96 on 50–100 nodes intra |
| **Robustness to seeds** | ✅ Good (NSFNET) | AUC std=0.0018, very stable |
| **v2b vs v0 advantage** | ✅ Consistent | +0.02–0.09 AUC across all settings |
| **Zero-shot to 100 nodes** | ⚠️ Challenging | Gap >0.13 AUC; recommend fine-tuning or calibration |

---

## 7. Recommendations for Phase 2

1. **For small topologies (≤50 nodes):** v2b zero-shot transfer works well (AUC>0.85). Platt Scaling is sufficient.
2. **For large topologies (80–100 nodes):** Zero-shot from NSFNET loses >0.13 AUC. Consider:
   - Fine-tuning on 2K–5K target samples (not just calibration)
   - Or training on a diverse mixture of topologies
3. **Temperature Scaling is unreliable** — recommend Platt Scaling as the default post-hoc method.
