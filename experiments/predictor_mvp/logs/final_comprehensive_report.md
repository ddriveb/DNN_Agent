# Final Comprehensive Report: v2b Predictor Validation & Transfer Learning
**Date:** 2026-04-27  
**Objective:** Validate v2b predictor robustness, generalization, and explore GNN as an alternative architecture.

---

## Part 1: Intra-Topology Performance (Train & Test on Same Topology)

| Topology | Nodes | v0 AUC | v2b AUC | Δ AUC | v0 ECE | v2b ECE |
|----------|-------|--------|---------|-------|--------|---------|
| NSFNET | 14 | 0.902 | **0.994** | +0.092 | 0.022 | 0.012 |
| USNET | 28 | 0.895 | **0.953** | +0.058 | 0.013 | 0.025 |
| COST266 | 28 | 0.909 | **0.953** | +0.044 | 0.017 | 0.029 |
| Random50 | 50 | 0.924 | **0.955** | +0.030 | 0.030 | 0.018 |
| Random80 | 80 | 0.938 | **0.958** | +0.020 | 0.033 | 0.020 |
| Random100 | 100 | 0.884 | **0.929** | +0.045 | 0.071 | 0.040 |

**Conclusion:** v2b consistently outperforms v0 across all scales. Native training achieves AUC > 0.92 even on 100-node random topology.

---

## Part 2: Cross-Topology Zero-Shot Transfer (NSFNET→X)

| Target | Nodes | v0 Cross | v2b Cross | v0 ECE | v2b ECE | v2b Gap (Intra−Cross) |
|--------|-------|----------|-----------|--------|---------|----------------------|
| USNET | 28 | 0.824 | **0.913** | 0.084 | 0.154 | 0.040 |
| COST266 | 28 | 0.812 | **0.910** | 0.066 | 0.128 | 0.043 |
| Germany50 | 50 | 0.806 | **0.864** | 0.113 | 0.272 | — |
| Random50 | 50 | 0.806 | **0.855** | 0.103 | 0.171 | 0.100 |
| Random80 | 80 | 0.785 | **0.815** | 0.099 | 0.261 | 0.143 |
| Random100 | 100 | 0.768 | **0.797** | 0.155 | 0.207 | 0.132 |

**Conclusion:** v2b retains superior discriminative power in zero-shot transfer, but calibration degrades severely (ECE up to 0.27). Transfer gap grows with topology scale.

---

## Part 3: Calibration Recovery (Platt Scaling, 2K Samples)

| Target | v2b Raw ECE | v2b Platt ECE | Recovery |
|--------|------------|--------------|----------|
| USNET | 0.154 | **0.026** | ✓ |
| COST266 | 0.128 | **0.029** | ✓ |
| Germany50 | 0.272 | **0.037** | ✓ |
| Random50 | 0.171 | **0.039** | ✓ |
| Random80 | 0.261 | **0.038** | ✓ |
| Random100 | 0.207 | **0.032** | ✓ |

**Conclusion:** Platt Scaling with 2K target-domain calibration samples consistently recovers calibration to ECE ≈ 0.03 across all topologies.

---

## Part 4: Fine-Tuning on Target Domain (Random100)

| Strategy | FT Samples | AUC | ECE (Platt) | vs Zero-shot |
|----------|-----------|-----|-------------|--------------|
| Intra-topology (upper bound) | — | **0.929** | 0.040 | — |
| Zero-shot | — | 0.797 | 0.032 | baseline |
| Fine-tune 500 | 500 | 0.785 | 0.186 | **−0.012** |
| Fine-tune 2K | 2,000 | 0.775 | 0.029 | **−0.022** |
| Fine-tune 5K | 5,000 | 0.807 | 0.086 | **+0.010** |

**Conclusion:** Fine-tuning is surprisingly ineffective. Small budgets (500–2K) actively hurt performance. Even 5K samples barely help (+0.01 AUC). The transfer gap to intra-topology remains >0.12 AUC.

---

## Part 5: Mixed-Topology Pre-Training

| Strategy | AUC | ECE | ECE (+Platt) |
|----------|-----|-----|--------------|
| NSFNET-only → Zero-shot | 0.797 | 0.207 | 0.032 |
| Mixed → Zero-shot | **0.824** | 0.146 | 0.033 |
| Mixed → FT 2K + Platt | 0.810 | — | 0.036 |

**Conclusion:** Mixed training (NSFNET+USNET+Random50) provides a modest +0.027 AUC improvement over NSFNET-only with zero additional target data. Fine-tuning the mixed model degrades performance.

---

## Part 6: GNN Architecture Exploration ⭐

| Model | AUC (Raw) | AUC (Platt) | ECE (Raw) | ECE (Platt) |
|-------|-----------|-------------|-----------|-------------|
| v2b (handcrafted) | **0.797** | **0.797** | 0.207 | **0.032** |
| GNN (2-layer GraphSAGE, 16 hidden) | 0.767 | 0.767 | 0.102 | 0.031 |

**Critical Finding:** The simple GNN performs **worse** than v2b in cross-topology transfer (−0.03 AUC). Although GNN has better raw calibration (ECE 0.102 vs 0.207), both achieve similar ECE after Platt Scaling (~0.032).

**Why GNN fails here:**
1. **Overfitting to small graph structure:** The GNN was trained on NSFNET (14 nodes, 21 edges). 2-layer message passing covers the entire graph diameter, causing the model to memorize NSFNET-specific propagation patterns.
2. **Node embeddings lack topology-agnostic initialization:** Randomly initialized node embeddings don't encode any structural similarity across different topologies.
3. **Edge features are too simple:** 3D handcrafted edge features (avail_frac, block_frac, frag) may not provide enough signal for the GNN to learn meaningful messages.
4. **No graph-level structural features:** The GNN doesn't explicitly know the graph size, diameter, or connectivity pattern.

**Implication:** Handcrafted path histogram (v2b) is actually a strong inductive bias for this task. It explicitly encodes the spectrum availability along candidate paths — exactly what the RSA algorithm needs. A naive GNN without careful inductive bias design performs worse.

---

## Part 7: Robustness (Multi-Seed Stability)

| Topology | AUC Mean±Std | AUC Range | Assessment |
|----------|-------------|-----------|------------|
| NSFNET | 0.9916 ± 0.0018 | 0.988–0.994 | Extremely stable |
| Random100 | 0.9385 ± 0.0138 | 0.918–0.957 | Acceptable |

---

## Part 8: Final Recommendations

### Deployment Strategy by Topology Scale

| Scale | Best Strategy | AUC | ECE | Data Needed |
|-------|--------------|-----|-----|------------|
| ≤28 nodes | v2b zero-shot + Platt | >0.91 | ~0.03 | 2K calib |
| 50 nodes | v2b zero-shot + Platt | >0.85 | ~0.03 | 2K calib |
| 80–100 nodes | Mixed pre-train + Platt | ~0.82 | ~0.03 | None |
| 80–100 nodes (best accuracy) | Native target training | ~0.93 | ~0.04 | 10K samples |

### Key Takeaways
1. **v2b is the recommended encoder** — its handcrafted path histogram provides strong inductive bias.
2. **Platt Scaling is essential** for cross-topology calibration — 2K samples sufficient.
3. **Fine-tuning is not recommended** unless >10K target samples available.
4. **Mixed pre-training helps slightly** (+0.03 AUC) for large topologies.
5. **Simple GNN does not beat v2b** — more sophisticated GNN designs (graph-level pooling, structural embeddings) would be needed.

### Future Work (If Continuing)
- **Topology-conditioned predictor:** Input graph statistics (diameter, avg degree, clustering) as additional features.
- **Hierarchical GNN:** Separate local (path-level) and global (graph-level) processing.
- **Contrastive pre-training:** Train encoder to distinguish feasible vs infeasible paths across diverse topologies.
