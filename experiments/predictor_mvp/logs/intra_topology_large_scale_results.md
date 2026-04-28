# Intra-Topology Large-Scale Results (New: +Random80/100)
**Date:** 2026-04-27  
**Setup:** Train + test on same topology, 10K samples, 30 epochs, v0 vs v2b

---

## Results

| Topology | Nodes | v0 Acc | v0 AUC | v0 ECE | v2b Acc | v2b AUC | v2b ECE | Δ AUC |
|----------|-------|--------|--------|--------|---------|---------|---------|-------|
| nsfnet | 14 | 0.8440 | 0.9020 | 0.0222 | 0.9620 | **0.9944** | 0.0117 | **+0.0924** |
| usnet | 28 | 0.8875 | 0.8952 | 0.0132 | 0.9300 | **0.9528** | 0.0249 | **+0.0576** |
| cost266 | 28 | 0.8665 | 0.9092 | 0.0174 | 0.9135 | **0.9528** | 0.0291 | **+0.0436** |
| random50 | 50 | 0.8700 | 0.9244 | 0.0303 | 0.9015 | **0.9545** | 0.0180 | **+0.0301** |
| random80 | 80 | 0.9455 | 0.9378 | 0.0326 | 0.9475 | **0.9580** | 0.0200 | **+0.0202** |
| random100 | 100 | 0.7960 | 0.8844 | 0.0712 | 0.8510 | **0.9294** | 0.0402 | **+0.0450** |

---

## Key Observations

1. **v2b consistently outperforms v0 across all scales** (14–100 nodes), with AUC advantage +0.02 to +0.09.

2. **Absolute v2b AUC remains high even at 100 nodes:** 0.929 (intra) vs 0.797 (cross from NSFNET). This confirms v2b *can* learn effectively on large topologies when trained directly.

3. **Random100 v0 AUC (0.884) is lower than Random80 v0 (0.938)** — likely due to higher load config (320 slots, arrival=30, ht=25) creating more challenging blocking scenarios.

4. **v2b ECE on Random100 intra (0.040) is much better than cross (0.207)** — calibration is only problematic in *transfer*, not when trained natively.

---

## Comparison: Intra vs Cross (NSFNET→X)

| Target | Nodes | v2b Intra | v2b Cross | v0 Intra | v0 Cross | v2b Gap | v0 Gap |
|--------|-------|-----------|-----------|----------|----------|---------|--------|
| usnet | 28 | 0.9528 | 0.9128 | 0.8952 | 0.8238 | 0.0400 | 0.0714 |
| cost266 | 28 | 0.9528 | 0.9101 | 0.9092 | 0.8116 | 0.0427 | 0.0976 |
| germany50 | 50 | — | 0.8641 | — | 0.8055 | — | — |
| random50 | 50 | 0.9545 | 0.8547 | 0.9244 | 0.8057 | 0.0998 | 0.1187 |
| random80 | 80 | 0.9580 | 0.8146 | 0.9378 | 0.7847 | **0.1434** | **0.1531** |
| random100 | 100 | 0.9294 | 0.7970 | 0.8844 | 0.7681 | **0.1324** | **0.1163** |

**Transfer gap analysis:**
- For small topologies (≤50 nodes), v2b has a **smaller transfer gap** than v0.
- For large topologies (80–100 nodes), both encoders suffer severe transfer gaps (>0.11 AUC). v2b's gap is comparable to v0's at this scale.
- **Key insight:** Zero-shot transfer from 14-node NSFNET to 80+ nodes is fundamentally hard for both encoders. The solution is either (a) fine-tuning on target data, or (b) collecting a small calibration set + Platt scaling (which fixes ECE but not AUC).
