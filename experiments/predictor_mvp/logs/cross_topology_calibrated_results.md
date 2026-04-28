# Cross-Topology Transfer with Post-Hoc Calibration — Full Results
**Date:** 2026-04-27  
**Experiment:** NSFNET(14 nodes) → 6 target topologies  
**Setup:** Train 15K samples on NSFNET, calibrate with 2K target samples, test on 3K target samples, 30 epochs  
**Encoders:** v0 (global frag index, 1D) vs v2b (binned path histogram, 10D)

---

## 1. AUC Summary (Discriminative Power)

| Test Topology | Nodes | v0 Raw | v0 +TempScale | v0 +Platt | v2b Raw | v2b +TempScale | v2b +Platt | Δ AUC (v2b−v0) |
|---------------|-------|--------|---------------|-----------|---------|----------------|------------|----------------|
| usnet         | 28    | 0.8238 | 0.8238        | 0.8238    | 0.9128  | 0.9128         | 0.9128     | **+0.089**     |
| cost266       | 28    | 0.8116 | 0.8116        | 0.8116    | 0.9101  | 0.9101         | 0.9101     | **+0.098**     |
| germany50     | 50    | 0.8055 | 0.8055        | 0.8055    | 0.8641  | 0.8641         | 0.8641     | **+0.059**     |
| random50      | 50    | 0.8057 | 0.8057        | 0.8057    | 0.8547  | 0.8547         | 0.8547     | **+0.049**     |
| random80      | 80    | 0.7847 | 0.7847        | 0.7847    | 0.8146  | 0.8146         | 0.8146     | **+0.030**     |
| random100     | 100   | 0.7681 | 0.7681        | 0.7681    | 0.7970  | 0.7970         | 0.7970     | **+0.029**     |

**Observation:** v2b consistently outperforms v0. The AUC advantage shrinks with topology scale (+0.09 at 28 nodes → +0.03 at 100 nodes), but remains significant even at 100 nodes.

---

## 2. ECE Summary (Calibration)

| Test Topology | Nodes | v0 Raw | v0 +TempScale | v0 +Platt | v2b Raw | v2b +TempScale | v2b +Platt | Raw ECE Ratio (v2b/v0) |
|---------------|-------|--------|---------------|-----------|---------|----------------|------------|------------------------|
| usnet         | 28    | 0.0835 | 0.0928        | **0.0282**| 0.1536  | 0.1996         | **0.0257** | 1.84×                  |
| cost266       | 28    | 0.0662 | 0.0432        | **0.0303**| 0.1275  | 0.1541         | **0.0293** | 1.93×                  |
| germany50     | 50    | 0.1133 | 0.1238        | **0.0408**| 0.2722  | 0.3099         | **0.0370** | 2.40×                  |
| random50      | 50    | 0.1025 | 0.0446        | **0.0325**| 0.1711  | 0.1604         | **0.0390** | 1.67×                  |
| random80      | 80    | 0.0992 | 0.0768        | **0.0131**| 0.2608  | 0.2771         | **0.0377** | 2.63×                  |
| random100     | 100   | 0.1550 | 0.0644        | **0.0379**| 0.2069  | 0.1370         | **0.0321** | 1.33×                  |

| Metric | v0 Raw | v2b Raw | v0 Platt | v2b Platt |
|--------|--------|---------|----------|-----------|
| Mean ECE | 0.1033 | 0.1987 | 0.0301 | 0.0335 |
| Std ECE  | 0.0308 | 0.0556 | 0.0094 | 0.0052 |

**Key Findings:**
- v2b suffers **1.3–2.6× worse raw ECE** than v0 in zero-shot transfer.
- **Platt Scaling** (2K calibration samples) recovers calibration for BOTH encoders to ~0.03 ECE.
- Post-calibration ECE is nearly identical between v0 and v2b (0.030 vs 0.034).
- **Temperature Scaling** is unreliable: helps some cases, worsens others.

---

## 3. Accuracy Summary

| Test Topology | Nodes | v0 Raw | v0 +Platt | v2b Raw | v2b +Platt |
|---------------|-------|--------|-----------|---------|------------|
| usnet         | 28    | 0.8137 | 0.8470    | 0.8247  | 0.8657     |
| cost266       | 28    | 0.7833 | 0.7833    | 0.8293  | 0.8543     |
| germany50     | 50    | 0.7977 | 0.8513    | 0.6993  | 0.8527     |
| random50      | 50    | 0.7487 | 0.7430    | 0.7453  | 0.7867     |
| random80      | 80    | 0.7603 | 0.7867    | 0.6900  | 0.7867     |
| random100     | 100   | 0.7020 | 0.7057    | 0.7053  | 0.7377     |

**Observation:** Platt Scaling adjusts decision threshold (via sigmoid rescaling), which can improve accuracy when raw predictions are miscalibrated.

---

## 4. Calibration Parameters

### Temperature Scaling (T)
| Test Topology | v0 T    | v2b T   |
|---------------|---------|---------|
| usnet         | 1.5600  | 5.0465  |
| cost266       | 1.8588  | 4.4869  |
| germany50     | 1.6060  | 6.4697  |
| random50      | 2.8057  | 5.1475  |
| random80      | 1.9074  | 8.8704  |
| random100     | 3.4654  | 6.0205  |

**v2b needs much higher T (4.5–8.9)** vs v0 (1.6–3.5), indicating v2b produces much "overconfident" logits in cross-topology transfer.

### Platt Scaling (a·σ(z) + b)
| Test Topology | v0 a      | v0 b       | v2b a     | v2b b      |
|---------------|-----------|------------|-----------|------------|
| usnet         | 0.4970    | 0.7207     | 0.2498    | 1.7197     |
| cost266       | 0.4602    | 0.4046     | 0.2618    | 1.3852     |
| germany50     | 0.4868    | 0.7103     | 0.2575    | 2.0262     |
| random50      | 0.3939    | −0.2044    | 0.2164    | 0.7999     |
| random80      | 0.4339    | 0.5871     | 0.2145    | 1.7839     |
| random100     | 0.3439    | −0.3274    | 0.2016    | 0.7472     |

**v2b requires smaller slope (a ≈ 0.20–0.26)** vs v0 (a ≈ 0.34–0.50), confirming stronger sigmoid compression is needed to correct overconfidence.

---

## 5. Training Metrics on Source (NSFNET)

### v0 Encoder (Final Epoch 30)
| Metric | Value |
|--------|-------|
| Loss   | 0.3087 |
| Acc    | 0.8477 |
| AUC    | 0.8991 |
| MAE    | 0.006553 |
| ECE    | 0.0202 |

### v2b Encoder (Final Epoch 30)
| Metric | Value |
|--------|-------|
| Loss   | 0.0708 |
| Acc    | 0.9537 |
| AUC    | 0.9926 |
| MAE    | 0.009435 |
| ECE    | 0.0156 |

**Intra-topology (NSFNET) baseline:** v2b dominates: AUC 0.993 vs 0.899, Acc 0.954 vs 0.848, ECE 0.016 vs 0.020.

---

## 6. Test Set Service Rates (Target Domain Load)

| Test Topology | Calib SR | Test SR | Notes |
|---------------|----------|---------|-------|
| usnet         | 0.823    | 0.844   | Moderate load |
| cost266       | 0.774    | 0.780   | Moderate load |
| germany50     | 0.809    | 0.854   | Moderate load |
| random50      | 0.647    | 0.674   | Higher load (more blocking) |
| random80      | 0.772    | 0.783   | Moderate load |
| random100     | 0.600    | 0.609   | Highest load (most blocking) |

---

## 7. Raw Log File

Full per-epoch training logs: `logs/cross_topology_calibrated_full_2026-04-27.log`
