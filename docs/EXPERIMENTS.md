# 实验方案与结果记录

> **状态**: Phase A-E 完成；Phase F (Yin 2024 协议仿真) 进行中  
> **日期**: 2026-05-09  
> **主方法**: Predictor-Guided Top-K Correction Agent (CorrectionNet v1)

---

## 一、核心实验矩阵（Phase A-E）

所有实验使用相同条件：5 seeds × 2000 requests，300 preload connections。

### Table 1: Main Results — Blocking Rate Comparison

| Method | NSFNET standard | NSFNET high load | USNET cross-topo |
|--------|-----------------|------------------|------------------|
| WO | 50.03±1.56% | 59.43±3.87% | 57.36±4.57% |
| DF | 53.10±2.27% | 60.56±3.35% | 57.39±4.98% |
| RF | 36.62±3.23% | 38.99±3.16% | 35.29±3.10% |
| IWD-Approx | 29.88±2.62% | 36.44±2.93% | 29.45±1.69% |
| YinLike | 19.48±2.07% | 33.08±1.63% | 32.53±3.10% |
| TopK2-30D | 15.51±0.83% | 27.49±1.18% | 26.85±2.69% |
| **CorrectionNet λ=0.2** | **13.04±1.58%** | **23.50±1.64%** | **21.77±0.89%** |

> 注：WO = Without Offloading（本地执行）；DF = Distance First（最近服务器）；RF = Resource First（最空闲服务器）；IWD-Approx = 启发式综合代价搜索。

### Table 2: Module Ablation (NSFNET standard)

| Component | Blocking Rate |
|-----------|---------------|
| None (YinLike) | 19.48±2.07% |
| + Neural proposal only | 20.53±0.93% |
| + Predictor reranking | 15.51±0.83% |
| + Residual correction | 13.04±1.58% |
| RL replaces selector (Direct DQN) | 38.23±2.34% |

### Table 3: Load Sweep (NSFNET, 32 slots, seed=42)

| Arrival Rate | TopK2-30D | CorrectionNet λ=0.2 | Improvement |
|--------------|-----------|---------------------|-------------|
| 2.0 | 6.30% | 4.30% | 2.00pp |
| 4.0 | 15.75% | 10.70% | 5.05pp |
| 6.0 | 19.65% | 15.50% | 4.15pp |
| 8.0 | 20.85% | 17.60% | 3.25pp |
| 10.0 | 27.95% | 24.95% | 3.00pp |

### Table 4: CorrectionNet Interpretability (NSFNET standard, seed=42)

| Metric | Value |
|--------|-------|
| Same decision as TopK2 | 470 (23.5%) |
| Changed decision | 1,530 (76.5%) |
| Changed → improved (fail→success) | 103 |
| Changed → worsened (success→fail) | 44 |
| Suppressed higher-predictor-score candidate | 58 |

**Correction Score vs Future Blocking (window=50)**:

| Correction Score | n | Future Block Rate | Immediate Success |
|------------------|---|-------------------|-------------------|
| < -0.5 | 23 | 0.613 | 0.522 |
| ≥ 0.5 | 23 | 0.240 | 0.957 |

---

## 二、Yin 2024 协议仿真（Phase F）

### 2.1 实验设置

- **拓扑**: Net-1 (9 nodes, 5 servers, 14 links), Net-2 (17 nodes, 6 servers, 23 links), Net-3 (20 nodes, 7 servers, 27 links)
- **频谱**: Net-1=100 slots, Net-2/3=150 slots; load_factor=0.6; frag=0.2/0.5
- **请求**: 15-60 requests/batch; subtasks=1-5; compute=1e7-1.5e7 Hz; data=8e7-6e8 bits; deadline=20-40ms
- **评估**: 5 seeds; 计算资源即时释放（模拟瞬时请求处理）

### 2.2 Net-1 结果（frag=0.2, 5 seeds）

| N_req | WO | DF | RF | IWD-Approx | YinLike |
|------:|---:|---:|---:|-----------:|--------:|
| 15 | 0.0±0.0% | 44.0±11.6% | 56.0±12.4% | 38.7±12.9% | 0.0±0.0% |
| 20 | 0.0±0.0% | 48.0±12.9% | 59.0±15.9% | 42.0±9.3% | 0.0±0.0% |
| 25 | 0.0±0.0% | 50.4±12.0% | 63.2±15.9% | 48.0±13.4% | 0.0±0.0% |
| 30 | 0.0±0.0% | 58.0±7.5% | 66.0±9.0% | 47.3±11.6% | 4.7±6.2% |
| 35 | 0.0±0.0% | 56.6±11.2% | 65.1±10.9% | 53.7±12.2% | 1.1±1.4% |
| 40 | 0.0±0.0% | 55.0±5.7% | 64.5±5.3% | 50.0±9.9% | 1.5±1.2% |
| 45 | 0.0±0.0% | 59.6±6.4% | 67.1±5.7% | 48.4±5.1% | 8.4±10.4% |
| 50 | 0.0±0.0% | 58.4±6.1% | 70.0±4.6% | 51.6±7.7% | 7.2±5.2% |
| 55 | 0.0±0.0% | 60.4±3.9% | 69.1±5.8% | 57.5±8.7% | 13.8±14.1% |
| 60 | 0.0±0.0% | 60.3±3.2% | 68.0±6.0% | 56.7±3.5% | 14.0±9.9% |

**相对顺序**: `WO (0%) < YinLike (0-14%) < IWD-Approx (39-57%) < DF (44-60%) < RF (56-70%)`

### 2.3 待完成实验

- [ ] Net-2 基线评估
- [ ] Net-3 基线评估
- [ ] frag=0.5 场景
- [ ] TopK2 / CorrectionNet zero-shot 迁移测试
- [ ] 绘制 Fig. 8 风格曲线

---

## 三、实验脚本清单

| 脚本 | 目的 | 输出 |
|------|------|------|
| `eval_yin_protocol_baselines.py` | Yin 协议基线对比（WO/DF/RF/IWD-Approx + 我们的方法） | `yin_protocol_baselines.json` |
| `eval_correction_net_aligned.py` | 5 seeds × 3 scenarios 主评估 | `correction_net_eval_aligned.json` |
| `run_ablation.py` | 消融实验（Random/ShortestPath/YinLike/Imitation） | `ablation_study.json` |
| `run_load_sweep.py` | 负载扫描（TopK2 vs CorrectionNet） | `load_sweep_topk_corrnet.json` |
| `analyze_correction_net.py` | 可解释性分析 | `correction_net_interpretability.json` |
| `compile_paper_results.py` (src/) | 整合所有结果到论文表格 | `paper_table_*.json`, `paper_summary.md` |

---

## 关键结论

1. **Baseline alignment verified**: λ=0.0 produces 0 mismatches vs native TopK2-30D (2000/2000 requests).
2. **WO/DF 验证了卸载必要性**: 不做卸载或仅按距离选服务器，阻塞率高达 50–60%，远高于智能基线。
3. **RF 单独不足**: 仅按负载选服务器（RF）阻塞率 35–39%，显著优于 DF，但仍不如同时考虑网络与计算的 YinLike。
4. **IWD-Approx 逼近 YinLike**: 综合代价启发式搜索（延迟+负载+频谱+截止期）将阻塞率降至 29–36%，在 USNET 上甚至略超 YinLike。
5. **Improvement scales with difficulty**: +2.47pp (standard) → +3.99pp (high) → +5.08pp (USNET) vs YinLike.
6. **Direct DQN failure confirmed**: 38.23% blocking vs 15.51% for TopK2 — RL must not replace the selector.
7. **Interpretability established**: Negative corrections correlate with 61.3% future blocking; positive corrections with 24.0%.
8. **Load sweep stable**: CorrectionNet improves across all tested arrival rates (2.0–10.0), with largest gains at moderate load (4.0–6.0).

---

## 原始实验记录

详细的历史实验日志和中间结果保留在：
- `experiments/agent_mvp/results/`
- `experiments/agent_mvp/logs/`
- `snapshots/20260509_181635_correctionnet_v1_aligned/`
