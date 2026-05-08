# 归档说明

本文档说明 `archive/` 目录中各组件的归档原因和实验结论。

---

## archive/ineffective/ — 被验证低效的组件

这些组件经过完整实验验证，性能不如当前方案，**不再推荐使用**。

### PINN (物理信息神经网络)

**文件**: `pinn_predictor.py`, `train_pinn.py`, `main_pinn.py`

**实验结论**: ❌ **完全失败**
- 物理损失（频谱占用约束）比分类损失大 **5 个数量级**
- 完全淹没数据梯度，模型无法学习
- 所有样本量下均输给 Baseline
- **原因**: PINN 的硬约束损失与可行性分类任务的软概率分布不匹配

### 独立 GNN 预测器

**文件**: `gnn_predictor.py`, `link_as_node_gnn_predictor.py`, `train_gnn.py`, `train_link_as_node_gnn.py`, `main_gnn.py`, `main_gnn_eval_only.py`, `main_link_as_node_gnn.py`

**实验结论**: ❌ **不如手工特征**
- 2-layer GraphSAGE 跨拓扑 AUC: **0.767**
- v2b 手工直方图跨拓扑 AUC: **0.797**
- GNN 比 v2b 差 **0.03 AUC**
- **原因**:
  1. 在 14 节点 NSFNET 上训练，2 层消息传递覆盖全图直径 → 过拟合到特定拓扑
  2. 随机初始化的节点嵌入无法编码跨拓扑结构相似性
  3. 边特征太简单（仅 3D），不足以支撑有意义的图消息传递

> **注意**: `link_as_node_gnn_predictor.py` 中的 `GATLayer` 被 `hybrid_predictor.py` 继续引用使用，只是 standalone 版本被废弃。

---

## archive/legacy/ — 历史实验脚本

这些是项目演进过程中各阶段的实验脚本，**已整合到最终方案中**。

### Baseline 模型 (Phase 0)

**文件**: `numpy_predictor.py`, `sklearn_predictor.py`, `predictor.py`, `selector.py`, `train.py`, `calibration_methods.py`, `main_numpy.py`, `main_sklearn.py`, `main.py`, `train_baseline_models.py`

**说明**: v0/v1 版本的简单预测器（单层 MLP、Sklearn、Numpy 实现）。被 v2b + Hybrid 完全取代。

### 迁移学习实验 (Phase 1-2)

**文件**: `main_cross_topology.py`, `main_cross_calibrated.py`, `main_finetune.py`, `main_same_scale_transfer.py`, `main_large_to_small.py`, `main_large_to_small_finetune.py`, `main_layerwise_finetune.py`, `main_l2sp_finetune.py`, `main_mixed_training.py`, `main_topology_conditioned.py`

**说明**: 验证 v2b 跨拓扑泛化、Platt Scaling 校准、Fine-tuning 策略、Mixed Pre-training 等。核心结论已写入 `logs/final_comprehensive_report.md`。

### 鲁棒性实验

**文件**: `main_robustness.py`, `main_noise_*.py`, `main_large_scale.py`

**说明**: 多 seed 稳定性、噪声注入、大规模训练等验证。

### 8-bin 实验

**文件**: `main_8bin_*.py`

**说明**: 粗粒度带宽分桶的迁移矩阵实验。

### 其他

**文件**: `main_topo_v2t_intra.py`, `main_topo_v2t_cross.py`, `main_usnet_quick.py`

**说明**: 拓扑内/跨拓扑 v2t 验证的快速脚本。

---

## archive/temp/ — 临时验证脚本

这些是为特定问题临时编写的脚本，**问题已解决**。

**文件**:
- `calibrate_load.py` — 负载参数搜索（已确定 arr=0.25, ht=8.0 for NSFNET）
- `validate_stability.py` — 多 seed 稳定性验证（结论：均值 +18.4%，标准差 7.6%）
- `validate_3000_5.py` — 3000样本/5epochs 配置测试（结论：比 1000/5 差，过拟合）
- `demo_selector_mapper_v2.py` ~ `v4.py` — 闭环 demo 的中间迭代版本
- `demo_full_pipeline.py`, `demo_dnn_offload.py`, `agent_interface.py` — 早期 pipeline 原型

---

## 核心系统（保留在根目录）

```
env.py                  # 光网络仿真环境
mapper.py               # KSP + First-Fit 频谱分配
encoder.py              # v2b 路径直方图编码（强基线）
hybrid_encoder.py       # Hybrid 编码器（v2b + GAT 输入）
hybrid_predictor.py     # Hybrid 预测模型（GAT + v2b + MLP）
hybrid_selector.py      # 路径选择器（max_prob / min_delay / weighted）
dataset.py              # 训练数据生成器（path-id bug 已修复）
train_hybrid.py         # 训练循环（BCE + MSE）
ksp_fast.py             # Yen K-最短路径算法
demo_selector_mapper_v5.py  # 最终版闭环仿真 demo
```

---

## 关键设计决策记录

| 决策 | 选项A | 选项B | 结论 |
|------|-------|-------|------|
| Encoder | v2b 手工 | GNN 学习 | **选 v2b**，AUC 更高 |
| Predictor | v2b 单独 | Hybrid (v2b+GAT) | **选 Hybrid**，跨拓扑 AUC +0.058 |
| 物理约束 | PINN 辅助头 | 无 | **不选 PINN**，损失淹没梯度 |
| 训练目标 | BCE+MSE | BCE+物理损失 | **选 BCE+MSE** |
| 选择策略 | MaxP | Frag-aware | **选 Frag-aware**，阻塞率降低 26.8% |
| 训练配置 | 1000/5epochs (欠拟合) | 3000/10epochs (过拟合) | **选欠拟合**，闭环更好 |
| K值 | K=3 | K=10 | **K=10 略优** (+2.3%)，但 K=3 已足够 |
