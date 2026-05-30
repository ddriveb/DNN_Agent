# 外出期间实验进度总览

> 生成时间: 2026-05-09 09:00
> 你出门期间，按 **C → A → B** 顺序推进，所有阶段已留痕。

---

## 一、Phase C: 错误分析 ✅ 已完成

### 运行
```bash
cd experiments/agent_mvp
python analyze_imitation_errors.py
```

### 核心发现

**C1 — Action Decomposition**
| 指标 | 数值 | 诊断 |
|------|------|------|
| Full Action Accuracy | 74.21% | |
| **Split Accuracy** | **81.94%** | ⚠️ **瓶颈！** |
| **Server Accuracy** | **87.54%** | 相对容易 |
| Top-3 Hit Rate | **96.32%** | Teacher 动作几乎总在 top-3 |

**C2 — Regret**
- Catastrophic Mismatch Rate: **仅 3.68%**
- 说明绝大多数错误是"软错误"

**C3 — 根因定位**

| Teacher Split | Full Acc | 诊断 |
|--------------|----------|------|
| split_0 | 82.28% | 正常 |
| split_1 | 65.23% | 偏低 |
| **split_2** | **50.84%** | 🚨 **近乎随机！** |

**NSFNET 标准负载崩的根因：split_2 只有 50.8% accuracy**
- split_2 对应高带宽需求（8 slots）
- state vector 只有 model_id（1 个 float），无法编码 split 差异
- 标准负载下网络资源充足，选错 split_2 不会立即失败，但累积次优分配 → compounding error

**USNET 跨拓扑好的原因**
- USNET 上 YinLike 本身极差 (32.53%)
- 即使 student 犯同样错误，也比 YinLike 好
- action space 更受限，错误后果较轻

### 输出文件
- `results/imitation_error_analysis.json` — 结构化数据
- `results/imitation_error_analysis.md` — 可读报告

---

## 二、Phase A: DQN 框架 ⚠️ 部分完成

### 代码
- `dqn_agent.py` — QNetwork + ReplayBuffer + DQNAgent (DDQN)
- `train_dqn.py` — 训练循环

### 关键设计
- **冻结前两层，只训练最后一层** — 防止 RL gradient 破坏 warm-start
- Warm-start eval: **19.90%**（比 ImitationAgent 21.84% 还好！）

### 训练结果

| Episode | Eval Blocking | 备注 |
|---------|--------------|------|
| 0 (warm-start) | **19.90%** | 最佳 |
| 25 | 33.60% | RL 破坏了性能 |
| 50 | 26.10% | 缓慢回升 |

**结论**: 即使是冻结 body 的保守 DQN，RL update 仍然无法超越 warm-start。

**可能原因**:
1. Reward 太稀疏（+1/-1）
2. 每个 request 是独立决策，TD target 不稳定
3. 21D state 不足以支持 Q-value 估计

### 输出文件
- `results/dqn_training_1000eps.log` — 训练日志
- `results/dqn_training_history.json` — 历史数据
- `checkpoints/dqn_best.pt` — 最佳检查点（warm-start 权重）

---

## 三、Phase B: 增强 State Vector ✅ 已完成

### 运行
```bash
cd experiments/agent_mvp
python enhance_state_and_retrain.py
```

### 增强方案
- 原始 state: **21D**（z[10] + model_id + deadline + src + frag + max_free + server_utils[5] + candidate_count）
- 增强 state: **30D**（+ 9D split profile: 3 splits × [bw_norm, compute_cost, interm_norm]）

### 结果对比

| 场景 | YinLike | Original (21D) | Enhanced (30D) |
|------|---------|---------------|---------------|
| NSFNET standard | 19.48±2.07% | 21.84±1.48% | **20.53±0.93%** ✅ |
| NSFNET high | 33.08±1.63% | 33.39±2.11% | **32.92±1.67%** |
| USNET cross-topo | 32.53±3.10% | **27.36±2.14%** | 28.34±2.40% ❌ |

### 关键发现
- **NSFNET standard 从 21.84% → 20.53%** (+6.0% 相对提升)
- **Val accuracy 从 73.69% → 74.60%** (+0.91%)
- USNET 反而变差 → 增强特征对跨拓扑场景引入 noise

### 输出文件
- `checkpoints/imitation_agent_enhanced.pt` — 增强模型权重
- `data/imitation_dataset_enhanced.pkl` — 增强数据集
- `results/enhanced_imitation_eval.json` — 评估结果
- `results/enhanced_retrain.log` — 训练日志

---

## 四、关键决策记录

| 决策 | 原因 | 结果 |
|------|------|------|
| 冻结 DQN body | 防止 RL gradient 破坏 warm-start | 改善（43% → 26%），但仍未超越 19.9% |
| 增强 state 到 30D | split_2 accuracy 仅 50.8% | NSFNET 提升，USNET 下降 |
| 终止 1000 eps DQN | Ep 50 eval 26.1%，趋势不明 | 节省计算资源 |

---

## 五、待你回来决策

### 选项 1: Soft-Label Imitation
- 用 teacher score 替代 one-hot label
- KL loss 替代 cross entropy
- 让 student 知道"A 最好，B 也还行，C 很危险"

### 选项 2: Top-K Selector
- 利用 96.3% top-3 hit rate
- Student 输出 top-3，RuleAgent scoring 做二次选择
- **零训练成本**

### 选项 3: GNN-based State
- 用图神经网络编码网络拓扑
- 替代当前的 10D encoder output

### 选项 4: 放弃 DQN，改用 Policy Gradient
- REINFORCE 直接优化策略
- 避免 Q-value 估计的不稳定性

### 选项 5: 冻结 imitation，只做数据增强
- 当前的 best checkpoint 是 `imitation_agent_enhanced.pt`
- 在 NSFNET 上已优于 original (20.53% vs 21.84%)
- 可能足以作为最终模型

---

## 六、文件索引

```
experiments/agent_mvp/
  analyze_imitation_errors.py      # Phase C 脚本
  dqn_agent.py                      # DQN 框架
  train_dqn.py                      # DQN 训练
  enhance_state_and_retrain.py      # Phase B 脚本
  PHASE_C_A_PROGRESS_REPORT.md      # 详细进度报告
  RETURN_README.md                  # 本文件
  
  results/
    imitation_error_analysis.*      # Phase C 输出
    enhanced_imitation_eval.json    # Phase B 输出
    enhanced_retrain.log            # Phase B 日志
    dqn_training_1000eps.log        # DQN 日志
    dqn_training_history.json       # DQN 历史
    
  checkpoints/
    imitation_agent.pt              # 原始 (21D)
    imitation_agent_enhanced.pt     # 增强 (30D)
    dqn_best.pt                     # DQN 最佳
    
  data/
    imitation_dataset.pkl           # 原始 (21D)
    imitation_dataset_enhanced.pkl  # 增强 (30D)
    
snapshots/
  20260509_081732_phaseAB_complete/   # 出门前的快照
  20260509_085907_phaseCAB_progress/  # 当前快照
```

---

## 七、下一步建议

基于当前证据，**最务实的路线**是：

1. **用 enhanced model (30D) 作为当前 best** — NSFNET 20.53% 已接近 YinLike 19.48%
2. **尝试 Soft-Label 训练** — 可能进一步降低 catastrophic mismatch
3. **如果仍不达标，做 Top-K Selector** — 利用 96.3% top-3 hit rate，零训练成本
4. **最后才考虑 DQN/Policy Gradient** — 当前 reward/MDP 设计不够成熟
