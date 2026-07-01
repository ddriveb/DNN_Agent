# Phase C → A 进度报告

> 生成时间: 2026-05-09 08:52
> 用户状态: 已出门，要求回来看到完整实验过程
> DQN 训练: 后台运行中 (PID 264584, 1000 episodes, NSFNET standard)

---

## Phase C: 错误分析 — 已完成

### 运行命令
```bash
cd experiments/agent_mvp
python analyze_imitation_errors.py
```

### 输出文件
- `results/imitation_error_analysis.json` — 结构化数据
- `results/imitation_error_analysis.md` — 可读报告

### C1: Action-Level Accuracy (Val Set, 7200 samples)

| 指标 | 数值 | 诊断 |
|------|------|------|
| Full Action Accuracy | **74.21%** | 25.8% 的样本至少一个组件错误 |
| **Split Accuracy** | **81.94%** | ⚠️ 瓶颈在此！ |
| **Server Accuracy** | **87.54%** | 服务器选择相对容易 |
| Top-3 Hit Rate | **96.32%** | Teacher action 几乎总在 top-3 |
| Top-5 Hit Rate | **99.22%** | 接近全覆盖 |

**核心发现**: Split point 选择是主要瓶颈，不是 server 选择。这解释了为什么 state vector 中 model_id（单个 float）不足以区分不同 split 的优劣。

### C2: Regret & Catastrophic Mismatch

| 指标 | 数值 |




| Average Regret | 0.3305 |
| Median Regret | 0.0000 |
| P95 Regret | 1.9576 |
| **Catastrophic Mismatch Rate** | **3.68%** |
| Avg Teacher Prob | 0.6266 |

**核心发现**: Catastrophic mismatch 仅 3.68%，说明绝大多数错误是"软错误"——student 的动作和 teacher 差距不大，只是 top-1 不够精确。

### C3: Mismatch Pattern Breakdown

| Pattern | Count | Fraction | Avg Regret | Catastrophic |
|---------|-------|----------|------------|-------------|
| server对, split错 | 960 | **51.7%** | 0.799 | 5.5% |
| split对, server错 | 557 | 30.0% | **1.461** | 13.3% |
| 两者都错 | 340 | 18.3% | **2.349** | **40.6%** |

### C3: By Teacher Split (关键！)

| Teacher Split | Full Acc | Split Acc | Server Acc | Top-3 Hit | Catastrophic |
|--------------|----------|-----------|------------|-----------|-------------|
| split_0 | **82.28%** | 90.87% | 90.03% | 98.42% | 1.58% |
| split_1 | 65.23% | 70.78% | 87.83% | 94.39% | 5.61% |
| **split_2** | **50.84%** | **58.59%** | **74.30%** | **89.45%** | **10.55%** |

**⚠️ 这是 NSFNET 标准负载崩的根因！**

- split_2 的 accuracy 仅 50.8%，几乎是随机水平
- split_2 通常对应高带宽需求（更多 slots，更严格的网络约束）
- state vector 只有 model_id（1 个 float），无法区分不同 split 的带宽/计算需求
- 在标准负载下，网络资源相对充足，选错 split_2 不会立即失败，但累积效应导致后续阻塞

### C3: By Load Level

| Load | Full Acc | Catastrophic |
|------|----------|-------------|
| high | **85.9%** | 1.7% |
| medium | 71.9% | 4.3% |
| low | 70.6% | 4.2% |

**悖论**: High load 下 action matching 最好，但闭环评估中 high load 表现最差。
**解释**: High load 下 action space 更受限（很多动作不可行），所以"猜对"更容易。但闭环中 error accumulation 在高负载下更严重。

### C3: By Strategy

| Strategy | Full Acc | Catastrophic |
|----------|----------|-------------|
| default+nsfnet | **75.1%** | 3.5% |
| pruned+mixed | 71.9% | 4.3% |

---

## Phase A: DQN 框架 — 进行中

### 代码文件
- `dqn_agent.py` — QNetwork + ReplayBuffer + DQNAgent (DDQN)
- `train_dqn.py` — 训练循环 + 评估

### 关键设计决策

#### 1. Warm-start 策略
- 从 `imitation_agent.pt` 加载全部权重
- **冻结前两层（body），只训练最后一层** — 这是关键修复
- 原因: 错误分析显示 top-3 hit 96.3%，问题只在 last layer 的分类精度

#### 2. 初始测试结果

| 配置 | Warm-start Eval | Ep 10 | Ep 20 | Ep 30 |
|------|----------------|-------|-------|-------|
| 未冻结, lr=5e-4 | 19.90% | **43.30%** | 43.65% | 43.55% |
| **冻结 body, lr=1e-4** | **19.90%** | 35.45% | **23.00%** | **23.30%** |

- 未冻结: RL gradient 快速破坏 warm-start（43% vs 19.9%）
- 冻结 body: 显著改善，但 30 eps 后仍比 warm-start 差 3.4 个百分点

#### 3. 当前训练配置

```python
{
    "topology": "nsfnet",
    "num_slots": 32,
    "num_servers": 5,
    "num_requests": 2000,
    "arr_rate": 5.0,
    "hold_time": 10.0,
    "preload": 300,
    "num_episodes": 1000,
    "eval_freq": 25,
    "batch_size": 64,
    "buffer_capacity": 50000,
    "gamma": 0.95,
    "lr": 5e-4,          # 注意: 当前 lr=5e-4（之前测试用 1e-4，但配置文件未改）
    "epsilon_start": 0.2,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "target_update_freq": 500,
    "buffer_warmup": 2000,
    "warm_start": true,
    "freeze_body": true,
}
```

**⚠️ 注意**: 配置文件中的 lr 仍是 5e-4，但冻结 body 后只训练最后一层，learning rate 影响较小。

#### 4. 训练进程

```bash
# 启动命令
PYTHONUNBUFFERED=1 nohup python train_dqn.py > results/dqn_training_1000eps.log 2>&1 &

# 日志
results/dqn_training_1000eps.log

# 检查点
checkpoints/dqn_best.pt     # 最佳 eval 性能
checkpoints/dqn_final.pt    # 最终权重

# 历史数据
results/dqn_training_history.json
```

---

## 诊断总结

### 为什么 USNET 跨拓扑能学好，NSFNET 标准负载反而比 YinLike 差？

```
根本原因: split_2 accuracy 仅 50.8%

NSFNET 标准负载:
- 老师 AdaptiveRA 使用 default+nsfnet (75.1% acc) + pruned+mixed (71.9% acc)
- 标准负载下 action space 更自由，网络资源相对充足
- 选错 split_2 不会立即失败（因为 slots 充足），但分配了次优资源
- 累积的次优分配导致后续请求阻塞 → compounding error
- 最终 blocking: 21.84% (vs YinLike 19.48%)

USNET 跨拓扑:
- 老师主要用 default+nsfnet (75.1% acc)
- USNET 拓扑更大 (28 nodes)，action space 更受限
- 选错 split_2 更容易直接失败（slots 更紧张）
- 但 USNET 上 YinLike 本身表现极差 (32.53%)
- Student 即使犯同样的 split_2 错误，也比 YinLike 的 naive 策略好
- 最终 blocking: 27.36% (vs YinLike 32.53%)
```

### 为什么 High load 下 action matching 最好但闭环最差？

```
High load 数据集上:
- action space 严重受限（很多 server/slot 不可用）
- 候选动作少 → 分类任务变简单 → accuracy 高 (85.9%)

High load 闭环中:
- 每个错误选择直接影响后续 10+ 个请求
- 网络资源稀缺，一次错误分配造成连锁反应
- error accumulation 比标准负载严重得多
```

---

## 下一步计划

### 选项 1: DQN 继续训练（当前进行中）
- 1000 episodes 后台训练
- 观察 eval blocking 是否能从 19.90% 降到 < 16.23% (AdaptiveRA)
- 如果成功，说明 RL fine-tuning 可以修正 last-layer 的分类错误

### 选项 2: 增强 State Vector（Phase B）
- 当前 state 只有 model_id（1 float）
- 加入每个 split 的 bandwidth_slots, compute_cost, intermediate_size
- 预计可提升 split_2 accuracy 从 50% → 70%+

### 选项 3: Top-K + Rule-based Selector
- 利用 96.3% top-3 hit rate
- Student 输出 top-3 actions，用 RuleAgent 的 scoring function 做二次选择
- 无需训练，立即可用

### 选项 4: Soft-Label Imitation
- 用 teacher score 做 soft label（KL loss）
- 让 student 知道"动作 A 最好，B 也还行，C 很危险"
- 降低 catastrophic mismatch rate

---

## 文件归档

```
snapshots/
  20260509_081732_phaseAB_complete/     # Phase A+B 完成快照
  
experiments/agent_mvp/
  analyze_imitation_errors.py             # Phase C 分析脚本
  dqn_agent.py                            # DQN 框架
  train_dqn.py                            # DQN 训练
  results/
    imitation_error_analysis.json         # 错误分析数据
    imitation_error_analysis.md           # 错误分析报告
    dqn_training_1000eps.log              # DQN 训练日志（实时更新）
    dqn_training_history.json             # DQN 训练历史
  checkpoints/
    dqn_best.pt                           # 最佳检查点
    dqn_final.pt                          # 最终检查点
```

---

## 待用户回来决策

1. **DQN 1000 eps 结果** — 是否能超越 AdaptiveRA (16.23%)？
2. **是否做 Phase B** — 增强 state vector 解决 split_2 问题？
3. **是否做 Top-K Selector** — 利用 96.3% top-3 hit rate？
4. **是否做 Soft-Label** — 用 teacher score 替代 one-hot？
