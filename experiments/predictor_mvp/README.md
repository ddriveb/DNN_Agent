# Selector + Mapper 闭环系统

基于 Hybrid Predictor (v2b + GAT) 的弹性光网络路由选择闭环。

## 系统架构

```
Request (src, dst, bw)
    │
    ▼
┌─────────────────────┐
│  HybridEncoder      │  ──→ v2b States + GAT Graph Features
│  (env + encoder)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  HybridSelector     │  ──→ 评估 K 条路径的 success_prob + delay
│  (predictor)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Strategy           │  ──→ max_prob / min_delay / weighted / frag-aware
│  (selector logic)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  KSPMapper          │  ──→ First-Fit 频谱分配
│  (mapper)           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  OpticalNetwork     │  ──→ 更新链路频谱状态
│  (env)              │
└─────────────────────┘
```

## 核心组件

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `env.py` | 光网络仿真环境 | `OpticalNetwork` |
| `mapper.py` | KSP + First-Fit 分配 | `KSPMapper.map()`, `execute_path()` |
| `encoder.py` | v2b 路径直方图编码 | `Encoder.encode_v2b()` |
| `hybrid_encoder.py` | Hybrid 编码（v2b + GAT输入） | `HybridEncoder.encode_full()` |
| `hybrid_predictor.py` | 路径可行性预测 | `HybridPredictor.forward()` |
| `hybrid_selector.py` | 路径选择 | `HybridSelector.select()` |
| `dataset.py` | 训练数据生成 | `DatasetGenerator.generate()` |
| `train_hybrid.py` | 模型训练 | `train_hybrid_predictor()` |
| `ksp_fast.py` | K-最短路径算法 | `k_shortest_paths()` |

## 快速开始

### 1. 运行闭环仿真

```bash
cd /home/lds/DNN_agent/experiments/predictor_mvp
/usr/bin/python3.10 demo_selector_mapper_v5.py
```

默认配置：
- 拓扑: NSFNET (14 nodes, 22 links)
- 频谱槽: 32 slots
- K: 10 条候选路径
- 训练: 1000 样本 / 5 epochs
- 负载: arr=0.25, ht=8.0 (Shortest 阻塞率 ~13%)
- 策略: Random / Shortest / Pred-MaxP / Pred-Frag

### 2. 自定义参数

```bash
# 更换拓扑
NUM_PATHS=5 TRAIN_SAMPLES=2000 TRAIN_EPOCHS=5 \
    /usr/bin/python3.10 demo_selector_mapper_v5.py

# 关键环境变量
TRAIN_SAMPLES=1000    # 训练样本数
TRAIN_EPOCHS=5        # 训练轮数（推荐 5，过拟合会恶化性能）
NUM_PATHS=10          # KSP 候选路径数（推荐 3~10）
```

### 3. 单独使用某个模块

```python
from env import OpticalNetwork
from mapper import KSPMapper
from hybrid_encoder import HybridEncoder
from hybrid_predictor import HybridPredictor
from hybrid_selector import HybridSelector

# 1. 初始化环境
net = OpticalNetwork(topology="nsfnet", num_slots=32, seed=42)
mapper = KSPMapper(net, k=10)
encoder = HybridEncoder(net, k=10)

# 2. 加载训练好的模型
model = HybridPredictor(
    state_dim=14, num_links=len(net.link_states),
    link_feat_dim=6, link_hidden=32,
    num_gat_layers=2, num_heads=4, dropout=0.1,
    num_paths=10, max_servers=128, hidden_dim=64,
)
# model.load_state_dict(torch.load("model.pt"))

# 3. 创建 Selector
selector = HybridSelector(model, encoder, strategy="max_prob")

# 4. 选择路径
best, candidates = selector.select(src=0, dst=5, bw=4)
# candidates = [{path_id, success_prob, delay}, ...]

# 5. 执行分配
result = mapper.execute_path(0, 5, 4, path_id=best["path_id"])
# result = {success, path, start_slot, delay, frag_change}
```

## 性能基准

| 场景 | Shortest BR | Pred-Frag BR | 改善 |
|------|-------------|--------------|------|
| NSFNET (14节点, K=10) | 13.05% | **9.55%** | **+26.8%** |
| Random100 (100节点) | 11~19% | **9~17%** | **+27.3%** (均值) |

## 关键设计决策

| 决策 | 结论 |
|------|------|
| Encoder | **v2b 手工路径直方图** 优于纯 GNN |
| Predictor | **Hybrid (v2b + GAT)** 跨拓扑 AUC 0.8545，最佳 |
| 选择策略 | **Frag-aware** (top-90% prob + min frag) 是唯一有效策略 |
| 训练配置 | **欠拟合** (1000样本/5epochs) 优于过拟合 |
| K值 | **K=10** 略优于 K=3，但边际收益递减 |

## 归档组件

见 [ARCHIVE.md](ARCHIVE.md) — 记录被验证低效的组件（PINN、独立GNN）和历史实验脚本。

## 已知问题

1. **类别不平衡**: 训练数据正样本 85-95%，负样本不足
2. **MaxP 策略失败**: 单纯选概率最高的路径在闭环中表现灾难性
3. **AUC ≠ 闭环性能**: 高 AUC 模型反而更差（过拟合悖论）
4. **DatasetGenerator 原 bug**: 已修复（path-id 标签与 map() 不匹配）

## 依赖

- Python 3.10
- PyTorch 2.11+ (CPU)
- numpy, networkx, scikit-learn

无需 GPU，纯 CPU 运行。
