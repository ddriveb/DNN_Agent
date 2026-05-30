# 环境条件化 DNN 分布式推理卸载框架 — 项目计划书

> **版本**：v1.0  
> **日期**：2026-04-24  
> **定位**：从"双时间尺度 MARL"转向"环境条件化策略 + 执行映射层"的轻量可泛化框架

---

## 一、可行性逻辑评估

### 1.1 为什么这个方案可行？

| 论证点 | 分析 |
|--------|------|
| **问题本质匹配** | 光网/MEC 环境变化（负载、碎片、拓扑扰动）是**可观测且可量化**的，用低维统计向量 z 表征具备信息论基础 |
| **条件策略的合理性** | π_θ(a|x,z) 在 RL 领域已被验证（如域随机化、环境条件化策略），本质是让策略学会**状态-环境联合空间**中的决策边界 |
| **动作空间解耦的必要性** | path-slot 分配是组合优化问题，直接让 RL 输出会导致动作空间爆炸；分层解耦（粗动作 + 确定性映射）是工业界验证过的有效模式 |
| **训练规模可控** | 第一阶段仅 2 拓扑 × 3 负载 × 3 碎片 = 18 个环境，DQN 训练在单卡 GPU/CPU 上完全可行 |
| **与 Yin 2024 / Bao 2025 的差异化空间** | Yin 依赖迁移学习适应新环境，Bao 是统一优化无显式环境感知；本方案的核心假设——"策略能学会根据 z 自适应"——是两个工作的合理中间态 |

### 1.2 潜在风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| z 表征信息不足，策略无法区分环境 | 中 | 高 | 第一阶段先做单变量消融，确保打乱 z 后性能显著下降；若不行则增加 z 维度或引入轻量 GNN 编码 |
| Execution Mapper 成为瓶颈，频繁 blocking | 中 | 高 | Mapper 中保留 K-shortest + path risk 排序 + best-fit 多级回退；若仍高则引入轻量 ILP 子问题求解 |
| 多环境训练收敛困难，策略在各环境上都是次优 | 中 | 中 | 采用课程学习（curriculum learning），从简单环境开始逐步增加难度；或引入环境embedding的辅助监督 loss |
| 在线 z_t 计算延迟过大 | 低 | 中 | z 是统计量，计算复杂度 O(E+V)，可每 N 个请求更新一次而非每个请求；或设计增量更新版本 z_{t+1} = λ z_t + (1-λ) z_new |
| 新环境严重 OOD（超出训练分布） | 低 | 高 | 保留轻量 Transfer Fallback，当检测到性能下降时触发微调 |

### 1.3 关键假设验证清单

- [ ] **假设 1**：打乱 z 后策略性能显著下降 → 证明策略确实在使用 z
- [ ] **假设 2**：环境变化后更新 z 优于固定旧 z → 证明实时 z_t 的价值
- [ ] **假设 3**：π_θ(x, z_new) 性能接近迁移后的 π_θ'(x) → 证明条件化策略可减少迁移依赖
- [ ] **假设 4**：Mapper 使用 z 排序路径优于纯 shortest-path → 证明 z 对执行层也有帮助

---

## 二、预期效果分析

### 2.1 相对 Baseline 的预期提升

| 指标 | 普通 DRL (π(x)) | Yin 2024 (迁移) | **本方案 (π(x,z) + Mapper)** | 预期优势来源 |
|------|-----------------|-----------------|------------------------------|--------------|
| 环境变化后延迟 | 显著恶化 | 需迁移时间 | **基本稳定** | z 实时驱动策略适应 |
| 迁移/更新频率 | 无 | **频繁** | **极低** | 策略本身学会泛化 |
| Blocking Probability | 中等 | 中等 | **降低 10-30%** | Mapper 受约束优化 + z 感知路径选择 |
| 在线决策时间 | 快 | 快 | **快** | DQN 前向传播 O(1)，Mapper 是确定性算法 |
| 训练复杂度 | 低 | 中（需保存旧环境参数） | **中** | 多环境训练，但单次训练后即固定 |

### 2.2 最有价值的预期结果

> **核心卖点**：在环境持续变化的场景中，本方案能以**零参数更新**的方式维持性能，而对比方案需要频繁迁移或重训练。

具体预期：

1. **跨环境泛化**：在训练未见过的新负载/碎片组合上，π_θ(x, z) 能达到该环境上专门训练普通 DRL 的 **85-95%** 性能。
2. **零迁移成本**：环境变化时无需任何梯度更新，仅需重新计算 z_t（毫秒级）。
3. **可解释性增强**：通过分析 z 各维度对策略输出的影响，可解释"为什么在这个环境下选择这种 partition"。

---

## 三、项目目录结构

```
DNN_agent/
├── docs/                       # 文档
│   ├── PROJECT_PLAN.md         # 本计划书
│   ├── ARCHITECTURE.md         # 系统架构详细设计（待补充）
│   └── EXPERIMENTS.md          # 实验方案与结果记录（待补充）
│
├── src/                        # 源代码
│   ├── environment/            # 仿真环境
│   │   ├── topology.py         # 光网拓扑生成与管理
│   │   ├── spectrum.py         # 频谱/链路状态模型
│   │   ├── mec.py              # MEC server 资源模型
│   │   ├── traffic.py          # 请求生成与分布
│   │   └── env_generator.py    # 多环境批量生成器（18个环境）
│   │
│   ├── encoder/                # 环境编码器
│   │   ├── metrics.py          # 各项指标计算（碎片、负载熵等）
│   │   └── encoder.py          # z_t = g(Env_t) 实现
│   │
│   ├── agent/                  # 推理 Agent（条件策略）
│   │   ├── network.py          # DQN / Double DQN 网络结构
│   │   ├── policy.py           # π_θ(a|x,z) 策略封装
│   │   ├── replay_buffer.py    # 经验回放（支持环境标签）
│   │   └── trainer.py          # 训练循环（多环境采样）
│   │
│   ├── mapper/                 # 执行映射层
│   │   ├── ksp.py              # K-shortest paths 计算
│   │   ├── spectrum_allocator.py # first-fit / best-fit 频谱分配
│   │   └── mapper.py           # M(a_high, s_net, z) 封装
│   │
│   ├── training/               # 训练流程编排
│   │   ├── curriculum.py       # 课程学习调度（可选）
│   │   ├── multi_env_trainer.py # 多环境联合训练主循环
│   │   └── reward.py           # 奖励函数 r_t 实现
│   │
│   ├── evaluation/             # 评估与测试
│   │   ├── metrics.py          # 评价指标计算
│   │   ├── tester.py           # 单环境测试流程
│   │   └── generalization.py   # 泛化性测试（OOD 环境）
│   │
│   └── baselines/              # 对比基线实现
│       ├── greedy.py           # 贪心启发式
│       ├── vanilla_drl.py      # 普通 DRL (π(x))
│       └── transfer_drl.py     # Yin 2024 风格迁移学习
│
├── configs/                    # 配置文件
│   ├── env_config.yaml         # 环境生成参数（拓扑、负载档位）
│   ├── agent_config.yaml       # Agent 超参数（网络结构、学习率等）
│   └── training_config.yaml    # 训练超参数（批次大小、训练步数等）
│
├── experiments/                # 实验输出
│   ├── data/                   # 生成的环境数据、轨迹数据
│   ├── results/                # 实验结果（指标、图表）
│   └── logs/                   # 训练日志、TensorBoard 日志
│
├── tests/                      # 单元测试
│   ├── test_environment.py
│   ├── test_encoder.py
│   ├── test_agent.py
│   └── test_mapper.py
│
├── notebooks/                  # Jupyter  notebooks（分析、可视化）
│   ├── visualize_topology.ipynb
│   └── analyze_results.ipynb
│
├── requirements.txt            # Python 依赖
└── README.md                   # 项目简介与快速开始
```

---

## 四、分阶段实施计划

### ✅ Phase 0：基础设施（Week 1）— 已完成

**目标**：搭建可运行的最小仿真环境。

| 任务 | 产出 | 验收标准 | 状态 |
|------|------|----------|------|
| 搭建光网拓扑模型 | `experiments/predictor_mvp/env.py` | 支持 2 种以上拓扑（NSFNET、USNET），可输出邻接矩阵 | ✅ |
| 搭建频谱链路模型 | `experiments/predictor_mvp/mapper.py` | 支持固定栅格频谱，记录每条链路的 slot 占用状态 | ✅ |
| 搭建 MEC 资源模型 | `experiments/predictor_mvp/env.py` | 支持多 server，记录算力、负载、缓存状态 | ✅ |
| 搭建请求生成器 | `experiments/agent_mvp/traffic_generator.py` | 支持按分布生成推理请求，含 DNN 模型类型、数据量、时延约束 | ✅ |
| 环境集成测试 | `experiments/agent_mvp/eval_fixed.py` | 能跑通 2000 个请求的仿真，无异常 | ✅ |

### ✅ Phase 1：环境表征与编码器（Week 2）— 已完成

**目标**：实现 z_t = g(Env_t)，输出 10-30 维统计向量。

| 任务 | 产出 | 验收标准 | 状态 |
|------|------|----------|------|
| 定义 z 维度 | `experiments/predictor_mvp/encoder.py` | 实现 fragmentation index、server load entropy、path risk vector、deployment score、resource pressure | ✅ |
| 编码器实现 | `experiments/predictor_mvp/encoder.py` | 输入环境状态，输出 z（numpy 或 tensor）| ✅ |
| 多环境生成器 | `experiments/agent_mvp/fixed_trace.py` | 生成固定 trace 用于公平对比 | ✅ |
| 编码器测试 | `experiments/predictor_mvp/main_cross_topology.py` | 不同拓扑的 z 向量可区分 | ✅ |

**关键决策点**：
- z 维度最终版为 **10 维**（link-state vector），v2b 编码器被验证为最优。
- 每个指标有明确的物理意义和计算公式。

### ✅ Phase 2：Inference Agent（Week 3-4）— 已完成（路线调整）

**目标**：实现 π_θ(a|x,z)。

**路线调整**：未采用端到端 DQN，改为 **Imitation Learning + Predictor reranking** 的分层架构：

| 任务 | 产出 | 验收标准 | 状态 |
|------|------|----------|------|
| 状态空间定义 | `experiments/agent_mvp/state_builder.py` | x_t = [请求特征, 快状态]，30D enhanced | ✅ |
| 动作空间定义 | `experiments/agent_mvp/dnn_models.py` | a_high = (split_id, server_id)，15 actions | ✅ |
| Imitation Agent | `experiments/agent_mvp/train_imitation.py` | 行为克隆 RuleAgent，top-3 hit rate > 95% | ✅ |
| TopK Selector | `experiments/agent_mvp/topk_selector_agent.py` | 神经 top-K + predictor 重排序 | ✅ |
| 单元测试 | `experiments/agent_mvp/eval_imitation.py` | 网络输入输出维度正确 | ✅ |

**关键决策**：
- Imitation + Predictor 已显著优于基线（15.51% vs 19.48%），无需端到端 DQN。
- 动作空间: 3 splits × 5 servers = 15 actions。

### ✅ Phase 3：Execution Mapper（Week 4-5）— 已完成

**目标**：实现 M(a_high, s_net, z) → [path, modulation, slot_block]。

| 任务 | 产出 | 验收标准 | 状态 |
|------|------|----------|------|
| KSP 实现 | `experiments/predictor_mvp/ksp_fast.py` | Yen's algorithm, K=3 | ✅ |
| 频谱分配器 | `experiments/predictor_mvp/mapper.py` | first-fit + best-fit | ✅ |
| Mapper 主逻辑 | `experiments/predictor_mvp/mapper.py` | 输入高层动作，输出 path-slot；失败返回 blocking | ✅ |
| 约束检查 | `experiments/predictor_mvp/mapper.py` | 频谱连续性、连续性约束；server 资源不超发 | ✅ |
| Mapper 测试 | 集成在 `eval_*.py` 中 | 2000+ 请求无异常 | ✅ |

**关键决策**：
- K=3 shortest paths + first-fit 足够；更复杂的分配器收益有限。
- Mapper 不引入 blocking 错误（已通过所有实验验证）。

### ✅ Phase 4：多环境联合训练（Week 5-6）— 已完成（路线调整）

**目标**：训练 CorrectionNet 学习长期价值残差。

| 任务 | 产出 | 验收标准 | 状态 |
|------|------|----------|------|
| 奖励函数 | `experiments/agent_mvp/env_wrapper.py` | r_t = +1.0 - delay_norm (success), -2.0 (fail) | ✅ |
| 离线数据收集 | `experiments/agent_mvp/collect_replay.py` | 30K transitions from TopK2 policy | ✅ |
| CorrectionNet 训练 | `experiments/agent_mvp/train_correction_net.py` | 5K 参数网络，val loss < 0.6 | ✅ |
| 训练监控 | `experiments/agent_mvp/logs/train_correction_net.log` | loss 曲线稳定下降 | ✅ |

**关键决策**：
- 不训练端到端条件策略；改为在强基线（TopK2）上叠加残差网络。
- CorrectionNet 输入 10D，输出 1D correction score。
- 目标: episode return G_t，γ=0.95。

### ✅ Phase 5：评估与消融实验（Week 7-8）— 已完成

**目标**：验证假设，产出论文可用图表。

| 任务 | 产出 | 验收标准 | 状态 |
|------|------|----------|------|
| 基线实现 | `experiments/agent_mvp/baselines.py` | Random, ShortestPath, YinLike, LoadBalanced | ✅ |
| 单环境对比 | `src/results/paper_table_1_main.json` | 3 场景 × 6 方法 | ✅ |
| 跨环境泛化测试 | `experiments/agent_mvp/results/` | USNET 零迁移: 21.77±0.89% | ✅ |
| 消融实验 | `src/results/paper_table_2_ablation.json` | 5 组消融 | ✅ |
| 可解释性分析 | `src/results/paper_table_4_interpretability.json` | Decision change + future blocking correlation | ✅ |

**关键决策**：
- 消融实验证实了: predictor reranking 是最大单一增益源；CorrectionNet 额外贡献 2.47pp。
- 直接 DQN 替换 selector 失败（38% blocking），证明了残差架构的必要性。

### 🔄 Phase 6：Yin 2024 协议仿真（Week 9-10）— 进行中

**目标**：在 Yin 2024 论文的协议设置上评估基线 + 我们的方法。

| 任务 | 产出 | 验收标准 | 状态 |
|------|------|----------|------|
| 拓扑重建 | `experiments/yin2024_sim/yin2024_network.py` | Net-1/2/3 从 Fig. 7 数字化 | ✅ |
| 频谱初始化 | `experiments/yin2024_sim/yin2024_network.py` | load_factor=0.6, frag=0.2/0.5 | ✅ |
| 请求生成器 | `experiments/yin2024_sim/yin2024_requests.py` | 1-5 subtasks, 20-40ms deadline | ✅ |
| 基线评估 (Net-1) | `experiments/yin2024_sim/results/yin2024_net1_results.json` | WO/DF/RF/IWD-Approx/YinLike | ✅ |
| 基线评估 (Net-2/3) | — | 同上 | 🔄 |
| frag=0.5 评估 | — | 同上 | 🔄 |
| 我们的方法 zero-shot | — | TopK2 / CorrectionNet 零迁移 | 🔄 |
| 绘制 Fig. 8 曲线 | — | Blocking vs N_req | ❌ |

**关键问题**：
- YinLike 在 Net-1 上阻塞率极低（0-14%），可能与小型网络 + 低请求数有关。需要在 Net-2/3 上验证趋势。
- RF 阻塞率高于 DF（68% vs 60%），与直觉相反。可能原因是 RF 选择远距离低负载服务器，导致路径频谱阻塞。

---

## 五、技术栈建议

| 组件 | 推荐方案 | 理由 |
|------|----------|------|
| 深度学习框架 | PyTorch | 灵活，DQN 实现简单 |
| 图/网络算法 | NetworkX | 拓扑、KSP 成熟 |
| 科学计算 | NumPy, SciPy | 环境仿真主力 |
| 配置管理 | YAML + dataclasses | 人类可读，类型安全 |
| 日志/可视化 | TensorBoard, Matplotlib | 训练监控 + 论文图表 |
| 单元测试 | pytest | 轻量 |
| 环境管理 | venv / conda | 隔离依赖 |

---

## 六、里程碑与检查点

| 里程碑 | 时间 | 关键产出 | Go/No-Go 标准 |
|--------|------|----------|---------------|
| M1：环境可运行 | Week 1 结束 | 仿真环境跑通 100 请求 | 无崩溃，blocking 率可统计 |
| M2：Agent 单环境收敛 | Week 3 结束 | 单环境上 blocking 率随训练下降 | 训练 10k episodes 后 blocking < 20% |
| M3：Mapper 零阻塞错误 | Week 5 结束 | Mapper 通过全部单元测试 | 资源约束 100% 满足 |
| M4：多环境训练完成 | Week 6 结束 | 条件策略训练收敛 | 各环境平均 reward 为正且稳定 |
| M5：消融验证通过 | Week 7 结束 | 5 组消融实验完成 | 假设 1-3 至少 2 个显著成立 |
| M6：论文图表就绪 | Week 8 结束 | 8 张核心图 | 包含对比、消融、泛化、训练曲线 |

---

## 七、核心创新点论文表述（预留）

### 创新点 1
提出一种**环境条件化的 DNN 分布式推理卸载框架**，将光网/MEC 环境编码为低维表征 z，训练条件策略 π_θ(x,z)，使同一策略在不同环境下自动调整高层推理编排决策，无需参数更新即可适应环境变化。

### 创新点 2
设计一种**策略层与执行层解耦机制**：Inference Agent 仅输出高层粗动作（partition、target server），Execution Mapper 负责 path-slot 细粒度映射，既避免 RL 动作空间爆炸，又保证光网资源约束严格满足。

### 创新点 3
构建一种**环境表征驱动的泛化适应机制**，通过多环境联合训练让策略学习 z → 决策的映射规律，实验验证在未见环境上的零迁移性能达到专门训练模型的 85% 以上，显著减少迁移学习开销。

---

## 八、附录：最小可做版本（MVP）范围

若时间紧张，优先保以下范围：

1. **z 维度**：仅保留 fragmentation index + server load entropy（2 维）。
2. **动作空间**：仅 partition point（3-4 档）+ target server（2-3 个）。
3. **Mapper**：K=3 shortest paths + first-fit。
4. **环境**：1 个拓扑 × 2 负载 × 2 碎片 = 4 个环境。
5. **基线**：仅 Greedy + Vanilla DRL。
6. **消融**：只做"去掉 z"和"打乱 z"两组。

MVP 验证的核心假设：**即使只有 2 维 z，策略仍能学会区分环境并调整决策。**

---

*本文档随项目进展持续更新。下一版补充：ARCHITECTURE.md（模块接口定义）、EXPERIMENTS.md（实验参数与结果）。*
