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

### Phase 0：基础设施（Week 1）

**目标**：搭建可运行的最小仿真环境。

| 任务 | 产出 | 验收标准 |
|------|------|----------|
| 搭建光网拓扑模型 | topology.py | 支持 2 种以上拓扑（如 NSFNET、GEANT），可输出邻接矩阵 |
| 搭建频谱链路模型 | spectrum.py | 支持固定栅格频谱，记录每条链路的 slot 占用状态 |
| 搭建 MEC 资源模型 | mec.py | 支持多 server，记录算力、负载、缓存状态 |
| 搭建请求生成器 | traffic.py | 支持按分布生成推理请求，含 DNN 模型类型、数据量、时延约束 |
| 环境集成测试 | test_environment.py | 能跑通 100 个请求的仿真，无异常 |

### Phase 1：环境表征与编码器（Week 2）

**目标**：实现 z_t = g(Env_t)，输出 10-30 维统计向量。

| 任务 | 产出 | 验收标准 |
|------|------|----------|
| 定义 z 维度 | metrics.py | 实现 fragmentation index、server load entropy、path risk vector、deployment score、resource pressure |
| 编码器实现 | encoder.py | 输入环境状态，输出 z（numpy 或 tensor）|
| 多环境生成器 | env_generator.py | 生成 18 个环境（2 拓扑 × 3 负载 × 3 碎片），每个环境可独立加载 |
| 编码器测试 | test_encoder.py | 不同环境的 z 向量有显著差异（余弦相似度或距离可区分）|

**关键决策点**：
- z 维度第一版控制在 **15 维左右**。
- 每个指标需有明确的物理意义和计算公式。

### Phase 2：Inference Agent（Week 3-4）

**目标**：实现 π_θ(a|x,z)，Double DQN。

| 任务 | 产出 | 验收标准 |
|------|------|----------|
| 状态空间定义 | - | x_t = [请求特征, 快状态]，明确各维度含义和范围 |
| 动作空间定义 | - | a_high = [partition_point, target_server, offloading_mode]；动作空间大小 < 1000 |
| DQN 网络实现 | network.py | 输入 (x, z)，输出 Q(s,a)；支持 dueling 或双层网络（可选）|
| 经验回放 | replay_buffer.py | 支持多环境混合采样，每条经验记录对应的 z |
| 训练循环 | trainer.py | 单环境上能收敛，blocking 随训练下降 |
| Agent 单元测试 | test_agent.py | 网络输入输出维度正确，训练过程无 NaN |

**关键决策点**：
- 第一版用 Double DQN，不急于上 PPO/SAC。
- 动作空间若仍太大，可先只优化 partition + target（2 维决策），offloading mode 内嵌。

### Phase 3：Execution Mapper（Week 4-5）

**目标**：实现 M(a_high, s_net, z) → [path, modulation, slot_block]。

| 任务 | 产出 | 验收标准 |
|------|------|----------|
| KSP 实现 | ksp.py | Yen's algorithm 或简化的 K-shortest，K=3~5 |
| 频谱分配器 | spectrum_allocator.py | 实现 first-fit、best-fit、fragmentation-aware best-fit |
| Mapper 主逻辑 | mapper.py | 输入高层动作，输出 path-slot；失败返回 blocking |
| 约束检查 | - | 每条路径满足频谱连续性、连续性约束；server 资源不超发 |
| Mapper 测试 | test_mapper.py | 各种边界情况（资源不足、多请求竞争）下行为正确 |

**关键决策点**：
- path risk 排序用 z 中的 path risk vector。
- 若所有路径失败，记录 blocking，不强行分配（保证约束）。

### Phase 4：多环境联合训练（Week 5-6）

**目标**：在 18 个环境上训练条件策略，使其学会根据 z 自适应。

| 任务 | 产出 | 验收标准 |
|------|------|----------|
| 奖励函数 | reward.py | r_t = -αT - βB - γC_frag - ηC_res + μS，参数可调 |
| 多环境采样 | multi_env_trainer.py | 每轮从 18 个环境均匀或按难度采样 |
| 完整训练流程 | - | 端到端：环境 → z → Agent → Mapper → reward → 回传 |
| 训练监控 | - | TensorBoard 记录各环境的 loss、Q 值、reward、blocking rate |

**关键决策点**：
- 训练时环境切换频率：每 episode 换一个环境，还是每 N step 换？
- 是否引入课程学习？先在中等环境训练，再逐步加入极端环境。

### Phase 5：评估与消融实验（Week 7-8）

**目标**：验证假设，产出论文可用图表。

| 任务 | 产出 | 验收标准 |
|------|------|----------|
| 基线实现 | baselines/ | Greedy、Vanilla DRL、Transfer DRL 均可运行 |
| 单环境对比 | results/ | 在同一环境上，本方案 vs 基线的延迟、blocking、吞吐 |
| 跨环境泛化测试 | generalization.py | 在 OOD 环境（新负载组合、轻微拓扑扰动）上测试零迁移性能 |
| 消融实验 | results/ablation/ | 去掉 z、打乱 z、固定旧 z、Mapper 不用 z 等 5 组实验 |
| 可视化 | notebooks/ | 训练曲线、拓扑热力图、z 空间 t-SNE（若维度允许）|

**关键决策点**：
- OOD 环境如何构造？建议：负载/碎片取训练档位的中间值（如训练中只有低/高，测试用中等）。
- 消融实验必须做，这是论文最核心的实证支撑。

### Phase 6：Transfer Fallback 与收尾（Week 8-9，可选）

**目标**：在严重 OOD 场景下兜底。

| 任务 | 产出 | 验收标准 |
|------|------|----------|
| OOD 检测 | - | 基于 z 的分布距离（如训练 z 的均值方差）检测是否 OOD |
| 轻量微调 | transfer_drl.py | 仅微调最后 1-2 层，少量梯度步数 |
| 对比实验 | results/transfer/ | π(x,z_new) vs 微调后 π_θ'(x) 性能差距 < 5% 为理想 |

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
