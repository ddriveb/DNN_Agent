# Environment-Conditioned DNN Distributed Inference Offloading

环境条件化 DNN 分布式推理卸载框架。

> 在光网/MEC 动态环境下，训练一个条件策略 π_θ(x,z)，根据实时环境表征 z 输出高层推理编排动作，再由 Execution Mapper 完成 path-slot 细粒度资源分配。

## 核心思路

- **Environment Encoder**: 将光网/MEC 环境编码为低维统计向量 z
- **Inference Agent**: 条件策略 π_θ(a|x,z)，输出高层粗动作（partition、target server）
- **Execution Mapper**: 受约束映射层，将粗动作转换为 path、modulation、slot block
- **关键优势**: 环境变化时零参数更新，仅需重新计算 z

## 快速开始

（待补充）

## 文档

- [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — 完整项目计划书
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 系统架构详细设计
- [EXPERIMENTS.md](docs/EXPERIMENTS.md) — 实验方案与结果记录

## 目录结构

```
DNN_agent/
├── src/           # 源代码
├── configs/       # 配置文件
├── experiments/   # 实验数据与结果
├── tests/         # 单元测试
├── notebooks/     # 分析笔记本
└── docs/          # 文档
```
