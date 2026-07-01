# Environment-Conditioned DNN Distributed Inference Offloading

> 练手项目：在光网/MEC 动态环境下，训练条件策略 π_θ(x,z) 进行 DNN 分布式推理卸载。

## 核心思路

- **Environment Encoder**: 将光网/MEC 环境编码为低维统计向量 z
- **Inference Agent**: 条件策略 π_θ(a|x,z)，输出高层粗动作（partition、target server）
- **Execution Mapper**: 受约束映射层，将粗动作转换为 path、modulation、slot block

## 目录结构

```
├── configs/       # 配置文件
├── docs/          # 文档
├── experiments/   # 实验代码（无产物，可重新运行）
│   ├── agent_mvp/      # Agent 实验（RuleAgent → Imitation → TopK → CorrectionNet）
│   ├── predictor_mvp/  # Predictor 训练与校准
│   ├── paper_compare/  # 论文对比实验
│   └── yin2024_sim/    # Yin 2024 协议仿真
├── src/           # 核心源代码
│   ├── agents/         # Agent 实现
│   ├── baselines/      # 启发式基线
│   ├── run_paper_eval.py
│   └── compile_paper_results.py
├── tests/         # 单元测试
└── notebooks/     # 分析笔记本
```

## 快速开始

```bash
# 1. 激活环境
source .venv/bin/activate

# 2. 运行论文评估
python src/run_paper_eval.py
```

## 文档

- `docs/PROJECT_STATUS.md` — 项目状态与完整实验记录
- `docs/ARCHITECTURE.md` — 系统架构设计
- `docs/EXPERIMENTS.md` — 实验方案与结果

---

*此为练手项目，已归档核心代码。接下来开始正式科研项目。*
