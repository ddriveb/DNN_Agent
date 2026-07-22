# SA-HMARL

SA-HMARL 是面向光网络 + MEC 场景的 DNN 分布式推理卸载系统。

这个仓库里同时保留了两条研究线，所以以前最容易混乱的地方不是代码，
而是“同一个东西在不同报告里叫法不一致”。从现在开始，统一按下面的
规范阅读和引用。

## 先看这里

- 命名总表：`sa_hmarl/experiments/CANONICAL_NAMES.md`
- 实验索引：`sa_hmarl/experiments/README.md`
- checkpoint 索引：`sa_hmarl/checkpoints/README.md`
- dataset 索引：`sa_hmarl/datasets/README.md`

## 当前对外主线

当前推荐作为系统主线的组合是：

```text
PPO-C + v1.2 static counterfactual ranker
```

含义：

- `PPO-C`：高层计算卸载策略，选择 `split_id, server_id`
- `v1.2 static counterfactual ranker`：低层 R 端静态排序器，选择
  `path, modulation, slot block`
- `v1.2` 不是在线 rollout policy，而是把离线 H-step counterfactual
  planner 偏好蒸馏成一个候选动作打分网络

主实验的对外推荐结果见：

- `sa_hmarl/experiments/main_s100_system_comparison.md`
- `sa_hmarl/experiments/main_s100_ksp_ff_k50_hops_strict_fixrerun.md`

## 两条研究线怎么区分

### A. 当前系统主线

用于论文主表、fix rerun、XLRON transfer、公平性审查。

统一叫法：

```text
PPO-C + v1.2 static counterfactual ranker
```

不要再单独说：

- `final v1.2`
- `mixed low`
- `ranker-only`
- `counterfactual_rank_only`

除非同时给出 checkpoint 或脚本路径。

### B. 历史 PPO-R 研究线

这一支主要用于说明“独立 PPO-R 能否达到强 R 端基线水平”，不是当前
paper-facing 主系统。

统一叫法：

```text
Typed-Mean-Field Agent-C + independent PPO-R
```

不要把它和 v1.2 主线混写。凡是讨论 `BC-PPO-R`、`teacher transfer`、
`same-state diagnostic`、`Oracle-R ceiling` 的文档，默认都属于这一支。

## 当前最重要结论

主实验配置：

- topology: `snap24_gnutella_reach`
- slots: `100`
- seeds: `3030,4040,5050,6060,7070`
- episodes: `20`
- requests per episode: `80`

bug 修复后的严格对比结果：

| Method | Blocking | NSB | Overload | Delay mean/P95 |
|---|---:|---:|---:|---:|
| PPO-C + v1.2 static counterfactual ranker | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms |
| PPO-C + KSP-FF K=50 hops | 1.05% | 0.94% | 0.11% | 7.919/15.229 ms |
| PPO-C + DeepRMSA | 1.29% | 1.21% | 0.07% | 10.183/18.040 ms |
| PPO-C + KSP-FF | 7.38% | 5.94% | 1.44% | 8.201/16.280 ms |

## 代码结构

```text
sa_hmarl/
├── sa_hmarl/
│   ├── env/            # 请求、事件环境、C/R 观测构造、动作解码
│   ├── mec/            # MEC server/cluster 资源模型
│   ├── network/        # 光网络、KSP、调制格式、频谱块
│   ├── agents/         # PPO-C、PPO-R、v1.2 ranker、DeepRMSA wrapper
│   ├── training/       # PPO-C/R 训练、v1.2 ranker 训练
│   ├── evaluation/     # 主实验、诊断实验、数据集生成
│   └── baselines/      # RMSA 启发式 baseline
├── checkpoints/        # 模型索引见 checkpoints/README.md
├── datasets/           # 数据集索引见 datasets/README.md
├── experiments/        # 报告索引见 experiments/README.md
├── scripts/            # 常用脚本
└── tests/              # 单元测试
```

## 在线推理主数据流

推荐阅读顺序：

1. C 端观测  
   `sa_hmarl/env/observation_builder.py::build_agent_c_observation`
2. PPO-C 选择切分点和服务器  
   `sa_hmarl/agents/ppo_agents.py::PPOAgentC.select_action`
3. C 动作解码  
   `sa_hmarl/env/observation_builder.py::decode_agent_c_action`
4. R 端观测构造  
   `sa_hmarl/env/observation_builder.py::build_agent_r_observation`
5. legal R action 特征枚举  
   `sa_hmarl/agents/r_agent.py::AgentR.build_action_features`
6. v1.2 候选动作打分  
   `sa_hmarl/agents/r_ranker_policy.py::CounterfactualRRankerPolicy.score_legal_actions`
7. 选最高分动作  
   `sa_hmarl/agents/r_ranker_policy.py::CounterfactualRRankerPolicy.select_action`
8. R 动作解码与执行  
   `sa_hmarl/env/observation_builder.py::decode_agent_r_action`  
   `sa_hmarl/env/event_env.py::EventDrivenDNNEnv.step`

主实验入口：

```text
sa_hmarl/evaluation/eval_main_s100_system_comparison.py
```

## v1.2 离线训练数据流

v1.2 的统一叙述是：

```text
planner-distilled amortized MPC
```

训练流程：

```text
当前请求 -> 枚举候选 R 动作 -> H-step counterfactual rollout
-> 得到每个候选动作的 planner return
-> 用 listwise ranking loss 训练 ranker
```

关键文件：

- 数据集生成：`sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py`
- 候选特征：`sa_hmarl/evaluation/r_ranker_features.py`
- 25 维候选特征：`sa_hmarl/evaluation/generate_r_post_decision_dataset.py`
- ranker 网络：`sa_hmarl/agents/counterfactual_r_ranker.py`
- policy wrapper：`sa_hmarl/agents/r_ranker_policy.py`
- ranker 训练：`sa_hmarl/training/train_r_counterfactual_ranking.py`

## 核心 checkpoint

当前主线 v1.2 checkpoint：

```text
checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt
```

常用 baseline checkpoint：

```text
checkpoints/deep_rmsa_snap24_s100_k5m10_ai015_h4_10_s5_30.pt
checkpoints/agent_r_mixed.pt
checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt
```

## 运行环境

```bash
cd /mnt/d/project/DNN_Agent
export PYTHONPATH=/mnt/d/project/DNN_Agent/sa_hmarl
```

安装依赖：

```bash
pip install -r sa_hmarl/requirements.txt
```

## 复现当前主实验

```bash
cd /mnt/d/project/DNN_Agent
PYTHONPATH=/mnt/d/project/DNN_Agent/sa_hmarl \
python -m sa_hmarl.evaluation.eval_main_s100_system_comparison
```

输出：

```text
sa_hmarl/experiments/main_s100_system_comparison.json
sa_hmarl/experiments/main_s100_system_comparison.md
```

## 运行测试

```bash
cd /mnt/d/project/DNN_Agent/sa_hmarl
pytest tests
```

也可以只跑核心单测：

```bash
pytest tests/test_ppo_agents.py tests/test_agent_r.py tests/test_observation_builder.py
```

## 术语说明

- `Blocking`: 总阻塞率
- `Raw empty`: C 端原始 mask 为空导致无可行动作
- `NSB`: No Suitable Block，无合适频谱块
- `Overload`: MEC server 计算资源不足导致的阻塞
- `v1.2`: 当前主线静态 counterfactual ranker
- `DeepRMSA`: 强 R 端 baseline
- `independent PPO-R`: 历史研究线中的独立训练 R policy
- `BC-PPO-R`: 教师迁移 / warm-start R baseline，不是当前系统主线
