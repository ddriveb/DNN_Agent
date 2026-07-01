# SA-HMARL

SA-HMARL 是当前主线工程：面向光网络 + MEC 场景的 DNN 分布式推理卸载系统。

当前推荐系统是：

```text
PPO-C + planner-distilled v1.2 R-ranker
```

也就是：

- C 端用 PPO 学习高层切分与服务器选择：`split_id, server_id`
- R 端用 v1.2 planner-distilled ranker 选择 RMSA 动作：`path, modulation, slot block`
- v1.2 不是在线 rollout，而是把离线 H-step planner 的偏好蒸馏成一个候选动作打分网络

## 当前最重要结论

主实验配置：

- topology: `snap24_gnutella_reach`
- slots: `100`
- seeds: `3030,4040,5050,6060,7070`
- episodes: `20`
- requests per episode: `80`

推荐主表见：

- `experiments/main_s100_system_comparison.md`
- `experiments/main_s100_c_heuristic_r_matrix.md`

核心结果：

| Method | Blocking | NSB | Overload | Delay mean/P95 |
|---|---:|---:|---:|---:|
| PPO-C + final v1.2 | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms |
| PPO-C + DeepRMSA | 1.29% | 1.21% | 0.07% | 10.183/18.040 ms |
| PPO-C + PPO-R | 3.66% | 3.25% | 0.41% | 8.088/14.672 ms |
| PPO-C + KSP-BF | 7.68% | 6.26% | 1.41% | 8.546/15.536 ms |
| PPO-C + KSP-FF | 7.38% | 5.94% | 1.44% | 8.201/16.280 ms |

## 代码结构

```text
sa_hmarl/
├── sa_hmarl/
│   ├── env/            # 请求、事件环境、C/R 观测构造、动作解码
│   ├── mec/            # MEC server/cluster 资源模型
│   ├── network/        # 光网络、KSP、调制格式、频谱块
│   ├── agents/         # PPO-C、PPO-R、v1.2 ranker、DeepRMSA wrapper
│   ├── training/       # PPO-C/R 训练、v1.2 planner distillation 训练
│   ├── evaluation/     # 主实验、诊断实验、数据集生成
│   └── baselines/      # RMSA 启发式 baseline
├── checkpoints/        # 已保留的关键 checkpoint
├── datasets/           # 小规模离线数据集/报告
├── experiments/        # 实验结果和表格
├── scripts/            # 常用脚本
└── tests/              # 单元测试
```

## 在线推理数据流

推荐阅读顺序：

1. 构造 C 端观测  
   `sa_hmarl/env/observation_builder.py::build_agent_c_observation`

2. PPO-C 选择切分点和服务器  
   `sa_hmarl/agents/ppo_agents.py::PPOAgentC.select_action`

3. 解码 C 动作  
   `sa_hmarl/env/observation_builder.py::decode_agent_c_action`

4. 根据 C 决策构造 R 端观测  
   `sa_hmarl/env/observation_builder.py::build_agent_r_observation`

5. 枚举 legal R actions 并构造候选特征  
   `sa_hmarl/agents/r_agent.py::AgentR.build_action_features`

6. v1.2 policy wrapper 调用 ranker 打分  
   `sa_hmarl/agents/r_ranker_policy.py::CounterfactualRRankerPolicy.score_legal_actions`

7. 选择最高分 R 动作  
   `sa_hmarl/agents/r_ranker_policy.py::CounterfactualRRankerPolicy.select_action`

8. 解码 R 动作并执行环境 step  
   `sa_hmarl/env/observation_builder.py::decode_agent_r_action`  
   `sa_hmarl/env/event_env.py::EventDrivenDNNEnv.step`

主实验入口：

```text
sa_hmarl/evaluation/eval_main_s100_system_comparison.py
```

## 离线 v1.2 训练数据流

v1.2 的论文叙述建议叫：

```text
planner-distilled amortized MPC
```

它的训练不是 TD bootstrap，而是：

```text
当前请求 -> 枚举候选 R 动作 -> H-step counterfactual rollout
-> 得到每个候选动作的 planner return
-> 用 listwise ranking loss 训练 ranker
```

关键文件：

- 数据集生成：`sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py`
- 候选特征：`sa_hmarl/evaluation/r_ranker_features.py`
- 25 维候选特征定义：`sa_hmarl/evaluation/generate_r_post_decision_dataset.py`
- ranker 网络：`sa_hmarl/agents/counterfactual_r_ranker.py`
- ranker policy wrapper：`sa_hmarl/agents/r_ranker_policy.py`
- ranker 训练：`sa_hmarl/training/train_r_counterfactual_ranking.py`

## 核心 checkpoint

当前 final v1.2 推荐 checkpoint：

```text
checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt
```

常用 baseline checkpoint：

```text
checkpoints/deep_rmsa_snap24_s100_k5m10_ai015_h4_10_s5_30.pt
checkpoints/agent_r_mixed.pt
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

## 复现主实验

```bash
cd /mnt/d/project/DNN_Agent
PYTHONPATH=/mnt/d/project/DNN_Agent/sa_hmarl \
python -m sa_hmarl.evaluation.eval_main_s100_system_comparison
```

输出会写入：

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
- `NSB`: No Spectrum Blocking，光谱/路径/频隙资源不足导致的阻塞
- `Overload`: MEC server 计算资源不足导致的阻塞
- `v1.2`: 当前 final planner-distilled R-ranker，不是旧的在线规则 rerank
- `DeepRMSA`: R 端 baseline，用于对比 RMSA 决策能力

## 当前阅读建议

如果你已经看完 C 端，下一步按这个顺序读：

1. `agents/r_agent.py`
2. `agents/r_ranker_policy.py`
3. `agents/counterfactual_r_ranker.py`
4. `evaluation/generate_r_counterfactual_ranking_dataset.py`
5. `training/train_r_counterfactual_ranking.py`
6. `evaluation/eval_main_s100_system_comparison.py`

