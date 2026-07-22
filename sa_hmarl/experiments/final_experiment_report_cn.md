# SA-HMARL 当前完整实验报告（中文注释版）

> 作用说明：这份报告属于历史 `independent PPO-R` 研究线，不是当前
> `PPO-C + v1.2 static counterfactual ranker` 主线总表。统一命名见
> `CANONICAL_NAMES.md`。

> 更新日期：2026-06-11。本文把当前关键实验整理成一份中文报告，
> 重点区分 C 端贡献、R 端贡献、教师迁移基线和诊断性实验。

## 0. 核心结论

当前最推荐作为论文主方法描述的是：

```text
增强型计算卸载智能体 Agent-C + 独立训练的 PPO-R 路由与频谱分配智能体
```

一句话总结：

```text
Agent-C 负责高层 split/server 选择，独立训练的 PPO-R 负责低层 RMSA 执行。
PPO-R 已达到 DeepRMSA 级别的 R 端阻塞性能；剩余阻塞主要由 C 端选择了
没有任何可行 R 动作的 split/server 组合导致。
```

需要特别注意：

- `BC-PPO-R` 不是当前主贡献，而是“行为克隆教师迁移 / 等价性验证”基线。
- 独立训练的 `PPO-R` 才是 R 端主贡献。
- 一些早期主表使用 `BC-PPO-R` 作为稳定后端，是因为当时它用于固定 R 端做 C 端对比；后续同状态诊断证明独立 `PPO-R` 与 `BC-PPO-R` 在成功/失败决策上几乎完全等价。

## 1. 名词全称与含义

| 缩写 | 全称 / 中文解释 | 在本文中的作用 |
|---|---|---|
| Agent-C | Computation Agent，计算卸载智能体 | 选择 DNN split 点和边缘服务器 |
| Agent-R | Routing and Spectrum Assignment Agent，路由与频谱分配智能体 | 选择光路、调制格式和频谱块 |
| PPO-R | Proximal Policy Optimization Agent-R，基于近端策略优化的 R 端智能体 | 本文主要 R 端方法，独立训练 |
| BC-PPO-R | Behavior-Cloned PPO-R，行为克隆初始化的 PPO-R | 教师迁移 / 稳定性验证基线 |
| DeepRMSA | Deep Reinforcement Learning for Routing, Modulation and Spectrum Assignment | 强 R 端教师式基线 |
| KSP-BF | K-Shortest Paths + Best-Fit Spectrum Assignment | K 条最短路径 + 最佳适配频谱分配启发式 |
| IWD-C | Inverse Workload Distance C baseline | 反工作负载距离计算卸载启发式 |
| DF-C | Delay-First C baseline | 延迟优先计算卸载启发式 |
| WO-C | Workload-Only C baseline | 仅考虑工作负载的计算卸载启发式 |
| RF-C | Random Feasible C baseline | 随机可行计算卸载启发式 |
| Greedy-C | Greedy Computation baseline | 贪心计算卸载基线 |
| Blocking | 阻塞率 | 请求无法成功完成的比例，越低越好 |
| Delay | 端到端时延 | 单位为毫秒，越低越好 |
| AvgFS | 平均占用频谱槽数 | 越低表示频谱使用越省 |
| noC% | C 端无有效动作比例 | Agent-C 找不到可行 split/server 动作的比例 |
| avgRacts | R 端平均有效动作数 | 每个请求平均可选 R 动作数量 |
| NSB | No Suitable Block，无合适频谱块 | R 端失败的主要物理约束原因 |

## 2. 实验 A：C 端对比，不同 C 算法 + 相同 R 后端

**实验目的：** 固定 R 端为 `BC-PPO-R`，只改变 C 端策略，观察高层 split/server 决策质量。

**设置：**

| 项目 | 值 |
|---|---|
| 拓扑 | `snap24_gnutella_reach` |
| 频谱槽数 | S16，即每条链路 16 个频谱槽 |
| 请求负载 | R80，即每个 episode 80 个请求 |
| split 配置 | `complex5_v2_lite` |
| 候选路径数 | `k_paths = 5` |
| 最大频谱块数 | `max_blocks = 10` |
| 频谱块排序 | `block_sort = mixed` |
| 随机种子 | 5 个种子 |
| 每种子 episode 数 | 10 |
| R 后端 | 行为克隆 PPO-R（BC-PPO-R） |

| 算法组合 | 阻塞率 Blocking | 时延 Delay | 平均频谱槽 AvgFS | C 端无动作 noC% | 说明 |
|---|---:|---:|---:|---:|---|
| 增强型 Agent-C（PPO） + BC-PPO-R | 0.3955 | 7.7ms | 2.05 | 45.3% | 最优 C 端方法 |
| IWD-C 反工作负载距离启发式 + BC-PPO-R | 0.4373 | 8.7ms | 2.17 | 49.4% | 最强传统 C 基线 |
| DF-C 延迟优先启发式 + BC-PPO-R | 0.4567 | 7.4ms | 2.35 | 51.6% | 时延低，但阻塞更高 |
| Greedy-C 贪心计算卸载 + BC-PPO-R | 0.4647 | 7.5ms | 2.43 | 52.1% | 弱于 Agent-C |
| WO-C 仅工作负载启发式 + BC-PPO-R | 0.4748 | 9.0ms | 2.47 | 52.8% | 弱于 Agent-C |
| RF-C 随机可行启发式 + BC-PPO-R | 0.4750 | 8.7ms | 2.48 | 53.0% | 弱于 Agent-C |
| 增强型 Agent-C（PPO） + KSP-BF | 0.5205 | 7.0ms | 2.90 | 58.3% | 弱 R 后端对照 |

**中文注释：**

- 在同一个 R 后端下，增强型 Agent-C 的阻塞率为 `0.3955`，最佳传统 C 基线 IWD-C 为 `0.4373`。
- 因此 Agent-C 相比最佳传统 C 基线降低阻塞约 `4.18` 个百分点。
- 把 R 后端从 `BC-PPO-R` 换成 `KSP-BF` 后，阻塞率从 `0.3955` 上升到 `0.5205`，退化约 `12.50` 个百分点。
- 这说明 C 端和 R 端都有贡献：Agent-C 比传统 C 选择更好，学习型 R 后端也明显强于 KSP-BF。

## 3. 实验 B：K 值消融，不同 C 算法 + 独立训练 PPO-R

**实验目的：** 固定 R 端为独立训练的 `PPO-R`，比较不同 C 端算法，并观察候选路径数 K 对结果的影响。

**设置：**

| 项目 | 值 |
|---|---|
| 拓扑 | `snap24_gnutella_reach` |
| 频谱槽数 | S20，即每条链路 20 个频谱槽 |
| 请求负载 | R80 |
| split 配置 | `default3` |
| K 值 | 4、5、7 |
| 随机种子 | 5 个种子 |
| 每种子 episode 数 | 5 |
| R 后端 | 独立训练的 PPO-R |

| K 值 | 增强型 Agent-C + 独立 PPO-R | IWD-C + 独立 PPO-R | DF-C + 独立 PPO-R | 增强型 Agent-C + KSP-BF |
|---:|---:|---:|---:|---:|
| 4 | 0.1815（8.2ms） | 0.2080（9.3ms） | 0.2155（7.5ms） | 0.2820（7.3ms） |
| 5 | 0.1830（8.3ms） | 0.2035（9.7ms） | 0.2130（7.8ms） | 0.2605（7.5ms） |
| 7 | 0.1805（8.7ms） | 0.2060（10.2ms） | 0.2130（8.4ms） | 0.2405（8.1ms） |

**中文注释：**

- 在 S20/default3 条件下，增强型 Agent-C + 独立 PPO-R 的阻塞率稳定在 `0.1805` 到 `0.1830` 之间。
- Agent-C 相比 IWD-C 的优势约为 `2.05` 到 `2.65` 个百分点。
- K 从 4 增加到 7，对 Agent-C + PPO-R 的阻塞率影响很小，说明独立 PPO-R 在当前动作空间下已经比较稳定。
- K 增大对 `KSP-BF` 帮助更明显，阻塞率从 `0.2820` 降到 `0.2405`，但仍明显落后于学习型 PPO-R。
- 这组实验最适合支撑“独立训练 PPO-R 是强 R 后端”的论文叙述，因为这里不是依赖 BC-PPO-R。

## 4. 实验 C：R 端对比，相同 C + 不同 R 后端

**实验目的：** 固定 Agent-C，只更换 R 端方法，验证 PPO-R、DeepRMSA 和传统启发式 R 后端的差异。

**设置：**

| 项目 | 值 |
|---|---|
| 拓扑 | `snap24_gnutella_reach` |
| 频谱槽数 | S24 |
| 请求负载 | R60 |
| split 配置 | `default3` |
| C 端 | 固定 Agent-C checkpoint |
| 评估规模 | 5 个种子 × 20 episodes × 60 请求 |

| R 后端 | 阻塞率 Blocking | 成功率 Success | 平均奖励 | 时延 Delay | 平均频谱槽 AvgFS | 说明 |
|---|---:|---:|---:|---:|---:|---|
| 独立训练 PPO-R | 0.224 | 0.776 | +0.074 | 7.9ms | 2.09 | 本文主 R 方法，最优/并列最优 |
| Dual-Critic v2 Agent-R | 0.224 | 0.776 | +0.074 | 7.9ms | 2.09 | 与 PPO-R 无增益差异 |
| DeepRMSA | 0.225 | 0.775 | +0.075 | 8.5ms | 2.02 | 教师式强基线，阻塞几乎相同 |
| KSP-BF | 0.264 | 0.736 | +0.041 | 7.7ms | 2.47 | 传统 R 启发式，阻塞明显更高 |

**失败原因统计：**

| R 后端 | 无合适频谱块 no_suitable_block | 服务器过载 server_overload |
|---|---:|---:|
| 独立训练 PPO-R | 93.4% | 6.6% |
| DeepRMSA | 90.7% | 9.3% |
| KSP-BF | 96.7% | 3.3% |

**中文注释：**

- 独立训练 PPO-R 的阻塞率为 `0.224`，DeepRMSA 为 `0.225`，两者几乎完全一致。
- KSP-BF 的阻塞率为 `0.264`，比 PPO-R 高约 `4.0` 个百分点。
- 这说明 PPO-R 不只是模仿教师，而是独立学习到了 DeepRMSA 级别的 RMSA 决策能力。
- PPO-R 和 DeepRMSA 的差异更多体现在频谱槽使用、路径长度或时延偏好，而不是阻塞率。

## 5. 实验 D：PPO-R 与 BC-PPO-R 的同状态诊断

**实验目的：** 排除“轨迹漂移”造成的假差异，直接在相同 pre-step 环境状态下比较独立 PPO-R 和 BC-PPO-R 的动作结果。

**设置：**

| 项目 | 值 |
|---|---|
| 比较方式 | `copy.deepcopy(env)` 后在相同状态下分别执行两个 R 策略 |
| 评估规模 | 2000 个请求状态 |
| 场景 | S20 / K=5 / mixed |

| 结果类型 | 数量 | 比例 |
|---|---:|---:|
| 两者都成功 | 1438 | 71.90% |
| 两者都失败 | 561 | 28.05% |
| BC-PPO-R 成功、独立 PPO-R 失败 | 0 | 0.00% |
| 独立 PPO-R 成功、BC-PPO-R 失败 | 1 | 0.05% |

**中文注释：**

- 没有任何状态满足“BC-PPO-R 能成功但独立 PPO-R 失败”。
- 这说明针对独立 PPO-R 做 targeted DAgger 没有可纠正样本。
- 早先看到的小幅 blocking 差异主要来自并行 episode 的轨迹漂移，而不是同状态决策质量差异。
- 因此，论文主线应该强调“独立 PPO-R 已达到 BC-PPO-R / DeepRMSA 级别”，而不是把 BC-PPO-R 当成主方法。

## 6. 实验 E：Exhaustive Oracle-R 分析

**实验目的：** 判断 R 端是否还有 blocking 改进空间。方法是固定 Agent-C 已经选出的 split/server，然后穷举所有可行 R 动作，构造一个理想 Oracle-R。

**设置：**

| 项目 | 值 |
|---|---|
| 拓扑 | `snap24_gnutella_reach` |
| 频谱槽数 | S20 |
| 请求负载 | R80 |
| 候选路径数 | `k_paths = 5` |
| 最大频谱块数 | `max_blocks = 10` |
| 频谱块排序 | `block_sort = mixed` |
| 评估规模 | 5 个种子 × 5 episodes × 80 请求 = 2000 请求 |
| Oracle 规则 | 成功优先，其次更少频谱槽、更低时延、更低浪费、更短路径 |

| 指标 | 数值 | 含义 |
|---|---:|---|
| PPO-R 阻塞率 | 0.2950 | 诊断设置下 PPO-R 的阻塞率 |
| Oracle-R 阻塞率 | 0.2950 | 穷举所有 R 动作后的理论阻塞率 |
| Oracle-R blocking gain | 0.00 pp | R 端阻塞改进空间为零 |
| PPO-R 失败但 Oracle-R 成功 | 0.0000 | PPO-R 不会在有可行替代动作时失败 |
| PPO-R 成功但 Oracle-R 更优 | 0.1890 | R 端仍有少量 FS/Delay 效率空间 |
| 平均有效 R 动作数 | 3.7 / 200 | 物理可行动作空间非常稀疏 |
| 失败原因 | 100% no_valid_r | 所有阻塞都因为没有任何可行 R 动作 |

**中文注释：**

- Oracle-R 是一个“诊断上限”，不是论文方法。它不学习，也不参与真实部署。
- 如果 Oracle-R 的阻塞率低于 PPO-R，说明 R 策略还会错过可行动作。
- 当前结果是 Oracle-R 与 PPO-R 阻塞率完全相同，增益为 `0.00` 个百分点。
- 因此，在固定 C 端选择的前提下，R 端已经达到 blocking 下界。
- 剩余 blocking 是因为 C 端选出的 split/server 组合本身没有任何可行 R 动作，而不是 R 策略选错了。

## 7. 实验 F：不同 R 后端的 Oracle-R 上限对照

**实验目的：** 验证 `DeepRMSA` 在相同固定 C 决策下是否也已经达到 R 端 blocking 上限，同时观察传统 `KSP-BF` 的差异来自哪里。

**设置：**

| 项目 | 值 |
|---|---|
| 拓扑 | `snap24_gnutella_reach` |
| 频谱槽数 | S24 |
| 请求负载 | R60 |
| 候选路径数 | `k_paths = 3` |
| 最大频谱块数 | `max_blocks = 1` |
| 频谱块排序 | `block_sort = mixed` |
| C 端 | 固定 Agent-C checkpoint |
| R 后端 | 独立 PPO-R、DeepRMSA、KSP-BF |
| 评估规模 | 5 个种子 × 5 episodes × 60 请求 = 1500 请求 |

说明：该设置与 DeepRMSA checkpoint 的训练动作空间对齐。DeepRMSA checkpoint 元数据为 `num_slots=24, k_path=3, m_blocks=1`，因此不能强行放到 S20/K5/max_blocks=10 中比较。

| R 后端 | 后端阻塞率 | Oracle-R 阻塞率 | Oracle Gain | 后端失败但 Oracle 成功 | 后端成功但 Oracle 更优 | 平均频谱槽 后端/Oracle | 平均时延 后端/Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| 独立训练 PPO-R | 0.2840 | 0.2840 | 0.0000 | 0.0000 | 0.1587 | 2.22 / 2.01 | 7.5 / 7.5 |
| DeepRMSA | 0.2813 | 0.2813 | 0.0000 | 0.0000 | 0.2567 | 2.02 / 2.01 | 8.4 / 7.5 |
| KSP-BF | 0.3120 | 0.3120 | 0.0000 | 0.0000 | 0.3173 | 2.51 / 2.01 | 7.5 / 7.4 |

失败原因：

| R 后端 | 后端失败原因 | Oracle 失败原因 |
|---|---|---|
| 独立训练 PPO-R | `no_valid_r`: 426 | `no_valid_r`: 426 |
| DeepRMSA | `no_valid_r`: 422 | `no_valid_r`: 422 |
| KSP-BF | `no_valid_r`: 468 | `no_valid_r`: 468 |

**中文注释：**

- 独立训练 PPO-R 和 DeepRMSA 的 `Oracle Gain` 都是 `0.0000`，说明二者在该固定 C 决策和当前 R 动作空间下都没有 blocking 改进空间。
- `后端失败但 Oracle 成功 = 0.0000` 表明：当 PPO-R 或 DeepRMSA 失败时，不存在另一个当前可行 R 动作能把请求救成功。
- 因此，这个实验进一步支持：剩余阻塞不是 PPO-R 或 DeepRMSA 执行器能力不足，而是 C 端已经选到了没有可行 R 动作的 split/server 状态。
- KSP-BF 的阻塞率更高，但它的即时 `Oracle Gain` 也是 0。这说明 KSP-BF 的问题更多是成功时频谱效率差，逐步造成未来状态更差，而不是在失败当步错过了一个可行成功动作。
- 这一点由 `后端成功但 Oracle 更优 = 0.3173` 支撑：KSP-BF 有约 31.73% 请求在成功时可以被 Oracle 选到更省频谱或更优的动作。

结论：

```text
在与 DeepRMSA 动作空间对齐的 S24/K3/max_blocks=1 设置下，
独立 PPO-R 与 DeepRMSA 均达到 R 端 blocking 下界。
KSP-BF 的劣势主要体现为频谱效率和长期状态退化，而非单步失败可被 Oracle-R 救回。
```

Source: `oracle_r_backend_compare_s24_k3.md`。

## 8. 消融实验：激活函数没有带来有效提升

**实验目的：** 测试 C 端 PPO actor 的激活函数是否影响 blocking。

| 激活函数 | 阻塞率 Blocking | 时延 Delay | 平均频谱槽 AvgFS | C 端无动作 noC% | 相对 Tanh |
|---|---:|---:|---:|---:|---:|
| Tanh | 0.246 | 8.0ms | 2.14 | 30.5% | 0 |
| LeakyReLU | 0.242 | 8.2ms | 2.11 | 29.7% | -0.4 pp |
| GELU | 0.249 | 8.1ms | 2.14 | 30.7% | +0.3 pp |
| SiLU | 0.260 | 8.8ms | 2.15 | 31.8% | +1.4 pp |

**中文注释：**

- LeakyReLU 只有 `0.4` 个百分点的表面改善，属于噪声范围内。
- GELU 和 SiLU 没有改善，SiLU 反而更差。
- 结论是：不需要继续做激活函数小改动，保留 Tanh 即可。

## 9. 消融实验：冻结 R 的 C 端保守微调没有提升

**实验目的：** 尝试在冻结 R 端的情况下，通过 C 端 pressure shaping 改善 split/server 选择。

| 配置 | 阻塞率 Blocking | 时延 Delay | 平均频谱槽 AvgFS | C 端无动作 noC | 平均 R 动作数 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 原始 Agent-C + 原始 R | 0.2760 | 8.4ms | 2.09 | 0.34 | 5.3 | 基线 |
| 微调 Agent-C + 原始 R | 0.2802 | 8.5ms | 2.12 | 0.35 | 5.0 | 退化 0.42 pp |
| 微调 Agent-C + 微调 R | 0.2802 | 8.5ms | 2.12 | 0.35 | 5.0 | 与微调 C 相同，说明 R 未改变 |
| 原始 Agent-C + 微调 R | 0.2760 | 8.4ms | 2.09 | 0.34 | 5.3 | 证明 R 实际冻结 |
| IWD-C + BC-PPO-R | 0.2975 | 9.9ms | 2.25 | 0.37 | 4.2 | 传统基线 |
| DF-C + BC-PPO-R | 0.3118 | 8.0ms | 2.44 | 0.38 | 3.5 | 传统基线 |

**中文注释：**

- 这个实验更准确地说是“冻结 R 的 C 端微调”，不是真正同时更新 C 和 R 的联合训练。
- 微调后阻塞率从 `0.2760` 上升到 `0.2802`，退化 `0.42` 个百分点。
- `原始 C + 微调 R` 与 `原始 C + 原始 R` 完全相同，说明 R 端没有变化。
- 结论：小幅 reward shaping 或保守微调不值得继续；如果要提升，需要 C 端结构性改动。

## 10. 为什么不同实验的阻塞率差很多

不同表格的阻塞率不能直接横向比较，除非设置完全相同。

| 现象 | 原因 |
|---|---|
| S16 场景阻塞接近 0.40 | 频谱槽少，资源紧张，C 端无有效动作比例高 |
| S20/default3 场景阻塞约 0.18 | 资源更宽松，split 配置也不同 |
| S24/R60 R 端对比阻塞约 0.224 | 请求数、slots、固定 C checkpoint 与 S20/S16 不同 |
| metro24 接近零阻塞 | 拓扑更容易，短链路占优 |
| 早期 K=7 接近零阻塞 | 小样本诊断，后续 5-seed 正式实验已证明有种子偏差 |
| 0.215、0.2305、0.246、0.276 同时存在 | 脚本、checkpoint、slots、负载、episode 数和 split profile 都不同 |

正式比较规则：

```text
只有在同一脚本、同一拓扑、同一 slots/load、同一 split profile、同一 seeds、
同一请求数、同一 k_paths/max_blocks/block_sort、同一 C/R checkpoint family 下，
阻塞率才可以直接比较。
```

## 11. 最终方法排序

| 排名 | 方法 | 定位 |
|---:|---|---|
| 1 | 增强型 Agent-C + 独立训练 PPO-R | 主方法，核心贡献 |
| 2 | 增强型 Agent-C + BC-PPO-R | 教师迁移 / 等价性验证基线 |
| 3 | 增强型 Agent-C + DeepRMSA | 外部强 R 基线 |
| 4 | IWD-C + PPO-R 或 BC-PPO-R | 最强传统 C 基线 |
| 5 | DF-C + PPO-R 或 BC-PPO-R | 低时延但阻塞更高 |
| 6 | WO-C / RF-C / Greedy-C | 较弱传统 C 基线 |
| 7 | KSP-BF / SP-HM-FF | 较弱传统 R 后端 |
| 8 | Oracle-R / targeted DAgger / 激活函数 / 冻结 R 微调 | 诊断或负结果，不作为最终方法 |

## 12. 论文叙述建议

推荐中文表述：

```text
本文提出一种层次化学习框架：增强型 Agent-C 进行频谱感知的 DNN split 与
边缘服务器选择，独立训练的 PPO-R 执行低层路由、调制格式与频谱分配。
实验表明，独立 PPO-R 在不依赖教师蒸馏的情况下达到了 DeepRMSA 级别的
RMSA 执行能力；进一步的穷举 Oracle-R 分析显示，在固定 C 端决策下，
PPO-R 已达到 R 端阻塞下界。剩余阻塞主要来自 C 端选择了没有任何可行
R 动作的 split/server 组合，而不是 R 端动作选择错误。
```

贡献拆分：

- **C 端贡献：** 增强型 Agent-C 在困难 snap24 场景中稳定优于传统 C 启发式。
- **R 端贡献：** 独立训练 PPO-R 达到 DeepRMSA 级别，并显著优于 KSP-BF。
- **诊断结论：** Oracle-R 的 blocking gain 为 `0.00 pp`，说明当前 R 端 blocking 已到上限。
- **剩余瓶颈：** 需要 C 端结构性改进，而不是继续微调 R 端或改激活函数。

## 13. 当前止损与继续方向

建议停止：

- R 端 blocking 微调。
- R 端 targeted DAgger。
- 激活函数小规模扫参。
- 冻结 R 的 C 端 pressure shaping 小系数微调。
- frag-aware / future-feasibility / Oracle-R rerank 这类 R 端 reranking 方向。

如需继续提升，建议只考虑：

- C 端可行性建模，例如显式预测 `(split, server)` 是否会导致 `no_valid_r`。
- C 端结构改动，例如加入更强的图结构、链路占用传播信息或服务器-路径联合特征。
- 在最终论文表中补一行同脚本的 `增强型 Agent-C + 独立 PPO-R`，避免审稿人误以为主方法依赖 BC-PPO-R。

## 14. 最终一句话

```text
R 端已经解决 blocking 问题：独立 PPO-R 是主贡献，并达到 DeepRMSA 级别；
当前真正限制系统上限的是 C 端 split/server 可行性选择。
```
