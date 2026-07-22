# SA-HMARL v1.3 后决策体系优化分析报告

> 目标：从“后决策体系本身”出发，系统分析 v1.3 post-decision ranker 在 COST239 上优势未能充分拉开的原因，并给出可优先落地的改进方案与实验计划。

---

## 0. 执行摘要

当前 v1.3 post-decision ranker 在 COST239（matched K=50, PPO-C）上把阻塞从 plain KSP-FF 的 **7.92%** 降到 **6.29%**，已经显著优于 KSP-FF 与 DeepRMSA-style proxy。但在启发式 C 端（`df_c`/`rf_c`）或固定 split 场景下，R-side 增益几乎消失。这说明：**ranker 有效，但天花板很低**。

通过代码审计、训练数据元信息、以及轻量级诊断，我们发现当前体系的最主要短板依次是：

1. **label 与实际失败模式不一致**：训练 label 的 H-step return 没有惩罚 future `server_overload`，而 COST239 上 100% 的阻塞正是 `server_overload`。
2. **特征缺少 post-allocation 信息**：ranker 只能看到当前频谱/服务器摘要，看不到“执行该动作后”的碎片化、瓶颈链路、服务器负载等长期信号。
3. **候选池过窄且依赖 PPO-R**：训练/在线都使用 `ppo_r_topk_only`（Top-30 PPO-R 提案），若 PPO-R 本身漏掉高价值动作，ranker 无补。
4. **H=5 太短**：未来 server overload 很难在 5 步内显现，导致 label 噪声大、区分度低。
5. **模型结构不是当前瓶颈**：per-candidate MLP 虽然简单，但在 label/特征/候选池修好之前，升级为 SetTransformer 收益不确定。
6. **C-side 可行域决定 R-side 上限**：当 C 端选得很差（df_c/rf_c/fixed split）时，R-side 即使换成 oracle 也难以弥补 server overload 主导的损失。

**最建议优先做的两个改动**：
- **改动 A（高优先级、低成本）**：在 label 中加入 future `server_overload` 惩罚，并把 H 从 5 延长到 12–20（或多 horizon label）。
- **改动 B（高优先级、中等成本）**：给 ranker 增加 post-allocation optical/server 特征（lfb_after、frag_after、free_block_delta、bottleneck margin、server_util_after 等）。

**不建议现在做的事**：直接换 SetTransformer/attention ranker、训练 all-legal 候选集、重新训练 PPO-C。这些要么收益不确定，要么超出“后决策体系本身”的优化范围。

---

## 一、当前 v1.3 后决策体系审计结论

### 1.1 Ranker 输入特征（25 维）

当前 checkpoint `r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt` 的 feature schema 如下（来自 `metadata.json`）：

| 分组 | 特征名 | 含义 |
|---|---|---|
| 基础动作特征 | `path_length_km`, `hop_count`, `lfb`, `free_ratio`, `frag_index`, `spectral_efficiency`, `reach_km`, `required_fs`, `block_size`, `block_waste`, `path_mod_feasible` | 候选 path/mod/block 的当前摘要 |
| C-side 上下文 | `split_norm`, `server_norm`, `deadline_norm`, `holding_norm`, `intermediate_size_norm`, `server_utilization`, `selected_valid_r_ratio` | 当前选定的 split/server、服务器利用率等 |
| 全局场特征 | `k_c_valid_ratio`, `k_r_total_ratio`, `phi_spec_norm`, `raw_r_valid_ratio` | 当前 C/R 可行域规模、 spectrum viability |
| 动作索引 | `path_idx_norm`, `mod_idx_norm`, `block_idx_norm` | 动作在候选列表中的位置编码 |

完整代码实现见：
- `sa_hmarl/sa_hmarl/evaluation/generate_r_post_decision_dataset.py`（`_r_feature_vector`，第 129–213 行）
- `sa_hmarl/sa_hmarl/evaluation/r_ranker_features.py`（在线特征封装，第 27–143 行）

### 1.2 是否包含真实 afterstate 特征？

**不包含**。`phi_after`（`_compute_phi_after_obs_c_after`）只用于 **label 计算**，不会进入 ranker 输入。ranker 输入里的 `lfb`/`free_ratio`/`frag_index` 是 **执行动作前** 的全局或 path-level 统计。

代码证据：`generate_r_counterfactual_ranking_dataset.py` 第 684–699 行：先算 `phi_after`，再把它加到 return 里；但 `_r_feature_vector` 没有用 `phi_after`。

### 1.3 是否包含 server overload risk / post-load？

**几乎没有**。只有 `server_utilization` 是 **当前** 利用率，没有：
- 执行动作后目标服务器的利用率；
- 队列延迟/EMA；
- 各服务器间负载差异；
- 未来 overload 风险指标。

### 1.4 是否包含路径共享边、瓶颈链路、post-allocation fragmentation、释放事件？

**不包含**。当前特征只有全局 `frag_index` 和 path-level `lfb`，没有：
- 动作占用后 path 上 largest free block 的变化；
- 动作占用后 free block count / fragmentation index 的变化；
- 瓶颈边（最满边）的剩余 margin；
- 动作与当前 active lightpaths 的共享边冲突程度；
- 资源释放时间重叠（release overlap）预测。

`use_structured_features=True` 可选地加入 `block_start/end/center` 和拓扑边 bitmap，但当前 checkpoint 未启用。

### 1.5 候选池如何构造？

训练数据集元信息（`metadata.json`）：

```json
{
  "candidate_mode": "ppo_r_topk_only",
  "max_candidates": 30,
  "ppo_top_k": 50,
  "ensure_ksp_action": false
}
```

即：**只取 PPO-R logit 最高的 50 个合法动作，再截断到 30 个**。在线策略（v13 fair comparison）因 `ranker_candidate_mode=null`，复用 checkpoint 默认值，也是 `ppo_r_topk_only`。实现见：
- `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py` `_select_candidate_actions_ppo_r_topk_only`（第 530–547 行）
- `sa_hmarl/sa_hmarl/agents/r_ranker_policy.py` `_select_online_candidates`（第 70–83 行）

### 1.6 标签如何生成？H-step rollout 的 continuation policy 是什么？

标签公式（`generate_r_counterfactual_ranking_dataset.py` `_compute_return`，第 344–361 行）：

```text
return = -3.0 * current_blocked
         -4.0 * future_blocked
         -3.0 * future_nsb
         -0.0 * future_server_overload   <-- 关键：系数为 0
         -0.03 * future_delay_mean
         -0.05 * future_avg_fs
```

- H = 5；
- continuation policy = frozen **PPO-C + PPO-R**（`future_rollout_policy=ppo_r`）；
- 对每个候选动作：deepcopy 环境 → 强制执行动作 → rollout 未来 5 个请求。

### 1.7 损失函数

`sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py`：

- **Listwise KL**（`_ranking_loss`，第 69–79 行）：softmax over returns vs softmax over scores；
- **回归损失**（`_regression_loss`，第 82–91 行）：Smooth-L1 到标准化 return；
- **Pairwise margin**（`_pairwise_ranking_loss`，第 94–114 行）：默认 `lambda_pair=0.5`；
- **Hard-negative 重加权**（`_hard_negative_weights`，第 116–141 行）：默认 `lambda_hard=1.0`。

总损失：

```text
L = L_listwise + 0.1 * L_reg + 0.5 * L_pairwise + 1.0 * L_hard
```

### 1.8 Ranker 结构与候选间关系建模

当前 checkpoint 使用 **per-candidate MLP**：

```python
CounterfactualActionValueRanker(input_dim=25, hidden_dims=(128, 64))
```

代码中虽然已实现了 `DeepSetCounterfactualRRanker` 和 `SetTransformerCounterfactualRRanker`（`counterfactual_r_ranker.py` 第 59–280 行），但当前 checkpoint 的 `model_type` 是 `mlp`，因此**没有建模候选间相对关系**。

---

## 二、为什么 v1.3 优势没有完全拉开：根因分析

### A. Label 问题（最核心）

#### A1. 当前 label 与实际阻塞模式错位

COST239 v13 K50 最终实验显示：

| 方法 | Blocking | Overload | NSB |
|---|---:|---:|---:|
| ppo_c+v13_k50_hops | 6.29% | 6.29% | 0.00% |
| ppo_c+ksp_ff_plain_k50_hops | 7.92% | 7.92% | 0.00% |

**所有阻塞都是 `server_overload`**，但训练 label 中 `return_future_server_overload_coef=0.0`。这意味着 ranker 学习的是“未来 5 步内是否发生 NSB/一般阻塞”，而完全不惩罚未来 server overload。label 与最终指标脱钩。

#### A2. H=5 太短

在 H=5 的 rollout 中，C-side PPO-C 仍有充足调整空间避开 overload，因此 `future_server_overload_count` 在训练集中几乎为 0。数据集诊断：

```json
"future_server_overload_positive_rate": 0.0,
"future_server_overload_variance": 0.0
```

过载是一种“慢累积”现象，H=5 很难捕捉到动作选择对 20–50 步后 overload 的影响。

#### A3. Label 区分度低

训练集诊断：

```json
"nonzero_return_range_rate": 0.61,
"mean_return_range": 0.084,
"median_return_range": 0.012,
"top1_top2_return_gap": 0.010
```

约 40% 的候选组在所有候选上得到几乎相同的 label；即使区分度非零，top1-top2 差距也非常小。模型很难从这种弱信号中学出稳定优势。

#### A4. 训练指标验证 label 质量差

Ranker 训练报告：

```text
Best val top-1: 16.62%
Test top-1: 25.54%
Test Spearman: 0.058
Test KL: 0.0355
```

一个 25 维 MLP 在 4000+ 组上 top-1 只有 25%，Spearman 接近 0，说明 teacher label 本身要么噪声大、要么与输入特征相关性弱。

### B. 特征问题

当前特征只能回答“这个动作现在看起来怎么样”，不能回答“执行这个动作后网络会变成什么样”。缺少的关键特征包括：

| 特征类 | 具体例子 | 为什么重要 |
|---|---|---|
| Post-allocation 频谱 | `lfb_after`, `frag_index_after`, `free_block_count_delta` | 判断动作是否造成碎片化 |
| 瓶颈链路 | `min_edge_lfb_margin`, `bottleneck_edge_util` | 共用边多的动作更容易在未来制造阻塞 |
| 路径冲突 | `path_conflict_count`, `active_lightpaths_sharing_edge` | 高冲突动作长期风险大 |
| 释放事件 | `expected_release_overlap`, `time_to_next_release_on_path` | 资源何时回流 |
| 服务器 post-load | `server_util_after`, `queue_delay_after`, `server_margin_after` | COST239 是 overload 主导 |

这些特征都可以通过一次“候选动作 trial step + 环境状态快照”在线计算，不需要 rollout，因此**在线推理开销增加很小**。

### C. 候选池问题

#### C1. 训练/在线候选池过窄

`ppo_r_topk_only` 把候选池限制在 PPO-R 的前 30 个提案。若 PPO-R 没有提出某个长期更优动作，ranker 永远看不到它。

#### C2. 缺少 optical-aware 与 server-aware anchors

当前候选构造没有显式加入：
- 最小化 post-fragmentation 的动作；
- 保留最大空闲块的动作；
- 最短可行路径动作（虽然 heuristic 里已有 shortest，但不是基于 post-state）；
- 对目标服务器负载最友好的动作。

#### C3. 训练数据集 oracle headroom 极小

训练集：

```json
"oracle_headroom_pp": 0.00156
```

即便在训练候选池内，oracle 相对 PPO 的每步阻塞优势只有 **0.00156 个百分点**。这有两个解释：
- PPO-R 的 top-30 提案本身已经很好；
- label 没有 capture 真正的长期差异（特别是 overload）。

结合 val/test top-1 很低，我们更倾向后者：teacher label 没有区分出真正高价值的动作。

### D. 模型结构问题

per-candidate MLP 有两个局限：
1. 每个候选独立打分，没有显式比较同一状态下其他候选；
2. 对输入特征的绝对数值敏感，训练/在线候选数量不一致时稳定性差。

但代码审计显示，**模型不是当前最大短板**：即使换成 SetTransformer，如果 label 还是 H=5 且不惩罚 overload，模型仍然学不到真正的长期价值。

### E. 训练目标问题

当前 loss 已经混合了 listwise + regression + pairwise + hard-negative，结构合理。但存在两个问题：
1. **hard-negative 权重基于 teacher return gap**：如果 teacher label 本身没有 capture overload，hard-negative 只是强化错误信号；
2. **回归目标使用标准化 return**：return 的绝对数值小（median range 0.012），regression loss 可能主导训练，导致模型过度拟合绝对值而非排序。

### F. C-R 耦合问题

启发式 C 端结果（`FINAL_HEURISTIC_C_UNIFIED_REPORT.md`）：

| C 端 | COST239 blocking | v1.3 vs KSP-FF Δ |
|---|---:|---:|
| PPO-C | 7.68% | -1.63 pp |
| df_c | 14.84% | 0.00 pp |
| rf_c | 15.17% | 0.00 pp |
| fixed split | ~74% | ~0 pp |

这说明：
- 当 C 端把 split/server 选得好时，R-side 才有优化空间；
- 当 C 端已经让服务器长期处于高负载，任何 R-side 动作都难逃 overload；
- 固定 split 直接把阻塞拉到 70%+，R-side 完全失效。

**结论**：v1.3 的 R-side 增益是“conditional”的，必须同时说明 C-side 质量。

---

## 三、优化方案：按优先级排序

| 优先级 | 方案 | 实现成本 | 预期收益 | 风险 | 推荐度 |
|---|---|---|---|---|---|
| **P0** | **A. 修正 label：加入 future server_overload 惩罚 + 延长 H** | 低 | 高 | 低 | ★★★ |
| **P0** | **B. 增加 post-allocation optical/server 特征** | 中 | 高 | 低 | ★★★ |
| **P1** | **C. 候选池加入 optical/server-aware anchors + 保证 KSP** | 中 | 中 | 低 | ★★☆ |
| **P1** | **D. Multi-horizon label + regret-weighted hard-negative** | 低 | 中 | 低 | ★★☆ |
| **P2** | **E. 训练/在线候选一致性 + candidate recall 诊断** | 低 | 中 | 低 | ★★☆ |
| **P2** | **F. 候选集上下文模型（DeepSets/SetTransformer）** | 中 | 中-低 | 中 | ★☆☆ |
| **P3** | G. 图/路径感知的 encoder | 高 | 不确定 | 高 | ☆☆☆ |
| **不建议** | H. all_legal 候选训练 | 高 | 低 | 高 | ☆☆☆ |
| **不建议** | I. 重新训练 PPO-C | 高 | 超出范围 | - | ☆☆☆ |

### 方案 A：修正 label / 延长 horizon

**具体改动**：
1. `generate_r_counterfactual_ranking_dataset.py` 第 1404 行：把 `--return_future_server_overload_coef` 从 `0.0` 改为 `4.0–8.0`（与 `future_block` 同量级）。
2. 同文件第 1392 行：`--horizon` 从 5 改为 12–20。
3. 可选：在 `_compute_return` 中加入 multi-horizon：
   ```text
   Q = 0.5 * Q_H5 + 0.5 * Q_H20
   ```
   这样短期信号不丢失，长期 overload 也能被捕捉。

**为什么重要**：直接对齐 COST239 上 100% server_overload 的失败模式。

**测试方式**：
- 重新生成数据集，观察 `future_server_overload_positive_rate` 是否显著 > 0；
- 观察 `oracle_headroom_pp`、`mean_return_range`、`top1_top2_return_gap` 是否提升；
- 重新训练 ranker，看 val/test top-1 与 Spearman 是否提升。

### 方案 B：增加 post-allocation optical/server 特征

**新增特征建议**：

| 特征名 | 计算方式 | 涉及文件 |
|---|---|---|
| `lfb_after` / `lfb_ratio_after` | trial step 后 path/global 最大空闲块 | `r_ranker_features.py`, `generate_r_post_decision_dataset.py` |
| `frag_index_after` | trial step 后 path/global 碎片化指数 | 同上 |
| `free_block_count_delta` | 执行前后 free block count 差 | 同上 |
| `min_edge_lfb_margin` | path 上最紧边的剩余 slot 数 / num_slots | 同上 |
| `bottleneck_edge_util` | path 上最满边利用率 | 同上 |
| `path_conflict_count` | 与当前 active lightpaths 共享的边数 | 同上 |
| `server_util_after` | 选定服务器 trial step 后的利用率 | 同上 |
| `server_queue_delay_after` | trial step 后目标服务器队列延迟 EMA | 同上 |
| `expected_release_overlap` | path 上近期释放事件数量 / 重叠度 | 同上 |

**实现位置**：
- 离线：`generate_r_post_decision_dataset.py` 的 `_r_feature_vector`；
- 在线：`r_ranker_features.py` 的 `_build_default_feature_batch` / `build_r_ranker_feature`；
- 需要在 trial step 后调用 `env.net.get_path_spectrum_stats(path)`、`env.net.get_global_spectrum_stats()`、`env.mec.servers[server_id].utilization`。

**测试方式**：
- 在 `diagnose_v13_ranker_label_quality.py` 中加入这些特征，训练前先用随机森林 / XGBoost 估计它们与 H20 return 的 feature importance；
- 重新训练同结构 MLP，对比 val top-1 / Spearman；
- 在线评估 blocking / overload / decision latency。

### 方案 C：改进候选池 recall

**具体改动**：
1. 在 `r_ranker_policy.py` / `generate_r_counterfactual_ranking_dataset.py` 的候选构造中加入以下 anchors：
   - `lowest_post_fragmentation`：trial 后 `frag_index_after` 最小的动作；
   - `largest_free_block_preserving`：trial 后 global/path `lfb` 最大的动作；
   - `server_safe_compatible`：trial 后目标服务器利用率增量最小的动作；
   - `shortest_feasible_path`：已存在；
   - `lowest_required_fs`：已存在（隐含在 best-fit）。
2. 始终保证 KSP-FF（plain + highest-mod）在候选池中：`ensure_ksp_action=True`。

**测试方式**：
- 统计 oracle top1/top5 recall 在训练候选池 vs 在线候选池中的比例；
- 对比改进前后阻塞率。

### 方案 D：Multi-horizon label + regret-weighted hard-negative

**具体改动**：
1. 在 `train_r_counterfactual_ranking.py` 中把 hard-negative 权重改为基于 **regret gap**：
   ```python
   weight = 1 + alpha * max(0, teacher_best_return - model_top1_return)
   ```
   当前实现已经是这样（`_hard_negative_weights`），但只在 listwise 上重加权。可以额外对 pairwise loss 重加权。
2. Multi-horizon：生成 `returns_h5` 和 `returns_h20`，训练时同时预测两个 horizon，用共享 MLP + 两个 head。

**测试方式**：
- 训练时分别看 H5 与 H20 的 val top-1；
- 在线评估时选择 H20 head 的分数。

### 方案 E：训练/在线候选一致性 + recall 诊断

**问题**：当前 heuristic-C 统一协议使用 `legalctx48+ensure_ksp`，而 v13 fair comparison 因 `ranker_candidate_mode=null` 使用 checkpoint 默认值 `ppo_r_topk_only`。两者不一致会导致论文中不同表格的 v1.3 实际上使用不同候选池。

**改动**：
1. 统一使用 `legalctx48+ensure_ksp`（与 heuristic-C 报告一致），并重新训练 ranker；
2. 或者，在 v13 fair comparison 中显式设置 `ranker_candidate_mode=ppo_r_topk_only` 以保持与训练一致。

**建议**：选择前者，因为 `legalctx48` 候选池更丰富，且包含 KSP anchor。

### 方案 F：候选集上下文模型（DeepSets/SetTransformer）

**改动**：
- `train_r_counterfactual_ranking.py` 第 575 行 `model_type` 选 `deepset` 或 `set_transformer`；
- 注意 `context_scale=0.0` 初始化可保护已有 MLP 能力；
- 在线 `score_legal_actions` 需要一次前向传整个候选集（当前已实现）。

**评估条件**：只有在方案 A/B 收益见顶、且 label 区分度足够时才做这个，否则模型复杂度只是拟合噪声。

### 方案 G/H/I（不建议现在做）

- **G. 图/路径 encoder**：需要把 spectrum 矩阵或链路状态作为输入，工程量大，且当前 25 维摘要远未用满。
- **H. all_legal 候选训练**：COST239 K=50 时合法动作可达数千，H-step rollout 成本极高；且 label 噪声会进一步放大。
- **I. 重新训练 PPO-C**：超出“后决策体系”范围，应作为独立工作。

---

## 四、各方案涉及代码文件

| 方案 | 主要修改文件 | 关键函数/行 |
|---|---|---|
| A | `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py` | `_compute_return` (344–361), parser `--horizon` (1392), `--return_future_server_overload_coef` (1406) |
| A | `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py` | 若 multi-horizon，需改 `_evaluate`、数据加载、loss |
| B | `sa_hmarl/sa_hmarl/evaluation/generate_r_post_decision_dataset.py` | `_r_feature_vector` (129–213), `FEATURE_NAMES` (68–85) |
| B | `sa_hmarl/sa_hmarl/evaluation/r_ranker_features.py` | `_build_default_feature_batch` (45–111), `build_r_ranker_feature_batch` (114–143) |
| C | `sa_hmarl/sa_hmarl/agents/r_ranker_policy.py` | `_build_core_candidates` (85–160), `_ensure_ksp_action` (162–182) |
| C | `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py` | `_select_candidate_actions_v1` (364–439), `_ensure_ksp_action_in_candidates` (550–565) |
| D | `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py` | `_hard_negative_weights` (116–141), `_pairwise_ranking_loss` (94–114) |
| E | `sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py` | ranker_policy 构造处 (597–603) |
| F | `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py` | `build_counterfactual_r_ranker` 调用 (375–377) |
| F | `sa_hmarl/sa_hmarl/agents/counterfactual_r_ranker.py` | `DeepSetCounterfactualRRanker` / `SetTransformerCounterfactualRRanker` (59–280) |

---

## 五、最小可行实验计划（Stage 0–4）

### Stage 0：诊断-only（已完成/进行中）

目标：不改模型，只量化问题。

已产出/可产出指标：
- [x] 训练集 label 范围、nonzero range rate、top1-top2 gap；
- [x] H=5 label 与 H=20/H=50 的 per-group Spearman；
- [x] oracle headroom（训练集 0.00156 pp / step）；
- [x] KSP / PPO / ranker 的 regret 分布；
- [x] 按 server utilization 分桶的 R-side headroom。

交付物：
- `sa_hmarl/experiments/v13_ranker_label_quality_diagnostic.json`
- 本报告 Section 二 的量化表格

### Stage 1：Feature-only upgrade

- 在 `_r_feature_vector` / `_build_default_feature_batch` 中加入 post-allocation optical/server 特征（约 8–12 维）。
- **不改** label、horizon、loss、model structure。
- 用当前 `ppo_r_topk_only` 候选池重新生成数据集并训练同结构 MLP。

输出指标：
| 指标 | 当前 v1.3 | Stage 1 |
|---|---|---|
| Val top-1 | 16.6% | ? |
| Test Spearman | 0.058 | ? |
| Test model regret | 0.028 | ? |
| COST239 blocking | 6.29% | ? |
| Overload | 6.29% | ? |
| Decision latency | ~12 ms | ? |

成功标准：Spearman > 0.15 或 val top-1 > 25%，或 blocking 下降 > 0.5 pp。

### Stage 2：Label + loss upgrade

- H 从 5 延长到 12–20；
- `return_future_server_overload_coef` 设为 4.0–8.0；
- 可选 multi-horizon（H5 + H20）；
- 使用 Stage 1 的特征，同结构 MLP。

输出指标同上，重点看：
- `future_server_overload_positive_rate` 是否 > 0；
- `oracle_headroom_pp` 是否显著提升；
- 在线 `server_overload_rate` 是否下降。

成功标准：blocking 相对 Stage 1 再降 > 0.5 pp，或 overload 成为可观测的 label 信号。

### Stage 3：Candidate recall upgrade

- 候选池从 `ppo_r_topk_only` 改为 `legalctx48+ensure_ksp`；
- 加入 post-frag / lfb-preserving / server-safe anchors；
- 保持 Stage 2 的 label 与特征。

输出指标：
- oracle top1/top5 recall；
- KSP inclusion rate；
- blocking / overload / NSB；
- decision latency（候选数 48，需确认 < 20 ms）。

成功标准：oracle recall > 90%，blocking 不劣于 Stage 2。

### Stage 4：Attention/Set ranker

- 仅当前三步收益见顶时执行；
- 比较 `mlp` vs `deepset` vs `set_transformer`；
- 控制特征/label/候选池不变。

输出指标：
- val top-1、Spearman、pairwise accuracy；
- decision latency（SetTransformer 可能比 MLP 慢 2–3 倍，需确认可接受）。

停止条件：若 Stage 1+2+3 已经把 COST239 blocking 降到 < 5.5%，Stage 4 收益可能很小，可放 future work。

---

## 六、论文叙事建议

### 6.1 当前论文可写的内容

1. **主结果**：v1.3 K50 vs KSP-FF K50 公平比较（已有 `v13_k50_final_fair_comparison_cost239.md`）。
2. **失败模式分析**：COST239 是 server_overload 主导，解释为什么 R-side 增益有上限。
3. **启发式 C 端消融**（`FINAL_HEURISTIC_C_UNIFIED_REPORT.md`）：说明 R-side 增益对 C-side 质量敏感。
4. **诊断性讨论**：指出当前 label 未惩罚 server_overload、H=5 较短，这是 v1.3 未能进一步拉开差距的根本原因。

### 6.2 建议的论文段落

> Under matched K=50 hop-ordered path support, the v1.3 planner-distilled ranker reduces blocking from 7.92% (plain KSP-FF) to 6.29% on COST239. However, the gain is smaller under heuristic C-side policies and disappears when the split is fixed. We trace this ceiling to a mismatch between the ranker's training objective and the actual failure regime: COST239 blocking is entirely `server_overload`, whereas the H=5 counterfactual return used to train the ranker does not penalize future server overload. Consequently, the distilled scorer learns to avoid short-term spectrum blocking rather than to manage long-term compute load. A staged upgrade—adding post-allocation server and optical features, extending the planning horizon, and re-weighting labels toward overload—directly addresses this mismatch and is the focus of the next optimization round.

### 6.3 哪些放 future work

- SetTransformer/attention ranker architecture；
- 图/路径级 encoder；
- all-legal 候选训练；
- 跨拓扑（German17/NSFNET/JPN48）上验证 overload-aware label 的泛化性。

---

## 七、最终结论

### 7.1 v1.3 当前最可能的短板排序

1. **Label 与实际阻塞模式不一致**（未惩罚 server_overload，H 太短）—— 最核心；
2. **特征缺少 post-allocation 信息** —— 次核心；
3. **候选池依赖 PPO-R 且缺少 optical/server anchors** —— 第三；
4. **C-side 可行域瓶颈** —— 决定 R-side 是否有空间，但不是 ranker 本身的问题；
5. **模型结构** —— 当前不是主要瓶颈；
6. **Loss 设计** —— 结构合理，但受 label 质量拖累。

### 7.2 最值得优先做的两个改动

1. **修正 label**：加入 future `server_overload` 惩罚，延长 horizon 到 12–20，必要时用 multi-horizon label。
2. **增加 post-allocation 特征**：尤其是 `lfb_after`、`frag_index_after`、`min_edge_lfb_margin`、`server_util_after`、`server_queue_delay_after`。

### 7.3 不建议现在做的事

- 直接换 SetTransformer/attention（收益不确定，先修 label/feature）；
- all_legal 候选训练（成本过高且 label 噪声会放大）；
- 重新训练 PPO-C（超出后决策体系优化范围）；
- 在 label 修好之前做大量超参搜索。

---

## 附录 A：关键数据速查

### A.1 v1.3 K50 公平比较（COST239）

| 方法 | Blocking | Overload | NSB |
|---|---:|---:|---:|
| ppo_c+v13_k50_hops | 6.29% | 6.29% | 0.00% |
| ppo_c+ksp_ff_plain_k50_hops | 7.92% | 7.92% | 0.00% |
| ppo_c+ksp_ff_highest_mod_k50_hops | 9.93% | 9.93% | 0.00% |
| ppo_c+deep_rmsa_style_k50_hops | 9.62% | 9.62% | 0.00% |

### A.2 训练数据集诊断（`r_counterfactual_ranking_postdec_topk_k50_s100`）

| Split | Groups | Avg cand | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future overload+ |
|---|---:|---:|---:|---:|---:|---:|
| train | 4612 | 17.6 | 61.1% | 53.8% | 0.00156 | 0.0% |
| val | 385 | 20.3 | 70.1% | 47.8% | 0.00052 | 0.0% |
| test | 368 | 14.3 | 43.5% | 65.8% | 0.00598 | 0.0% |

### A.3 Ranker 训练指标

| Metric | Value |
|---|---:|
| Best val top-1 | 16.62% |
| Test top-1 | 25.54% |
| Test Spearman | 0.058 |
| Test model regret | 0.0281 |
| Test PPO regret | 0.1391 |

### A.4 启发式 C 端结果（COST239）

| C 端 | v1.3 blocking | KSP-FF blocking | Δ |
|---|---:|---:|---:|
| PPO-C | 7.68% | 7.92% | -1.63 pp |
| df_c | 14.84% | 14.84% | 0.00 pp |
| rf_c | 15.17% | 15.17% | 0.00 pp |

（启发式 C 表格来自 `FINAL_HEURISTIC_C_UNIFIED_REPORT.md`，使用 `legalctx48+ensure_ksp` 配置。）

---

*报告生成时间：2026-07-12*
*相关脚本：*
- `sa_hmarl/sa_hmarl/evaluation/diagnose_v13_ranker_label_quality.py`
- `sa_hmarl/sa_hmarl/evaluation/diagnose_v13_postdecision_root_causes.py`
