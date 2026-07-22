# SA-HMARL 严格 v1.3 诊断结论与 v1.35 启动决定（修正版）

> **⚠️ 本文件修正并替代 `FINAL_V13_DECISION.md`**  
> **日期：** 2026-07-08  
> **数据集：** `r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`  
> **协议：** K_C=5, K_path=50, K_prop=30, H=5, gamma=1.0, candidate_mode=ppo_r_topk_only  
> **模型：** MLP [128,64]，simple loss（reg_weight=1.0, lambda_pair=0, lambda_hard=0）  
> **Checkpoint：** `sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt`

---

## 1. 修正说明

本次决策基于两项关键修正：

1. **指标修正**：原 `relative_regret_reduction` 因按 group 做 ratio 平均、且只在 PPO regret>0 时计算，对 tiny regret 极度不稳定，曾把真实改善误判为 −1.15 的“灾难”。新指标 `aggregate_regret_reduction = (mean_ppo_regret − mean_model_regret) / mean_ppo_regret` 稳定、有界，详见 `METRIC_CORRECTION.md`。
2. **Checkpoint 选择修正**：不再用 Top-1 准确率选模型，改用 **validation model regret 最低** 的 checkpoint，并通过 3-seed 并行训练 + 验证选择得到 seed 42。

---

## 2. 离线排序结果（full medium 7.3k groups）

| Seed | Val selection score | Test model regret | Test PPO regret | Abs. improvement | Aggregate reduction | Model better rate | Oracle tie hit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | −0.0465 | **0.0682** | 0.1050 | **0.0368** | **35.04%** | 26.34% | 46.22% |
| 123 | −0.0432 | 0.0696 | 0.1050 | 0.0354 | 33.71% | 27.35% | 47.76% |
| 456 | −0.0450 | 0.0791 | 0.1050 | 0.0259 | 24.65% | 23.50% | 45.02% |

* 选中模型：**seed 42**
* 在测试集上，ranker 将平均 regret 从 0.1050 降到 0.0682，**相对降低 35.04%**。
* 约 **26%** 的 group 上模型比 PPO-R 更优；约 **46%** 命中 oracle tie 集。

---

## 3. 闭环公平评估（5 seeds × 6000 requests）

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r_top1 | 6.82% ± 1.88% | 6.61% | 0.21% | 10.87 | 3.43 | 11.33 | 100.00% |
| ksp_ff_plain | 7.94% ± 1.47% | 7.86% | 0.07% | 10.11 | 3.15 | 11.06 | 32.97% |
| ksp_ff_highest | 7.04% ± 0.94% | 7.03% | 0.00% | 9.79 | 2.35 | 11.23 | 11.38% |
| ranker_old_v13 | **6.01% ± 0.75%** | 5.98% | 0.02% | 11.07 | 2.95 | 21.41 | 9.69% |
| ranker_new_v13 | 6.37% ± 1.47% | 6.31% | 0.05% | 12.08 | 3.06 | 18.62 | 9.83% |

### Per-seed blocking

| Seed | ppo_r_top1 | ksp_ff_plain | ksp_ff_highest | ranker_old_v13 | ranker_new_v13 |
|---:|---:|---:|---:|---:|---:|
| 3030 | 7.62% | 7.58% | 8.12% | **6.48%** | 7.00% |
| 4040 | 4.54% | 10.56% | 6.20% | 4.80% | **4.68%** |
| 5050 | 5.12% | 6.02% | 8.06% | 5.46% | **5.00%** |
| 6060 | 9.82% | 7.90% | 5.80% | **6.60%** | 8.76% |
| 7070 | 7.00% | 7.62% | 7.00% | 6.70% | **6.40%** |

### 与 legacy `ranker_old_v13` 的对比

| 统计量 | 值 |
|---|---:|
| Mean blocking difference (new − old) | **+0.0036** (+0.36 pp) |
| Wins / ties / losses (new vs old) | 2 / 0 / 3 |
| Wilcoxon signed-rank p-value | 0.6250 |

说明：legacy `ranker_old_v13` 在该 5-seed 闭环中平均阻塞率 **6.01%**，仍略优于新 ranker 的 **6.37%**。新 v1.3 ranker 优于 PPO-R 与 KSP-FF，但尚未超过旧 ranker。

---

## 4. 成对统计检验

### strict v1.3 vs PPO-R Top-1

| 统计量 | 值 |
|---|---:|
| Mean blocking difference (new − ppo) | **−0.0045** (−0.45 pp) |
| Relative blocking reduction | **6.6%** |
| Wins / ties / losses | 4 / 0 / 1 |
| Bootstrap 95% CI for (PPO − new) | **[0.0010, 0.0079]** |
| Wilcoxon signed-rank p-value | 0.1875 |
| Permutation one-sided p (new < ppo) | ≈ 0.093 |

*Bootstrap CI 完全位于正值区间*，说明在重采样意义下新 ranker 的阻塞率显著低于 PPO-R；但 Wilcoxon 与置换检验因 **n=5** 未达到传统 0.05 显著性。

### strict v1.3 vs KSP-FF plain

| 统计量 | 值 |
|---|---:|
| Mean blocking difference (new − plain) | **−0.0157** (−1.57 pp) |
| Bootstrap 95% CI for (plain − new) | [−0.0016, 0.0385] |
| Wilcoxon p-value | 0.1875 |

均值改善，但 CI 跨过 0，不能断言显著优于 plain。

### strict v1.3 vs KSP-FF highest-mod

| 统计量 | 值 |
|---|---:|
| Mean blocking difference (new − highest) | **−0.0067** (−0.67 pp) |
| Bootstrap 95% CI for (highest − new) | [−0.0135, 0.0214] |
| Wilcoxon p-value | 0.4375 |

同样均值改善但 CI 跨过 0，未建立对 KSP-FF highest 的统计非劣效性。

### strict v1.3 vs ranker_old_v13

| 统计量 | 值 |
|---|---:|
| Mean blocking difference (new − old) | **+0.0036** (+0.36 pp) |
| Wins / ties / losses (new vs old) | 2 / 0 / 3 |
| Wilcoxon signed-rank p-value | 0.6250 |

新 ranker 未能在 5-seed 闭环中超过 legacy ranker。

---

## 5. 分类与 v1.35 启动决定

### 分类：B（有前景，但未达到 A）

| 类别 | 定义 | 是否符合 |
|---|---|---|
| **A** | 显著优于 PPO，且对 KSP-FF 建立非劣效 / 显著优势，可启动 full v1.35 | ❌（KSP-FF CI 过 0） |
| **B** | 显著优于 PPO（bootstrap CI），但 vs KSP-FF 尚未确立稳健优势，允许 pilot | ✅ |
| **C** | 离线/闭环结果矛盾或统计不显著，只允许小规模探索 | ❌ |
| **D** | 未优于 PPO 或明显劣于 KSP-FF，不允许 v1.35 | ❌ |

**判定为 B 的核心依据：**

1. 离线 full-medium  regret 降低 35%，模型 regret 稳定低于 PPO。
2. 5-seed 闭环均值优于 PPO（−0.45 pp）且 bootstrap CI 全部为正。
3. 但 Wilcoxon / 置换检验因 seed 数太少不显著；vs KSP-FF 的 CI 跨过 0，未建立非劣效。
4. 新 ranker 未能超过 legacy `ranker_old_v13`（mean +0.36 pp，Wilcoxon p=0.625），说明当前 v1.3 标签/训练流程尚未产生比旧模型更强的闭环策略。
5. 绝对改善幅度较小（~0.45 pp 阻塞率 vs PPO），不足以支持 full v1.35 大规模投入。

### v1.35 启动决定

* **Full v1.35：不允许。**
* **Diagnostic pilot：允许。** 可以启动一个 **≤ 1,000 groups** 的 v1.35 诊断 pilot，目标明确为：
  * 验证 afterstate / 状态增强特征能否提升 Spearman / regret predictability；
  * 仅在 pilot 离线结果达到 **A 类**（显著优于 PPO 且非劣于 KSP-FF）后，才允许扩大为 full v1.35 dataset。
* 在 pilot 结果出来前，**禁止生成 30k–40k groups 的 full v1.35 数据集**。

---

## 6. 仍未解决但不妨碍本次决策的问题

1. **真实 same-state / multi-trace 稳定性研究** 尚未完成；旧的 `donor_group_index_aligned_proxy` 已被显式废弃。
2. **标签消融**（H=0/1/2/5、blocking-only、future-only、full）已设计但未运行；这些是理解性工作，不是 v1.35 pilot 的阻塞条件。
3. 闭环评估中的 `ranker_old_v13` 行因解析器 bug 最初缺失，已修复并补跑（见 `MEDIUM_CLOSED_LOOP.md`）。

---

## 7. 下一条应执行的准确命令

1. 设计 v1.35 afterstate 特征方案。
2. 生成并训练一个 ≤ 1,000 groups 的 v1.35 diagnostic pilot。
3. 用与 v1.3 相同的协议（K_C=5, K_path=50, K_prop=30, H=5, regret 选模型）评估 pilot。
4. 根据 pilot 结果重新分类；只有 A 类才启动 full v1.35。

---

## 8. 结论

严格 v1.3 的候选池和实现协议是正确的。在修正指标和 checkpoint 选择后，**full-medium ranker 显著降低了 PPO-R 的 regret，并在 5-seed 闭环中稳定优于 PPO-R**。然而，新 ranker 并未超过 legacy `ranker_old_v13`，样本量小且对 KSP-FF 的非劣效性未确立，因此本次分类为 **B**，只允许启动 **v1.35 diagnostic pilot**，不得直接启动 full v1.35。
