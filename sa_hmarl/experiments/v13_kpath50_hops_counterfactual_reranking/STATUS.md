# SA-HMARL v1.3 诊断流水线状态（修正版）

> **⚠️ 本文件已更新，替代 2026-07-14 版本。**  
> 旧版 `FINAL_V13_DECISION.md` 与 `MEDIUM_TRAINING_REPORT.md`（3 seeds × 4 规模）仍保留，但已标注“过期”。请以此修正版为准。

---

## 已完成

| 任务 | 输出 | 状态 |
|---|---|---|
| 生成器扩展：保存原始标签分量、PPO logits、PPO rank、trace hash | `generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py` | ✅ |
| 回归测试：新旧输出 action IDs/features/returns 一致、标签可重建 | `LABEL_COMPONENT_AUDIT.md/.json` | ✅ PASS |
| 中等规模数据生成：10 train / 3 val / 3 test shards，1200 req/shard | `MEDIUM_DATASET_REPORT.md` | ✅ |
| 标签可学习性分析 | `MEDIUM_LABEL_LEARNABILITY.md/.json` | ✅ |
| 指标修正：`aggregate_regret_reduction` 替代 `relative_regret_reduction` | `METRIC_CORRECTION.md/.json` | ✅ |
| 指标单元测试 | `tests/test_regret_metrics.py` | ✅ 4/4 PASS |
| 3 seeds × full 7.3k groups regret-selected 重训练 | `MEDIUM_TRAINING_REPORT_CORRECTED.md/.json` | ✅ |
| 5-seed 闭环公平评估 | `MEDIUM_CLOSED_LOOP.md/.json` | ✅ |
| 修正版最终决策 | `FINAL_V13_DECISION_CORRECTED.md/.json` | ✅ |

---

## 关键数字

### 离线（full medium，seed 42）

- **Train groups:** 7303
- **Val groups:** 1750
- **Test groups:** 2079
- **Test model regret:** 0.0682
- **Test PPO regret:** 0.1050
- **Absolute regret improvement:** 0.0368
- **Aggregate regret reduction:** **35.04%**
- **Model better than PPO rate:** 26.34%
- **Oracle tie hit rate:** 46.22%

### 闭环（5 seeds × 6000 requests）

| Mode | Mean blocking | Std |
|---|---:|---:|
| ppo_r_top1 | 6.82% | 1.88% |
| ksp_ff_plain | 7.94% | 1.47% |
| ksp_ff_highest | 7.04% | 0.94% |
| **ranker_old_v13** | **6.01%** | **0.75%** |
| ranker_new_v13 | 6.37% | 1.47% |

- vs PPO-R: mean diff −0.45 pp，bootstrap 95% CI [0.0010, 0.0079]（PPO − new，全为正），Wilcoxon p=0.1875
- vs KSP-FF plain: mean diff −1.57 pp，CI 过 0
- vs KSP-FF highest: mean diff −0.67 pp，CI 过 0
- vs ranker_old_v13: mean diff +0.36 pp，Wilcoxon p=0.625（新 ranker 未超过旧 ranker）

---

## 分类与决定

- **分类：B**
- **v1.35 full launch：不允许**
- **v1.35 diagnostic pilot：允许**（≤ 1,000 groups，验证 afterstate 可预测性）

---

## 未执行 / 后续工作

- **真实 same-state / multi-trace 稳定性研究**：尚未完成；旧的 `donor_group_index_aligned_proxy` 已废弃。
- **标签消融**：H=0/1/2/5、blocking-only、future-only、full 已设计，未运行；属于理解性工作，不是 pilot 阻塞条件。
- **v1.35 pilot**：待设计 afterstate 特征后启动。
