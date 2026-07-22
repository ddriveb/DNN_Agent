# Experiments Index

这个目录里的文件很多，最重要的是先区分“主线证据”和“历史诊断”。

## 1. 论文主线 / 当前推荐引用

优先读这些：

| 文件 | 作用 |
|---|---|
| `main_s100_system_comparison.md` | 主系统基础对比表 |
| `main_s100_ksp_ff_k50_hops_strict_fixrerun.md` | bug 修复后的严格主表，当前最重要 |
| `main_s100_v12_k50_hops_diagnostic_fixrerun.md` | 修 bug 后的诊断对照 |
| `final_v12_ranker_recommendation.md` | v1.2 主线结论汇总 |
| `xlron_topology_fixrerun_summary.md` | 四个 XLRON 拓扑 fix rerun 汇总 |
| `xlron_topology_fairness_review_note.md` | XLRON 公平性审查说明 |

## 2. XLRON transfer 逐拓扑结果

| 文件 | 备注 |
|---|---|
| `xlron_cost239_s100_r_compare_fixrerun.md` | COST239 修 bug 后结果 |
| `xlron_german17_s100_r_compare_fixrerun.md` | German17 修 bug 后结果 |
| `xlron_nsfnet_deeprmsa_s100_r_compare_fixrerun.md` | NSFNET 修 bug 后结果 |
| `xlron_jpn48_s100_r_compare_fixrerun.md` | JPN48 修 bug 后结果 |

旧版未修 bug 的 `*_r_compare.md` 仍保留，但不应再作为主证据。

## 3. 历史 PPO-R 研究线

这些文件不是当前 v1.2 主线，而是回答“独立 PPO-R 是否已经足够强”：

| 文件 | 备注 |
|---|---|
| `current_experiment_record_complete.md` | 历史 PPO-R 主记录 |
| `final_experiment_report_cn.md` | 历史 PPO-R 中文总报告 |
| `deeprmsa_comparison_report.md` | PPO-R 与 DeepRMSA 的历史对照 |
| `oracle_ladder_v12_s100_report.md` | Oracle / ladder 诊断 |
| `r_bottleneck_hardpair_signal_diagnostic.md` | R 端瓶颈诊断 |

## 4. 诊断 / 消融 / 可行性分析

| 文件模式 | 含义 |
|---|---|
| `*_diagnostic*.md` | 诊断实验 |
| `*_findings.md` | 小结或筛选结论 |
| `*_summary.md` | 多实验汇总 |
| `r_ranker_v1_3_*` | v1.3 探索，未成为主线 |
| `lyapunov_*` | Lyapunov rerank 探索，未成为主线 |

## 5. 阅读顺序建议

如果你只想快速抓主线：

1. `CANONICAL_NAMES.md`
2. `main_s100_ksp_ff_k50_hops_strict_fixrerun.md`
3. `xlron_topology_fixrerun_summary.md`
4. `final_v12_ranker_recommendation.md`

如果你要追历史分支：

1. `current_experiment_record_complete.md`
2. `final_experiment_report_cn.md`
3. 再回头看具体诊断文件
