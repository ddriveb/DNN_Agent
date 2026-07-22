# Dataset Index

这个目录既有主线数据集，也有很多历史探索数据集。目录名通常是生成配置摘要，
不是推荐对外方法名。

统一方法名见：

`sa_hmarl/experiments/CANONICAL_NAMES.md`

## 1. 当前主线相关

| 路径 | 含义 |
|---|---|
| `r_counterfactual_ranking_v1_2_mixed_low/` | v1.2 主线训练数据集 |
| `r_counterfactual_ranking_s100_v1_2_mixed_low/` | S100 相关主线数据集 |

## 2. 主线变体 / 消融

| 路径模式 | 含义 |
|---|---|
| `r_counterfactual_ranking_s100_v1_2_legalctx*` | legal-context 特征变体 |
| `r_counterfactual_ranking_s100_v1_2_deepset*` | DeepSet 架构变体 |
| `r_counterfactual_ranking_s100_v1_2_settransformer*` | SetTransformer 架构变体 |
| `r_counterfactual_ranking_s100_v1_2_dagger1*` | DAgger 变体 |
| `r_counterfactual_ranking_s100_v1_2_k50_hops*` | K=50 hops 诊断变体 |

## 3. XLRON 重训

| 路径 | 含义 |
|---|---|
| `r_counterfactual_ranking_xlron_cost239_v1_2_retrain/` | COST239 第一次重训数据集 |
| `r_counterfactual_ranking_xlron_cost239_v1_2_retrain_v2/` | COST239 第二次重训数据集 |

## 4. 其他

| 路径 | 含义 |
|---|---|
| `c_post_decision/` | C 端 post-decision 数据 |
| `test_structured_legalctx48/` | 小规模测试数据 |

## 5. 使用规范

- 目录名视为数据集 id，不视为方法名
- 引用数据集时最好同时给：
  1. 对应方法规范名
  2. 数据集路径
  3. 生成报告 `generation_report.md`
