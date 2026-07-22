# Canonical Names

这份文档只做一件事：统一项目里最容易混淆的名字。

原则：

- 对外写作、汇报、论文、PPT，一律优先用“规范名”
- 旧脚本名、旧 checkpoint 名、旧实验文件名可以保留，但要被视为
  artifact id，而不是方法名
- 同一个对象不要同时用“方法名 + 文件夹名 + 临时备注名”三套叫法

## 1. 主线分支

| 分支 | 规范名 | 用途 | 不要再单独写 |
|---|---|---|---|
| 当前系统主线 | `PPO-C + v1.2 static counterfactual ranker` | 主实验、fix rerun、XLRON transfer、公平性审查 | `final v1.2`, `rank_only`, `mixed_low`, `counterfactual_rank_only` |
| 历史 PPO-R 研究线 | `Typed-Mean-Field Agent-C + independent PPO-R` | 研究 PPO-R 是否能达到强 R 端水平 | `main method`、`final method`，除非明确说明是 PPO-R 分支 |

## 2. R 端方法统一叫法

| 规范名 | 旧别名 / 代码名 | 备注 |
|---|---|---|
| `v1.2 static counterfactual ranker` | `v1.2`, `counterfactual_rank_only`, `planner-distilled ranker` | 当前主线 R 后端 |
| `independent PPO-R` | `PPO-R`, `agent_r_mixed` | 历史研究线主角；单独写 `PPO-R` 容易和 BC-PPO-R 混 |
| `BC-PPO-R` | `ppo_r_deeprmsa_bc`, `teacher-transfer PPO-R` | 行为克隆 warm-start / 教师迁移基线 |
| `DeepRMSA baseline` | `deep_rmsa` | 强教师/强启发式学习基线 |
| `KSP-FF K=50 hops` | `ksp_ff_k50_hops` | bug 修复后必须保留完整名 |
| `KSP-FF` | `ksp_ff` | 默认 KSP-FF，不等于 K=50 hops 版本 |
| `KSP-BF` | `ksp_bf` | 独立启发式 baseline |

## 3. C 端方法统一叫法

| 规范名 | 旧别名 / 代码名 | 备注 |
|---|---|---|
| `PPO-C` | `ppo_c` | 当前系统主线 C policy |
| `Typed-Mean-Field Agent-C` | `typed_mean_field`, `mean-field Agent-C` | PPO-R 研究线中的结构改进版本 |
| `Greedy-C` | `greedy_c` | 传统对照 |
| `IWD-C` | `inverse workload distance` | 传统对照 |
| `DF-C` | `delay-first` | 传统对照 |

## 4. 关键 checkpoint 规范引用

引用 checkpoint 时，推荐写法是：

```text
规范名 + checkpoint 路径
```

例如：

```text
v1.2 static counterfactual ranker
(sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt)
```

关键映射：

| 规范名 | 路径 |
|---|---|
| `PPO-C` | `sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt` |
| `v1.2 static counterfactual ranker` | `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt` |
| `independent PPO-R` | `sa_hmarl/checkpoints/agent_r_mixed.pt` |
| `BC-PPO-R` | `sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt` |
| `DeepRMSA baseline` | `sa_hmarl/checkpoints/deep_rmsa_snap24_s100_k5m10_ai015_h4_10_s5_30.pt` |

## 5. 实验文件命名规则

以后新增实验文件，推荐按下面的前缀来：

| 前缀 | 含义 |
|---|---|
| `main_` | 论文主表或主线结果 |
| `xlron_` | XLRON transfer / topology 实验 |
| `diag_` | 诊断实验 |
| `ablation_` | 消融实验 |
| `hist_` | 历史记录，仅归档，不作主证据 |

推荐命名模板：

```text
{prefix}_{scenario}_{method_or_focus}_{status}.md
```

例如：

```text
main_s100_r_backend_strict_fixrerun.md
xlron_cost239_r_backend_fixrerun.md
diag_v12_residual_blocking_floor.md
```

## 6. 哪些名字最容易误导

下面这些词本身不是错，但单独出现时容易歧义：

- `v1.2`
- `old v1.2`
- `new v1.2`
- `mixed low`
- `best model`
- `main method`
- `teacher model`
- `PPO-R`
- `transfer model`
- `strict`
- `fixrerun`

正确做法：

- `v1.2` 后面补上“static counterfactual ranker”或 checkpoint
- `PPO-R` 后面补上 `independent` 或 `BC-PPO-R`
- `teacher model` 后面补上 `DeepRMSA baseline` 或 `BC-PPO-R`
- `strict`/`fixrerun` 要说明修的是哪个 bug、比较的是哪组方法

## 7. 当前推荐引用顺序

1. 先引用主线方法名
2. 再给 checkpoint 或脚本路径
3. 最后给实验文件路径

推荐示例：

```text
我们当前主线使用 PPO-C + v1.2 static counterfactual ranker，
其中 v1.2 checkpoint 为
sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt，
主结果见 sa_hmarl/experiments/main_s100_ksp_ff_k50_hops_strict_fixrerun.md。
```
