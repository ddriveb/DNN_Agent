# SA-HMARL 严格 v1.3 最终诊断

## 一句话结论

**当前严格 v1.3 下，PPO-R Top-30 候选集中存在 headroom，但 H=5 common-future 标签缺乏稳定、可学习的 future-blocking 排序信号；7.3k 训练 groups 仍未让 MLP [128,64] 学会优于 PPO-R 的排序。因此不允许进入 v1.35。**

## 关键证据

| 维度 | 结果 | 说明 |
|---|---|---|
| 数据规模 | 1.3k → 7.3k groups | Spearman/Kendall 仍接近 0，未随数据增长改善 |
| Offline ranking | full Top-1 6.2%±1.8% | 远低于可用水平；relative regret reduction 为负 |
| Model vs PPO regret | 0.084 vs 0.105 | Model regret **高于** PPO regret |
| PPO headroom | P_headroom = 61.15% | PPO Top-1 经常不在 oracle tie set |
| Future-blocking variance | 仅 1.47% groups 候选间不同 | 绝大多数 group 的 future blocking 完全相同 |
| 标签主导 | 83.88% groups 只有 delay/FS 不同 | H=5 排序信号主要来自 delay/FS，而非阻塞 |
| Trace 稳定性 | Kendall tau-b = 0.018 | 替换共同未来 trace 后候选排序剧烈变化 |

## 判定

落入 **情况 5：候选有 headroom，但模型仍没学会**。

## 禁止

- 禁止启动 v1.35 full。
- 禁止以“再加点数据就会好”为由继续生成 full dataset。
- 禁止把 offline Top-1 的随机波动解释为显著改善。

## 下一步

先执行 **标签消融**：H=0/1/2/5、blocking-only、future-only、full，确认哪类标签更稳定可学习。只有在消融证明 afterstate 特征能显著提升 predictability 后，才允许生成小规模 v1.35 diagnostic pilot。

详细数字与分类依据见：
- `FINAL_V13_DECISION.md`
- `FINAL_V13_DECISION.json`
