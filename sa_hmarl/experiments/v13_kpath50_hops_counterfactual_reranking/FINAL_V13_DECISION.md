> ⚠️ **本报告已过期并被 `FINAL_V13_DECISION_CORRECTED.md` 替代。**  
> 原报告依赖的 `relative_regret_reduction` 指标存在严重缺陷（tiny-regret 分母导致不稳定），且未包含修正后的 regret-selected 重训练与 5-seed 闭环公平评估。请勿依据本文件做决策。

# SA-HMARL 严格 v1.3 诊断结论与 v1.35 启动决定

**日期：** 2026-07-14  
**数据集：** `r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`  
**协议：** K_C=5, K_path=50, K_prop=30, H=5, gamma=1.0, candidate_mode=ppo_r_topk_only  
**模型：** MLP [128,64]，simple loss（reg_weight=1.0, lambda_pair=0, lambda_hard=0）

---

## 1. 实际落入的情况

**情况 5：候选有 headroom，但模型仍没学会。**

---

## 2. 支撑判断的具体数字

### A. 离线排序未学会

| 训练规模 | Top-1 | Tie-aware Top-1 | Spearman | Kendall tau-b | Model regret | PPO regret | Relative regret reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.3k groups | 4.3%±1.0% | 45.8%±1.5% | 0.029±0.029 | 0.024±0.023 | 0.070±0.002 | 0.105 | **-0.659±0.023** |
| 3k groups | 6.8%±0.9% | 45.0%±0.3% | -0.000±0.002 | 0.000±0.001 | 0.075±0.009 | 0.105 | **-0.253±0.053** |
| 5k groups | 6.4%±0.4% | 44.9%±0.5% | 0.001±0.009 | 0.001±0.007 | 0.086±0.009 | 0.105 | **-0.136±0.035** |
| **full 7.3k** | **6.2%±1.8%** | **44.8%±0.5%** | **0.006±0.014** | **0.004±0.011** | **0.084±0.002** | **0.105** | **-1.154±0.134** |

关键观察：
- Spearman 和 Kendall tau-b 围绕 0 波动，95% 经验区间包含负值。
- Model regret **始终高于** PPO regret（relative reduction 为负）。
- 把训练数据从 1.3k 增加到 7.3k groups，Top-1 没有单调上升，full 的 regret 反而最差。
- Pairwise accuracy ≈ 51–53%，接近随机。

### B. 闭环未执行

由于离线指标未满足门控条件（Spearman 未稳定正、model regret 未低于 PPO regret、learning curve 未改善），**未启动 MEDIUM_CLOSED_LOOP**。

### C. 候选集存在真实空间

| 指标 | 数值 |
|---|---:|
| Train nonzero return range | 85.35% |
| PPO top-1 不在 oracle tie set（P_headroom） | 61.15% |
| PPO regret mean / P50 / P90 | 0.0471 / 0.0102 / 0.0751 |
| Future blocking 在候选间不同的 group 比例 | **1.47%** |
| Only delay/FS 不同的 group 比例 | **83.88%** |
| Future blocking 完全相同的 group 比例 | **98.53%** |

结论：PPO-R Top-30 内**存在**更优动作（oracle headroom 非零），但这些 headroom 主要来自 delay/FS，而非 future blocking。

### D. 标签主要由 delay/FS 决定

| Component | Explained fraction of return variance |
|---|---:|
| future_block | 97.90% |
| future_delay | 0.73% |
| future_fs | 0.11% |
| current_block / future_nsb | 0.00% |

注意：future_block 的 explained fraction 高，是因为它在少数（1.47%）group 中一旦变化就占主导；在绝大多数 group 中 future blocking 没有变化，因此实际可学习的排序信号几乎只剩下 delay/FS。

### E. 多 future trace 稳定性极差

| Proxy | Value |
|---|---:|
| Future-trace swap Kendall tau-b | 0.018 ± 0.194 |
| Top-1 agreement under swapped trace | 18.00% |

说明：即使保持当前动作不变、只替换共同未来请求序列，候选排序就会剧烈变化。H=5 标签对未来 trace 高度敏感，缺乏稳定监督信号。

---

## 3. 是否允许启动 v1.35

**不允许启动 v1.35 full。**

依据：当前结果属于情况 5，v1.35 启动规则只允许情况 1 直接启动 full，情况 2/3 启动小规模 paired pilot。情况 4–7 不得启动大规模 v1.35。

---

## 4. 如果允许，是 full 还是 pilot

当前**不允许**。在情况 5 下，只有在完成标签消融并获得 afterstate 能提升 predictability 的证据后，才允许生成一个 **500–1000 group 的 v1.35 diagnostic pilot**，且该 pilot 仅用于判断 afterstate 特征是否提高 Spearman/regret，不跑大规模闭环。

---

## 5. 下一条应执行的准确命令

根据情况 5 的下一步，应先做标签消融，判断当前 H=5 全标签是否比短 horizon / 阻塞-only / 未来-only 标签更可学习：

```bash
cd /mnt/d/project/DNN_Agent
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TORCH_NUM_THREADS=1
export PYTHONPATH=sa_hmarl
# 示例：在 medium 数据上训练 H=5 blocking-only 标签
python3 -m sa_hmarl.training.train_r_counterfactual_ranking \
  --dataset_dir sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium \
  --output_dir sa_hmarl/checkpoints/v13_label_ablation_blocking_only \
  --reg_weight 1.0 --lambda_pair 0.0 --lambda_hard 0.0 \
  --seed 42 --epochs 80 --patience 15
```

更系统的方法是新增一个 `run_label_ablations.py` 脚本，统一对比：
- `H=0` immediate-only
- `H=1`
- `H=2`
- `H=5`
- `blocking-only`
- `future-only`
- `full v1.3`

在确认哪一类标签更稳定、更可学习之前，**不启动 v1.35**。

---

## 6. 明确禁止的下一步

- **禁止**直接启动严格 v1.35 full dataset。
- **禁止**扩大 medium 数据集到 30k–40k groups 以期望“数据多了就会好”。
- **禁止**把 offline Top-1 从 4% 到 8% 的波动解释为“显著改善”。
- **禁止**在标签-闭环一致性问题未解决前，把 afterstate 特征当作救命稻草。
- **禁止**修改候选池（加入 anchors / fillers / diversity）并仍称为当前严格 v1.3。

---

## 7. 对六个核心问题的明确回答

1. **当前 pilot 差是否主要由数据规模造成？**  
   **不是。** 把训练 groups 从 1.3k 提升到 7.3k 后，Spearman/Kendall 仍接近 0，regret 相对 PPO 仍为负，说明主要瓶颈不是数据量。

2. **H=5 是否提供可学习的 future-blocking 排序信号？**  
   **没有。** 仅 1.47% 的 group 中候选间 future blocking 不同；98.53% group 的 future blocking 完全相同。H=5 标签主要由 delay/FS 决定，且多 trace 稳定性极差。

3. **25-d 特征能否预测这些标签？**  
   **当前不能稳定预测。** 全局 Spearman 部分特征中等（如 lfb 0.32、free_ratio 0.33、server_utilization -0.38），但 per-group 平均绝对相关普遍低于 0.28；ranker 在 7.3k groups 上仍未学会排序。

4. **PPO-R Top-30 内是否存在真实、可实现的 reranking headroom？**  
   **存在 headroom，但不可实现。** P_headroom=61%，PPO regret>0；然而 headroom 主要由不稳定/微弱的 delay/FS 差异驱动，模型无法稳定学习并转化为 regret 改善。

5. **扩大数据后 strict v1.3 是否稳定优于 PPO-R？**  
   **否。** Offline 指标未改善；未满足闭环门控，因此未跑闭环，但现有证据不支持“扩大数据即可稳定优于 PPO-R”。

6. **是否值得继续生成 full dataset？**  
   **不值得。** 在 H=5 标签不稳定、future-blocking 信号微弱的前提下，继续生成 full dataset 不会解决 label predictability 问题。

---

## 8. 结论

严格 v1.3 的候选池和实现协议是正确的，但 **H=5 common-future 标签在当前 traffic/load 下缺乏稳定、可学习的 future-blocking 排序信号**。问题不是 candidate recall（PPO-R Top-30 已包含更优动作），而是 **label predictability / feature sufficiency**。在通过标签消融确认更强的监督信号之前，**不得进入 v1.35**。
