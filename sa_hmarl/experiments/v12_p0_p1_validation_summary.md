# v1.2 P0 / P1 最小验证实验汇总

## 背景

- **P1（候选集/视野假设）**：`xlron_jpn48` 上 `ppo_c+v1.2` 落后于 `ppo_c+KSP-FF K50 hops`，怀疑主因是 v1.2 ranker 的候选集视野不足（`k=5, km`）。
- **P0（continuation / overload-aware return 假设）**：`xlron_cost239_ptrnet_real` 高压场景（`paper_style_heavy`）下所有方法 blocking ~12–13% 且几乎全是 `server_overload`，怀疑瓶颈不是 path coverage，而是 future rollout continuation 未能预见 overload，且 return 中对 overload 惩罚不足。

## 基线参考

| 拓扑 | 方法 | blocking | NSB | overload | deadline |
|---|---|---:|---:|---:|---:|
| `xlron_jpn48` | `ppo_c+v12` | 6.19% | 4.59% | 1.05% | 0.51% |
| `xlron_jpn48` | `ppo_c+ksp_ff_k50_hops` | 5.04% | 1.94% | 2.81% | — |
| `snap24_gnutella_reach` | `v12_k50_hops` | 0.94% | — | — | — |
| `snap24_gnutella_reach` | `v12` | 0.92% | — | — | — |
| `xlron_cost239_ptrnet_real`（高压 `paper_style_heavy`） | `ppo_c+v12` | 13.40% | 0.14% | 13.26% | 0.00% |
| `xlron_cost239_ptrnet_real`（高压 `paper_style_heavy`） | `ppo_c+v12_k50_hops` | 13.39% | 0.00% | 13.39% | 0.00% |
| `xlron_cost239_ptrnet_real`（高压 `paper_style_heavy`） | `ppo_c+ksp_ff_k50_hops` | 12.35% | 0.00% | 12.35% | 0.00% |

## P1 实验：`xlron_jpn48` 三方法对照（已完成）

- **命令**：`eval_main_s100_system_comparison --topology xlron_jpn48 --methods ppo_c+v12,ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops`
- **输出**：`sa_hmarl/experiments/jpn48_v12_k50_validation.json/.md`
- **结果**（5 seeds × 20 episodes × 80 requests）：

| Method | Blocking | NSB | Overload | Deadline | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|
| `ppo_c+v12` | 6.19% | 4.59% | 1.05% | 0.51% | 36.4/57.0 ms |
| `ppo_c+v12_k50_hops` | **2.75%** | 1.56% | 0.84% | 0.33% | 93.7/236.8 ms |
| `ppo_c+ksp_ff_k50_hops` | 5.04% | 1.94% | 2.81% | 0.27% | 28.5/45.3 ms |

- **结论**：
  - 仅把候选集环境扩到 `k=50, hops, start_asc`、让 v1.2 ranker 在更大候选集上打分，就把 JPN48 blocking 从 6.19% 降到 **2.75%**，甚至优于 `KSP-FF K50 hops` 的 5.04%。
  - 这说明 JPN48 上 v1.2 落后的主因确认是 **候选集视野不足（k=5, km）**，而不是 scorer 本身不会选。
  - 代价是 R-side 决策时延显著上升（36 ms → 94 ms 均值，237 ms P95），需要在工程/推理效率与性能之间权衡。
  - **P1 优先级高**：下一步应把 `k=50, hops, start_asc`（或更轻量的 `legalctx48` + ensure KSP）集成到在线 ranker 候选集构建中，并在 JPN48 上完整训练/评估。

## P0 实验：`xlron_cost239_ptrnet_real` 高压场景

### 关键纠正

P0 的“高压”应指 `paper_style_heavy`：`arrival_interval=0.07, holding=4-6, size=15-50`。已有结果如下：

| Method | Blocking | NSB | Overload |
|---|---:|---:|---:|
| `ppo_c+v12` | 13.40% | 0.14% | 13.26% |
| `ppo_c+v12_k50_hops` | 13.39% | 0.00% | 13.39% |
| `ppo_c+ksp_ff_k50_hops` | 12.35% | 0.00% | 12.35% |

- 扩候选集在高压下几乎无收益（v12_k50_hops 仅 -0.01 pp），符合 P0 假设的验证前提。

### P0 smoke（主配置，负载偏轻）

- **脚本**：`sa_hmarl/experiments/run_p0_cost239_smoke.sh`
- **分析脚本**：`sa_hmarl/experiments/analyze_p0_smoke.py`
- **配置**：`num_slots=100, arrival_interval=0.15, holding=4-10, size=5-30, k=5, km, mixed, horizon=1, train 2 episodes x 60 requests`
- **代码改动**：为 `generate_r_counterfactual_ranking_dataset.py` 增加了 `future_server_overload_counts` 的落盘与诊断统计（`future_server_overload_variance`、`future_server_overload_positive_rate`），并把 `future_server_overload` 系数写入 metadata/report。
- **三组对照**：
  1. `a_baseline`：`agent_r_mixed.pt` + `future_rollout_policy=ppo_r` + `overload_coef=0`
  2. `b_strong_ppo_r_overload8`：`ppo_r_deeprmsa_bc_snap24_best.pt` + `future_rollout_policy=ppo_r` + `overload_coef=8`
  3. `c_ranker_future_overload8`：`future_rollout_policy=ranker`（COST239 retrain v2） + `overload_coef=8`
- **smoke 结果**（`analyze_p0_smoke.py`）：

| Run | groups | nonzero_range | mean_range | blocked_var | overload_var | overload+ | top1_top2_gap | ol_coef |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline_ppo_r_no_ol | 360 | 63.61% | 0.0954 | 0.0000 | 0.0000 | 0.00% | 0.0057 | 0.0 |
| B_strong_ppo_r_ol8 | 360 | 68.33% | 0.0778 | 0.0000 | 0.0000 | 0.00% | 0.0040 | 8.0 |
| C_ranker_future_ol8 | 360 | 59.72% | 0.0680 | 0.0000 | 0.0000 | 0.00% | 0.0036 | 8.0 |

- **观察**：主配置下没有 future overload 事件，说明负载不够重；后续 P0 数据集应在 `paper_style_heavy` 下进行。

### P0 closed-loop probe（已完成）

- **主配置 probe**（`bash-kef1vern`）：`ppo_c+v12=0.12%`, `ppo_c+ppo_r(strong)=0.42%`, `KSP-FF=0.00%`。主配置下没有 overload 问题，strong PPO-R 甚至略差于 v12。
- **高压 probe**（`bash-yt58gef2`，`paper_style_heavy`）：

| Method | Blocking | NSB | Overload | Δ vs v12 |
|---|---:|---:|---:|---:|
| `ppo_c+v12` | 13.92% | 0.00% | 13.92% | — |
| `ppo_c+ppo_r`（strong BC-PPO-R） | 15.17% | 0.00% | 15.17% | **+1.25 pp** |
| `ppo_c+ksp_ff_k50_hops` | 13.75% | 0.00% | 13.75% | -0.17 pp |

- **P0 结论**：
  - 在 COST239 高压下，**更强的 PPO-R continuation（`ppo_r_deeprmsa_bc_snap24_best.pt`）反而更差**（+1.25 pp overload），扩候选集收益也很小（-0.17 pp）。
  - 这说明当前可用的“强 continuation”并不能缓解 overload；高压 overload 更可能是 **C-side 服务器分配 / 容量** 或 **拓扑/负载本身** 的限制，而不是 R-side scoring/continuation 的短板。
  - 因此 **P0 暂不成立**，不需要立即投入 overload-aware return / stronger continuation；应先把精力放在 P1（候选集升级）。

## 最终结论与下一步

1. **P1 已确认，优先级最高**：在 `xlron_jpn48` 上，把候选集扩到 `k=50, hops, start_asc` 即可让 v1.2 ranker 大幅超越 KSP-FF。下一步：
   - 在 `CounterfactualRRankerPolicy` 中支持 `k=50/hops/start_asc` 候选集构建（或 `legalctx48` + ensure KSP 的折中方案）。
   - 在 JPN48 上做完整训练/评估，权衡 blocking 收益与决策时延。
2. **P0 暂不成立**：COST239 高压 overload 不是简单地靠“更强的 R continuation”或“overload-aware return”就能解决；需要重新定位是 C-side、环境负载还是拓扑容量问题。
3. **工程/数据改进**：本次为数据集生成脚本增加了 `future_server_overload_counts` 的落盘与诊断，后续如需再探 P0 可直接使用。
