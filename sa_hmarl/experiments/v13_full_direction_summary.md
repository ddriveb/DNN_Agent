# v1.3 方向总结：JPN48 在线候选集 + COST239 C-side 根因诊断

## 1. JPN48 推荐在线候选集配置

**推荐方案：`v12_k50_hops_legalctx48_ensure_ksp`**

- R-side环境使用 `k=50, path_sort_strategy=hops, block_sort_strategy=start_asc`。
- Ranker 候选集使用 `legalctx48`（最多 48 个精选候选），并启用 `ensure_ksp_action` 强制包含 KSP-FF 动作。
- 这是 **效果-时延折中最优** 的配置，适合作为 v1.3 默认在线候选集。

## 2. 相对 `baseline_v12` 的收益与代价

| 指标 | baseline_v12 | v1.3 推荐 | 变化 |
|---|---:|---:|---:|
| Blocking | 6.19% | 3.61% | **-2.57 pp** |
| NSB | 4.59% | 1.85% | **-2.74 pp** |
| Overload | 1.05% | 1.31% | +0.26 pp |
| Decision mean | 34.28 ms | 39.64 ms | +5.35 ms |
| Decision P95 | 55.62 ms | 55.30 ms | -0.32 ms |

- 主要收益：blocking 下降 2.57 pp，NSB 下降 2.74 pp。
- 主要代价：mean decision time 增加 5.35 ms；P95 基本持平（甚至略降）。
- 性价比：每降低 1 pp blocking 仅额外付出约 **2.08 ms** mean decision time，显著优于 k=30/k=50 all_legal 方案。

## 3. COST239 `paper_style_heavy` 的 overload 根因

**结论：主要是 C-side server allocation / capacity 问题，而非 R-side continuation/path coverage。**

关键证据：
- 100% 的 `server_overload` 事件集中在 **Server 0**，且全部发生在 **Split 0**。
- Servers 1–3 从未触发 overload，说明它们仍有容量。
- R-side 差异对 blocking 影响极小：`ppo_c+ksp_ff_k50_hops` 仅比 `ppo_c+v12` 好 0.17 pp，强 PPO-R continuation 反而更差（+1.25 pp）。
- Server utilization 在平均值上看似均衡（0.53–0.74），但 Server 0 是实际瓶颈；当 PPO-C 持续把 Split-0 请求分到 Server 0 时，就会触发 overload。

因此，COST239 高压 overload 不能通过继续优化 R-side 候选集/continuation 解决。需要转向：
- C policy 看到 server utilization / capacity 特征；
- 允许/鼓励 C policy 把 Split-0 请求分散到其它服务器；
- 或调整 server placement / capacity。

## 4. 下一阶段主线结论

**正式建议：**

> **v1.3 主线 = v1.2 + better online candidate set（`v12_k50_hops_legalctx48_ensure_ksp`）**
>
> **COST239 高压 overload 作为独立 C-side 支线处理，不再与 v1.3 R-side 主线绑定。**

### 后续工作拆分

1. **v1.3 主线（高优先级）**
   - 在 `CounterfactualRRankerPolicy` 中固化 `legalctx48 + ensure_ksp` 在线候选集逻辑。
   - 确保该配置在 JPN48 及其它拓扑（snap24, german17, nsfnet）上训练/评估一致。
   - 如需进一步降低时延，可尝试 `max_candidates=32/40` 的 lighter legalctx，但当前 48 已足够高效。

2. **COST239 C-side 支线**
   - 为 C policy 引入 server utilization / available capacity / split-server 压力特征。
   - 在 `paper_style_heavy` 上重新训练/微调 PPO-C，观察是否能将 Split-0 负载从 Server 0 分散出去。
   - 如 C policy 已无法分散（例如拓扑/部署约束导致只能走 Server 0），则考虑调整 server placement 或容量。

## 5. 生成文件清单

- JPN48 完整 sweep：`sa_hmarl/experiments/jpn48_candidate_latency_sweep_full.md` / `.json`
- JPN48 单个配置原始结果：`sa_hmarl/experiments/jpn48_candidate_latency_sweep/*.json` / `*.md`
- COST239 C-side 诊断：`sa_hmarl/experiments/cost239_cside_overload_diagnostic_full.md` / `.json`
- 脚本：`sa_hmarl/scripts/run_jpn48_candidate_latency_sweep.sh`
- 脚本：`sa_hmarl/scripts/aggregate_jpn48_candidate_latency_sweep.py`
- 代码改动：`sa_hmarl/sa_hmarl/evaluation/server_diagnostics.py`
- 代码改动：`sa_hmarl/sa_hmarl/evaluation/eval_main_s100_system_comparison.py`（增加 `--collect_server_diagnostics`）
