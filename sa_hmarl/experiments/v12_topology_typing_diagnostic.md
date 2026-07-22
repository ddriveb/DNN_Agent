# v1.2 拓扑判型诊断：PPO-C + v1.2 vs PPO-C + KSP-FF K50 hops

> 目标：根据已有 S100 主实验、XLRON topology fixrerun、COST239 高压 probe 以及拓扑结构统计，判断每个拓扑上 `PPO-C + v1.2` 与 `PPO-C + ksp_ff_k50_hops` 的 blocking 主导瓶颈类型，并给出后续升级优先级。

## 1. 数据来源

| 实验 | 文件 |
|---|---|
| S100 snap24 主实验（严格 fixrerun） | `main_s100_ksp_ff_k50_hops_strict_fixrerun.json/.md` |
| S100 snap24 v1.2 K=50 hops 诊断 | `main_s100_v12_k50_hops_diagnostic_fixrerun.json/.md` |
| XLRON 四个拓扑 fixrerun | `xlron_topology_fixrerun_summary.md` 及 `xlron_*_s100_r_compare_fixrerun.json/.md` |
| COST239 高压 probe | `v12_high_pressure_probe.json/.md` |
| 拓扑结构统计 | `sa_hmarl/scripts/compute_topology_stats.py`（本次生成） |

## 2. 判型方法

对每个拓扑，按 **v1.2 自身的失败原因占比** 判型：

- **NSB-dominant**：`no_suitable_block_rate / blocking_rate > 50%`，即 spectrum/fragmentation 是主因。
- **overload-dominant**：`server_overload_rate / blocking_rate > 50%`，即 server compute / 长期负载是主因。
- **mixed**：两者都显著（>20% 且相差不大）。
- **near-ceiling**：v1.2 blocking 与当前已知最优（KSP-FF K50 hops 或 v1.2 自身）差距 < ~0.2 pp，且绝对值 < 1%。

> 注意：`server_overload` 在代码中发生在 Agent-C 选 server 之后、Agent-R 选 path 之前，因此 R backend 的 overload 差异是 **R 决策通过成功率间接改变未来 server 负载** 造成的，不是 R 直接选错 server。

## 3. 拓扑指纹

| 拓扑 | 节点 | 边 | 平均度 | 平均 SP (km) | 直径 (hops) | 全局边连通度 | K=5 km 平均路径数 | K=50 hops 平均路径数 | 最短路径 SE 均值 | 16QAM% | 8QAM% | QPSK% | BPSK% | 服务器平均 SP (km) | 瓶颈代理（max edge betweenness） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `snap24_gnutella_reach` | 24 | 40 | 3.33 | 708 | 3 | 1 | 4.33 | 41.83 | 3.08 | 23.2 | 62.0 | 14.9 | 0.0 | 598 | 0.272 |
| `xlron_cost239_ptrnet_real` | 11 | 26 | 4.73 | 777 | 3 | 4 | 5.00 | 50.00 | 2.95 | 23.6 | 47.3 | 29.1 | 0.0 | 813 | 0.218 |
| `xlron_german17` | 17 | 24 | 2.82 | 805 | 6 | 2 | 5.00 | 42.01 | 2.89 | 27.2 | 34.6 | 38.2 | 0.0 | 229 | 0.338 |
| `xlron_nsfnet_deeprmsa` | 14 | 22 | 3.14 | 1995 | 3 | 3 | 5.00 | 50.00 | 1.81 | 6.6 | 12.1 | 37.4 | 44.0 | 1175 | 0.247 |
| `xlron_jpn48` | 48 | 82 | 3.42 | 721 | 14 | 2 | 5.00 | 50.00 | 3.12 | 38.3 | 37.9 | 21.9 | 2.0 | 367 | 0.433 |

关键观察：

- **NSFNET** 是典型的“长距-低调制”拓扑：平均最短路径 1995 km，44% 的节点对只能使用 BPSK，候选 action 空间最小（mean total valid R actions ≈ 40）。
- **COST239** 最稠密、路径最多，调制限制最小，资源瓶颈几乎不在 spectrum。
- **JPN48** 最大、直径最长（14 hops），存在大量长跳数路径；虽然平均距离不大，但 `K=50 hops` 能显著降低 NSB，说明 path diversity / hop-count 是主要杠杆。
- **German17** 最稀疏，max edge/node betweenness 高，存在明显结构性桥边，容易形成 directional bottleneck。

## 4. 各拓扑判型与证据

### 4.1 `snap24_gnutella_reach` — NSB-dominant / low-load near-ceiling

| 方法 | Blocking | NSB | Overload | 说明 |
|---|---:|---:|---:|:---|
| `ppo_c+v12` | **0.92%** | 0.84% | 0.09% | NSB 占 blocking 91% |
| `ppo_c+ksp_ff_k50_hops` | 1.05% | 0.94% | 0.11% | KSP 也主要是 NSB |
| `ppo_c+v12_k50_hops` | 0.94% | 0.81% | 0.12% | 给 v1.2 同样 50-hop 候选集后，表现与 KSP 几乎一致 |

- **判型**：NSB-dominant（spectrum / fragmentation）。
- **证据**：失败原因中 `no_suitable_block` 占绝对多数；把 v1.2 的候选集也扩到 K=50 hops 后，blocking 与 KSP-FF 基本持平（0.94% vs 1.05%），说明 **scoring 不是瓶颈，spectrum 可用性才是**。
- **拓扑对应**：平均跳数仅 2.2、SE 高（3.08），整体负载低，blocking 已接近当前天花板（≈0.9%）。
- **升级路线**：优先级低；如需再降，可关注 block waste / fragmentation 建模，而非单纯扩路径。

---

### 4.2 `xlron_cost239_ptrnet_real` — overload-dominant / near-zero-ceiling

| 方法 | Blocking | NSB | Overload | 说明 |
|---|---:|---:|---:|:---|
| `ppo_c+v12` | 0.20% | 0.00% | 0.20% | 100% overload |
| `ppo_c+ksp_ff_k50_hops` | **0.00%** | 0.00% | 0.00% | KSP 完全消除失败 |

高压 probe（`arrival_interval=0.09, holding_max=13.0`）：

| Probe | Blocking | NSB | Overload |
|---|---:|---:|---:|
| old_v12 | 12.51% | 0.00% | 12.51% |
| k50_all_legal | 12.91% | 0.00% | 12.91% |
| k50_ensure_ksp | 12.91% | 0.00% | 12.91% |
| KSP-FF K50 hops | 12.53% | 0.00% | 12.53% |

- **判型**：**overload-dominant**。在主配置和高压配置下，所有失败原因都是 `server_overload`；扩路径、注入 KSP 动作都无法降低 blocking。
- **证据**：COST239 稠密（avg degree 4.73）、全局边连通度 4、K=50 hops 总能找到 50 条路径，spectrum 不是限制。v1.2 的 0.20% 全部来自 server overload，而 KSP-FF 通过更保守的 path/mod/block 选择可以把 overload 压到 0。
- **拓扑对应**：高连通、短路径、调制灵活 → 瓶颈在 **MEC compute / server load dynamics**。
- **升级路线**：
  1. 优先修改 reward/return，使 overload 失败获得更大权重（当前 `server_overload_penalty=1.0`、`no_block_penalty=1.2` 在高压下无法拉开差距）。
  2. 用更强的 future rollout labeler（如 `ppo_r_deeprmsa_bc_snap24_best.pt` / BC-PPO-R）替代当前 ranking teacher，提升 continuation 对 server 负载的预测。
  3. 在 Agent-C 侧加入显式的 server load balancing / overload 风险项。

---

### 4.3 `xlron_german17` — overload-dominant / mixed / near-ceiling

| 方法 | Blocking | NSB | Overload | Δ vs v1.2 |
|---|---:|---:|---:|---|
| `ppo_c+v12` | **0.97%** | 0.27% | 0.70% | — |
| `ppo_c+ksp_ff_k50_hops` | 1.09% | 0.07% | 1.01% | NSB -0.20 pp, Overload +0.31 pp |

- **判型**：**overload-dominant（mixed）**。Overload 占 v1.2 blocking 的 72%，NSB 也存在但较小。
- **证据**：v1.2 胜过 KSP-FF 主要因为它把 **overload 从 1.01% 压到 0.70%**（-0.31 pp），代价是 NSB 略升。KSP-FF 的 hop-first 策略虽然降低 NSB，却让更多请求成功，反过来加剧了 server 负载。
- **拓扑对应**：German17 稀疏（avg degree 2.82）、max edge betweenness 0.338，桥边多；server 间平均距离只有 229 km（很短），spectrum 相对充足，server 负载是主要矛盾。
- **升级路线**：与 COST239 类似，优先强化 overload-aware return / continuation；同时保持 v1.2 在 server 选择上的优势，不要让 KSP 式“成功更多”把负载推向饱和。

---

### 4.4 `xlron_nsfnet_deeprmsa` — mixed-overload-dominant / spectrum-constrained

| 方法 | Blocking | NSB | Overload | Deadline/Other |
|---|---:|---:|---:|---:|
| `ppo_c+v12` | **3.09%** | 0.76% | 2.18% | 0.15% |
| `ppo_c+ksp_ff_k50_hops` | 3.26% | 0.83% | 2.23% | 0.21% |

- **判型**：**mixed-overload-dominant**。Overload 占 70%，NSB 占 25%，还有少量 deadline/other。
- **证据**：NSFNET 平均最短路径 1995 km，44% 节点对只能用 BPSK，mean total valid R actions ≈ 40（远低于 snap24 的 190、JPN48 的 214）。这导致每条成功请求占用大量 FS 和传输/计算资源，既压 spectrum 又压 server。两种方法差距很小，说明当前 v1.2 在这个拓扑上已接近“结构上限”。
- **拓扑对应**：长距、低 SE、候选空间小 → **spectrum 与 compute 双重紧张**。
- **升级路线**：
  1. 同时优化 NSB（mod-aware ranking、FS 效率）和 overload（server load 预测）。
  2. 考虑在该拓扑上做 per-topology fine-tune 或引入 extended modulation profile（PM-BPSK）以降低长距路径的 FS 需求。
  3. 由于 blocking 基数高（~3%），这里任何 reward/return 改进的绝对收益都会比较明显。

---

### 4.5 `xlron_jpn48` — NSB-dominant / topology-complex

| 方法 | Blocking | NSB | Overload | Deadline | Other |
|---|---:|---:|---:|---:|---:|
| `ppo_c+v12` | 6.19% | **4.59%** | 1.05% | 0.51% | 0.04% |
| `ppo_c+ksp_ff_k50_hops` | **5.04%** | 1.94% | 2.81% | 0.27% | 0.01% |

- **判型**：**NSB-dominant / topology-complex**。v1.2 的 NSB 占 blocking 的 74%，是最大单一失败源。
- **证据**：KSP-FF K50 hops 把 NSB 从 4.59% 降到 1.94%（-2.65 pp），同时 overload 上升 1.76 pp，净 blocking 下降 1.15 pp。这说明 v1.2 的 **k=5 km 候选集在 JPN48 上严重不足**——它错过了大量更短/更可行的 hop-first 路径。
- **拓扑对应**：48 节点、直径 14 hops、max edge betweenness 0.433，存在长路径和局部瓶颈；但平均距离仅 721 km，调制不是问题，**路径选择空间本身**是问题。
- **升级路线**：
  1. 直接给 v1.2 配 K=50 hops 候选集（`v12_k50_hops`），验证是否能复现 snap24 的“追上 KSP”现象。
  2. 若 scoring 无法利用 50-hop 候选集，需要针对 JPN48 重新蒸馏 ranking teacher，或加入显式的 hop-count / path-diversity / edge-congestion 特征。
  3. 在降低 NSB 的同时，通过 overload-aware return 抑制因成功率上升带来的 server load 反弹。

## 5. 全局对照表

| 拓扑 | v1.2 Blocking | KSP K50 Blocking | v1.2 主因 | KSP 相对 v1.2 的收益来源 | 判型 |
|---|---:|---:|:---|:---|:---|
| `snap24_gnutella_reach` | 0.92% | 1.05% | NSB (91%) | 几乎无收益；v1.2 已领先/持平 | NSB-dominant / near-ceiling |
| `xlron_cost239_ptrnet_real` | 0.20% | **0.00%** | Overload (100%) | KSP 通过更保守选择消除 overload | Overload-dominant |
| `xlron_german17` | **0.97%** | 1.09% | Overload (72%) | v1.2 反而更好：KSP 降 NSB 但增 overload | Overload-dominant / mixed |
| `xlron_nsfnet_deeprmsa` | **3.09%** | 3.26% | Overload (70%) + NSB (25%) | 两者都接近结构上限 | Mixed-overload-dominant |
| `xlron_jpn48` | 6.19% | **5.04%** | NSB (74%) | KSP 通过 50-hop 候选集大幅降低 NSB | NSB-dominant / topology-complex |

## 6. 后续升级优先级

### P0：修复 overload-aware return / continuation

- **为什么**：COST239 高压 probe 显示，即使把候选集扩到 K=50 并注入 KSP，blocking 仍 ~12.5% 且 **100% server_overload**。这说明当前 return 中 overload/NSB 的权重不足以让策略学到“拒绝短期可成功但会压垮 server 的动作”。
- **做什么**：
  1. 增大 `server_overload_penalty` / `server_fail_extra`，让 overload 失败的代价显著高于 NSB。
  2. 引入 BC-PPO-R（`ppo_r_deeprmsa_bc_snap24_best.pt`）作为 future rollout labeler，提供更强的 continuation。
  3. 对 Agent-C 加入 server load / queue pressure 的显式长期惩罚。

### P1：针对 NSB-dominant 拓扑扩展/优化候选集

- **目标拓扑**：`xlron_jpn48`（及 `snap24` 的进一步优化）。
- **做什么**：
  1. 对 v1.2 启用 `v12_k50_hops`（K=50, `path_sort_strategy=hops`），先看是否复现 KSP 收益。
  2. 若 scoring 跟不上，用 hop-aware / edge-congestion 特征重新训练 ranking model。
  3. 对 NSFNET 这种长距拓扑，优化调制/FS 效率，而非单纯扩路径。

### P2：per-topology 微调

- 对于 NSFNET/JPN48 等 transfer 性能明显差于 snap24 的拓扑，考虑收集 topology-specific counterfactual 数据并 fine-tune ranker，而不是纯靠 snap24 训练的检查点 transfer。

## 7. 结论：v1.2 更适合“复杂拓扑”还是“未来影响复杂”场景？

**当前证据不支持“v1.2 更适合复杂拓扑”**。在拓扑最复杂的 `xlron_jpn48`（48 节点、14 hops 直径）上，v1.2 明显输给 KSP-FF K50 hops，失败以 NSB 为主；在拓扑最稠密、资源最充足的 `xlron_cost239_ptrnet_real` 上，v1.2 也输给 KSP，失败以 overload 为主。

**v1.2 的设计目标（counterfactual future rollout + ranking）更适合“未来影响复杂”的场景**——即今天的 path/server 选择会通过资源占用、碎片化、server 负载等机制显著影响未来 blocking。但在当前实现中，future rollout teacher/continuation 不够强、return 对 overload 的惩罚不够重，导致它还没有在“未来影响复杂”场景（如 COST239 高压）里兑现优势。

因此：

- 如果论文要宣称 v1.2 的强项，最安全的表述是：
  > “v1.2 在概念上针对未来决策影响进行蒸馏，适用于 server/spectrum 耦合强的动态场景；在当前实现下，它在 `snap24` 和 `german17` 等中等规模拓扑上具有竞争力，但在 JPN48 这类 hop-diverse 大拓扑上仍需候选集与 scoring 的联合升级。”
- 下一步应优先把主线从 **“扩路径/KSP 注入”** 转向 **“更强的 future continuation + overload-aware return”**，同时对 JPN48 这类 NSB-dominant 拓扑补做候选集扩展实验。
