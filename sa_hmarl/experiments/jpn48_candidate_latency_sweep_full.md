# JPN48 Online Candidate-Set / Latency Sweep (v1.3 Selection)

## Table 1: Full Results

| Method | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_v12 | 6.19% | 6.19% | 4.59% | 1.05% | 0.51% | 0.04% | 16.066/28.934 ms | 34.284/55.618 ms |
| v12_k20_hops | 3.96% | 4.05% | 2.43% | 1.12% | 0.35% | 0.06% | 16.381/29.453 ms | 40.104/81.028 ms |
| v12_k30_hops | 3.40% | 3.54% | 1.91% | 1.10% | 0.36% | 0.03% | 16.598/29.796 ms | 48.528/108.653 ms |
| v12_k50_hops_all_legal | 2.75% | 2.89% | 1.56% | 0.84% | 0.33% | 0.03% | 16.848/29.979 ms | 67.228/169.881 ms |
| v12_k50_hops_legalctx48 | 4.98% | 5.17% | 3.63% | 1.03% | 0.30% | 0.03% | 16.216/29.135 ms | 38.947/54.356 ms |
| v12_k50_hops_legalctx48_ensure_ksp | 3.61% | 3.87% | 1.85% | 1.31% | 0.42% | 0.03% | 16.352/29.654 ms | 39.637/55.302 ms |

## Table 2: Delta vs baseline_v12

| Method | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) | Δ Decision mean (ms) | Δ Decision P95 (ms) |
|---|---:|---:|---:|---:|---:|
| baseline_v12 | +0.00 | +0.00 | +0.00 | +0.000 | +0.000 |
| v12_k20_hops | -2.23 | -2.16 | +0.07 | +5.820 | +25.410 |
| v12_k30_hops | -2.79 | -2.67 | +0.05 | +14.244 | +53.034 |
| v12_k50_hops_all_legal | -3.44 | -3.02 | -0.21 | +32.944 | +114.263 |
| v12_k50_hops_legalctx48 | -1.21 | -0.96 | -0.03 | +4.663 | -1.262 |
| v12_k50_hops_legalctx48_ensure_ksp | -2.57 | -2.74 | +0.26 | +5.353 | -0.316 |

## Table 3: Blocking-Latency Trade-off

| Method | Blocking | Decision mean | Decision P95 | mean ms / 1pp blocking | P95 ms / 1pp blocking |
|---|---:|---:|---:|---:|---:|
| baseline_v12 | 6.19% | 34.284 ms | 55.618 ms | nan | nan |
| v12_k20_hops | 3.96% | 40.104 ms | 81.028 ms | 2.62 | 11.42 |
| v12_k30_hops | 3.40% | 48.528 ms | 108.653 ms | 5.11 | 19.03 |
| v12_k50_hops_all_legal | 2.75% | 67.228 ms | 169.881 ms | 9.58 | 33.24 |
| v12_k50_hops_legalctx48 | 4.98% | 38.947 ms | 54.356 ms | 3.85 | -1.04 |
| v12_k50_hops_legalctx48_ensure_ksp | 3.61% | 39.637 ms | 55.302 ms | 2.08 | -0.12 |

## Findings & Recommendation

### 1. k = 20 → 30 → 50 是否存在收益递减？

Yes. 从 baseline 到 k=20 的 blocking 降幅最大（-2.23 pp），而时延增幅相对可控（+5.8 ms mean，+25.4 ms P95）。继续扩到 k=30 再降 0.56 pp，但 mean decision time 再涨 7.4 ms；扩到 k=50 all_legal 只再降 0.65 pp，mean decision time 却几乎翻倍（+18.7 ms）。边际收益明显递减。

### 2. legalctx48 是否显著降低时延？

是。`v12_k50_hops_all_legal` mean/P95 decision time 为 67.2 ms / 169.9 ms；切换到 `legalctx48` 后降到 38.9 ms / 54.4 ms，接近 baseline 水平。但代价是 blocking 仅改善 -1.21 pp（4.98%），说明纯 legalctx48 会漏掉一些高价值候选。

### 3. ensure_ksp_action 是否几乎不损失 blocking，却能逼近 all_legal？

是。`legalctx48_ensure_ksp` 把 blocking 从 4.98% 进一步压到 3.61%，同时 decision mean/P95 仍只有 39.6 ms / 55.3 ms。与 k=20 hops（3.96% blocking，40.1 ms / 81.0 ms）相比，ensure_ksp 在相近时延下 blocking 更好，且 P95 明显更低。

### 4. 推荐配置

**推荐 `v12_k50_hops_legalctx48_ensure_ksp` 作为 v1.3 默认在线候选集方案。**

理由：
- 在 k=50/hops/start_asc 的宽视野下，只评估 48 个精选候选 + 强制 KSP-FF 动作，决策时延接近 baseline。
- blocking 从 6.19% 降到 3.61%，NSB 从 4.59% 降到 1.85%，效果接近 k=20/30 全候选，但 P95 时延更低。
- 每降低 1 pp blocking 仅额外付出约 2.08 ms mean decision time，性价比最优。

### 5. 推荐方案相对 baseline_v12 的收益与代价

| 指标 | baseline_v12 | 推荐方案 | 变化 |
|---|---:|---:|---:|
| Blocking | 6.19% | 3.61% | **-2.57 pp** |
| NSB | 4.59% | 1.85% | **-2.74 pp** |
| Overload | 1.05% | 1.31% | +0.26 pp |
| Decision mean | 34.28 ms | 39.64 ms | +5.35 ms |
| Decision P95 | 55.62 ms | 55.30 ms | -0.32 ms |

Overload 微增可能是 k=50/hops 让更多长路径候选进入，导致 server 选择压力略增；但总体 blocking 仍大幅改善。后续如要压 overload，应配合 C-side 优化（见 COST239 诊断），而不是继续扩 R 候选集。