# N1-100 — 100-slot Neural Opportunity（协议记录 + 结果）

**Status: 对比实验完成。** 结构 = N1 的 110→16→50 在 100 slots 下的扩展（210→16→100），
监督目标 = Opportunity Teacher 稠密价格，损失 = price + 0.5 listwise + 0.5 pairwise。

## 协议要素
- PROTOCOL_ID: `pure_rmsa_neural_opportunity_n1_100_v1`
- FEATURE_SCHEMA_ID: `neural_opportunity_conflict_pressure_map_100slot_v1`
- 结构: features(210) → relu → hidden(16) → price(100)；部署单个 NumPy 网络
- 拓扑: nsfnet（最快拓扑），K=50 hops 路径池
- load: **100 → 250 Erlang（2026-08-03 协议修订）**——100-slot 频谱过剩（100 Erlang 下 KSP 与 N1 均 0 阻塞，对比无区分度）；负载扫描（150-400）确定 250 Erlang 为区分度区间（KSP 阻塞 ~9%）。同负载重新采集+重训。
- seed: train 61301-61303 / valid 61350 / smoke 61399 / diagnosis 61401-61403
  （全新段，与全部历史 split 不相交；formal 23001-23010 未触）
- 训练: 30000 样本（3×10000，Opportunity-only continuation），60 epochs，AdamW lr 1e-3
- 部署: w2×target_scale 折叠，target_scale=164.1

## 结果（100 slots, nsfnet, 250 Erlang —— 修订后正式配置，同负载）

### 训练（250-Erlang 同负载重训）
- 采集：4 seeds 各 ~9200 样本（Teacher 阻塞 ~8%，blocked 状态跳过）
- 训练：60 epochs；validation regret 0.140、top1 recall 0.351、target_scale 147.0
- （100-Erlang 版本 regret 0.019/recall 0.447——拥挤状态定价更难，recall 下降符合预期）

### 对比（61401-61403，同 trace 配对）
| seed | parity | KSP blk | N1 blk | Δpp | 相对降 | N1/KSP 时延 |
|---|---|---|---|---|---|---|
| 61401 | 0 | 0.09700 | 0.08490 | −1.21 | 12.5% | 2.95× |
| 61402 | 0 | 0.08570 | 0.07810 | −0.76 | 8.9% | 2.93× |
| 61403 | 0 | 0.08620 | 0.07720 | −0.90 | 10.4% | 3.30× |
| 汇总 | 0/30000 | 0.0896 | 0.0801 | **−0.96 pp** | **10.6%** | **3.06×** |

- OOD 参考（100-Erlang 模型直接 250 评估）：−1.14 pp/12.6%——同负载重训后略降但同量级
- 与 50-slot（100 Erlang）对比：阻塞优势 10.6% vs 15.0%（略小）；时延 3.06× vs 1.69×（更大，网络 210→16→100）

### 正式 confirmatory（2026-08-03 审计修复后，全新 seeds 61701-61705）
- mean_delta = **−0.782 pp**，95% 配对 t-CI = **[−1.119, −0.445]**（完全 <0）
- N1 赢 **5/5** seeds，相对降 **8.86%**，时延比 **3.14×**
- provenance 完整：model SHA-256 `ab6ce1a248332e79...`、native=None、load 250、slots 100、warmup 1000/eval 10000
- 注：之前 61401-3 的 −0.957pp/10.6% 有种子复用偏差（61401-3 曾用于压力测试），
  现以 61701-61705 为正式数字（方向稳定，量级略小）
- 50-slot NSFNET 同拓扑对照（61201-3, 100 Erlang）：**−1.587 pp / 19.69% / 2.22×**
  （修正此前混拓扑的 JPN48 口径；跨槽位对比仅作压力匹配参考）

### 诚实解读
- 训练: validation regret 0.0191、top1 recall 0.447、price MAE 12.0
- parity（FUSE0-100 vs plain N1-100）: **0/30000 mismatches**
- **阻塞: KSP-FF K50 与 N1 在 61401-61403 全部 0.00000**（10000 请求全接入）——100 slots
  频谱充足，对比无区分度，相对降低无定义（分母 0）
- 时延（selector-only, 单 worker, native unset）: KSP 0.0171ms vs N1-FUSE0 0.0735ms
  = **4.30×**；FUSE0 相对 plain N1（0.0885ms）省 17%

## 诚实解读
1. **100 slots + 100 Erlang 是"频谱过剩"配置**：两种策略都零阻塞。N1 的阻塞优势
   （50-slot 下 -1.28pp/15%）在此配置下不可观测。
2. **时延成本真实**：100-slot 网络（210→16→100）比 50-slot 大，N1-FUSE0 4.30× KSP。
   若要评估 N1 在 100 slots 的价值，需提高负载（>100 Erlang）重新采集+训练
   （新配置，属协议修订）。
3. 基础设施交付：位图 >64 放宽（compiled_ksp_ff.py、bitparallel_path_opportunity.py，
   50-slot 路径行为不变，回归通过）、n1_100 包、FUSE0-100、对比 runner。
