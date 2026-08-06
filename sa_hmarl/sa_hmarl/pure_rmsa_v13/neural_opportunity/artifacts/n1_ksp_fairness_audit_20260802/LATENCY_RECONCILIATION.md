# KSP Latency Reconciliation — A / B / C 三组数字

## 三组数字及其原始出处

| 组 | KSP-FF K=50 (ms) | 出处 | seeds | warmup/eval | backend |
|---|---|---|---|---|---|
| **A** | 0.0162 / 0.0195 / 0.0250 | `artifacts/direct_sketch_exact_optimization/phaseA_confirmatory_10seed/per_seed.csv`（2026-08-02 09:44 revalidation run；N1 最终报告引用此组） | 23001-23010 | 1000/10000 | python_bitparallel_early_exit |
| **B** | 0.0197 / 0.0230 / 0.0279 | `artifacts/neural_pricing_kernel_v2_residual/direct_vs_ksp_k5_k50_python_early_exit_10seed_v1_20260801/`（2026-08-01 修正报告原跑） | 23001-23010 | 1000/10000 | python_bitparallel_early_exit |
| **C** | 0.0210 / 0.0193 / 0.0247 | `neural_opportunity/artifacts/latency_ceiling_conflict_v1/LATENCY_CEILING.json` | **60201**（非 23xxx） | **300/5000** | python_bitparallel_early_exit |

## A vs B：同实现、同 seed、同计时边界 → 机器/运行时刻差异

PROTOCOL_AUDIT.md（confirmatory revalidation 时生成）对 A 与 B 逐项对比：

- blocking：**+0.00% 差异**（9/9 方法-seed 全等）——同一 trace、同一行为
- selector_ms / end_to_end_ms / init_s：**全部方法、全部指标同时快 15-20%**（ksp_k5 -18.2%、ksp_k50 -18.0%、full_direct -19.6% 等）
- 全指标同幅度偏移 = 机器负载/CPU 频率/运行时刻的系统性差异，**不是实现不一致**（两侧均为 python_bitparallel_early_exit，源码同一类 `PythonKSPFFSelector`）

结论：A 与 B 都有效，但**不能混用**；报告引用时须注明取自哪个 run。A 组（revalidation）是 N1 最终报告采用的一组。

## C：不同 seed、不同配置 → 不可直接比较

C 是 latency ceiling gate（seed 60201、warmup 300、eval 5000）的测量，目的只是验证 N1 ≤ 0.060 ms 门禁（passes_absolute_gate=true），不是论文主表数字。nsfnet 的 0.0210 高于 A/B（0.0162/0.0197）而 usnet 的 0.0193 低于 B（0.0230）——seed 与配置差异的正常波动，不构成三组冲突。

## 计时边界（两侧代码核实）

- `rollout_lab/eval_direct_sketch_study.py` `_evaluate`（L106-134）：`started = perf_counter()` 只包 `selector.select`；`advance_external` 在外；init（route/selector）单独计时
- `direct_sketch_exact_optimization/run_phaseA.py` `evaluate_arm`（L157-188）：同一结构
- 均值分母 = warmup + eval（含 warmup 请求）；p50/p95/p99 用 eval 段
- 单 worker、native kernel unset（manifest 记录）

## 对 N1/KSP 时延比（1.91-2.77x）的判定

- 公平性：PASS（边界一致、同实现、同 trace、单 worker、native unset）
- 可复现性：在 ±15-20% 机器波动内可复现（A/B 两组即是该波动的上下界）
- 精确性：引用时须注明数字来自 A（revalidation fact table）还是 B（20260801 原跑），不可无来源引用
- 措辞修正：N1 最终报告 "within roughly 2x of KSP"（实测 2.77x）应改为 "within 2-3x"
