# N1 vs KSP-FF 公平性独立审计报告

**Audit ID:** `n1_ksp_fairness_audit_20260802`
**Date:** 2026-08-02
**Verdict: PASS_WITH_REPORT_CORRECTIONS**

---

## Findings（按严重性排序）

### F1 [HIGH] TCHR 线的 delta 符号误读已确认；N1/KSP 线未受影响

- 证据：`tchr_rl/evaluator.py` `paired_deltas` 计算 `treat − base`（正 = treatment 更差）。`eval_v2_calibration/EVALUATION.json` 的 `paired_delta_n1_vs_tchr_joint = +0.490 pp` 意味着 **TCHR 比 N1 阻塞多 0.49pp（TCHR 更差）**，而此前的 TCHR 报告（含 `V2_FREEZE.md` 的 calibration_result 表）将其读成"TCHR 击败 N1（5/5 seeds）"——方向完全颠倒。
- 复核：逐 seed blocking（63101: n1=0.06370 vs tchr=0.07050）确认 TCHR 更高（更差）；validation 选型（v3−v2 = −0.208 → v3 好）方向正确。
- 污染范围：**仅 TCHR 侧文字/表格解读**；N1/KSP 的 `CONFIRMATORY_RESULTS.json` 使用 `(N1 − fact)×100`（负 = N1 好，run_confirmatory.py L109-131 核实），方向正确，报告解读正确（"CI 全负"→beats KSP）。
- 修正动作（不覆盖旧产物）：TCHR 侧报告/冻结文档需出修正记录；`run_select_tchr` 等字段名（`mean_delta_pp_v3_minus_v2`、`paired_delta_n1_vs_tchr_joint`）须带显式公式。

### F2 [MEDIUM] Provenance 缺口：无 trace hash、无历史 artifact hash

- `phaseA_confirmatory_10seed/` 与 `direct_vs_ksp_k5_k50_python_early_exit_10seed_v1_20260801/` 两目录**无 request trace hash / action hash / 存档 SHA-256**（全仓 grep NOT FOUND；仅 experiments/ 下 deeprmsa 等旧协议有 `request_trace_hash`）。
- `deployment_weights.npz` 的 SHA-256 为审计新算（见 AUDIT_RESULTS.json），**无历史 hash 可对比**；冻结时间线（checkpoint 12:13 → deployment 12:17 → pilot 12:30 → confirmatory 12:40）支持"冻结先于 confirmatory"的声明，但这是 mtime 链证据，不是 hash 证据。
- 处置：如实标注为 provenance 风险，**不伪造历史 hash**。补 hash 属新协议/新目录工作。

### F3 [MEDIUM] N1 最终报告时延措辞偏松

- "The final method is within roughly 2x of KSP" —— 实测 N1/KSP = 2.77/2.28/1.91x，nsfnet 的 2.77 不是 "roughly 2x"。应改为 "within 2-3x"。"same sub-0.05-ms order of magnitude"（0.0448 vs 0.0162，均几十微秒）成立。

### F4 [LOW] 报告引用 KSP 时延数字未注明 run 来源

- N1 报告引用 0.0162/0.0195/0.0250（A 组 = revalidation fact table），而 20260801 修正报告为 0.0197/0.0230/0.0279（B 组）。两者同实现同 seed 同 blocking（+0.00%），差异（-15~-20%，全方法全指标同幅度）归因于机器/运行时刻（PROTOCOL_AUDIT.md 已记录）。论文引用须注明取自哪次 run，或并列报告机器偏移。

### F5 [LOW] 字段名无显式公式（全仓）

- 审计要求"delta/gain/retention 类字段必须带显式公式"——`run_select_tchr`（`mean_delta_pp_v3_minus_v2`）、`tchr_rl/evaluator.py`（`paired_delta_n1_vs_tchr_joint`）、`direct_sketch_residual_rl/statistics.py`（`mean_delta_pp` 约定 B）等字段名未在 schema/报告中写明方向。本次审计已在测试中固化公式字符串（test_audit_fairness.py `FORMULA`）。

---

## 一、阻塞公平性审计 — PASS

从原始 per-seed 行独立重算（`audit_recompute_blocking.py` → `PAIRWISE_BLOCKING_RECOMPUTE.csv`）：

| topology | delta_pp (N1−KSP, 负=N1好) | 配对 t-CI | 存档 bootstrap CI | W/T/L | seeds | CI 全负 |
|---|---|---|---|---|---|---|
| NSFNET | **-1.412** | [-1.598, -1.226] | [-1.569, -1.263] | 10/0/0 | 10/10 | ✓ |
| USNET | **-0.892** | [-1.070, -0.714] | [-1.039, -0.745] | 10/0/0 | 10/10 | ✓ |
| JPN48 | **-1.065** | [-1.281, -0.849] | [-1.242, -0.885] | 10/0/0 | 10/10 | ✓ |

- 与 `CONFIRMATORY_RESULTS.json` 存档 delta_pp 逐位一致（abs<1e-9）
- 报告宣称的 -1.412/-0.892/-1.065 **精确可复现**
- 公平性要素核实：同 topology/load/seed/warmup(1000)/eval(10000)/trace/50 slots/同一 K=50 hops 路径池/路径序 hops→km→tuple；KSP 为 First-Fit 最低合法 start、bit-parallel early-exit、不调用 `env.build_candidates()`（`PythonKSPFFSelector.select` 源码核实）；`ksp_ff_backend=python_bitparallel_early_exit` 两侧一致
- seed 隔离：train 60001-60003 / valid 60101 / smoke 60201 / pilot 60301-60305 / confirmatory 23001-23010 严格不相交（protocol.py 核实）

## 二、时延公平性审计 — PASS（详见 LATENCY_RECONCILIATION.md）

- 三套 KSP 数字全部定位出处：A=revalidation fact table（N1 报告引用）、B=20260801 修正报告、C=latency ceiling（seed 60201, warmup 300, eval 5000）
- A vs B：同实现同 seed 同 blocking，全指标 -15~-20% 机器偏移 → 非实现差异
- 计时边界两侧一致（selector-only、init 排除、mean 分母含 warmup、percentiles 用 eval 段——两侧 runner 源码核实）
- 单 worker、native kernel unset（manifest 记录）、无 GPU 计时、无旧 build_candidates 混入
- 1.91-2.77x 时延比公平且可复现（±15-20% 机器波动内）

## 三、N1 算法身份审计 — PASS

`selector.py`：first-three-feasible（`MAX_FEASIBLE_PATHS=3`）→ 110 维特征（10 标量 + 50 common-free + 50 conflict-pressure，`FEATURE_DIM=10+2×50`）→ 共享 `110→16→50` ReLU MLP（`TinyOpportunityWeights`）→ 非法 start 严格 mask（`=np.inf`）→ `argmin(path_rank + neural_price)` → tie-break 确定（path-major、行内 start 升序，`np.argmin` 取首个）。在线**无** exact Opportunity / Shadow pricer / compressed sketch / rollout / GNN / Teacher fallback / PyTorch / C kernel。

## 四、行为与 trace parity — 部分证据

- blocking 逐 seed 一致（A vs B：+0.00%，PROTOCOL_AUDIT.md 9/9 项）
- **trace/action hash 不存在**（F2）——"同 blocking 异轨迹被误称 parity"无法排除，也无法证实；标注为证据缺口，新审计重跑未补充 hash（属新协议工作）

## 五、历史问题复核 — 全部确认

| # | 主张 | 复核结果 |
|---|---|---|
| 1 | 旧 1.13-1.19x 来自 legacy materialized KSP | ✓ 确认（20260801 报告 L77-79 原文；legacy 目录保留） |
| 2 | 当前 KSP 是 bit-parallel early-exit | ✓ 确认（backend 字段 + 源码） |
| 3 | C kernel 不可与 Python KSP 直接比 | ✓ 确认（AGENTS.md；manifest native=unset；compiled 目录独立） |
| 4 | TCHR +0.490 误读 | ✓ 确认（F1；未污染 N1 线） |
| 5 | N1 vs Teacher +0.023/+0.002/+0.041 | ✓ 复现（存档值） |
| 6 | N1 vs FD +0.210/+0.009/+0.201 | ✓ 复现（存档值） |
| 7 | retention 87.1/99.0/84.1% | ✓ 公式 `(mean_KSP−mean_N1)/(mean_KSP−mean_FD)` 从原始行重建一致 |
| 8 | retention 分母/方向 | ✓ 无零分母；方向 = N1 保留 KSP→FD 增益的比例 |
| 9 | "same order of magnitude" 措辞 | ⚠️ 绝对量级成立；"roughly 2x" 偏松（F3） |
| 10 | N1 是监督 amortized pricing | ✓ 确认（Huber price 回归 + KL + margin；非 RL 非动作分类蒸馏） |

## 六、测试

`test_audit_fairness.py`：**20 passed**（delta 方向、paired CI 双方法全负、W/T/L、seed 完整性、KSP early-exit 无 build_candidates、native kernel unset、timing scope 源码断言、deployment shape/schema、retention 重建、legacy 隔离、TCHR 符号方向）。输出见 `PYTEST_OUTPUT.txt`。

## Verdict 依据

- 阻塞公平性：PASS（独立重算精确复现、方向正确、CI 全负、seed 完整）
- 时延公平性：PASS（边界一致、三组数字归因闭合）
- Provenance：有缺口（无 hash），但**无不一致**（原始数据与报告逐位吻合）→ 按指令"缺 hash 标注风险"，不构成 FAIL_PROVENANCE
- 需要修正的报告文字：F1（TCHR 侧）、F3、F4、F5

→ **PASS_WITH_REPORT_CORRECTIONS**
