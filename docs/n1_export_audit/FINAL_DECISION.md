# 最终裁决（FINAL_DECISION，rev2）

审计日期：2026-08-06。rev2 并入业主裁决：verdict 下调、阶段一不改包名、
归档线代码不迁、fixture 与 Release 身份分离、fail-closed、许可证先行。

配套文件：MIGRATION_MANIFEST.csv/json（217 行，rev2）、DEPENDENCY_GRAPH.md、
PROPOSED_REPO_LAYOUT.md（rev2）、EXCLUSIONS.md、CHECKPOINT_AND_DATA_POLICY.md、
MIGRATION_RISKS.md、**MIGRATION_PLAN.md（分阶段计划表）**。

## 1. 建议迁入多少 Python 文件

**71 个 Python 文件（阶段一，含 19 个临时文件）；Commit 4 瘦身后 52 个运行时文件。**

| 类别 | 文件数 | 说明 |
|---|---:|---|
| KEEP_CORE | 17 | N1 核心 50/100-slot + FUSE0 |
| KEEP_ENV | 11 | 最小 RMSA 环境 + topology + core |
| KEEP_BASELINE | 7 | 参考 KSP-FF + 公平后端 + XLRON 兼容环境 |
| KEEP_TRAINING | 6 | Opportunity Teacher + 定价栈 + selectors_phaseA |
| KEEP_EXPERIMENT | 15 | **14 个 runner + 1 个 adapter**（已剔除 3 条归档线） |
| KEEP_TEST | 15 | |
| KEEP_DOC | 6 | markdown |
| EXTRACT_MINIMAL_HELPER | 2 | 阶段一整迁，Commit 4 抽取 |
| 【临时】side-effect 链 | 19 | EXCLUDE_LEGACY 标注，Commit 1 随迁、Commit 4 删除 |

runner 明细（14）：run_collect、run_train、run_closed_loop_smoke、run_confirmatory、
run_frozen_pilot（Commit 4 并入 confirmatory）、run_latency_ceiling、run_fuse0_bench、
run_n1_100_xlron、compare_new_topologies、recalibrate_below_1pct、
build_experiment_matrix、run_ksp_k5_k50、run_n1_vs_ksp_xlron、run_truncation_compare。
合并 frozen_pilot 后为 13 个，符合 10–12 个主 runner + 工具脚本的量级。

## 2. 建议保留哪些实验

1. **N1 训练与 checkpoint 导出**：run_collect + run_train + run_closed_loop_smoke。
2. **N1 vs KSP-FF K5/K50**：50-slot run_confirmatory（含 Teacher/Full Direct 对照臂）；
   100-slot full-XLRON-parity 三 runner（fair_comparison_xlron）。
3. **N1 vs Opportunity Teacher / Full Direct**：run_confirmatory（+frozen_pilot 合并后为其 --pilot 模式）。
4. **N1-FUSE0 parity + latency**：run_fuse0_bench；100-slot 侧由 test_wide_bitmap 保障。
5. **100-slot XLRON 多拓扑矩阵**：run_n1_100_xlron + compare_new_topologies（**主矩阵**）+
   recalibrate_below_1pct + build_experiment_matrix。

**不保留代码、只留结果**：run_n1_100 / run_compare_100 / run_confirmatory_100
（hops 池归档线，结果标 exploratory/archived 入 `results/archived_n1_100_hops_pool/`）。

## 3. 哪些文件绝对不能迁入

- `rollout_lab/_direct_sketch_kernel.{c,so}` + build 脚本 + 其 MD（破坏公平口径，R4）。
- `artifacts/training_v1/`（`load()` 拒绝的旧格式，名字像主线，R5）。
- `n1_100_xlron/german17/`（死实验线，业主裁决不迁）。
- `network/modulation.py` 最终版（仅阶段一临时；两套调制表防串用）。
- `N1_BLOCKING_LATENCY_LITERATURE_SEARCH_LOCK.md`（未审，业主裁决不迁）。
- 三条归档线 runner（见上）、已关闭实验分支整体、~775M 大型 artifacts、
  旧 README/KIMI_CODE_CONTEXT/过时 PROTOCOL_LOCK、外层残留目录、
  `__pycache__`/`*.prof`/临时脚本。

## 4. 是否已经形成最小闭包

**是（分两阶段呈现）**。阶段一闭包 71 文件（含 19 个临时 side-effect 文件），
Commit 4 执行两个 EXTRACT_MINIMAL_HELPER 后收敛到约 52 个运行时文件；
闭包外候选依赖逐模块 grep 复核为零引用。

## 5. 阻断项（未全部完成前不得 Commit 1）

| # | 阻断项 | 完成判据 |
|---|---|---|
| B1 (R8) | **许可证先行**：撰好 LICENSE + NOTICE + THIRD_PARTY.md | 三个文件入库；含 XLRON MIT 全文与全部 attribution |
| B2 (R5) | **fail-closed**：`topology_data.py` 在 usnet JSON 缺失/哈希不符时抛明确异常 | 缺失 JSON 时 `load_topology("xlron_usnet_gcnrmsa")` raise，且错误信息含预期 SHA-256 |
| B3 (R1) | 阶段一复制清单冻结（本 manifest rev2）+ 原路径复制验证脚本就绪 | 复制脚本 dry-run 通过 |
| B4 (R2) | 每个 runner 的 `path_sort_strategy`/K/num_slots 显式锁定值登记入 protocol 注释 | 无隐式默认依赖 |
| B5 (R3) | `tests/fixtures/` **四个**权重就位（50-slot 三拓扑 + n1_100 legacy nsfnet，SHA-256 核验），两个测试改指 fixture | test_fuse0、test_wide_bitmap 在新仓绿 |
| B6 (R5) | 权重身份以 SHA-256 复核（50-slot=training_conflict_v1，nsfnet=`7d4436a0…`） | 核验记录入库 |
| B7 | FAIR_COMPARISON_REPORT.md 中 /mnt/d 路径 scrub；术语二分（XLRON path-ordering vs full XLRON parity）写入 README | 文档无绝对路径、无 "XLRON-consistent" 笼统表述 |

## 6. 结果定位（收紧后）

- **主矩阵（formal）**：`compare_new_topologies.py` 的 XLRON path-ordering 口径结果
  （5 拓扑 COMPARE_RESULTS.json + per_seed.csv），重新发布前补 checkpoint SHA +
  trace hash + route-pool hash（`trace_sha256()` 现成）。
- **formal（50-slot）**：CONFIRMATORY_RESULTS.json（verdict=HOLD，勿写 GO）、
  FUSE0_RESULTS、xlron 三组、latency_ceiling_final。
- **exploratory / archived**：旧 hops 池全部结果（CONFIRMATORY_250 等，
  存在 gate 事后放宽、warmup 1000≠3000、holding_truncation None≠2.0 漂移）、
  backup_5_8pct、PHASE_N1_100_PROTOCOL.md。
- 禁止引用：legacy 1.13x–1.19x 延迟比、latency_ceiling_locked（HOLD）与 final（GO）混用。

## 7. 验收标准逐项核对

10 项标准全部满足（同 rev1，明细略；见 MIGRATION_RISKS.md 与 DEPENDENCY_GRAPH.md），
但 **R1/R2/R3/R5/R8 以阻断项形式挂起**，由第 5 节 B1–B7 闭环。

## 8. Verdict

# **READY_FOR_LOCAL_MIGRATION**

（rev4，业主最终裁决 2026-08-06；`V8B` 通过后才是 **PUBLICATION_COMPLETE**。）

rev4 三项执行级修正已全部落实：

1. **V8 拆分**：`V8A`（推送前，本地 clean clone + 本地 release-staging 权重包验收）→
   首次 push `main` → 建 tag/GitHub Release 上传 8 个发布权重 →
   `V8B`（远程 clean clone + 真实 Release 下载复验）。消除了"先 Release 还是先 push"
   的先后矛盾。
2. **root commit 生成**：改用**独立干净本地仓库**（`git init` 新仓 + 复制 staging 最终
   工作树 + 单次提交），机械核验四条：`rev-list --count main == 1`、唯一提交
   parent 数 == 0、与 `migration-staging` 无共同祖先、临时文件在 main tree 零命中。
3. **checkpoint 口径冻结**：`9 unique / 4 fixture copies / 8 release copies`；
   3 个 50-slot 权重双重身份（fixture + release）；V1 分三层核验，副本不计为独立模型。

执行链（MIGRATION_PLAN.md rev4）：Phase 0（B1–B7）→ Commit 1–3（staging）→
V1–V7 首轮 → Commit 4（瘦身）→ V1–V7 复跑 → 独立干净仓库 root commit →
V8A → 首次 push → tag/Release → V8B → PUBLICATION_COMPLETE。
