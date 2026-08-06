# N1 迁移分阶段计划表（MIGRATION_PLAN，rev4）

核心原则：**迁移与重构不同时做**。先原路径复制、验证 0-mismatch，再瘦身。
前置阻断项 B1–B7 见 FINAL_DECISION.md §5，全部完成前不动工。

> **rev4（业主裁决 2026-08-06，READY_FOR_LOCAL_MIGRATION 的前置修正）**：
> 1. **V8 拆分为 V8A/V8B**，消除"Release 下载"与"首次 push"的先后矛盾：
>    V8A 在推送前用**本地 release-staging 权重包**做 clean-clone 验收；
>    首次 push、建 tag/Release、上传权重后，再以 V8B 做**真实远程**验收。
> 2. **root commit 生成改为独立干净仓库**（推荐方案），并给出四条机械核验
>    （rev-list=1、parent=0、与 staging 无共同祖先、临时文件零命中）。
> 3. **checkpoint 口径冻结**：`9 unique checkpoints` / `4 fixture copies` /
>    `8 release copies`（5 个 N1-100 + 3 个 50-slot Release 副本；50-slot 三权重
>    同时拥有 fixture 与 release 两种身份）；V1 分三层分别核验，不把重复副本
>    算成独立模型。
>
> 历史裁决（仍有效）：临时文件不得进入最终 Git 历史（Commit 1–4 仅在本地
> `migration-staging`，staging 永不推送）；fixture 为四权重。

## 完整执行链

```text
Phase 0：完成 B1–B7（含 rev4 三项修正的落地）
  ↓
本地 migration-staging
  Commit 1：原路径代码、环境、许可证
  Commit 2：测试与四个 fixture
  Commit 3：结果摘要与复现入口
  ↓
V1–V7 首轮 parity
  ↓
Commit 4：抽 helper、删 19 个临时文件、合并 runner
  ↓
再次执行 V1–V7
  ↓
独立干净仓库生成单个 root commit（main）
  ↓
V8A：本地 clean-clone 全量验收（本地 release-staging 权重包）
  ↓
首次 push main → git@github.com:ddriveb/N1.git
  ↓
创建 tag 和 GitHub Release，上传 8 个发布权重
  ↓
V8B：远程 clean-clone + 真实 Release 下载验收 → PUBLICATION_COMPLETE
```

## 阶段总览

| 阶段 | 位置/分支 | 内容 | 完成判据 |
|---|---|---|---|
| **Phase 0** | 旧仓/工作区 | 阻断项 B1–B7 | B1–B7 逐条签核 |
| **Commit 1** | `migration-staging`（本地） | 原路径复制 + 环境/许可证 | import 冒烟通过 |
| **Commit 2** | `migration-staging`（本地） | 测试与四个 fixture | 全部单测绿 |
| **Commit 3** | `migration-staging`（本地） | 结果摘要与复现脚本 | 结果可读、脚本可跑 |
| **验证门 V1–V7（首轮）** | `migration-staging`（本地） | 旧/新仓库 0-mismatch 对拍 | action/blocking/route hash 全一致 |
| **Commit 4** | `migration-staging`（本地） | 抽 helper、删临时文件、合并 runner | 瘦身后 V1–V7 仍全过 |
| **Root commit** | 独立干净本地仓库 | staging 最终工作树 → 单个无父 root commit | 四条机械核验全过 |
| **V8A** | 本地干净目录 | 本地 clean-clone + release-staging 权重包验收 | 见 V8A 判据 |
| **首次推送** | `main` → GitHub | 只推 `main` | 远程仅含干净历史 |
| **Release 发布** | GitHub | 建 tag + Release，上传 8 个发布权重 | 权重 SHA-256 与台账一致 |
| **V8B** | 远程干净目录 | 从 GitHub clone + 真实 Release 下载复验 | 见 V8B 判据 → **PUBLICATION_COMPLETE** |

## Phase 0：阻断项（在旧仓/工作区完成）

| # | 动作 | 位置 | 备注 |
|---|---|---|---|
| P0-1 | 撰写 LICENSE / NOTICE / THIRD_PARTY.md | 新仓根 | XLRON MIT 全文 + OFC2024/JOCN2025 + TopologyBench attribution |
| P0-2 | `network/topology_data.py` 改 fail-closed：sidecar JSON 缺失时 raise（含预期 SHA-256），加载后校验边表 hash | 旧仓小修，随 Commit 1 带入 | 当前静默返回空拓扑，发布仓不可接受 |
| P0-3 | 冻结复制清单（manifest rev2 已冻结）+ 写复制脚本（含 `__pycache__`/`.prof` 过滤、SHA 校验） | tmp 工具 | dry-run 核对 217 行 |
| P0-4 | runner 锁定参数登记：逐 runner 写明 num_slots/K/path_sort_strategy/truncation/warmup/seeds | 新仓 docs/protocols/ | 消除 'km'/'hops' 双默认依赖 |
| P0-5 | 制备 `tests/fixtures/` **四权重**并核 SHA-256（nsfnet=`7d4436a0…`）；同时制备**本地 release-staging 权重包**（8 个发布权重 + SHA256SUMS） | 工作区 | fixture 身份≠Release 身份；release-staging 包供 V8A 使用 |
| P0-6 | scrub FAIR_COMPARISON_REPORT.md 的 /mnt/d 路径；README 写入术语二分 | 随 Commit 1 带入 | "XLRON path-ordering" vs "full XLRON parity" |

## Commit 1（migration-staging）：原路径复制 + 环境/许可证

| 项 | 内容 |
|---|---|
| 复制 | manifest 中全部 KEEP_* + EXTRACT_MINIMAL_HELPER + 19 个【临时】文件 + SOURCE_REQUIRED（usnet JSON、fact table csv），**路径与原仓完全一致**（`sa_hmarl/...`） |
| 新增 | LICENSE、NOTICE、THIRD_PARTY.md、README.md（新写）、requirements.txt（numpy≥2/scipy/networkx/torch/pytest） |
| 不复制 | 全部 EXCLUDE/LEGACY 行（除标注【临时】者）、`_direct_sketch_kernel.*`、german17、大 artifacts |
| 验证 | 锁定 venv 下 `PYTHONPATH=. python -c "import sa_hmarl.pure_rmsa_v13.neural_opportunity.selector"` 等 6 入口 import 冒烟；`SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL` 保持 unset |

注：19 个【临时】文件只存在于 staging 历史；root commit 所在的干净仓库历史不含它们（见下）。

## Commit 2（migration-staging）：测试与小型 fixture

| 项 | 内容 |
|---|---|
| 复制 | 15 个 KEEP_TEST 文件（已在 Commit 1 包树内者除外）+ `tests/fixtures/` **四权重** |
| 改写（仅 2 处，行为保持） | `test_fuse0.py`、`test_wide_bitmap.py` 的权重路径 → `tests/fixtures/`（`__file__` 相对） |
| skip guard | `test_path_opportunity_pricing.py` 的 `_direct_sketch_kernel` 用例在无 .so 时 skip |
| 验证 | `pytest sa_hmarl/pure_rmsa_v13/neural_opportunity sa_hmarl/pure_rmsa_v13/fair_comparison_xlron sa_hmarl/pure_rmsa_v13/tests sa_hmarl/pds_rmsa/tests sa_hmarl/pure_rmsa_v13/rollout_lab/test_demand_shadow_pricing.py -q` 全绿 |

## Commit 3（migration-staging）：结果摘要与复现脚本

| 项 | 内容 |
|---|---|
| 复制 | 33 个 RESULT_SUMMARY_SMALL + 8 个 SUMMARY_ONLY（入 `results/` 与 `results/archived_n1_100_hops_pool/`）+ 6 份 KEEP_DOC |
| 标注 | archived 目录 README 写明 "exploratory / archived：gate 事后放宽、warmup/truncation 口径漂移"；主矩阵（compare_new_topologies）标 formal |
| 复现脚本 | `scripts/reproduce.md`：每实验一条命令（锁定种子/负载/路径池口径），并列出待补的 checkpoint/trace/route-pool hash 记录项 |
| 验证 | 结果文件 JSON 可解析；EXPERIMENT_MATRIX 引用的数字与 results/ 一致 |

## 验证门 V1–V7（staging 上，Commit 3 之后首轮；Commit 4 之后复跑）

| # | 对拍项 | 方法 | 判据 |
|---|---|---|---|
| V1 | checkpoint 身份（**三层分别核验**，冻结口径 9/4/8） | (a) 9 个 unique checkpoint 源文件 SHA-256 对台账；(b) `tests/fixtures/` 4 个 fixture 副本 SHA-256 对台账；(c) 本地 release-staging 包 8 个 release 副本 SHA-256 对 SHA256SUMS | 三层各自全一致；副本不计为独立模型 |
| V2 | 50-slot action parity | 旧仓 vs 新仓各跑 `run_closed_loop_smoke`（同种子同 trace）+ `run_fuse0_bench` parity 段 | action 签名 0-mismatch（3 拓扑×3 种子×10k 决策） |
| V3 | 50-slot blocking parity | 旧/新各跑 confirmatory 前 2 种子（23001/23002） | per-seed blocking 与 `results/n1_50/neural_per_seed.csv` 逐值一致；KSP blocked 逐种子等于 fact table |
| V4 | route-pool hash | 对 6 拓扑×锁定口径（hops/xlron, K=50）计算路由池（路径集合+顺序）SHA-256，旧/新对拍 | 全一致 |
| V5 | 100-slot parity | 旧/新各跑 test_wide_bitmap + 1 种子 compare_new_topologies（nsfnet，seed 62021） | 高位槽 0-mismatch；blocking 一致 |
| V6 | KSP 公平后端 | 确认后端常量为 `python_bitparallel_early_exit`、.so 不存在 | backend 字段=python |
| V7 | trace hash | 对锁定种子调用 `trace_sha256()` 旧/新对拍 | 一致 |

**任一不过：停在原地排查，不得进入下一阶段。**

## Commit 4（migration-staging）：瘦身重构

| # | 动作 | 影响 |
|---|---|---|
| C4-1 | 抽 `make_env()` → `pure_rmsa_v13/env_factory.py`（保持 'hops' 默认），runner 改 import | 删 agents/4 + env/r_frag_aware + network/modulation |
| C4-2 | 抽 `_prewarm_routes/_bootstrap_ci/_wtl` → 保留模块；删 `run_phaseA.py`、`rollout_lab/block_heuristics.py` | rollout_lab 收敛到 6 文件 |
| C4-3 | 删 pds_rmsa/{training,features,networks} 临时文件 | 运行时闭包 → 约 52 文件 |
| C4-4 | `run_frozen_pilot.py` 折叠为 `run_confirmatory.py --pilot` | runner 14→13 |
| C4-5 | `register_extra_topologies()` 改显式调用；可选补 `n1_100/__init__.py` | 消除 import 副作用/PEP 420 依赖 |
| C4-6 | 复跑验证门 V1–V7 | 全过才允许生成 root commit |

## 生成 root commit：独立干净仓库（推荐）

**不得在 `migration-staging` 所在仓库内 `git checkout -b main`**——那会继承
staging 历史，产生有父提交。采用独立干净仓库：

```bash
git init /path/to/N1-public            # 全新的、与 staging 仓库无关联的本地仓库
# 将 migration-staging 的最终工作树（不含 .git）完整复制进 /path/to/N1-public
cd /path/to/N1-public && git add -A && git commit -m "N1: initial public release"
```

（备选：`git checkout --orphan main` + 清空 index 后重建；但必须通过下列核验。）

机械核验（四条全过才算完成）：

```text
git rev-list --count main == 1                       # 唯一提交
git cat-file -p main | grep -c '^parent ' == 0       # 无父提交（root）
git merge-base main migration-staging 为空/命令失败    # 与 staging 无共同祖先（独立仓库天然满足）
git ls-files | grep -c -E 'sa_hmarl/agents/|pds_rmsa/(training|features|networks)/|network/modulation\.py|block_heuristics|r_frag_aware' == 0
                                                     # 19 个临时文件在 main tree 中零命中
```

`migration-staging` 保留在本地备查，**永不推送**。

## V8A：本地 clean-clone 验收（首次 push 前）

| # | 步骤 | 判据 |
|---|---|---|
| V8A-1 | `git clone /path/to/N1-public /tmp/n1-clean`（干净空目录） | clone 成功 |
| V8A-2 | 全程**不得**引用 `/mnt/d/project/DNN_Agent`（环境变量、PYTHONPATH、配置文件均不含） | 环境检查为零 |
| V8A-3 | 用 **P0-5 制备的本地 release-staging 权重包**（8 个发布权重 + SHA256SUMS）模拟 Release 下载流程，按 README 指令放置并校验 SHA-256 | 与 CHECKPOINT_AND_DATA_POLICY 台账一致 |
| V8A-4 | 在干净 clone 中运行：6 入口 import smoke、全量 pytest、route-pool/trace hash 对拍、action/blocking parity（V2–V5、V7 的干净环境版） | 全部通过 |
| V8A-5 | 全仓搜索 `/mnt/d`、`D:\`、`D:/`、`/home/lds`、旧仓依赖字符串 | **命中数为零** |

**V8A 全过 → 才允许 `git push origin main`（首次公开推送，只推 `main`）。**

## Release 发布（push 之后、V8B 之前）

1. 在 `main` 上打 tag（如 `v1.0.0`）并推送 tag；
2. 创建 GitHub Release，上传 **8 个发布权重**（5 个 N1-100 + 3 个 50-slot Release 副本）
   及 `SHA256SUMS`；
3. README 的下载指令指向该 Release。

## V8B：远程 clean-clone 验收（发布后）

| # | 步骤 | 判据 |
|---|---|---|
| V8B-1 | 从 `git@github.com:ddriveb/N1.git` 重新 clone 到另一干净目录 | clone 成功；`git rev-list --count HEAD` = 1 |
| V8B-2 | 全程不引用 `/mnt/d/project/DNN_Agent` | 环境检查为零 |
| V8B-3 | 按 README **实际从 GitHub Release 下载** 8 个发布权重并校验 SHA-256 | 与台账一致 |
| V8B-4 | 复跑 import smoke、全量 pytest、关键 parity（V2/V3/V5 的干净环境版）与 route/trace hash | 全部通过 |
| V8B-5 | 全仓搜索绝对路径与旧仓依赖字符串 | **命中数为零** |

**V8B 全过 → PUBLICATION_COMPLETE。**

## 后续（不在本迁移窗口）

- 主矩阵重新发布：为 compare_new_topologies 补 checkpoint SHA + trace hash + route-pool hash 后重跑 10 种子。
- 包结构重命名（`n1/`、`rmsa/`、`teacher/`）：独立 PR，附 import 改写与完整回归。
- 私有接口转正（`_cached_k_paths`、`_DirectStaticRoutes`、pricer 内部字段）：与 Teacher/学生同步演进。
- german17 如需复活：单独校准+训练+评估，不入本次。
