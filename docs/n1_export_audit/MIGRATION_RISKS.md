# 迁移风险登记（MIGRATION_RISKS，rev2）

按严重度排序。R1–R5 为**复制前必须修复**；R6–R10 为迁移时/发布后必须控制。

> **rev2（业主裁决，2026-08-06）**：verdict 下调为 CONDITIONALLY_READY，
> R1/R2/R3/R5/R8 升格为阻断项 B1–B7（FINAL_DECISION.md §5），未闭环不得 Commit 1。
> 关键修正：
> - **阶段一不改包名**：原 `sa_hmarl.*` 路径原样复制，R1 的"包名改写"推迟到验证门
>   之后的独立重构（不再提议 `n1/`、`rmsa/`、`teacher/` 一次性改名）。R1 的 helper
>   抽取（make_env、run_phaseA 三函数）相应改为 Commit 4 动作；阶段一改以
>   **19 个临时 side-effect 文件随迁**保证 import 不断，Commit 4 再删。
> - **fail-closed 升格为阻断项 B2**：`topology_data_usnet_gcnrmsa.json` 缺失时
>   当前**静默返回空拓扑**——发布前必须改为抛明确异常并校验边表 SHA-256。
> - **术语二分**：只允许 "XLRON path-ordering"（仅路径池 (hops,tuple) 排序）与
>   "full XLRON parity"（另含 5-candidate rejection traffic，仅 fair_comparison_xlron），
>   禁止笼统的 "XLRON-consistent"（消除 R2/R9 的口径混用根源）。
> - **测试不依赖 Release checkpoint**：FUSE0/wide-bitmap 所需 **4 个**权重
>   （50-slot 三拓扑 + n1_100 legacy nsfnet）以
>   TEST_FIXTURE_SMALL 身份入 `tests/fixtures/`；正式部署 checkpoint 才走 Release。
> - **结果定位收紧**：旧 hops 池结果（含 CONFIRMATORY_250）一律标
>   exploratory/archived；主矩阵（compare_new_topologies）重新发布前须补
>   checkpoint SHA + trace hash + route-pool hash。
> - **许可证先行**：LICENSE/NOTICE/THIRD_PARTY.md 是第一次公开 push 的前置；
>   german17、`network/modulation.py`（仅阶段一临时）、未审文献锁文档均不迁。
> - **归档线代码不迁**：run_n1_100/run_compare_100/run_confirmatory_100 只留结果摘要；
>   run_frozen_pilot 于 Commit 4 并入 run_confirmatory。

## R1. import 路径破坏（阻断性）

N1 不是自包含的，存在两类闭包外硬依赖，直接复制 KEEP 清单文件会得到 import error：

1. `train_proposer.py` 模块级 import `agents.ppo_agents._MaskedPPOBase`（私有）、
   `pds_rmsa.training.trainers`、`torch`——而 N1 只需要其中 25 行的 `make_env()`。
   **修复**：抽取 `make_env` 到 `rmsa/env_factory.py`，保持
   `path_sort_strategy="hops"` 默认（make_env='hops' vs `PDSRMSAEnv` 构造默认='km'，
   绕过 make_env 直接建 env 会改变路径序，违反锁定协议）。
2. `run_phaseA.py` 顶层 import 了 N1 不用的 arm 依赖——N1 只需要
   `_prewarm_routes/_bootstrap_ci/_wtl`（~30 行）。
   **修复**：抽取到 `experiments/phaseA_helpers.py`，签名原样保留。
3. 迁移后包名从 `sa_hmarl.pure_rmsa_v13...` 变为新根包，所有绝对 import 需机械改写；
   `n1_100/` 及其 `tests/` 无 `__init__.py`（PEP 420 命名空间子包），打包或改结构会断 import——
   显式保留 PEP 420 或补 `__init__.py`，二选一。
4. `n1_100/protocol.py` import 即执行 `register_extra_topologies()` 改全局 registry
   （`import protocol  # noqa: F401` 触发）；迁移时改为显式注册调用，否则依赖 import 顺序。

## R2. 路径排序语义变化（行为漂移，最隐蔽）

- `path_sort_strategy` 三值语义：`xlron` = (hops, tuple)，无 km tie-break；
  `hops` = (hops, km, tuple) 且收集同跳数边界内全部路径再截断；`km` = (km, hops, tuple)。
  同一 K 下三种池在稠密拓扑（COST239）上差异显著。
- 现行主线口径：50-slot 全部经 make_env → **hops**；N1-100 训练/评估 → **xlron**；
  归档线（run_compare_100/run_confirmatory_100/run_compare_xlron_topologies）= hops 池
  评 xlron 训练模型（口径混用，归档原因）。fair_comparison_xlron/env.py 内联等价 xlron 排序。
- 迁移必须锁定：每个 runner 显式传 `path_sort_strategy` 并在 manifest 记录；
  禁止依赖任何地方的双默认（'km'/'hops' 不一致）隐式行为。

## R3. 50/100-slot 位图行为差异

- 50-slot：uint64 word；100-slot：**object-dtype Python 大整数**（`_full_word_int=(1<<100)-1`，
  `int.to_bytes(13,'little')`+`np.unpackbits` 解码）。曾评估 uint64 双字方案因 numpy 小数组
  调用开销被放弃（n1_100/selector.py:84-90 注释）。
- 高 64–99 槽路径只有 `n1_100/tests/test_wide_bitmap.py` 保障（250E 轨迹几乎碰不到 ≥64 槽）。
- 该测试与 test_fuse0 均**依赖 artifacts 权重 npz + cwd 相对路径**——权重不入 Git 则测试红；
  迁移时必须把 fixture 权重（RELEASE_ASSET 中 3 个）与路径改写一并完成。
- `np.bitwise_count` 要求 numpy≥2.0，requirements 须写明。

## R4. KSP 后端不公平风险

- 公平口径锁定：`PythonKSPFFSelector`（`python_bitparallel_early_exit`）。
  `CompiledKSPFFSelector` 在 .so 存在时**无视环境变量**自动走 C（≤64 槽）——
  因此 `_direct_sketch_kernel.{c,so}` 与 build 脚本**坚决不迁移**。
- `fair_comparison_xlron/ksp_ff.py` 为 cumsum 向量化全路径扫描后取首条可行
  （决策等价 early-exit，延迟口径≠AGENTS.md §8 参考）；正式延迟声明只能引用
  pds_rmsa 环境的 `PythonKSPFFSelector` 数字（NSFNET 0.0128/0.0197ms @K5/K50 等）。
- 旧 `1.13x–1.19x` Direct/KSP 延迟比（legacy 全候选物化 KSP）不得再引用。

## R5. checkpoint schema 不匹配

- `load()` 强校验 protocol_id + feature_schema_id；`training_v1/` 权重无这两个键，
  **当前代码无法加载**——按目录名直迁会把坏的当好的。以 SHA-256 为准
  （50-slot 部署权重=training_conflict_v1，nsfnet=`7d4436a0…`）。
- 结果文件未记录权重 hash（见 R6），迁移后为每个发布权重在 README 钉 SHA-256。
- `n1_100/run_confirmatory_100.py` 重跑会以新口径（225E/61921-61925）
  `os.replace` **覆盖**磁盘上 250E 正式档案且文件名（CONFIRMATORY_250.json）继续撒谎——
  归档线迁移后应改输出名并加防覆盖。

## R6. trace / route-pool provenance 缺失

- 仅 run_confirmatory_100 记录 model_sha256；无任何 N1 runner 记录 trace hash 或
  route-pool hash；`traffic_trace.trace_sha256()` 现成但无人调用。
- CONFIRMATORY_RESULTS.json / FROZEN_PILOT_RESULTS.json 未记录所用权重——
  现行结果只能靠时间线推断权重身份。
- 迁移后 runner 统一写：checkpoint SHA-256、trace_sha256、route-pool（K+sort+topology）hash、
  protocol_id、backend 常量。

## R7. 私有字段耦合（结构性，迁移后须冻结接口）

清单见 DEPENDENCY_GRAPH.md §4：`env._cached_k_paths`（12+ 处跨模块）、
`_DirectStaticRoutes` 私有基类、`teacher.sketch.pricer._arc_words_int`、
`feasible_window_sketch` 对 pricer 15+ 个 `_` 字段的访问、跨模块下划线 import。
这批文件必须同包整体迁移；迁移后第一件事是把被外部依赖的私有成员登记为
"冻结接口"（或转正为公开 API），否则任何重构静默打挂 Teacher/学生一致性。

## R8. 外部许可证

仓库无任何 LICENSE；XLRON（MIT）衍生拓扑数据与 clean-room 复刻代码将随发布再分发。
复制前必须：选定本项目许可证、撰写 NOTICE（XLRON MIT 全文 + OFC2024/JOCN2025 引用 +
TopologyBench 引用）、保留源码注释中的 source URL/commit。FAIR_COMPARISON_REPORT.md
含 `/mnt/d` 复现命令需 scrub。

## R9. protocol post-hoc 修改与文档漂移

- `n1_100/PHASE_N1_100_PROTOCOL.md` 记录的种子段/负载（613xx/250E）已被
  2026-08-06 重校准取代（618xx/619xx/620xx、225E 等），文档未更新——以
  EXPERIMENT_MATRIX.md / EXPERIMENT_REPORT.md 为准，旧文档仅存档。
- gate 放宽（recall 0.80→0.20、regret 0.10→0.25）已在 EXPERIMENT_REPORT 披露，
  但 `run_n1_100.py` 的 gate 键名仍是旧值；`LATENCY_MEAN_MS_GATE=0.060` 定义了却无脚本
  强制执行（N1 实测 88–153μs 超 gate）。
- `run_confirmatory_100.py` docstring 种子（61701-61705）与实际常量（61921-61925）不一致。
- confirmatory_100 warmup 1000 vs 训练 3000、holding_truncation None vs 训练 2.0：
  口径漂移需在报告中显式声明。
- "XLRON-consistent" 双轨：run_n1_100_xlron 实际用 make_env 的 hard-resample 截断，
  真正的 XLRON 5-candidate rejection 只在 fair_comparison_xlron/traffic.py。

## R10. 环境/杂项

- 两套调制表并存：`pds_rmsa/protocol.py`（BPSK 10000km…16QAM 625km，N1 用）vs
  `network/modulation.py`（4000km 表，N1 不用）；迁移只带前者，防止串用。
- `NUM_SLOTS` 在 6+ 处分别定义（50 与 100）、`K_PATHS` pds_rmsa=5 vs N1=50 并存——
  新仓以各 protocol.py 为唯一来源，删除重复定义。
- `protocol.py`（50-slot）含死配置 DAGGER_SEEDS/MAX_DAGGER_ROUNDS（无实现）；
  `neural_opportunity/selector.py` 有死 import `_legal_starts_from_word`——迁移时顺手清理。
- `core.execute()` 对非法 action 是 raise AssertionError：selector 输出必然合法这一
  不变量必须保持，否则 parity 语义破坏。
- requirements.txt 只有 8 项最小版本、无钉死；N1 闭包实际需要且仅需要
  numpy≥2 / scipy / networkx / torch / pytest。
- 全仓 `pure_rmsa_v13` 未被 git 跟踪——迁移即复制工作树，无历史；同时确认
  不携带任何 `__pycache__`/`.prof`。

## 迁移后必须运行的 parity 测试清单（验收标准 10）

1. `tests/n1/test_core.py`（model/training/protocol 单测，含 numpy↔torch parity、
   checkpoint round-trip）——环境无关，首先跑。
2. `tests/rmsa/test_core.py` + `test_phase_a1.py`（env/execute/baseline 行为不变）。
3. `tests/xlron/test_{env,ksp_ff,traffic}.py`（19 项：调制表、required_slots、KSP hops 序、
   截断分布 CDF 对照理论值）。
4. `tests/teacher/test_{demand_shadow_pricing,path_opportunity_pricing}.py`
   （Teacher 定价回归；无 .so 时 native 用例 skip）。
5. `tests/fuse0/test_fuse0.py`（fused 手算价格 allclose 1e-3 + action 0-mismatch +
   静态表只读）——需 fixture 权重。
6. `tests/n1_100/test_wide_bitmap.py`（高 64–99 槽 parity、`_ksp_first_fit_python`、
   `_legal_starts_from_word` 随机对照）——需 fixture 权重。
7. 冒烟：`run_closed_loop_smoke.py`（单种子 3 臂）+ `fuse0_bench.py` parity 段
   （3 拓扑×3 种子×10k 决策 0 mismatch），两者均断言 native env var unset。
8. 数值锚点复核：50-slot confirmatory 前 2 个种子（23001/23002）的 N1 与 KSP-FF
   per-seed blocking 与 `results/n1_50/neural_per_seed.csv` 完全一致；
   KSP blocked 逐种子等于 `assets/fact_tables/phaseA_confirmatory_10seed_per_seed.csv`。
