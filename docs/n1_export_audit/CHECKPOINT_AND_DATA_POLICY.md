# Checkpoint 与数据发布策略（CHECKPOINT_AND_DATA_POLICY，rev2）

原则：大型 checkpoint/dataset 默认不入 Git；必要 checkpoint 走 GitHub Release
（首选，体量小）或 Git LFS；拓扑数据文件随源码入 Git；第三方衍生数据需 NOTICE。

> **rev2（业主裁决）**：**测试 fixture 与正式部署 checkpoint 身份分离**——
> FUSE0/wide-bitmap 测试所需的 **4 个**权重（50-slot training_conflict_v1 三拓扑 +
> n1_100 legacy nsfnet，合计 ~68KB）以 TEST_FIXTURE_SMALL 身份直接入
> `tests/fixtures/`；正式部署 checkpoint（5 个 N1-100 权重，及 50-slot 三权重的
> Release 副本）以 RELEASE_ASSET 身份经 GitHub Release 发布（SHA-256 钉死），
> 不进 Git 历史。同一文件可有两种身份，但存放位置与用途严格分开。
> 另：`topology_data_usnet_gcnrmsa.json` 的加载必须 fail-closed（缺失/哈希不符即
> 抛异常），不得再静默返回空拓扑。

## 1. 必要 checkpoint（冻结口径：9 unique / 4 fixture copies / 8 release copies）

**数量口径（rev4 冻结，任何文档不得再出现其他算法）**：

- **9 unique checkpoints**：50-slot 三拓扑（training_conflict_v1）+ 100-slot 五拓扑
  （n1_100_xlron）+ n1_100 legacy nsfnet，合计 ~170KB；只有这 9 个是独立模型。
- **4 fixture copies**：50-slot 三拓扑 + n1_100 legacy nsfnet 的副本，
  以 TEST_FIXTURE_SMALL 身份直接入 `tests/fixtures/`（Git 内）。
- **8 release copies**：5 个 N1-100 权重 + 3 个 50-slot 权重的副本，
  以 RELEASE_ASSET 身份经 GitHub Release 发布（SHA-256 钉死），不进 Git 历史。
- **双重身份**：3 个 50-slot 权重同时拥有 fixture 与 release 两种身份；
  n1_100 legacy nsfnet 仅 fixture 身份（归档线不发布）。
- V1 验证分三层：(a) 9 unique 源文件；(b) 4 fixture 副本；(c) 8 release-staging
  副本（V8B 时改为真实 Release 下载）——副本不得计为独立模型。

发布方式：**GitHub Release 附件**（每个 ~11–21KB，无需 LFS）；
README 提供下载 + SHA-256 校验指令。

### 50-slot N1（schema：`pure_rmsa_neural_opportunity_v1` / `neural_opportunity_conflict_pressure_map_v1`）

| 拓扑 | 源路径（artifacts/ 下） | 大小 | SHA-256 |
|---|---|---:|---|
| nsfnet | `training_conflict_v1/nsfnet/deployment_weights.npz` | 11,174 B | `7d4436a0dad4330a5ece731793c741c709854e24745446c764957c15cd7ada60` |
| usnet | `training_conflict_v1/usnet/deployment_weights.npz` | ~11 KB | `c3cdd0dc40435aed6966b9d7665654635831ded309f89632c613aeb27648871f` |
| jpn48 | `training_conflict_v1/jpn48/deployment_weights.npz` | ~11 KB | `2ecdbd46e37bad2361dcc0eace8906e8f2922fae27bbb22ca5882985f9bf6949` |

nsfnet 权重 SHA 与 `causal_selective_rerank` 钉死的 `N1_DEPLOYMENT_SHA256` **完全一致**——
这是真正的部署权重；`training_v1/` 是 `load()` 拒绝的旧格式，禁止误迁。

### 100-slot N1-100（schema：`pure_rmsa_neural_opportunity_n1_100_v1` / `neural_opportunity_conflict_pressure_map_100slot_v1`）

| 拓扑 | 源路径 | SHA-256 |
|---|---|---|
| cost239 | `n1_100_xlron/cost239/training/deployment_weights.npz` | `725c6101dd5cc7cf410399ec165ce2a7d9696b60090f559c17ff9086354d2737` |
| abilene | `n1_100_xlron/abilene/training/deployment_weights.npz` | `63404aab6f0401150d3ada50f1b69e2823bad45c77093388b2a2e9be0722ad0d` |
| nsfnet | `n1_100_xlron/nsfnet/training/deployment_weights.npz` | `c891af753890ab50b5efba37dc03021804533c962cb3f8cec83fef85ce1f4dc3` |
| jpn48 | `n1_100_xlron/jpn48/training/deployment_weights.npz` | `b1b24db06b108d22bcaf3b4e5891fb94ff83880d03fa7cd3165ecfdb94ec7639` |
| usnet | `n1_100_xlron/usnet/training/deployment_weights.npz` | `f6b0d3acd7ee608d610acda32b1eb97b1afb5fe2701b4f509db47b9b10f9ec75` |
| （归档）nsfnet legacy | `n1_100/nsfnet/training/deployment_weights.npz` | `ab6ce1a248332e7907780f8148cc70ad1ec0a0ed6b6da687c022c92fd203a2ee` |

german17 权重（`b6f86646…`）：该线无对比结果，**业主裁决不迁、不发布**（EXCLUDE_LEGACY）。

### 权重 schema（npz 键）

`protocol_id, feature_schema_id, w1(FEATURE_DIM,16)[转置], b1(16,), w2(16,NUM_SLOTS)[转置，已乘 target_scale], b2(NUM_SLOTS,)`。
`TinyOpportunityWeights.load()` 对两个 ID **强校验**，不匹配即 raise——
代码常量与全部已发布权重绑定，改 ID 等于作废全部权重。
`training_checkpoint.pt`（torch）只写不读，不发布（REGENERATABLE_EXCLUDE）。

## 2. 训练数据

| 数据 | 大小 | 策略 |
|---|---:|---|
| `datasets_conflict_v1/`（50-slot teacher 数据集，12 npz） | 66M | SOURCE_REQUIRED；**不入 Git**。建议冷存储或 Release 打包；可由 `run_collect.py --mode train/validation` 按锁定种子再生成（极慢） |
| `n1_100_xlron/*/datasets/`（100-slot） | ~485M | REGENERATABLE_EXCLUDE；`run_n1_100_xlron.py` 再采集 |
| `n1_100/nsfnet/datasets/` | 36M | 同上 |
| 数据集元数据缺陷 | — | 两代数据集（v1 / conflict_v1）元数据逐字段相同、无 `feature_schema_id`，**无法从文件自身区分**；新仓数据目录命名必须带 schema 后缀 |

## 3. 拓扑与事实表（随源码入 Git）

| 文件 | 策略 | 许可证风险 |
|---|---|---|
| `network/topology_data_usnet_gcnrmsa.json` | SOURCE_REQUIRED，随包入 Git | 含 XLRON commit `d07980b3` 与 source_url；XLRON 为 MIT，需 NOTICE；缺失时 USNET **静默变空边表** |
| `network/topology_data.py` 内联边表（NSFNET/COST239/German17/JPN48） | 随源码 | 衍生自 XLRON/TopologyBench 数据；NOTICE + 引用 TopologyBench/Real-Topologies |
| `n1_100/topologies.py`（Abilene） | 随源码 | 注释含 XLRON raw URL；NOTICE |
| `phaseA_confirmatory_10seed/per_seed.csv`（fact table） | SOURCE_REQUIRED，入 `assets/fact_tables/` | 本项目自产，无风险 |

## 4. 结果摘要（RESULT_SUMMARY_SMALL，全部入 Git，合计 <1.5M）

- `results/n1_50/`：CONFIRMATORY_RESULTS.json（**verdict=HOLD**，非全拓扑通过——文档勿写 GO）、
  FROZEN_PILOT_RESULTS.json、per_seed csv、LATENCY_CEILING.json（final=GO；
  注意 locked 轮=HOLD，勿混）、3×TRAINING_RESULTS.json。
- `results/n1_100/`：5 拓扑 CALIBRATION_RESULTS.json + COMPARE_RESULTS.json + per_seed.csv
  （现行 2026-08-06 <1% 校准口径）；`backup_5_8pct/` 旧口径结果仅存档（SUMMARY_ONLY）。
- `results/fuse0/`：FUSE0_RESULTS.json（parity 0 mismatch；fused 0.038–0.044ms vs
  original 0.045–0.065ms vs KSP-FF 0.018–0.024ms）。
- `results/xlron/`：3 组 fair-comparison json/csv。
- `results/n1_ksp_fairness_audit_20260802/`：整目录（含审计脚本）。
- provenance 缺口：CONFIRMATORY/FROZEN_PILOT 结果**未记录权重路径/hash**；
  仅 run_confirmatory_100 记录 model_sha256；`trace_sha256()` 已存在但无 runner 使用。
  新仓 runner 应统一记录 checkpoint SHA + trace hash + route-pool hash（见 MIGRATION_RISKS R6）。

## 5. 许可证总览

- 本仓库（DNN_Agent）**根及子目录均无 LICENSE/COPYING**——发布 N1 前必须先为本项目代码选定许可证。
- XLRON 上游：**MIT, © Michael Doherty 2023**（github.com/micdoh/XLRON）。
  `fair_comparison_xlron/` 为 clean-room 语义复刻（无复制代码）；拓扑边表为 MIT 数据的衍生再发布。
- 必须动作：新仓 `NOTICE` 收录 XLRON MIT 许可证全文 + 引用 XLRON OFC 2024 论文、
  Doherty 2025 (JOCN "Hype or Hope")、TopologyBench/Real-Topologies；保留源码注释中的
  source URL 与 commit hash。
- venv 未安装 xlron 包，无运行时第三方依赖。
