# N1 新仓库布局（PROPOSED_REPO_LAYOUT，rev2）

> **rev2 变更（业主裁决）**：迁移第一阶段**不改包名**。新仓库保持
> `sa_hmarl.pure_rmsa_v13...` / `sa_hmarl.pds_rmsa...` 原始 import 路径，
> 先达成旧仓库/新仓库 0-mismatch；包结构重构（`n1/`、`rmsa/`、`teacher/` 改名）
> 是独立的后续提交，不在迁移窗口内做。迁移与重构不得同时进行。

目标远程：`git@github.com:ddriveb/N1.git`

## 阶段一布局（Commit 1–3，最终仓库的源码结构）

```text
N1/
├── README.md                     # 新写：N1 简介 + 复现指引（勿照搬旧仓 README）
├── LICENSE                       # 新写：第一次公开 push 前必须存在
├── NOTICE                        # XLRON MIT 全文 + OFC2024 / JOCN2025 引用
├── THIRD_PARTY.md                # XLRON（拓扑数据+clean-room 复刻）与 TopologyBench 明细
├── requirements.txt              # numpy>=2, scipy, networkx, torch, pytest（N1 闭包仅需这 5 个）
│
├── sa_hmarl/                     # 与原仓库完全同构的包树
│   ├── __init__.py
│   ├── pure_rmsa_v13/
│   │   ├── __init__.py  core.py  train_proposer.py    # train_proposer 阶段一整迁，Commit 4 抽 make_env
│   │   ├── tests/test_core.py
│   │   ├── neural_opportunity/
│   │   │   ├── __init__.py  protocol.py  model.py  features.py
│   │   │   ├── selector.py  dataset.py  training.py
│   │   │   ├── run_collect.py  run_train.py  run_closed_loop_smoke.py
│   │   │   ├── run_confirmatory.py  run_frozen_pilot.py  run_latency_ceiling.py
│   │   │   ├── PHASE_N1_PROTOCOL_LOCK.md  NEURAL_OPPORTUNITY_FINAL_REPORT.md
│   │   │   ├── tests/{__init__.py, test_core.py}
│   │   │   ├── n1_fuse0/{__init__.py, selector.py, run_fuse0_bench.py, README.md,
│   │   │   │             tests/{__init__.py, test_fuse0.py}}
│   │   │   └── n1_100/{protocol.py, model.py, features.py, selector.py,
│   │   │                 fused_selector.py, dataset.py, training.py, topologies.py,
│   │   │                 run_n1_100_xlron.py, compare_new_topologies.py,
│   │   │                 recalibrate_below_1pct.py, build_experiment_matrix.py,
│   │   │                 EXPERIMENT_MATRIX.md, EXPERIMENT_REPORT.md,
│   │   │                 tests/test_wide_bitmap.py}
│   │   ├── fair_comparison_xlron/
│   │   │   ├── __init__.py  env.py  ksp_ff.py  traffic.py  n1_adapter.py
│   │   │   ├── run_ksp_k5_k50.py  run_n1_vs_ksp_xlron.py  run_truncation_compare.py
│   │   │   ├── FAIR_COMPARISON_REPORT.md                  # 发布前 scrub /mnt/d 路径
│   │   │   └── tests/{__init__.py, test_env.py, test_ksp_ff.py, test_traffic.py}
│   │   ├── rollout_lab/            # 仅 6+2 个文件，不是整个 rollout_lab
│   │   │   ├── demand_shadow_pricing.py  incremental_shadow_pricing.py
│   │   │   ├── path_opportunity_pricing.py  bitparallel_path_opportunity.py
│   │   │   ├── feasible_window_sketch.py  compiled_ksp_ff.py
│   │   │   ├── block_heuristics.py                        # 【临时】run_phaseA 顶层 import 所需，Commit 4 删
│   │   │   └── test_{demand_shadow_pricing, path_opportunity_pricing}.py
│   │   └── direct_sketch_exact_optimization/
│   │       ├── selectors_phaseA.py  run_phaseA.py         # run_phaseA 阶段一整迁，Commit 4 抽 3 个 helper 后删
│   ├── pds_rmsa/
│   │   ├── protocol.py
│   │   ├── env/{__init__.py, rmsa_env.py, spectrum.py, topology.py,
│   │   │        traffic_trace.py, reservations.py}
│   │   ├── baselines/{__init__.py, ksp_ff.py}
│   │   ├── tests/{__init__.py, conftest.py, test_phase_a1.py}
│   │   ├── training/{__init__.py, trainers.py, agents.py, fast_afterstate.py,
│   │   │             pred_features.py, replay.py}         # 【临时】side-effect 链，Commit 4 删
│   │   ├── features/{__init__.py, afterstate.py}          # 【临时】Commit 4 删
│   │   └── networks/{__init__.py, mlp.py}                 # 【临时】Commit 4 删
│   ├── network/
│   │   ├── __init__.py  ksp.py  topology_data.py          # topology_data 须先改 fail-closed（见下）
│   │   ├── topology_data_usnet_gcnrmsa.json               # SOURCE_REQUIRED
│   │   └── modulation.py                                  # 【临时】side-effect 链，Commit 4 删
│   ├── agents/                                            # 【临时】4 文件，Commit 4 删
│   │   ├── __init__.py  action_feature_builders.py  c_agent.py  ppo_agents.py  r_agent.py
│   └── env/
│       ├── __init__.py  r_frag_aware.py                   # 【临时】Commit 4 删
│       # fs_demand/mean_field/request.py 不迁：lazy import，N1 从不执行
│
├── tests/
│   └── fixtures/                   # TEST_FIXTURE_SMALL（直接入 Git，合计 ~68KB）
│       ├── n1_50/{nsfnet,usnet,jpn48}/deployment_weights.npz      # test_fuse0 / smoke 用
│       └── n1_100_legacy/nsfnet/deployment_weights.npz            # test_wide_bitmap 用
│
├── assets/
│   └── fact_tables/phaseA_confirmatory_10seed_per_seed.csv        # run_confirmatory 断言用
│
├── results/                        # RESULT_SUMMARY_SMALL（全部 <1.5M，入 Git）
│   ├── n1_50/  fuse0/  xlron/  n1_100/  n1_ksp_fairness_audit_20260802/
│   └── archived_n1_100_hops_pool/  # SUMMARY_ONLY：旧 hops 池结果，标注 exploratory/archived
│
└── docs/
    ├── PHASE_N1_PROTOCOL_LOCK.md  NEURAL_OPPORTUNITY_FINAL_REPORT.md
    ├── EXPERIMENT_MATRIX.md  EXPERIMENT_REPORT.md  FAIR_COMPARISON_REPORT.md
    └── history/PHASE_N1_100_PROTOCOL.md           # 过期，仅存档
```

部署 checkpoint 的正式身份不进 Git 历史：5 个 N1-100 权重为 RELEASE_ASSET，
经 GitHub Release 发布（SHA-256 钉死，见 CHECKPOINT_AND_DATA_POLICY.md）；
**四个测试权重**（50-slot 三拓扑 + n1_100 legacy nsfnet）以 **fixture 身份**入
`tests/fixtures/`（50-slot 三权重同一 SHA-256 同时挂 Release），
两种身份与用途严格分开。

## Commit 4 瘦身（验证通过后的独立重构提交）

1. `train_proposer.py` → 抽 `make_env()` 为 `pure_rmsa_v13/env_factory.py`
   （保持 `path_sort_strategy="hops"` 默认），各 runner 改 import；
2. `run_phaseA.py` → 抽 `_prewarm_routes/_bootstrap_ci/_wtl` 保留，
   删除 `run_phaseA.py` 与 `rollout_lab/block_heuristics.py`；
3. 删除全部【临时】文件（agents/ 4、env/r_frag_aware、network/modulation、
   pds_rmsa/{training,features,networks}）共 19 个；
4. `run_frozen_pilot.py` 折叠为 `run_confirmatory.py --pilot`（评估后执行）；
5. `n1_100/protocol.py` 的 `register_extra_topologies()` 改为显式调用；
6. 可选：补 `n1_100/__init__.py`（消除 PEP 420 依赖）。

完成后运行时闭包从阶段一的 71 文件降到约 52 个，runner 从 14 个降到 13 个。

## 布局纪律

1. 阶段一**禁止**任何包名/目录名改动；所有 import 保持 `sa_hmarl.*` 原样。
2. 同名实现只留一份；不迁 `_direct_sketch_kernel.{c,so}` 与 build 脚本。
3. 不迁任何 `__pycache__`、`.pytest_cache`、`*.prof`。
4. 源仓 `pure_rmsa_v13` 未被 git 跟踪，迁移即复制工作树，无 git 历史。
5. "XLRON" 术语二分（写进 README 与 protocol 注释）：
   - **XLRON path-ordering**：仅路径池按 (hops, tuple) 排序（`path_sort_strategy="xlron"`）；
   - **full XLRON parity**：另含 5-candidate rejection traffic（仅 `fair_comparison_xlron/`）。
   禁止再笼统使用 "XLRON-consistent"。
