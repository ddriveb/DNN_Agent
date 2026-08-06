# N1 迁移依赖闭包图（DEPENDENCY_GRAPH）

审计日期：2026-08-06。方法：对 15 个入口模块做 AST 递归本地 import 闭包
（脚本 `/mnt/d/project/tmp/n1_audit/import_closure.py`，机器可读结果
`/mnt/d/project/tmp/n1_audit/import_closure.json`），再由人工语义复核。

源路径前缀：`/mnt/d/project/DNN_Agent/sa_hmarl/sa_hmarl/`（下文省略）。

## 1. 六个主入口及其闭包规模

| # | 入口 | 模块 | 闭包文件数 | 第三方依赖 |
|---|---|---|---:|---|
| 1 | N1 训练 | `neural_opportunity.run_train` | 4 | numpy, torch |
| 1b | N1 采集 | `neural_opportunity.run_collect` | 44 | numpy, scipy, networkx, torch |
| 2 | N1 selector | `neural_opportunity.selector` | 19 | numpy, scipy, networkx, torch |
| 2b | N1 确认评估 | `neural_opportunity.run_confirmatory` | 44 | numpy, scipy, networkx, torch |
| 3 | N1-FUSE0 selector | `neural_opportunity.n1_fuse0.selector` | 20 | numpy, scipy, networkx, torch |
| 3b | N1-FUSE0 bench | `neural_opportunity.n1_fuse0.run_fuse0_bench` | 44 | 同上 |
| 4 | N1/KSP 公平比较 | `fair_comparison_xlron.run_n1_vs_ksp_xlron` / `run_ksp_k5_k50` | 17 / 5 | numpy, networkx, torch |
| 5 | N1/Teacher 比较 | 由 `run_confirmatory`（Teacher=OpportunityOnlyDirectSelector，Full Direct=ExactOptimizedDirectCompressedSketchSelector）覆盖，见 2b | — | — |
| 6 | N1-100 XLRON 训练/比较 | `n1_100.run_n1_100_xlron` / `compare_new_topologies`（及 run_n1_100 / run_compare_100 / run_confirmatory_100 / run_compare_xlron_topologies 旧线） | 44–49 | 同上 + scipy |

**15 个入口的并集 = 71 个 Python 文件。** 第三方依赖全并集：
`numpy, scipy, networkx, torch`（pytest 仅测试）。闭包内**无** pandas/yaml/tqdm/matplotlib import。

## 2. 分层依赖结构

```
experiments 层 (run_*.py, compare_*.py, n1_fuse0/run_fuse0_bench.py, fair_comparison_xlron/run_*.py)
   │
N1 学生层   neural_opportunity/{model,features,selector,dataset,training,protocol}.py
            neural_opportunity/n1_100/{model,features,selector,fused_selector,dataset,training,protocol,topologies}.py
            neural_opportunity/n1_fuse0/selector.py
            fair_comparison_xlron/n1_adapter.py
   │
Teacher/定价栈 (rollout_lab + dseo)
   direct_sketch_exact_optimization/selectors_phaseA.py   ← _DirectStaticRoutes（两个学生 selector 的基类）
                                                          ← OpportunityOnlyDirectSelector（Opportunity Teacher）
   direct_sketch_exact_optimization/run_phaseA.py         ← _prewarm_routes / _bootstrap_ci / _wtl（建议抽取）
   rollout_lab/feasible_window_sketch.py                  ← sketch + Full Direct + _legal_starts_from_word
   rollout_lab/bitparallel_path_opportunity.py            ← bit-parallel opportunity pricing
   rollout_lab/path_opportunity_pricing.py                ← 精确机会定价基类（唯一 scipy 运行依赖）
   rollout_lab/incremental_shadow_pricing.py              ← 增量影子价格
   rollout_lab/demand_shadow_pricing.py                   ← 影子价格基座
   rollout_lab/compiled_ksp_ff.py                         ← PythonKSPFFSelector（公平 KSP-FF 后端）
   │
环境层 (pds_rmsa + network + pure_rmsa_v13.core)
   pure_rmsa_v13/core.py                    execute() 物理执行
   pure_rmsa_v13/train_proposer.py          make_env()（仅此函数被需要；建议抽取）
   pds_rmsa/protocol.py                     Request / required_fs / MODULATION_TABLE
   pds_rmsa/env/{rmsa_env,spectrum,topology,traffic_trace,reservations}.py
   pds_rmsa/baselines/ksp_ff.py             朴素参考 KSP-FF
   network/{ksp,topology_data}.py + topology_data_usnet_gcnrmsa.json
```

XLRON 公平比较子图（自包含，不依赖 rollout_lab）：
`fair_comparison_xlron/{env,ksp_ff,traffic}.py` → 仅 `network/topology_data.get_topology_edges` + networkx；
`n1_adapter.py` → `n1_100.{features,model}`。

## 3. 共享模块（被 ≥9 个入口共享）

`pds_rmsa/env/{rmsa_env,spectrum,topology,traffic_trace,reservations}.py`、`pds_rmsa/protocol.py`、
`network/{ksp,topology_data}.py`、`pure_rmsa_v13/{__init__,core}.py`、`train_proposer.py`、
rollout_lab 7 文件中的 6 个（除 block_heuristics）、`selectors_phaseA.py`。
这些是新仓库的"公共地基"，任何签名改动会同时击穿全部入口。

## 4. 私有字段耦合（迁移高风险面）

- `env._cached_k_paths(src,dst,k)`：被 `core.py:53`、`features.py`(50/100 各 3 处)、
  `compiled_ksp_ff.py:114`、`demand_shadow_pricing.py`(4 处)、`path_opportunity_pricing.py:58`、
  `feasible_window_sketch.py:489`、`selectors_phaseA.py:65`、`run_phaseA._prewarm_routes` 使用；
  `fair_comparison_xlron/n1_adapter.py:57` 还**自定义同名私有方法**以满足该私有契约。
- `NeuralOpportunitySelector`（50/100 两代）继承私有基类 `_DirectStaticRoutes`（selectors_phaseA.py），
  依赖其 `_init_route_state/_precompute_static_routes/_detect_changed_arcs/_mark_versions_seen/_blocked_reasons`。
- Teacher 采集（`dataset.py`，50/100）直接触达
  `teacher._detect_changed_arcs/_scan_candidates/sketch.sync_state/sketch.pricer._arc_words_int`。
- `feasible_window_sketch.py` 深度访问两个 pricer 的 15+ 个 `_` 缓存字段；
  跨模块下划线 import：`_free_runs/_window_count`、`_popcount_u64`、`_legal_starts_from_word`、
  `run_phaseA._prewarm_routes/_bootstrap_ci/_wtl`。
- 结论：这批文件必须作为**同一包整体迁移、同步演进**，不能拆分发布或单独重构。

## 5. 副作用依赖（AST 闭包假象，可切断）

并集 71 文件中的约 20 个并非真实运行依赖，唯一污染源是
`train_proposer.py` 的模块级 import（`agents.ppo_agents._MaskedPPOBase`、
`pds_rmsa.training.trainers` 常量、`import torch`）：

- `agents/{action_feature_builders,c_agent,ppo_agents,r_agent}.py`、`network/modulation.py`、
  `env/r_frag_aware.py`：模块级副作用 import，N1 从不调用其中符号；
- `env/{fs_demand,mean_field,request}.py`、`pds_rmsa/training/potential_shaping.py`：
  函数级 lazy import，AST 假象，import 时根本不执行；
- `pds_rmsa/{training/*,features/afterstate,networks/mlp}.py`：经 `pds_rmsa/training/__init__.py`
  副作用放大进入闭包，N1 只需要 `trainers.py` 的 4 个常量。

**切断方案**：把 `make_env()`（train_proposer.py:24-47，25 行）抽取为独立
`env_factory.py` 后，以上 ~20 个文件全部移出闭包，纯 NumPy 推理不再需要 torch。

## 6. block_heuristics 的特殊情况

`rollout_lab/block_heuristics.py` 唯一闭包入边是 `run_phaseA.py:41` 的顶层 import，
其符号仅用于 N1 从不调用的 `ksp_bf_k50` arm。对 `run_phaseA.py` 执行
EXTRACT_MINIMAL_HELPER（抽 `_prewarm_routes/_bootstrap_ci/_wtl` 共 ~30 行）后，
`block_heuristics.py` 连同其测试一并移出闭包，rollout_lab 保留文件从 7 降至 6。

## 7. 外部数据/文件依赖

- `network/topology_data_usnet_gcnrmsa.json`：`topology_data.py:546` 运行时读取（`__file__` 相对，可移植）；
  缺失时 USNET 边表**静默为空**（不报错）。
- 权重 npz：`model.py:92`（50/100 两处）`np.load`，路径由调用方给；无代码内硬编码。
- Phase-A fact table：`run_confirmatory.py` 断言用
  `pure_rmsa_v13/artifacts/direct_sketch_exact_optimization/phaseA_confirmatory_10seed/per_seed.csv`（跨区依赖，须随迁）。
- rollout_lab 闭包内 6 个文件**自身零文件 I/O**。
- 全候选区活代码 **0 处硬编码绝对路径**；唯一两处 `/mnt/d` 在
  `fair_comparison_xlron/FAIR_COMPARISON_REPORT.md:76-77`（文档复现命令，发布前 scrub）。

## 8. 动态 import / 环境变量

- 无 importlib 动态加载；唯一 `__import__('numpy')` 在 `run_ksp_k5_k50.py:76`（格式化输出，无害）。
- `SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL`：运行开关仅在 `feasible_window_sketch.py:41`、
  `incremental_shadow_pricing.py:29`；几乎所有 N1 runner 断言其 unset。
  **注意**：`compiled_ksp_ff.py` 的 C 路径**不看**该变量——只要 `_direct_sketch_kernel.so`
  可导入且 ≤64 槽，`CompiledKSPFFSelector` 自动走 C。因此 **`.so/.c/build 脚本一律不迁移**；
  不搬 .so 时三个 backend 常量自动落到 `"python"`，恰为协议锁定状态。
