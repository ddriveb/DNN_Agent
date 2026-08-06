# 排除清单（EXCLUSIONS，rev2）

逐类说明不迁入 N1 新仓库的内容及原因。行级明细见 MIGRATION_MANIFEST.csv
（rev2：EXCLUDE_LEGACY 56 行、LEGACY_EXCLUDE 10 行、REGENERATABLE_EXCLUDE 19 行）。

> **rev2 变更**：(1) 三条归档线 runner（`n1_100/run_n1_100.py`、`run_compare_100.py`、
> `run_confirmatory_100.py`）由 KEEP_EXPERIMENT 改为 EXCLUDE_LEGACY——只留结果摘要
> （标 exploratory/archived），代码不迁；(2) german17、`network/modulation.py`、
> `N1_BLOCKING_LATENCY_LITERATURE_SEARCH_LOCK.md` 经业主裁决不迁；(3) 19 个
> side-effect 链文件（agents/ 4、env/r_frag_aware、network/modulation、
> pds_rmsa/{training,features,networks} 及 `rollout_lab/block_heuristics.py`）标注为
> **阶段一临时随迁、Commit 4 删除**——它们不是最终仓库内容。

## 1. 已关闭/已终结的实验分支（EXCLUDE_LEGACY，代码）

| 对象 | 原因 |
|---|---|
| `neural_opportunity/{cate_pilot, causal_selective_rerank, full_price_rl, full_price_kernel, full_direct_amortization, release_time_rt, rl_lite, hd0, sko0, sp0, sp1, shadow_residual_ceiling}/` | 默认排除清单中的侧线实验；grep 确认 N1 六入口对它们**零 import**（仅 run_load_scan 依赖 causal_selective_rerank 的钉死权重——该入口一并排除） |
| `pure_rmsa_v13/{direct_sketch_residual_rl, tchr_rl, ht_aware}/` | 已关闭的 Direct-Sketch RL / Teacher-RL 线 |
| `neural_opportunity/n1_100/{ns1_dataset, ns1_training, run_ns1_100}.py` | NS1 线（学 Full-Direct 全价格）已终结：NS1_RESULTS.json verdict=STOP（regret 相对降低 7.5%，gate ≥30%） |
| `n1_100/hybrid_selector.py` | T5 门控混合实验；全仓库无 runner 引用，死代码 |
| `n1_100/{calibrate_new_topologies, calibrate_new_topologies_extended}.py` | 5-8% 校准口径，已被 `recalibrate_below_1pct.py`（<1%）取代 |
| `n1_100/run_compare_xlron_topologies.py` | legacy 池评估，产物已被 fair-format 重跑取代 |
| `neural_opportunity/{run_k100_cal, run_k100_tri}.py` | 100-slot 非神经矩阵实验，与 N1 模型无关 |
| `neural_opportunity/run_load_scan.py` | 硬依赖 `causal_selective_rerank.protocol` 的钉死权重路径+sha256，随迁成本高、价值低 |
| `rollout_lab/` 其余 ~60 个 py（含 `neural_pricing_kernel.py`、`npk_dataset.py`、`potential_cost_variants.py`、全部 audit_*/eval_*/run_* 旧脚本、10+ 个 .sh） | 逐模块 grep 确认 N1 侧**零引用**；.sh 内含 /mnt/d 硬编码 |
| `rollout_lab/_direct_sketch_kernel.{c,so}`、`build_direct_sketch_kernel.py`、`DIRECT_SKETCH_NATIVE_KERNEL.md` | 可选 native kernel；不搬 .so 时 backend 自动落 python（正是协议锁定状态）；搬了反而引入 C 路径静默切换风险（compiled_ksp_ff 不看环境变量） |
| `rollout_lab/block_heuristics.py` + `test_block_heuristics.py` | 唯一闭包入边是 run_phaseA 顶层 import 的 N1 未用 arm；helper 抽取后出闭包 |
| `direct_sketch_exact_optimization/` 其余文件（run_phaseB/C、selector_phaseB/C、phase0_profile、phaseC_margin_precheck、run_k_path_sweep、run_phaseA_confirmatory、test_k_path_sweep、2 份 PROTOCOL_LOCK） | N1 只需 selectors_phaseA.py 与 run_phaseA.py 的 3 个 helper |
| `pds_rmsa/{training/, features/, networks/, exact_dp/, ht_pds/, ablations.py, degradation_diagnosis.py, run_*.py, evaluate_*.py, replay_audit.py, verify_*.py}` 及 10 份 PHASE_*.md | PDS 训练线；仅经 train_proposer 副作用 import 进入 AST 闭包，make_env 抽取后全部断开 |
| `sa_hmarl/{agents, env, dnn, mec, evaluation, datasets, path_pds_ff, scripts, training, utils, experiments, baselines}/` | 旧 SA-HMARL 主线；副作用依赖 only |
| `network/{modulation.py（REVIEW_REQUIRED，倾向排除）, optical_network.py, spectrum_blocks.py}` | N1 不用；modulation.py 仅副作用链拉入，make_env 抽取后可弃 |
| 外层残留 `sa_hmarl/pure_rmsa_v13/`、`sa_hmarl/pds_rmsa/`（工作根下一层） | 非源码残留目录，强制规则 3 |
| 根级 `DeepRMSA/`、`backups/`、`tmp/`、`tmp_*.py`、`mnt/` 杂散目录 | 参考工程/备份/临时物 |

若未来发现 N1 需要上述目录中的某个公共函数，按强制规则应 EXTRACT_MINIMAL_HELPER
（提取最小函数并列明来源），而不是搬入整条实验分支。当前审计未发现此类需求。

## 2. 大型 artifacts（LEGACY_EXCLUDE / REGENERATABLE_EXCLUDE）

| 对象 | 大小 | 分类 | 原因 |
|---|---:|---|---|
| `artifacts/n1_100_xlron/{5 拓扑}/datasets/` | ~485M | REGENERATABLE_EXCLUDE | teacher 数据集，`run_n1_100_xlron` 可再采集；不入 Git，需要时冷存储 |
| `artifacts/n1_100_xlron/{nsfnet,jpn48,usnet}/backup_legacy_pool/` | ~217M | LEGACY_EXCLUDE | 被取代的 hops 池数据集+checkpoint |
| `artifacts/n1_100_xlron/german17/` | 72M | REVIEW_REQUIRED | 死实验线：只有 datasets+training，无 COMPARE 结果、不在最终 5 拓扑矩阵；建议排除或仅留 64K 权重 |
| `artifacts/n1_100/nsfnet/datasets/` | 36M | REGENERATABLE_EXCLUDE | 可再采集 |
| `artifacts/n1_100/nsfnet/ns1/` | 39M | LEGACY_EXCLUDE | 随 NS1 代码线排除 |
| `artifacts/datasets_v1/` | 19M | LEGACY_EXCLUDE | 被 datasets_conflict_v1 取代；元数据无 feature_schema_id，易误用 |
| `artifacts/training_v1/` | 144K | LEGACY_EXCLUDE | **旧权重格式，当前 `TinyOpportunityWeights.load()` 直接拒绝**；名字像主线，实为 legacy |
| `artifacts/{datasets_smoke, training_smoke, closed_loop_smoke_conflict_v1}/` | ~144K | REGENERATABLE_EXCLUDE | smoke 输出，按 AGENTS.md 不得作正式结果 |
| `artifacts/latency_ceiling{,_v2,_v3,_v4,_conflict_v1,_locked}/` | ~24K | REGENERATABLE_EXCLUDE | 历史迭代轮（保留 latency_ceiling_final 即可；注意 locked 与 final verdict 相反，勿混用） |
| `artifacts/{k100_cal, k100_tri, load_scan}/` | ~550K | LEGACY_EXCLUDE | 随对应入口排除 |
| 全部 `training_checkpoint.pt`（10 个） | ~200K | REGENERATABLE_EXCLUDE | 全仓**无任何代码读回 .pt**，纯留档 |
| `artifacts/datasets_conflict_v1/` | 66M | SOURCE_REQUIRED（特殊） | 训练现行权重的真实数据源；**不入 Git**，建议冷存储/Release；可由 run_collect 再生成（极慢） |
| 所有 raw trajectory、完整候选 outcomes、profile 文件 | — | REGENERATABLE_EXCLUDE | 按任务规定不迁移 |

## 3. 文档类排除

- `DNN_Agent/README.md`、`KIMI_CODE_CONTEXT.md`、`sa_hmarl/README.md`：主线是旧
  PPO-C/v1.2 ranker / Phase-R0（已 STOP），与 N1 无关；新仓 README 需新写。
- `pure_rmsa_v13/PROTOCOL_LOCK.md`、`rollout_lab/PROTOCOL_LOCK.md`、`pds_rmsa/` 全部
  PHASE_*.md：祖先/侧线协议，不随迁（避免被当成当前主线）。
- `neural_opportunity/{PHASE_K100_CAL_PROTOCOL, PHASE_K100_TRI_PROTOCOL, PHASE_K100_TRI_REPORT, PHASE_LOAD_SCAN_PROTOCOL, PHASE_LOAD_SCAN_REPORT}.md`：随 k100/load_scan 实验排除。

## 4. 杂质

- 所有 `__pycache__/`、`.pytest_cache/`、`*.prof`、`.pyc`（含 artifacts 目录内的散落副本）。
- 工作根下的 `tmp_ksp_k5_k50_100*.py/json`、`test_truncation.py` 等临时脚本。
