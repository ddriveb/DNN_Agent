# Execution Log — n1_ksp_fairness_audit_20260802

## 环境摘要
- OS: Ubuntu WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- Python: /mnt/d/project/DNN_Agent/.venv/bin/python 3.14.4
- PYTHONPATH=. ; cwd = /mnt/d/project/DNN_Agent/sa_hmarl
- SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL: unset（全程）
- workers: 1（无并行实验）
- 无依赖安装/升级；无旧产物修改；未训练/调参/模型选择

## 实际执行命令（按序）
1. `cat AGENTS.md`；`.venv/bin/python --version` → 3.14.4；`ps aux | grep run_tchr|ceiling|run_select|probe` → 0 个重复进程
2. `mkdir -p n1_ksp_fairness_audit_20260802`
3. 读取：confirmatory_10seed/{CONFIRMATORY_RESULTS.json,neural_per_seed.csv}、phaseA_confirmatory_10seed/per_seed.csv、LATENCY_CEILING.json、NEURAL_OPPORTUNITY_FINAL_REPORT.md、DIRECT_VS_KSP_K5_K50_REPORT.md、PROTOCOL_AUDIT.md、EXPERIMENT_MANIFEST.json、selector.py、model.py、features.py、protocol.py、compiled_ksp_ff.py
4. `PYTHONPATH=. .venv/bin/python .../audit_recompute_blocking.py` → 阻塞重算（PAIRWISE_BLOCKING_RECOMPUTE.csv + AUDIT_BLOCKING_RECOMPUTE.json）
5. Explore agent：全仓 N1/KSP 比较清单 + 6 项证据补查（delta 公式/trace hash/runner 计时代码/tie-break/LATENCY_BREAKDOWN）
6. 时间线：`stat -c %y` 对 9 个关键 artifact
7. `sha256sum` deployment_weights.npz ×3
8. `PYTHONPATH=. .venv/bin/python -m pytest .../test_audit_fairness.py -v` → 20 passed（PYTEST_OUTPUT.txt）
9. 生成：AUDIT_RESULTS.json、PROVENANCE_MATRIX.csv、LATENCY_RECONCILIATION.md、COMPARISON_INVENTORY.csv、AUDIT_REPORT.md、EXECUTION_LOG.md

## 交付物清单
- AUDIT_REPORT.md（Findings 开头按严重性；verdict=PASS_WITH_REPORT_CORRECTIONS）
- AUDIT_RESULTS.json
- AUDIT_BLOCKING_RECOMPUTE.json
- COMPARISON_INVENTORY.csv
- PAIRWISE_BLOCKING_RECOMPUTE.csv
- LATENCY_RECONCILIATION.md
- PROVENANCE_MATRIX.csv
- test_audit_fairness.py + PYTEST_OUTPUT.txt（20 passed）
- audit_recompute_blocking.py
- EXECUTION_LOG.md（本文件）
