# Checkpoint Index

本目录及 `experiments/` 中所有现存 `.pt` 文件的版本、用途和状态统一记录在：

- `CHECKPOINT_MANIFEST.md`：人工阅读版，逐个 checkpoint 标注版本。
- `CHECKPOINT_MANIFEST.csv`：机器可读版，包含完整路径、角色、拓扑、seed、快照类型、架构、协议 ID、大小和 SHA-256。
- `../scripts/build_checkpoint_manifest.py`：清单生成与完整性检查脚本。

## 当前 Strict v1.3

Strict v1.3 在线 R 端只由以下两个核心 checkpoint 组成：

1. `agent_r_mixed.pt`：冻结 PPO-R 合法 Top-30 proposer。
2. `../experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt`：正式 25 维 Ranker。

原生 SA-HMARL 协议保留的 C 端支持 checkpoint 是：

- `agent_c_cost239_r_feasibility_safe_last.pt`

fixed-C / all-OD pure RMSA 实验不会调用 PPO-C。

## 规则

- `best`、`last`、`final` 只表示保存时刻，不是算法版本。
- `seed_42/43/44` 表示训练复现实例，不是算法版本。
- 只有 manifest 中 `canonical=yes` 的文件可以直接称为当前正式版本。
- 新增、删除或替换 checkpoint 后运行：

```powershell
python sa_hmarl/scripts/build_checkpoint_manifest.py
python sa_hmarl/scripts/build_checkpoint_manifest.py --check
```

生成器发现未分类 checkpoint 会立即失败，禁止用文件名猜版本后继续实验。
