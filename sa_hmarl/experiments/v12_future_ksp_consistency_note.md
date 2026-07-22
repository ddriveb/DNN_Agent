# v1.2 Future Continuation KSP 一致性说明

## 1. 问题

当前 state 的 ranking dataset 生成已经支持 `--ensure_ksp_action`，会把当前 `obs_r` 下 `KSP-FF K=50 hops` 会选中的 legal 动作显式注入候选集。

但 future rollout / continuation 里的 `future_ranker_policy` 在构造时只是从 checkpoint 加载，**没有显式开启 `ensure_ksp_action`**。如果 future continuation 使用 `ranker`，它在未来状态上构造 candidate set 时，就可能漏掉那些未来状态下的 KSP 强动作。

这会造成自洽升级不完整：
- 当前候选集里有 KSP 动作；
- future continuation 里的 ranker 却未必能看到 KSP 动作；
- 导致 counterfactual return 估计与未来部署策略不一致。

## 2. 改动点

### `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py`

在加载 `trajectory_ranker_policy` 和 `future_ranker_policy` 之后，新增以下覆盖逻辑：

```python
# trajectory ranker
if args.trajectory_ranker_candidate_mode is not None:
    trajectory_ranker_policy.candidate_mode = args.trajectory_ranker_candidate_mode
if args.trajectory_ranker_max_candidates is not None:
    trajectory_ranker_policy.max_candidates = args.trajectory_ranker_max_candidates
if args.trajectory_ranker_ensure_ksp is not None:
    trajectory_ranker_policy.ensure_ksp_action = args.trajectory_ranker_ensure_ksp
elif args.ensure_ksp_action:
    trajectory_ranker_policy.ensure_ksp_action = True

# future rollout ranker
if args.future_ranker_candidate_mode is not None:
    future_ranker_policy.candidate_mode = args.future_ranker_candidate_mode
if args.future_ranker_max_candidates is not None:
    future_ranker_policy.max_candidates = args.future_ranker_max_candidates
if args.future_ranker_ensure_ksp is not None:
    future_ranker_policy.ensure_ksp_action = args.future_ranker_ensure_ksp
elif args.ensure_ksp_action:
    future_ranker_policy.ensure_ksp_action = True
```

新增 CLI 参数：
- `--trajectory_ranker_ensure_ksp` / `--no-trajectory_ranker_ensure_ksp`
- `--future_ranker_ensure_ksp` / `--no-future_ranker_ensure_ksp`

这两个参数默认继承 `--ensure_ksp_action`（如果开了全局 ensure_ksp，则 trajectory/future ranker 也自动开），但可以被显式覆盖。

加载日志现在会打印 `ensure_ksp` 状态，例如：

```text
[DAgger-lite] Loaded future rollout ranker from ... (candidate_mode=legalctx48, max_candidates=48, ensure_ksp=True)
```

## 3. 为什么这样改

- **显式可控**：不偷偷默认，用户可以通过 `--future_ranker_ensure_ksp` 明确决定 future continuation 是否注入 KSP。
- **自洽**：当前 state 和 future continuation 使用同一套 `CounterfactualRRankerPolicy._ensure_ksp_action` 逻辑，KSP 动作注入点统一。
- **可继承**：大多数情况下用户只关心 `--ensure_ksp_action`，所以默认继承；需要单独控制时再显式覆盖。

## 4. Smoke 验证

运行命令：

```bash
PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset \
  --topology xlron_cost239_ptrnet_real --num_slots 100 \
  --k_paths 50 --path_sort_strategy hops --block_sort_strategy start_asc \
  --candidate_mode legalctx48 --max_candidates 48 --ensure_ksp_action \
  --trajectory_policy ppo_r --future_rollout_policy ranker \
  --future_ranker_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --future_ranker_candidate_mode legalctx48 --future_ranker_max_candidates 48 --future_ranker_ensure_ksp \
  --episodes 1 --horizon 1 --requests_per_episode 10 \
  --output_dir sa_hmarl/datasets/future_ksp_consistency_smoke
```

输出关键行：

```text
[DAgger-lite] Loaded future rollout ranker from sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt (candidate_mode=legalctx48, max_candidates=48, ensure_ksp=True)
```

验证结果：
- `future_ranker_policy.candidate_mode = legalctx48`
- `future_ranker_policy.max_candidates = 48`
- `future_ranker_policy.ensure_ksp_action = True`

future continuation 的 KSP 注入已补齐。

## 5. 已知限制

- `ranker` continuation 的 CPU 开销很大。即使把 future ranker 的 candidate mode 压到 `legalctx48`、max_candidates=48，1 个 episode × 10 requests × horizon=1 仍需约 100 秒。完整数据集生成（3 splits × 20 episodes × 80 requests × horizon=5）会非常慢，必须考虑并行化或更小的 continuation candidate set。
- 当前 smoke 数据集的 `oracle_headroom_pp=0`，因为样本太少，return range 主要由当前 blocked 决定，不代表正式训练时的质量。
