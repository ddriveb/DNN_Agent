> ⚠️ **本报告已过期。** 其中的 `Rel. regret red.` 列使用有缺陷的 `relative_regret_reduction` 指标，且 checkpoint 选择依据为 Top-1 准确率而非 regret。修正后的训练结果见 `MEDIUM_TRAINING_REPORT_CORRECTED.md`。

# SA-HMARL v1.3 Medium Dataset Training Report

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- Model: MLP [128,64]
- Loss: listwise KL + SmoothL1 MSE (reg_weight=1.0, lambda_pair=0, lambda_hard=0)
- Seeds: [42, 43, 44]

## Learning curve

| Train groups | Top-1 | Tie-aware Top-1 | Top-3 | Spearman | Kendall tau-b | NDCG@3 | Model regret | PPO regret | Rel. regret red. | PPO agree |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k3 | 0.043±0.010 | 0.458±0.015 | 0.115±0.023 | 0.029±0.029 | 0.024±0.023 | 0.634±0.012 | 0.070±0.002 | 0.105±0.000 | -0.659±0.023 | 0.047±0.010 |
| 3k | 0.068±0.009 | 0.450±0.003 | 0.166±0.016 | -0.000±0.002 | 0.000±0.001 | 0.620±0.003 | 0.075±0.009 | 0.105±0.000 | -0.253±0.053 | 0.091±0.020 |
| 5k | 0.064±0.004 | 0.449±0.005 | 0.154±0.001 | 0.001±0.009 | 0.001±0.007 | 0.626±0.001 | 0.086±0.009 | 0.105±0.000 | -0.136±0.035 | 0.082±0.001 |
| full | 0.062±0.018 | 0.448±0.005 | 0.151±0.035 | 0.006±0.014 | 0.004±0.011 | 0.630±0.007 | 0.084±0.002 | 0.105±0.000 | -1.154±0.134 | 0.084±0.031 |

## Per-seed details

| Subset | Seed | Top-1 | Spearman | Rel. regret red. | Output dir |
|---|---|---:|---:|---|---|
| 1k3 | 42 | 5.44% | 0.026 | -0.690 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/1k3_seed42 |
| 1k3 | 44 | 2.94% | 0.066 | -0.652 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/1k3_seed44 |
| 1k3 | 43 | 4.48% | -0.004 | -0.635 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/1k3_seed43 |
| 3k | 43 | 5.97% | -0.002 | -0.320 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/3k_seed43 |
| 3k | 42 | 8.04% | -0.000 | -0.250 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/3k_seed42 |
| 3k | 44 | 6.50% | 0.002 | -0.189 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/3k_seed44 |
| 5k | 42 | 6.55% | 0.012 | -0.098 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/5k_seed42 |
| 5k | 44 | 6.84% | -0.010 | -0.182 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/5k_seed44 |
| 5k | 43 | 5.92% | 0.002 | -0.127 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/5k_seed43 |
| full | 42 | 8.28% | -0.012 | -0.997 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/full_seed42 |
| full | 43 | 6.55% | 0.008 | -1.140 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/full_seed43 |
| full | 44 | 3.90% | 0.022 | -1.324 | /mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/full_seed44 |