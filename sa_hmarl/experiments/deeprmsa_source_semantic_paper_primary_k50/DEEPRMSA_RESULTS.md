# DeepRMSA Source-Semantic K=50 Results

> This is an unmasked PyTorch protocol port of the DeepRMSA source semantics,
> not unchanged execution of the upstream TensorFlow source.

* Mean blocking (aggregated over 3 training seeds and 12 test seeds): **45.8000%**
* Std across test seeds (aggregated): 0.0055%
* Training-seed variation (std of per-seed means): 0.0615%

## Per training seed
| Training seed | Mean blocking | Std |
|--------------:|--------------:|----:|
|            42 |      52.9058% | 0.0049 |
|           123 |      42.1867% | 0.0060 |
|           456 |      42.3075% | 0.0073 |

## Aggregated per test seed
| Test seed | Mean blocking | Std |
|----------:|--------------:|----:|
|      3030 |      46.7533% | 0.0617 |
|      4040 |      45.7733% | 0.0634 |
|      5050 |      45.9233% | 0.0629 |
|      6060 |      45.9867% | 0.0564 |
|      7070 |      46.4500% | 0.0587 |
|      8080 |      44.8300% | 0.0642 |
|      9090 |      45.7800% | 0.0614 |
|      1010 |      45.4033% | 0.0615 |
|      2020 |      45.7767% | 0.0602 |
|      3031 |      44.9800% | 0.0651 |
|      4041 |      45.7000% | 0.0601 |
|      5051 |      46.2433% | 0.0632 |

## Paired comparisons
* `ksp_ff_highest` (Formal KSP-FF K=50): baseline - method = -32.6000 pp, 95% CI [-32.9350, -32.2647] pp, p=1.000000, wins/ties/losses={'wins': 0, 'ties': 0, 'losses': 12}
* `strict_v13` (Strict v1.3 zero-shot): baseline - method = -29.7433 pp, 95% CI [-30.0733, -29.4331] pp, p=1.000000, wins/ties/losses={'wins': 0, 'ties': 0, 'losses': 12}
* `ppo_r_top1` (PPO-R Top-1 zero-shot): baseline - method = -20.5175 pp, 95% CI [-20.7878, -20.2461] pp, p=1.000000, wins/ties/losses={'wins': 0, 'ties': 0, 'losses': 12}
