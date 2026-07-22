<!-- SUPERSEDED AS ORIGINAL-DEEPRMSA CLAIM; RETAINED ONLY AS MASKED-ADAPTER ABLATION. This result uses a legal-action mask, which is not present in the upstream DeepRMSA source. It must not be called 'Original DeepRMSA' or 'Unmodified DeepRMSA'. -->

# DeepRMSA-Adapted PAPER_PRIMARY_K50 Results

> These are native COST239/K=50 PyTorch DeepRMSA-style adapter results, not an original DeepRMSA reproduction.

| Method | K | M | Mean blocking | Std |
|---|---:|---:|---:|---:|
| Formal KSP-FF K=50 | 50 | n/a | 13.2000% | 0.6111% |
| Strict v1.3 zero-shot | 50 | n/a | 16.0567% | 0.7865% |
| PPO-R Top-1 zero-shot | 50 | n/a | 25.2825% | 0.5113% |
| deeprmsa_k50_m1_native | 50 | 1 | 14.4083% | 0.4859% |

## Paired comparisons

- `deeprmsa_k50_m1_native_vs_ksp_ff_highest`: baseline-method=-1.2083 pp, 95% CI [-1.4642, -0.9550] pp, p=1.000000.
- `deeprmsa_k50_m1_native_vs_strict_v13`: baseline-method=1.6483 pp, 95% CI [1.3250, 1.9600] pp, p=0.000488.
- `deeprmsa_k50_m1_native_vs_ppo_r_top1`: baseline-method=10.8742 pp, 95% CI [10.6658, 11.0883] pp, p=0.000488.
