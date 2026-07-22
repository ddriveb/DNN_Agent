# Strict v1.3 Multi-Topology C-side Fair Evaluation Results (Verified)

## Per-topology aggregate blocking

| Topology | C-side | R-side | Blocking | C-no | R-no | NSB | Overload | Deadline | Other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xlron_cost239_ptrnet_real | df_c | ksp_ff_highest | 5.15% ± 1.51% | 5.15% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_cost239_ptrnet_real | df_c | ppo_r_top1 | 5.15% ± 1.51% | 5.15% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_cost239_ptrnet_real | df_c | strict_v13 | 5.15% ± 1.51% | 5.15% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_cost239_ptrnet_real | ppo_c | ksp_ff_highest | 7.84% ± 1.47% | 7.84% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_cost239_ptrnet_real | ppo_c | ppo_r_top1 | 5.61% ± 1.56% | 5.61% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_cost239_ptrnet_real | ppo_c | strict_v13 | 5.94% ± 1.23% | 5.94% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_german17 | df_c | ksp_ff_highest | 5.59% ± 1.41% | 5.59% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_german17 | df_c | ppo_r_top1 | 6.98% ± 1.52% | 6.98% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_german17 | df_c | strict_v13 | 6.21% ± 1.45% | 6.21% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_german17 | ppo_c | ksp_ff_highest | 43.22% ± 1.62% | 43.22% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_german17 | ppo_c | ppo_r_top1 | 23.48% ± 1.63% | 23.48% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_german17 | ppo_c | strict_v13 | 23.02% ± 1.54% | 23.02% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_jpn48 | df_c | ksp_ff_highest | 7.06% ± 1.35% | 7.06% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_jpn48 | df_c | ppo_r_top1 | 7.37% ± 1.30% | 7.37% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_jpn48 | df_c | strict_v13 | 7.55% ± 1.23% | 7.55% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_jpn48 | ppo_c | ksp_ff_highest | 35.88% ± 2.48% | 35.88% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_jpn48 | ppo_c | ppo_r_top1 | 20.34% ± 1.13% | 20.34% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_jpn48 | ppo_c | strict_v13 | 21.57% ± 1.25% | 21.57% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_nsfnet_deeprmsa | df_c | ksp_ff_highest | 12.20% ± 2.41% | 12.20% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_nsfnet_deeprmsa | df_c | ppo_r_top1 | 12.52% ± 2.62% | 12.52% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_nsfnet_deeprmsa | df_c | strict_v13 | 12.21% ± 2.22% | 12.21% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_nsfnet_deeprmsa | ppo_c | ksp_ff_highest | 40.18% ± 3.24% | 40.18% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_nsfnet_deeprmsa | ppo_c | ppo_r_top1 | 38.50% ± 2.14% | 38.50% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| xlron_nsfnet_deeprmsa | ppo_c | strict_v13 | 38.46% ± 2.04% | 38.46% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Per-topology paired comparisons

- **xlron_cost239_ptrnet_real / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=-0.32 pp, rel red=-5.77%, wins=1/0/4, bootstrap CI=[-0.0065, -0.0002], exact one-sided p=0.9375, two-sided p=0.1875, Holm adj. p=1.0
- **xlron_cost239_ptrnet_real / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=1.90 pp, rel red=24.25%, wins=5/0/0, bootstrap CI=[0.0153, 0.0231], exact one-sided p=0.0312, two-sided p=0.0625, Holm adj. p=0.5
- **xlron_cost239_ptrnet_real / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=0.00 pp, rel red=0.00%, wins=0/5/0, bootstrap CI=[0.0000, 0.0000], exact one-sided p=1.0000, two-sided p=1.0000, Holm adj. p=1.0
- **xlron_cost239_ptrnet_real / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=0.00 pp, rel red=0.00%, wins=0/5/0, bootstrap CI=[0.0000, 0.0000], exact one-sided p=1.0000, two-sided p=1.0000, Holm adj. p=1.0
- **xlron_german17 / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=0.47 pp, rel red=1.99%, wins=4/0/1, bootstrap CI=[0.0004, 0.0092], exact one-sided p=0.0938, two-sided p=0.1875, Holm adj. p=1.0
- **xlron_german17 / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=20.21 pp, rel red=46.75%, wins=5/0/0, bootstrap CI=[0.1835, 0.2188], exact one-sided p=0.0312, two-sided p=0.0625, Holm adj. p=0.5
- **xlron_german17 / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=0.77 pp, rel red=11.01%, wins=5/0/0, bootstrap CI=[0.0057, 0.0098], exact one-sided p=0.0312, two-sided p=0.0625, Holm adj. p=0.5
- **xlron_german17 / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=-0.62 pp, rel red=-11.02%, wins=0/0/5, bootstrap CI=[-0.0081, -0.0039], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_nsfnet_deeprmsa / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=0.04 pp, rel red=0.10%, wins=4/0/1, bootstrap CI=[-0.0100, 0.0084], exact one-sided p=0.4375, two-sided p=0.8750, Holm adj. p=1.0
- **xlron_nsfnet_deeprmsa / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=1.72 pp, rel red=4.27%, wins=4/0/1, bootstrap CI=[0.0058, 0.0260], exact one-sided p=0.0625, two-sided p=0.1250, Holm adj. p=0.75
- **xlron_nsfnet_deeprmsa / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=0.32 pp, rel red=2.52%, wins=2/0/3, bootstrap CI=[-0.0013, 0.0089], exact one-sided p=0.2500, two-sided p=0.5000, Holm adj. p=1.0
- **xlron_nsfnet_deeprmsa / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=-0.01 pp, rel red=-0.10%, wins=3/0/2, bootstrap CI=[-0.0024, 0.0020], exact one-sided p=0.5312, two-sided p=1.0000, Holm adj. p=1.0
- **xlron_jpn48 / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=-1.23 pp, rel red=-6.04%, wins=0/0/5, bootstrap CI=[-0.0193, -0.0052], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_jpn48 / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=14.31 pp, rel red=39.88%, wins=5/0/0, bootstrap CI=[0.1272, 0.1564], exact one-sided p=0.0312, two-sided p=0.0625, Holm adj. p=0.5
- **xlron_jpn48 / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=-0.18 pp, rel red=-2.44%, wins=3/0/2, bootstrap CI=[-0.0062, 0.0018], exact one-sided p=0.7188, two-sided p=0.6250, Holm adj. p=1.0
- **xlron_jpn48 / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=-0.49 pp, rel red=-6.97%, wins=0/0/5, bootstrap CI=[-0.0071, -0.0026], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0