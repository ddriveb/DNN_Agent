# Strict v1.3 Multi-Topology C-side Fair Evaluation Results (Verified)

## Per-topology aggregate blocking

| Topology | C-side | R-side | Blocking mean ± std | C-no | R-no/NSB | Overload | Other |
|---|---|---|---:|---:|---:|---:|---:|
| xlron_cost239_ptrnet_real | df_c | ksp_ff_highest | 11.72% ± 0.85% | 0.31% | 0.00% | 0.00% | 11.41% |
| xlron_cost239_ptrnet_real | df_c | ppo_r_top1 | 18.03% ± 0.89% | 0.22% | 0.00% | 0.00% | 8.98% |
| xlron_cost239_ptrnet_real | df_c | strict_v13 | 17.58% ± 0.87% | 0.10% | 0.00% | 0.00% | 9.18% |
| xlron_cost239_ptrnet_real | ppo_c | ksp_ff_highest | 10.85% ± 0.23% | 0.38% | 0.00% | 0.01% | 10.47% |
| xlron_cost239_ptrnet_real | ppo_c | ppo_r_top1 | 11.37% ± 0.55% | 0.35% | 0.00% | 0.00% | 5.16% |
| xlron_cost239_ptrnet_real | ppo_c | strict_v13 | 30.91% ± 1.41% | 0.10% | 0.00% | 0.00% | 25.65% |
| xlron_german17 | df_c | ksp_ff_highest | 13.33% ± 1.23% | 0.11% | 0.00% | 0.00% | 8.54% |
| xlron_german17 | df_c | ppo_r_top1 | 28.57% ± 0.91% | 0.00% | 0.00% | 0.00% | 15.15% |
| xlron_german17 | df_c | strict_v13 | 42.60% ± 0.52% | 0.01% | 0.00% | 0.00% | 8.30% |
| xlron_german17 | ppo_c | ksp_ff_highest | 8.78% ± 0.44% | 2.86% | 0.00% | 0.74% | 5.17% |
| xlron_german17 | ppo_c | ppo_r_top1 | 25.53% ± 0.47% | 1.08% | 0.00% | 0.13% | 12.98% |
| xlron_german17 | ppo_c | strict_v13 | 38.39% ± 1.72% | 0.00% | 0.00% | 0.00% | 5.77% |
| xlron_jpn48 | df_c | ksp_ff_highest | 17.53% ± 0.85% | 0.19% | 0.00% | 0.00% | 12.06% |
| xlron_jpn48 | df_c | ppo_r_top1 | 27.94% ± 0.47% | 0.07% | 0.00% | 0.00% | 9.58% |
| xlron_jpn48 | df_c | strict_v13 | 31.98% ± 0.67% | 0.06% | 0.00% | 0.00% | 5.31% |
| xlron_jpn48 | ppo_c | ksp_ff_highest | 12.08% ± 1.09% | 1.45% | 0.00% | 1.24% | 9.21% |
| xlron_jpn48 | ppo_c | ppo_r_top1 | 20.81% ± 0.78% | 0.60% | 0.00% | 0.20% | 9.38% |
| xlron_jpn48 | ppo_c | strict_v13 | 29.07% ± 1.39% | 0.26% | 0.00% | 0.21% | 4.49% |
| xlron_nsfnet_deeprmsa | df_c | ksp_ff_highest | 30.20% ± 0.52% | 0.00% | 0.00% | 0.00% | 29.94% |
| xlron_nsfnet_deeprmsa | df_c | ppo_r_top1 | 34.74% ± 1.18% | 0.00% | 0.00% | 0.00% | 24.12% |
| xlron_nsfnet_deeprmsa | df_c | strict_v13 | 61.74% ± 0.58% | 0.00% | 0.00% | 0.00% | 55.34% |
| xlron_nsfnet_deeprmsa | ppo_c | ksp_ff_highest | 28.34% ± 0.64% | 1.82% | 0.00% | 1.74% | 24.62% |
| xlron_nsfnet_deeprmsa | ppo_c | ppo_r_top1 | 33.31% ± 1.11% | 1.83% | 0.00% | 1.18% | 21.44% |
| xlron_nsfnet_deeprmsa | ppo_c | strict_v13 | 70.07% ± 0.99% | 0.06% | 0.00% | 0.40% | 63.96% |

## Per-topology paired comparisons

- **xlron_cost239_ptrnet_real / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=19.54 pp, rel red=-171.83%, wins=0/0/5, bootstrap CI=[-0.2081, -0.1808], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_cost239_ptrnet_real / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=20.06 pp, rel red=-184.85%, wins=0/0/5, bootstrap CI=[-0.2115, -0.1890], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_cost239_ptrnet_real / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=-0.44 pp, rel red=2.46%, wins=3/0/2, bootstrap CI=[-0.0044, 0.0136], exact one-sided p=0.2188, two-sided p=0.4375, Holm adj. p=1.0
- **xlron_cost239_ptrnet_real / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=5.87 pp, rel red=-50.09%, wins=0/0/5, bootstrap CI=[-0.0661, -0.0498], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_german17 / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=12.86 pp, rel red=-50.35%, wins=0/0/5, bootstrap CI=[-0.1414, -0.1156], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_german17 / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=29.61 pp, rel red=-337.22%, wins=0/0/5, bootstrap CI=[-0.3132, -0.2823], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_german17 / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=14.03 pp, rel red=-49.12%, wins=0/0/5, bootstrap CI=[-0.1474, -0.1336], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_german17 / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=29.27 pp, rel red=-219.63%, wins=0/0/5, bootstrap CI=[-0.3032, -0.2844], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_nsfnet_deeprmsa / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=36.76 pp, rel red=-110.38%, wins=0/0/5, bootstrap CI=[-0.3753, -0.3611], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_nsfnet_deeprmsa / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=41.74 pp, rel red=-147.29%, wins=0/0/5, bootstrap CI=[-0.4270, -0.4085], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_nsfnet_deeprmsa / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=27.00 pp, rel red=-77.72%, wins=0/0/5, bootstrap CI=[-0.2831, -0.2577], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_nsfnet_deeprmsa / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=31.54 pp, rel red=-104.46%, wins=0/0/5, bootstrap CI=[-0.3224, -0.3085], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_jpn48 / ppo_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=8.26 pp, rel red=-39.69%, wins=0/0/5, bootstrap CI=[-0.0988, -0.0668], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_jpn48 / ppo_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=17.00 pp, rel red=-140.74%, wins=0/0/5, bootstrap CI=[-0.1873, -0.1487], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_jpn48 / df_c / strict_v13 vs ppo_r_top1**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=4.04 pp, rel red=-14.44%, wins=0/0/5, bootstrap CI=[-0.0465, -0.0340], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0
- **xlron_jpn48 / df_c / strict_v13 vs ksp_ff_highest**: n=5, common=[3030, 4040, 5050, 6060, 7070], mean diff=14.45 pp, rel red=-82.41%, wins=0/0/5, bootstrap CI=[-0.1542, -0.1338], exact one-sided p=1.0000, two-sided p=0.0625, Holm adj. p=1.0