# Formal Pure RMSA Results

Positive delta means Frozen Strict v1.3 pure-RMSA transfer blocks more.

| Topology | Load | Comparator | Strict | Comparator | Delta (pp) | 95% CI (pp) | Verdict |
|---|---:|---|---:|---:|---:|---:|---|
| cost239 | 600 | KSP-FF K=50 hops | 18.972% | 1.906% | +17.067 | [+16.907, +17.236] | Strict significantly higher |
| cost239 | 600 | FF-KSP K=50 hops | 18.972% | 5.721% | +13.252 | [+13.065, +13.449] | Strict significantly higher |
| jpn48 | 300 | KSP-FF K=50 hops | 10.409% | 1.195% | +9.214 | [+9.043, +9.382] | Strict significantly higher |
| jpn48 | 300 | FF-KSP K=50 hops | 10.409% | 0.602% | +9.807 | [+9.621, +9.980] | Strict significantly higher |
| jpn48 | 375 | KSP-FF K=50 hops | 14.803% | 4.634% | +10.168 | [+10.005, +10.328] | Strict significantly higher |
| jpn48 | 375 | FF-KSP K=50 hops | 14.803% | 3.920% | +10.883 | [+10.753, +11.016] | Strict significantly higher |
| jpn48 | 475 | KSP-FF K=50 hops | 19.547% | 10.223% | +9.325 | [+9.166, +9.489] | Strict significantly higher |
| jpn48 | 475 | FF-KSP K=50 hops | 19.547% | 10.054% | +9.493 | [+9.256, +9.727] | Strict significantly higher |
| jpn48 | 650 | KSP-FF K=50 hops | 26.108% | 18.660% | +7.448 | [+7.241, +7.657] | Strict significantly higher |
| jpn48 | 650 | FF-KSP K=50 hops | 26.108% | 19.117% | +6.992 | [+6.851, +7.155] | Strict significantly higher |
| nsfnet | 250 | KSP-FF K=50 hops | 18.387% | 2.421% | +15.967 | [+15.855, +16.072] | Strict significantly higher |
| nsfnet | 250 | FF-KSP K=50 hops | 18.387% | 4.235% | +14.152 | [+14.012, +14.304] | Strict significantly higher |
| usnet | 450 | KSP-FF K=50 hops | 11.213% | 0.883% | +10.330 | [+10.163, +10.495] | Strict significantly higher |
| usnet | 450 | FF-KSP K=50 hops | 11.213% | 1.224% | +9.989 | [+9.840, +10.138] | Strict significantly higher |
| usnet | 550 | KSP-FF K=50 hops | 15.602% | 4.514% | +11.088 | [+10.859, +11.322] | Strict significantly higher |
| usnet | 550 | FF-KSP K=50 hops | 15.602% | 5.913% | +9.690 | [+9.460, +9.936] | Strict significantly higher |
| usnet | 650 | KSP-FF K=50 hops | 19.553% | 9.053% | +10.500 | [+10.332, +10.665] | Strict significantly higher |
| usnet | 650 | FF-KSP K=50 hops | 19.553% | 10.736% | +8.818 | [+8.655, +8.979] | Strict significantly higher |
| usnet | 900 | KSP-FF K=50 hops | 27.637% | 19.151% | +8.486 | [+8.233, +8.751] | Strict significantly higher |
| usnet | 900 | FF-KSP K=50 hops | 27.637% | 21.422% | +6.215 | [+5.998, +6.442] | Strict significantly higher |

## Blocking-Load AUC

| Topology | Method | Load-normalized AUC |
|---|---|---:|
| usnet | Frozen Strict v1.3 pure-RMSA transfer | 0.19994 |
| usnet | KSP-FF K=50 hops | 0.09942 |
| usnet | FF-KSP K=50 hops | 0.11575 |
| jpn48 | Frozen Strict v1.3 pure-RMSA transfer | 0.19022 |
| jpn48 | KSP-FF K=50 hops | 0.09968 |
| jpn48 | FF-KSP K=50 hops | 0.09773 |
