# Baseline Profile

CPU latency was measured serially after 500 warmup requests. Each topology used
1,000 evaluated requests and three repeats. Diagnostic-only KSP-FF/FF-KSP
coverage calls were excluded so this is the reference production selector, not
the older diagnostic evaluator timing.

| Topology | Reference selector | Reference end-to-end |
|---|---:|---:|
| NSFNET | 10.277 ms | 15.044 ms |
| USNET | 17.934 ms | 25.670 ms |
| JPN48 | 21.818 ms | 34.203 ms |

The dominant reference costs were the second full PPO-R action-feature build
and repeated request-level field computation inside the 30-candidate feature
loop.
