# C-Side Direction-Level Lyapunov Rerank Diagnostic

This is an inference-time diagnostic only: no training and no checkpoint updates.

| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Avg selected R | C changed | C eval avg | H max mean | H mean | Adj mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v12_rank_only | 6.94% | 6.94% | 5.69% | 1.25% | 7.664/15.307 ms | 14.07 | 0.00% | 0.0 | 0.00 | 0.00 | 0.000 |
| dir_lyap_lambda0.1 | 6.94% | 6.94% | 5.69% | 1.25% | 7.596/15.350 ms | 14.08 | 1.39% | 0.9 | 5.93 | 1.70 | 0.371 |
| dir_lyap_lambda0.3 | 7.08% | 7.08% | 5.00% | 2.08% | 7.441/15.409 ms | 14.10 | 3.47% | 0.9 | 5.93 | 1.49 | 1.153 |
| dir_lyap_lambda0.5 | 7.22% | 7.22% | 5.83% | 1.39% | 7.311/15.254 ms | 13.80 | 4.44% | 0.9 | 5.94 | 1.49 | 1.921 |
| dir_lyap_lambda1 | 6.81% | 6.81% | 4.44% | 2.36% | 7.290/15.330 ms | 13.93 | 5.97% | 0.9 | 5.94 | 1.49 | 3.619 |

## Delta vs v1.2

| Method | Δ Blocking | Δ Raw empty | Δ NSB | Δ Overload | Δ Delay mean | Verdict hint |
|---|---:|---:|---:|---:|---:|---|
| dir_lyap_lambda0.1 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp | -0.068 ms | FAIL/MARGINAL |
| dir_lyap_lambda0.3 | -0.14 pp | -0.14 pp | +0.69 pp | +0.83 pp | -0.223 ms | FAIL/MARGINAL |
| dir_lyap_lambda0.5 | -0.28 pp | -0.28 pp | -0.14 pp | +0.14 pp | -0.353 ms | FAIL/MARGINAL |
| dir_lyap_lambda1 | +0.14 pp | +0.14 pp | +1.25 pp | +1.11 pp | -0.374 ms | FAIL/MARGINAL |