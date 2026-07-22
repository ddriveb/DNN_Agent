# Final Decision

* Mean blocking rate: **45.8000%**
* Std across test seeds: 0.0055%
* Training-seed variation: 0.0615%
* Converged: `False`

## Answers to required questions

1. Is the implementation truly unmasked? **Yes.**
2. Remaining differences from upstream? COST239/K=50 topology, single-process A2C, PyTorch implementation.
3. PAPER_PRIMARY_K50 blocking rate: 45.8000%.
4. Did it converge? False.
5. Stable vs KSP-FF? mean diff = -32.6000 pp, wins/ties/losses = {'wins': 0, 'ties': 0, 'losses': 12}.
6. Stable vs zero-shot Strict v1.3? mean diff = -29.7433 pp, wins/ties/losses = {'wins': 0, 'ties': 0, 'losses': 12}.
7. Conclusion scope: **evaluation-mechanics-matched**; training regimes differ.
8. Paper-table ready? **No**, because Strict v1.3 was zero-shot while DeepRMSA was native-trained.
