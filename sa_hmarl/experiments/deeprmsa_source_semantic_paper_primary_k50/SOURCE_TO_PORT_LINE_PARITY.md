# Source-to-Port Line Parity: DeepRMSA → PyTorch Port

The port is intentionally not a line-by-line translation (the upstream is TensorFlow 1.x
and the environment/topology differ), but the following semantic elements are preserved.

| Upstream concept | Upstream file / lines | Port file / lines | Status |
|-------------------|----------------------|-------------------|--------|
| Source one-hot | DeepRMSA_Agent.py 360-363 | deep_rmsa_source_semantic_agent.py 50-55 | preserved |
| Destination one-hot | DeepRMSA_Agent.py 361,364 | deep_rmsa_source_semantic_agent.py 50-55 | preserved |
| State dim formula NODE_NUM*2 + k_path*(1+M*2+2) | Deep_RMSA_A3C.py 105-106 | deep_rmsa_source_semantic_agent.py 58 | preserved |
| Required FS normalization (x-5.5)/3.5 | DeepRMSA_Agent.py 395 | deep_rmsa_source_semantic_agent.py 73 | preserved |
| First-M block start normalization 2*(start-0.5*SLOT_TOTAL)/SLOT_TOTAL | DeepRMSA_Agent.py 402 | deep_rmsa_source_semantic_agent.py 77 | preserved |
| First-M block size normalization (size-8)/8 | DeepRMSA_Agent.py 403 | deep_rmsa_source_semantic_agent.py 78 | preserved |
| Total available FS normalization 2*(sum-0.5*SLOT_TOTAL)/SLOT_TOTAL | DeepRMSA_Agent.py 407 | deep_rmsa_source_semantic_agent.py 82 | preserved |
| Mean block size normalization (mean-4)/4 | DeepRMSA_Agent.py 408 | deep_rmsa_source_semantic_agent.py 83 | preserved |
| Unavailable path segment filled with -1 | DeepRMSA_Agent.py 380,392 | deep_rmsa_source_semantic_agent.py 62,70 | preserved |
| Action space k_path * M | Deep_RMSA_A3C.py 101-102 | deep_rmsa_source_semantic_agent.py (k_path=50, M=1) | preserved |
| Action decode path_id = action // M, FS_id = action % M | DeepRMSA_Agent.py 451-452 | deep_rmsa_source_semantic_agent.py via _decode_to_sahmarl inherited from deep_rmsa_agent.py 229-244 | preserved |
| No legal-action mask; invalid selection blocks | DeepRMSA_Agent.py 478 | deep_rmsa_source_semantic_agent.py 92-113 | preserved |
| Reward +1 admit / -1 block | DeepRMSA_Agent.py 478 | train_deeprmsa_source_semantic_k50.py 77,84 | preserved |
| 5-layer ELU MLP | AC_Net.py 85-93 | deep_rmsa_agent.py ACNetwork 23-48 | preserved |
| Hidden size 128 | AC_Net.py layer_size=128 | deep_rmsa_agent.py layer_size=128 | preserved |
| Policy head normalized-columns std=0.01 | AC_Net.py 36 | deep_rmsa_source_semantic_agent.py 39-42 | preserved |
| Value head normalized-columns std=1.0 | AC_Net.py 42 | deep_rmsa_source_semantic_agent.py 39-42 | preserved |
| Gradient clipping max_norm=40 | AC_Net.py 71,76 | deep_rmsa_agent.py 391-394 | preserved |
| Adam lr=1e-5 | Deep_RMSA_A3C.py 166 | train_deeprmsa_source_semantic_k50.py default | preserved |
| gamma=0.95 | Deep_RMSA_A3C.py 119 | train_deeprmsa_source_semantic_k50.py gamma=0.95 | preserved |
| entropy coefficient 0.01 | AC_Net.py 60 | deep_rmsa_agent.py entropy_coef=0.01 | preserved |

## Known differences from upstream

1. **Topology and K**: upstream uses NSFNET (14 nodes, K=5); this port uses COST239 (11 nodes, K=50).
2. **Training algorithm**: upstream uses asynchronous A3C with multiple CPU workers; this port uses a single-process A2C update and is labelled accordingly.
3. **Backbone weight initialization**: upstream relies on TensorFlow slim defaults; the PyTorch port uses normal(0,0.3) for hidden layers and normalized-column initialization only for output heads.
4. **No global network / local worker synchronization**: A2C maintains one set of parameters.
5. **No epsilon-greedy exploration schedule**: the A2C port relies on policy entropy for exploration.
