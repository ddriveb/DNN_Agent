# PROTOCOL LOCK — DeepRMSA vs KSP-FF Paper-Standard Pure RMSA Reproduction

Reproduction target: **"Reinforcement learning for dynamic resource allocation
in optical networks: hype or hope?" (Doherty et al., 2025)** — pure RMSA
protocol, DeepRMSA vs KSP-FF fair comparison.

This file pins every protocol degree of freedom. All pipeline stages
(A: unit tests, B: smoke, C: KSP parity, D: DeepRMSA training) must read their
parameters from `RUN_CONFIG.json`, which is generated from this lock.
Do not change any value here without regenerating every artifact in this
directory.

## 1. Published reference values (paper)

| Topology | Slots | Load (Erlang) | DeepRMSA (published) | KSP-FF K=5 km | KSP-FF K=5 hops | KSP-FF K=50 hops |
|---|---:|---:|---:|---:|---:|---:|
| NSFNET  | 100 | 250 | 4.00% | 5.00 ± 0.29% | 2.93 ± 0.22% | 2.33 ± 0.25% |
| COST239 | 100 | 600 | 5.75% | 6.69 ± 0.35% | 3.80 ± 0.39% | 2.61 ± 0.36% |

Parity acceptance criterion (Stage C): our 10-seed **mean** blocking
probability of *KSP-FF K=5 km* must fall inside the paper's published
`mean ± 2·std` interval. The K=5 hops and K=50 hops variants are reported
against the same interval as secondary evidence.

## 2. Topologies

| Name in code | Role | Nodes | Undirected links | Directed arcs |
|---|---|---:|---:|---:|
| `xlron_nsfnet_deeprmsa` | NSFNET | 14 | 22 | 44 |
| `cost239_deeprmsa` (local, `paper_topologies.py`) | COST239 | 11 | 26 | 52 |

- NSFNET edge lengths are identical (link-for-link, 1-based minus 1) to the
  upstream DeepRMSA `linkmap` in `DeepRMSA/K-SP-FF benchmark_NSFNET.py`
  (verified 2026-07-17; all 22 lengths match).
- COST239 uses the **DeepRMSA variant** (`cost239_deeprmsa`): distances from
  upstream `DeepRMSA/Cost_239.m` (2× great-circle, max 2620 km), identical to
  XLRON's `cost239_deeprmsa_undirected.json`. **Not** the repository's
  `xlron_cost239_ptrnet_real` variant — with PtrNet-real distances the 600
  Erlang K=5 km baseline blocks ~3.2% instead of the paper's 6.69±0.35%
  (discovered empirically in the first Stage C run, 2026-07-17; see
  PREFLIGHT_AUDIT.md addendum).
- All nodes may be source and destination.

## 3. Spectrum model

- 100 frequency-slot units (FSU) per link **per direction**.
- FSU width 12.5 GHz. Guard band = 1 FSU (included in the allocated block).
- **Dual-fiber direction-independent spectrum**: arc (u→v) and arc (v→u) have
  independent occupancy arrays. A lightpath u→v blocks slots only on the u→v
  arcs. (Matches upstream `LINK_NUM = 44` directed links for NSFNET.)
- Continuity: identical slot set on every arc of the path.
- Contiguity: one contiguous block per lightpath.

## 4. Modulation formats (paper-standard reaches)

| Format | Reach (km) | Spectral efficiency (b/s/Hz) |
|---|---:|---:|
| BPSK  | 10000 | 1 |
| QPSK  | 2500  | 2 |
| 8QAM  | 1250  | 3 |
| 16QAM | 625   | 4 |

- Reach test is **inclusive** (`path_len <= reach`), per upstream `cal_FS`.
- Required FS for a request on a path:
  `FS = ceil(bitrate_gbps / (SE · 12.5)) + 1`  (guard band included).
- Modulation per path: always the **highest-SE feasible** format
  (implicit in upstream `cal_FS`; explicit in our code).

## 5. Traffic model (paper-standard)

- Requests: `(src, dst, arrival_time, holding_time, bit_rate)` — optical only.
  **No** C-side, MEC, split, deadline, or server capacity semantics.
- OD selection: uniform over all `N·(N−1)` **ordered** pairs, src != dst,
  implemented as a single uniform pair-index draw
  (upstream: `Src_Dest_Pair[np.random.randint(0, num_src_dest_pair)]`).
- Inter-arrival: exponential with mean `arrival_interval` (Poisson arrivals);
  zero draws are resampled (upstream `while time_to == 0`).
- Holding time: exponential with mean 10 s; **resampled while
  `ttl == 0 or ttl >= 2 · mean`** (upstream truncation).
- Bit rate: integer uniform in [25, 100] Gbps (`randint(25, 101)`).
- Offered load: `arrival_interval = mean_holding / load_erlang`.
  - NSFNET: 250 Erlang → arrival_interval = 10/250 = 0.04
  - COST239: 600 Erlang → arrival_interval = 10/600 ≈ 0.0166667

## 6. Measurement protocol

- Warmup: 3000 requests (not scored). Measured: next 10000 requests.
- Blocking probability = blocked / 10000.
- Release discipline: event-based, at or before each arrival all lightpaths
  with `release_time <= arrival_time` are released first; ties broken FIFO
  (insertion order), matching upstream `request_set` iteration.
- Same `(topology, seed)` trace is shared by every method (KSP variants and
  DeepRMSA evaluation) — traces are regenerated from the seed, never
  re-sampled per method.
- 10 independent test seeds per topology: **101..110**.
- DeepRMSA training seeds: **42, 43, 44**; validation seeds 42001–42003.
  Training trace seed per episode = `train_seed · 100000 + episode`.

## 7. Methods

### 7.1 KSP-FF (three variants)
1. **KSP-FF K=5 km**: K=5 candidate paths ordered by total length (km).
2. **KSP-FF K=5 hops**: K=5 paths ordered by hop count, ties by km.
3. **KSP-FF K=50 hops**: K=50 paths ordered by hops, ties by km.

All variants: scan paths in order; per path use the highest-SE feasible
modulation; First-Fit = lowest-start-slot contiguous block with
`size >= FS`; allocate at the block start (left-fit). First feasible path
wins; if none, the request is blocked.

### 7.2 Local DeepRMSA (source-semantic reimplementation)
- K=5 candidate paths ordered by **km** (upstream precomputed k-shortest by
  length), M=1 → action space = 5 (path-only action).
- Policy/value networks: 5 × 128 ELU MLPs, upstream initializers
  (hidden layers N(0, 0.3²)/bias 0.1; normalized heads).
- **No invalid-action mask**: all 5 outputs always selectable; choosing a
  path without a feasible block blocks the request (reward −1).
- Modulation: highest-SE feasible on the chosen path. Spectrum: First-Fit.
- State: upstream model-1 encoding — src/dst one-hot + per path:
  `(FS−5.5)/3.5`; for the first M=1 block with `size >= FS`:
  `2·(start−50)/100`, `(size−8)/8`; total free FS `2·(Σsizes−50)/100` and
  mean block size `(mean−4)/4` computed over **all** free blocks on the path
  (not only FS-eligible ones); path with no eligible block → all −1.
- Training: single-agent A2C port of upstream A3C worker; γ = 0.95,
  Adam lr = **1e-4**, entropy coef 0.01, value coef 1.0, **grad-clip 5**,
  **raw (unnormalized) advantages** — see deviation notes D6/D7 in
  PREFLIGHT_AUDIT.md for the measured justification of each deviation from
  upstream optimizer hyperparameters (all algorithm semantics unchanged).
  Overlapping window updates: store transitions, update on a window of
  `2·batch−1 = 399` transitions, drop the oldest `batch = 200` after each
  update (upstream "window-based training"). Bootstrap value 0.0.
  First 3000 training requests are warmup (no experience stored).
  Exploration: upstream inverted ε-schedule — sample from the policy with
  probability ε, argmax otherwise; ε starts at 1.0 and decays by 1e-5 per
  gradient update, floor 0.05.
- Checkpoint selection: best validation mean blocking rate (argmax policy);
  each selected checkpoint is then evaluated on the 10 test seeds.
- ≥ 3 training seeds per topology.

## 8. Exclusions (hard constraints)

- NO `snap24_gnutella_reach` topology, NO `deep_rmsa_snap24_reach_mixed.pt`.
- NO K=3/M=1 masked DeepRMSA and no legal-action mask anywhere in DeepRMSA.
- NO SA-HMARL C/MEC request generator, split/server, C-mask, PPO-C, DF_C.
- NO deadline, server capacity, or server-overload semantics.
- The legacy SNAP24 masked result (25.7067%) must not enter the main
  comparison table; appendix only, labelled
  "Legacy masked DeepRMSA-style K=3/M=1 on custom SNAP24".

## 9. Known deviations from repo-default components (see PREFLIGHT_AUDIT)

1. Repo `DEFAULT_MODULATIONS` reaches (4000/2000/1000/500) differ from the
   paper (10000/2500/1250/625) → custom paper registry is used.
2. `OpticalNetwork` shares one spectrum array per undirected edge → this
   pipeline uses directed arcs (dual-fiber direction-independent).
3. `generate_od_requests` does not truncate holding times > 2·mean → this
   pipeline's generator implements the upstream resampling.
4. `DeepRMSASourceSemanticAgent.encode_state` computes the total-free-FS and
   mean-block-size features over the FS-eligible, max_blocks-truncated block
   list; upstream computes them over all free blocks. This pipeline subclasses
   the agent (`PaperDeepRMSAAgent`) restoring the exact upstream aggregation.
