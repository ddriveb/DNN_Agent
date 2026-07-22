# PREFLIGHT AUDIT — Component Reuse and Deviation Analysis

Audit date: 2026-07-17. Scope: every existing component the task lists as
"existing code to reuse", checked against the paper-standard protocol pinned
in `PROTOCOL_LOCK.md`. Conclusion: topology/KSP/agent/training scaffolding are
reused; spectrum state, traffic generation, and the DeepRMSA state aggregation
need paper-standard variants, implemented self-contained in this directory
(`paper_rmsa_core.py`, `paper_deeprmsa_agent.py`) without modifying shared
library code.

## Component-by-component verdict

| Component | Verdict | Reason |
|---|---|---|
| `network/topology_data.py` (`xlron_nsfnet_deeprmsa`) | REUSED as-is | NSFNET lengths verified identical to upstream DeepRMSA `linkmap` (all 22 links). |
| `cost239_deeprmsa` (local `paper_topologies.py`) | REPLACED repo variant | Upstream `Cost_239.m` / XLRON `cost239_deeprmsa` distances (2× real). The repo's `xlron_cost239_ptrnet_real` is a different COST239; see addendum below. |
| `network/ksp.py::get_k_shortest_paths` | REUSED as-is | Provides `sort_by="km"` (Yen, weighted) and `sort_by="hops"` (hop count, km tiebreak) exactly as the protocol needs. |
| `network/modulation.py::ModulationRegistry` / `ModulationFormat` | REUSED with custom table | Registry accepts a custom table; paper reaches (10000/2500/1250/625) are injected. Repo `DEFAULT_MODULATIONS` (4000/2000/1000/500) is **not** paper-standard and is not used. |
| `agents/deep_rmsa_source_semantic_agent.py` | REUSED via subclass | Unmasked upstream action semantics, upstream initializers, per-path feature layout all correct. One deviation fixed by subclassing, see D4 below. |
| `training/train_deeprmsa_source_semantic_k50.py` | Pattern reused, not called | It hard-codes COST239 K=50, episode-level (non-overlapping) updates and repo-default reaches. Our trainer implements the upstream overlapping-window schedule, K=5, both topologies. |
| `baselines/rmsa_baselines.py::ksp_ff_highest_mod_action` | Semantics replicated in core | It operates on the masked SA-HMARL observation with undirected shared spectrum; our core implements the same path-ordered/highest-mod/First-Fit rule on directed arcs (unit-tested against the upstream oracle). |
| `evaluation/optical_only_rmsa_env.py` / `optical_only_rmsa_evaluator.py` | Not used at runtime | See deviations D2/D3. The evaluator also loads PPO-C / ranker checkpoints that are excluded by the protocol. |
| `experiments/v135_r_only_standard_and_blocking_diagnosis/deep_rmsa_parity_adapter.py` | REUSED as test oracle | Verified upstream-behavior extraction (`deeprmsa_fs`, `deeprmsa_judge_availability`, `deeprmsa_ksp_ff_action`, …) used as the reference oracle in Stage A unit tests. |

## Deviations from repo defaults (required by the paper protocol)

- **D1 — Modulation reaches.** Paper: BPSK/QPSK/8QAM/16QAM = 10000/2500/1250/625 km
  (upstream `cal_FS`, boundaries inclusive). Repo default: 4000/2000/1000/500.
  Fix: custom `ModulationRegistry` table in `paper_rmsa_core.py`
  (`paper_modulation_registry()`).

- **D2 — Direction-independent spectrum.** Upstream NSFNET has
  `LINK_NUM = 44` directed links: counter-directional requests on a fiber pair
  never contend. Repo `OpticalNetwork` keys `link_states` by
  `(min(u,v), max(u,v))` — one shared array per undirected edge.
  Fix: `PaperOpticalNetwork` in `paper_rmsa_core.py` keeps one occupancy array
  per **directed arc**.

- **D3 — Holding-time truncation.** Upstream resamples TTL while
  `ttl == 0 or ttl >= 2 · mean`. Repo `generate_od_requests` draws plain
  exponentials (only floored at 1e-3). Fix: `generate_paper_requests` in
  `paper_rmsa_core.py` implements the resampling, plus the upstream
  pair-index OD draw and zero-inter-arrival resampling.

- **D4 — DeepRMSA state aggregation.** Upstream computes "total available FS"
  and "mean FS-block size" over **all** contiguous free blocks on the path
  (`slotscontinue` from `mark_vector`), while the first-M block features use
  only blocks with `size >= required_FS`. The existing source-semantic agent
  computes the aggregates over the FS-eligible, `max_blocks`-truncated
  candidate list. Fix: `PaperDeepRMSAAgent` (in `paper_deeprmsa_agent.py`)
  subclasses the source-semantic agent and restores the exact upstream
  aggregation using an `all_free_blocks_per_path` observation field supplied
  by our core; action selection, masking-free semantics, and checkpoint format
  are inherited unchanged.

- **D5 — Training schedule.** Upstream: streaming requests, warmup 3000 before
  storing experience, gradient update on every window of `2·batch−1 = 399`
  transitions with the oldest 200 dropped (overlap), bootstrap 0.0, and an
  inverted ε-schedule (sample w.p. ε, argmax otherwise; ε: 1.0 → 0.05 by
  1e-5/update). Existing repo trainers run episode-level updates.
  Fix: `train_deeprmsa_paper.py` implements the upstream schedule with the
  audited agent's `optimize()` (window snapshot/restore around the call).

- **D6 — Learning rate (disclosed adaptation).** Upstream trains with Adam
  lr=1e-5 for millions of requests using parallel A3C workers. At this
  session's CPU budget (1M requests/run, single worker), lr=1e-5 stalls at the
  SP-FF collapse (validation ≈12.4% NSFNET / ≈14.8% COST239 at 475k requests;
  verified SP-FF = always-path-0). A probe at lr=1e-4 escaped the collapse
  within 75k requests. Stage D therefore uses lr=1e-4.

- **D7 — Gradient clipping and advantages (disclosed adaptation).** At
  lr=1e-4 with upstream grad-clip 40, training was seed-unstable: one COST239
  seed diverged (62% blocking), two seeds stayed parked at the SP-FF collapse.
  Measured on the worst seed: per-window advantage *normalization* prevented
  divergence but removed the early gradient signal needed to escape the
  collapse (still parked at 250k requests, even at lr 3e-4/5e-4); a higher
  entropy coefficient (0.05/0.1) had no effect; a **tighter grad clip of 5**
  let the worst seed escape the collapse and descend cleanly. Final Stage D
  config: lr=1e-4, raw advantages, grad-clip 5, entropy 0.01. Every other
  algorithmic detail (γ, batch/window, bootstrap, ε-schedule, ±1 reward) is
  upstream-exact.

## Addendum 2026-07-17 — COST239 variant correction

The first Stage C run (with `xlron_cost239_ptrnet_real`) produced COST239
K=5 km blocking of ~3.2% versus the paper's 6.69±0.35%, while NSFNET matched
(5.04% vs 5.00±0.29%). Investigation showed XLRON ships multiple COST239
variants; the DeepRMSA comparison in the paper corresponds to
`cost239_deeprmsa` (upstream `Cost_239.m`, distances = 2× great-circle, max
link 2620 km), whose longer paths force lower-order modulations and hence
higher slot occupancy. The pipeline switched COST239 to `cost239_deeprmsa`
(local edges in `paper_topologies.py`, 0-based) and Stage C was rerun.

## Explicit exclusions honored

No `snap24_gnutella_reach`, no `deep_rmsa_snap24_reach_mixed.pt`, no
K=3/M=1 masked agent, no C/MEC request generator, no split/server/deadline
semantics anywhere in this pipeline. Grep guards are included in Stage A
(`test_no_excluded_components`) to fail the pipeline if an excluded module is
imported by any pipeline source file.

## Verification plan mapping

| Protocol property | Verified in |
|---|---|
| FS boundaries (reach inclusivity, FS formula) | Stage A `test_fs_boundaries` (oracle: parity adapter) |
| Direction-independent spectrum | Stage A `test_direction_independence` |
| Continuity + contiguity | Stage A `test_continuity_contiguity`, Stage B conservation check |
| First-Fit block selection | Stage A `test_first_fit` (oracle: parity adapter) |
| Release order (FIFO, release-before-arrival) | Stage A `test_release_fifo` |
| Holding truncation | Stage A `test_holding_truncation` |
| Uniform all-OD | Stage A `test_uniform_all_od` |
| Program/traffic/metrics wiring | Stage B smoke |
| KSP-FF absolute numbers vs paper | Stage C parity check |
