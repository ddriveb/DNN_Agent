# COST239 Paper-Style Diagnostic Summary

> **Scope**: This is a paper-style diagnostic, not a strict reproduction of the XLRON/Hype-or-Hope benchmark. The goal is to stress the local `xlron_cost239_ptrnet_real` environment closer to the optical-load regime reported in the cited papers, while keeping the local C-side/server/deadline/compute machinery unchanged.

## 1. Literature Alignment

### 1.1 Key settings reported in the two papers

**Doherty et al. (2025), "Reinforcement learning for dynamic resource allocation in optical networks: hype or hope"**

The paper explicitly states the DeepRMSA problem settings that it recreates (page 6, lines 631–643):

- Topologies: NSFNET, **COST239** (11 nodes, 52 directed single-fibre links, avg shortest path 1.56 hops / 1810 km), USNET, JPN48.
- **Task**: dynamic **RMSA** (routing, modulation and spectrum assignment).
- **Link resources**: dual-fibre links, **100 FSUs per fibre**, 12.5 GHz FSU width.
- **Candidate paths**: KSP-FF with **K = 50** and paths ordered by **number of hops** is the best heuristic.
- **Traffic model**: uniform traffic probability between each node pair; Poisson arrivals/exponential holding times.
- **Mean service holding time**: **10 time units**.
- **Data rate**: uniform random from **25 to 100 Gbps** in 1 Gbps steps.
- **Modulation formats**: BPSK/QPSK/8QAM/16QAM with maximum reaches 10 000 / 2 500 / 1 250 / **625 km**.
- **Evaluation**: 3 000-request warm-up + 10 000 requests per episode; blocking probability vs. traffic load in Erlangs.
- **Reported DeepRMSA COST239 point**: load = **600 Erlang**, SBP ≈ 5.75% (RL), KSP-FF K=5 6.75%, KSP-FF K=50 hops 2.61%.

**Doherty et al. (2026), "Graph Transformers and Stabilized Reinforcement Learning for Large-Scale Dynamic Routing Modulation and Spectrum Allocation"**

- Same COST239 topology (11 nodes, 52 directed links).
- Same RMSA task and 100-FSU link model for the standard topologies.
- Benchmark heuristic: **KSP-FF with K = 50 ordered by hops** (or FF-KSP for JPN48).
- The focus is on large-scale topologies (USA100, TataInd) with 320 FSUs, but the standard benchmarks are inherited from the 2025 paper.

### 1.2 What can / cannot be aligned in the local system

| Paper setting | Local implementation | Alignable? | Notes |
|---|---|---|---|
| Topology `COST239` | `xlron_cost239_ptrnet_real` | **Partially** | Same 11 nodes; exact directed-link count may differ from the paper’s 52. |
| `num_slots = 100` | `--num_slots 100` | **Yes** | Matches the DeepRMSA benchmark. |
| Task = RMSA | Local env supports modulation-dependent reach | **Yes** | Default modulation profile is used. |
| `K = 50` candidate paths | `--ksp_ff_k50_hops_k_paths 50` and `v12_k50_hops` | **Yes** | Implemented in `eval_main_s100_system_comparison.py`. |
| Path order = `#hops` | `v12_k50_hops` / `ksp_ff_k50_hops` use `path_sort_strategy=hops` | **Yes** | Confirmed in `_r_backend_env_config`. |
| Path order = `#km` | default `ppo_c+v12` uses `--path_sort_strategy km` | **Yes** | Used as the local main-experiment default. |
| Mean holding time = 10 | `--holding_min/--holding_max` | **Partially** | We can set the mean near 10, but the local env also models server holding/compute. |
| Traffic load in Erlangs | `--arrival_interval` | **Indirectly** | No direct Erlang parameter; load ≈ mean_holding / arrival_interval, but server compute changes the effective capacity. |
| Data rate 25–100 Gbps | `--size_min_mb/--size_max_mb` | **Partially** | Local request size is in MB; we keep the same numerical range (5–30 MB) or scale it (15–50 MB) to increase optical pressure. |
| Single-lightpath requests | Local requests are single | **Yes** | No grooming. |
| No compute / deadline / MEC | Local system has PPO-C, servers, deadline, edge cost | **No** | These are extra local constraints; they are the main reason this cannot be a strict reproduction. |

## 2. Design of the Three Paper-Style Workloads

I calibrated three points so that aggregate blocking enters the 1–15% regime (closer to the paper’s reported range) while avoiding the case where blocking is dominated by artificial server saturation.

| Point | `arrival_interval` | `holding_min` | `holding_max` | `size_min_mb` | `size_max_mb` | Approx. optical stress | Rationale |
|---|---:|---:|---:|---:|---:|---|---|
| `paper_style_light` | 0.09 | 4.0 | 8.0 | 5 | 30 | Mean holding ≈ 6, load ≈ 67 Erlang-equivalent | Entry-level stress; should produce low-single-digit blocking. |
| `paper_style_mid` | 0.08 | 4.0 | 8.0 | 5 | 30 | Mean holding ≈ 6, load ≈ 75 Erlang-equivalent | Moderate stress; targets the 5–10% blocking range. |
| `paper_style_heavy` | 0.07 | 4.0 | 6.0 | 15 | 50 | Mean holding ≈ 5, load ≈ 71 Erlang-equivalent but larger requests | Higher per-request spectrum pressure; pushes blocking above 10%. |

**Why these values?**
- I first tried holding 8–12 with arrival interval 0.06–0.08, but blocking jumped to >20% and the server/compute side became a large contributor.
- Keeping holding modest (4–8 or 4–6) and instead tightening `arrival_interval` and/or raising `size` keeps the bottleneck closer to the optical layer.
- The exact Erlang numbers above are rough because the local environment adds server capacity, deadline, and edge-compute constraints that do not exist in the paper.

## 3. Results

All three methods were run with:

- seeds: `3030,4040,5050,6060,7070`
- episodes: `20`
- requests_per_episode: `80`
- ranking checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`

### 3.1 Paper-style light

**Config file**: `sa_hmarl/experiments/xlron_cost239_paper_style_light.{json,md}`

| Method | Blocking | Raw empty | NSB | Overload | Deadline | Other |
|---|---:|---:|---:|---:|---:|---:|
| `ppo_c+v12` | **4.19%** | 4.34% | 0.00% | 4.19% | 0.00% | 0.00% |
| `ppo_c+v12_k50_hops` | **3.72%** | 3.89% | 0.00% | 3.72% | 0.00% | 0.00% |
| `ppo_c+ksp_ff_k50_hops` | **3.12%** | 3.23% | 0.00% | 3.12% | 0.00% | 0.00% |

### 3.2 Paper-style mid

**Config file**: `sa_hmarl/experiments/xlron_cost239_paper_style_mid.{json,md}`

| Method | Blocking | Raw empty | NSB | Overload | Deadline | Other |
|---|---:|---:|---:|---:|---:|---:|
| `ppo_c+v12` | **8.21%** | 8.50% | 0.00% | 8.21% | 0.00% | 0.00% |
| `ppo_c+v12_k50_hops` | **7.41%** | 7.71% | 0.00% | 7.41% | 0.00% | 0.00% |
| `ppo_c+ksp_ff_k50_hops` | **6.75%** | 6.94% | 0.00% | 6.75% | 00.0% | 0.00% |

### 3.3 Paper-style heavy

**Config file**: `sa_hmarl/experiments/xlron_cost239_paper_style_heavy.{json,md}`

| Method | Blocking | Raw empty | NSB | Overload | Deadline | Other |
|---|---:|---:|---:|---:|---:|---:|
| `ppo_c+v12` | **13.40%** | 13.80% | 0.14% | 13.26% | 0.00% | 0.00% |
| `ppo_c+v12_k50_hops` | **13.39%** | 13.84% | 0.00% | 13.39% | 0.00% | 0.00% |
| `ppo_c+ksp_ff_k50_hops` | **12.35%** | 12.73% | 0.00% | 12.35% | 0.00% | 0.00% |

### 3.4 Gap relative to the local fixrerun baseline

For reference, the existing COST239 fixrerun (`xlron_cost239_s100_r_compare_fixrerun`) uses `arrival_interval=0.15`, `holding 4–10`, `size 5–30`:

| Method | Fixrerun blocking | Paper-style mid blocking | Change |
|---|---:|---:|---:|
| `ppo_c+v12` | 0.20% | 8.21% | **+8.01 pp** |
| `ppo_c+ksp_ff_k50_hops` | 0.00% | 6.75% | **+6.75 pp** |

## 4. Answers to the Required Questions

### Q1: What are the key COST239 benchmark settings in the papers?

See Section 1.1 above. The most important points for this diagnostic are:
- RMSA on COST239, 100 FSUs per link.
- KSP-FF with **K = 50** and paths ordered by **hops** is the strongest simple heuristic.
- Mean holding time = 10 units; load reported in Erlangs (e.g. 600 Erlang for DeepRMSA).
- Request bandwidth uniform 25–100 Gbps.

### Q2: Which settings can be aligned and which cannot?

**Aligned**: topology name, node count (11), `num_slots=100`, RMSA task, `K=50`, hops-vs-km path ordering, dynamic traffic.

**Not aligned**: the local system adds PPO-C split/server selection, MEC server capacity, deadlines, edge-compute costs, and uses MB-based request sizes. There is no direct Erlang parameter; load is controlled indirectly via `arrival_interval` and `holding_min/max`.

### Q3: Which local paper-style points did you choose and why?

- `paper_style_light`: `arrival_interval=0.09`, holding 4–8, size 5–30 → blocking ~3–4%.
- `paper_style_mid`: `arrival_interval=0.08`, holding 4–8, size 5–30 → blocking ~6.75–8.2%.
- `paper_style_heavy`: `arrival_interval=0.07`, holding 4–6, size 15–50 → blocking ~12.4–13.4%.

These cover the 1–15% range and are closer to the paper’s reported operating points than the fixrerun baseline. I kept holding moderate to prevent the extra server/compute constraints from becoming the dominant failure mode.

### Q4: What are the blocking gaps between the three methods under paper-style stress?

| Point | `ppo_c+v12` vs `ppo_c+ksp_ff_k50_hops` | `ppo_c+v12_k50_hops` vs `ppo_c+ksp_ff_k50_hops` |
|---|---:|---:|
| light | +1.06 pp | +0.60 pp |
| mid | +1.46 pp | +0.66 pp |
| heavy | +1.05 pp | +1.04 pp |

In all three points, `ppo_c+ksp_ff_k50_hops` has the lowest blocking; the local v1.2 ranker is behind the heuristic.

### Q5: What is the main source of the gap?

The evidence points to **a combination of candidate-path-set differences and ranker capability**, with C-side effects held constant:

- **Candidate path set matters**: in light and mid, simply giving v1.2 the same K=50/hops candidate set (`ppo_c+v12_k50_hops`) closes roughly half of the gap to KSP-FF.
- **Ranker capability matters**: even with the same candidate set, `ppo_c+v12_k50_hops` is still worse than `ppo_c+ksp_ff_k50_hops` at every point (up to ~1 pp at heavy load). So the v1.2 ranker does not use the larger path set as effectively as first-fit.
- **C-side / offloading effects are not the main differentiator**: all three methods share the same PPO-C split/server selection; the only difference is the R-side decision. Blocking decomposition is dominated by `raw_empty`/`overload`, with negligible `deadline`/`other`.

### Q6: Does this diagnostic support the claim that the current COST239 fixrerun is too easy and underestimates paper-style optical differences?

**Yes, cautiously.**

- The fixrerun baseline shows **0.00%** blocking for KSP-FF and **0.20%** for v1.2, with almost no dynamic-range to distinguish methods.
- Under paper-style stress, KSP-FF K50 hops outperforms v1.2 by **1.0–1.5 percentage points**, and all methods enter a blocking range (3–15%) that is much closer to the paper’s reported regime.
- The caveat is that this is **not a strict reproduction**: the local server/deadline/compute layers mean the absolute blocking numbers are not directly comparable to the paper’s Erlang-load curves. The diagnostic therefore supports the qualitative judgment that the fixrerund workload is too light, but it cannot quantify the exact paper-style gap.

## 5. Conservative Conclusion

- The paper-style stress successfully moves COST239 evaluation out of the near-zero-blocking regime.
- It reveals that, under higher optical load, **KSP-FF K50 hops remains stronger than the local v1.2 ranker**, and that a non-trivial part of the gap is simply the wider/hops-ordered candidate path set.
- Because the local system adds compute/server/deadline constraints, these numbers should be treated as a **diagnostic**, not as a claim that the paper’s results have been reproduced.
- No old result files were overwritten; all diagnostic outputs are in `sa_hmarl/experiments/xlron_cost239_paper_style_*`.
