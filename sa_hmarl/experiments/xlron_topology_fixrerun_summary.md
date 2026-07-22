# XLRON Topology Fixrerun Summary

## Background

The observation-builder bug described in `xlron_topology_fairness_review_note.md`
has been fixed in `sa_hmarl/env/observation_builder.py`.  The fix ensures that
Agent-C and Agent-R candidate-path construction both explicitly respect
`env.path_sort_strategy`, eliminating the "observation sorted by km, execution
sorted by hops" mismatch.

After the S100 main-experiment fixrerun showed unchanged blocking/NSB/overload
conclusions, the four XLRON topology transfer experiments were rerun with the
same fixed code.

## Commands

The same command template was used for all four topologies:

```bash
PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology TOPOLOGY_NAME \
  --methods ppo_c+v12,ppo_c+ksp_ff_k50_hops \
  --episodes 20 \
  --requests_per_episode 80 \
  --seeds 3030,4040,5050,6060,7070 \
  --output_json sa_hmarl/experiments/TOPOLOGY_NAME_s100_r_compare_fixrerun.json \
  --output_md sa_hmarl/experiments/TOPOLOGY_NAME_s100_r_compare_fixrerun.md
```

Generated files:

| Topology | Markdown | JSON |
|---|---|---|
| `xlron_cost239_ptrnet_real` | `xlron_cost239_s100_r_compare_fixrerun.md` | `xlron_cost239_s100_r_compare_fixrerun.json` |
| `xlron_german17` | `xlron_german17_s100_r_compare_fixrerun.md` | `xlron_german17_s100_r_compare_fixrerun.json` |
| `xlron_nsfnet_deeprmsa` | `xlron_nsfnet_deeprmsa_s100_r_compare_fixrerun.md` | `xlron_nsfnet_deeprmsa_s100_r_compare_fixrerun.json` |
| `xlron_jpn48` | `xlron_jpn48_s100_r_compare_fixrerun.md` | `xlron_jpn48_s100_r_compare_fixrerun.json` |

## Results (after bug fix)

| Topology | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other |
|---|---|---:|---:|---:|---:|---:|---:|
| `xlron_cost239_ptrnet_real` | v1.2 | 0.20% | 0.20% | 0.00% | 0.20% | 0.00% | 0.00% |
| `xlron_cost239_ptrnet_real` | KSP-FF K50 hops | **0.00%** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `xlron_german17` | v1.2 | **0.97%** | 0.97% | 0.27% | 0.70% | 0.00% | 0.00% |
| `xlron_german17` | KSP-FF K50 hops | 1.09% | 1.62% | 0.07% | 1.01% | 0.00% | 0.00% |
| `xlron_nsfnet_deeprmsa` | v1.2 | **3.09%** | 3.15% | 0.76% | 2.18% | 0.14% | 0.01% |
| `xlron_nsfnet_deeprmsa` | KSP-FF K50 hops | 3.26% | 3.28% | 0.83% | 2.23% | 0.05% | 0.16% |
| `xlron_jpn48` | v1.2 | 6.19% | 6.19% | 4.59% | 1.05% | 0.51% | 0.04% |
| `xlron_jpn48` | KSP-FF K50 hops | **5.04%** | 6.56% | 1.94% | 2.81% | 0.27% | 0.01% |

## Comparison: before vs after bug fix

| Topology | Metric | v1.2 (old) | v1.2 (fix) | KSP-FF K50 (old) | KSP-FF K50 (fix) | Conclusion change |
|---|---|---:|---:|---:|---:|---|
| `xlron_cost239_ptrnet_real` | Blocking | 0.20% | 0.20% | 3.95% | 0.00% | KSP-FF now **better** |
| `xlron_german17` | Blocking | 0.97% | 0.97% | 2.96% | 1.09% | v1.2 still better, gap narrows |
| `xlron_nsfnet_deeprmsa` | Blocking | 3.09% | 3.09% | 9.55% | 3.26% | v1.2 still slightly better, gap narrows |
| `xlron_jpn48` | Blocking | 6.19% | 6.19% | 9.70% | 5.04% | KSP-FF now **better** |

## What changed

- **v1.2 blocking is unchanged** across all four topologies.  This makes sense:
  the observation-builder bug primarily affected the R-side backend that
  explicitly overrides `env.k` / `path_sort_strategy` / `block_sort_strategy`
  (i.e., KSP-FF K50 hops).  v1.2 uses the default `k_paths=5, km` setting and
  was therefore less exposed to the mismatch.
- **KSP-FF K50 hops improved dramatically** after the fix, especially on
  `xlron_cost239_ptrnet_real` (3.95% → 0.00%) and `xlron_nsfnet_deeprmsa`
  (9.55% → 3.26%).  This confirms the bug was heavily penalizing the tuned
  heuristic baseline.
- **Decision times are now more honest**: v1.2 decision latency is 43–100 ms
  because it runs a neural ranker over candidates, while KSP-FF stays at
  33–36 ms as a pure heuristic.

## Updated conclusion

Before the bug fix, the claim was:

> "v1.2 consistently beats KSP-FF K50 hops on the four imported XLRON
> topologies."

After the fix, this claim must be revised:

> **v1.2 no longer consistently beats KSP-FF K50 hops on the four XLRON
> topologies.**
> KSP-FF K50 hops achieves lower blocking on `xlron_cost239_ptrnet_real`
> (0.00% vs 0.20%) and `xlron_jpn48` (5.04% vs 6.19%).  v1.2 remains slightly
> better on `xlron_german17` (0.97% vs 1.09%) and `xlron_nsfnet_deeprmsa`
> (3.09% vs 3.26%), but the margins are small.

## Fairness caveats (unchanged)

- PPO-C and v1.2 are both transferred without per-topology retraining.
- KSP-FF is a tuned heuristic evaluated directly on each topology.
- DeepRMSA is still not included because topology-specific checkpoints are
  missing.

## Recommendation

Do **not** use the old XLRON results (`xlron_topology_r_backend_comparison.md`)
as evidence that v1.2 universally beats KSP-FF K50 hops across topologies.
The fixrerun files above supersede the old comparison for blocking-rate claims.

If the paper needs a positive v1.2 transfer claim, the safest formulation is:

> "On the original S100 topology, PPO-C + v1.2 achieves 0.92% blocking,
> outperforming DeepRMSA (1.29%) and KSP-FF K50 hops (1.05%).  On the four
> imported XLRON topologies, transferred v1.2 is competitive with the tuned
> KSP-FF K50 hops heuristic, with small advantages on German17 and NSFNET."

The stronger claim "v1.2 consistently beats KSP-FF on all XLRON topologies"
should be removed.
