# Protocol Lock

- fixed-C / all-OD pure RMSA; no PPO-C or DF_C.
- Strict v1.3 is the frozen PPO-R legal Top-30, 25-d common-future counterfactual Ranker.
- K_path=50, paths sorted by hops then km, four default modulations, max_blocks=10, blocks start_asc.
- No checkpoint, Top-30, feature, normalization, logits, gate, candidate-pool, or fallback change.
- Optimization is exact: one PPO feature build, vectorized candidate features, and optional diagnostic-only heuristic coverage.
