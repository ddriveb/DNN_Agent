# FF-KSP Implementation Audit

## Function

`ff_ksp_highest_mod_action` is defined in `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py`.

## Algorithm

1. For each path (in the provided hops/km order), determine the highest feasible modulation (smallest required FS).
2. Gather all candidate (path, best_mod, block) tuples whose action is legal.
3. Sort globally by `(start_slot, path_idx, action_idx)` and return the first.

## Distinction from KSP-FF

- KSP-FF is path-first: it scans paths in order, picks the first feasible block on the first path with a feasible modulation.
- FF-KSP is spectrum-first: it scans the lowest start slots across all paths and picks the path that can use that slot with the highest feasible modulation.

## Sanity check

The smoke test prints the KSP-FF and FF-KSP choices for a sample state and reports whether they agree. When the network is lightly loaded, the two heuristics often choose the same first-fittable block on the shortest path; under contention they diverge because FF-KSP prioritizes the globally lowest start slot.
