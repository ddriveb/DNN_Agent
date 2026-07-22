# Path Ordering Audit

- K_path = 50
- path_sort_strategy = `hops`
- Same-hop tie-break: by km (ascending).

The observation builder uses `get_k_shortest_paths(..., sort_by='hops')` and the underlying KSP implementation preserves path order by hops, with shorter km paths appearing earlier among equal-hop paths. This is verified in the smoke test by printing the first 10 paths for a sample OD pair and confirming that hops are non-decreasing and km is non-decreasing within equal-hop groups.

## Sample first-10 paths

See the smoke-test stdout for per-topology sample paths.
