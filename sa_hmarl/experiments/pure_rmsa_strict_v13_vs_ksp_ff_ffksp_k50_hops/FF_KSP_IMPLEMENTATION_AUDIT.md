# FF-KSP Implementation Audit

Canonical function: `ff_ksp_highest_mod_action`. It first selects the highest feasible modulation per path, then sorts legal candidates by physical start slot, path order and action index. Thus it is spectrum-start-first, unlike path-first KSP-FF.
