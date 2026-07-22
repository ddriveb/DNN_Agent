# KSP-FF Implementation Audit

Canonical function: `ksp_ff_highest_mod_action`. Loop order is path first, paths are K=50 in hops/km order, modulation minimizes required FS on that path, and the legal block with the smallest physical start slot is selected. It is not flat BPSK-first and not K=5.
