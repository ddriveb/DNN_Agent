# H=1 All-Zero Mechanical Proof

* Snapshots analyzed: 10
* Future traces per snapshot: 2
* Total trace records: 20
* H=1 invariant violations: 0
* Passed: True

All H=1 trace records satisfy the mechanical invariant:
* first action in both forks is successful,
* zero future requests are processed (H=1 = first action only),
* both forks report exactly 0 blocks, and
* ΔB_1 = blocks_strict - blocks_ksp = 0 for every record.
