# Historical Trace Compatibility Audit

The original multi-topology pilot and the optimized rerun share checkpoint hashes, topology hashes, request settings, and request traces, but their manifest code hashes differ. The historical evaluator was an untracked evolving script, so its exact source snapshot cannot be reconstructed from Git. Its warmup spectrum state differs from the current code even when the first evaluated flat action matches.

Therefore historical per-request traces are not used as the exact-optimization oracle. Exactness is established against the current reference implementation on identical pre-decision states and independent paired closed-loop environments. Historical aggregate blocking remains useful as prior experimental context, not as a byte-for-byte software regression target.
