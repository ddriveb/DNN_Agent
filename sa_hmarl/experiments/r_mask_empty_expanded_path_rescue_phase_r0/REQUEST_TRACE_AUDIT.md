# Request Trace Audit

Each task uses the same generated request trace for all four methods, ensuring a paired comparison. The per-task `.triad.jsonl.gz` files contain block identity records linking the four method outcomes per request.

- Output trace: `paired_block_identity.jsonl.gz`
- Probe events: `mask_empty_rescue_events.jsonl.gz`
