# Request Trace Parity

Overall: **PASS**

- `request_trace_exact_parity`: PASS; `{"states": 1000}`
- `fixed_seed_reproducibility`: PASS; `{"states": 1000}`
- `holding_truncation`: PASS; `{"min": 0.05638405472599321, "max": 19.95256774737728, "mean": 6.90742558073687}`
- `load_changes_arrival_trace`: PASS; `{"load100_mean_interarrival": 0.09966526117332182, "load600_mean_interarrival": 0.016610876862220257}`
- `pure_request_schema`: PASS; `{"fields": ["arrival_time", "bitrate_gbps", "dst_node", "holding_time", "req_id", "src_node"]}`
- `required_fs_known_values`: PASS; `{"25Gbps_BPSK": 3, "100Gbps_16QAM": 3}`
