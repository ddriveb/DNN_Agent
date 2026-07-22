# Same-State Action Difference Summary

These are shadow comparisons on the Strict trajectory; heuristic actions are computed but not executed.

| Comparison/category | Count | Share of decisions |
|---|---:|---:|
| strict_vs_ffksp:block | 80265 | 6.69% |
| strict_vs_ffksp:mod | 79751 | 6.65% |
| strict_vs_ffksp:mod+block | 35945 | 3.00% |
| strict_vs_ffksp:path | 116178 | 9.68% |
| strict_vs_ffksp:path+block | 20249 | 1.69% |
| strict_vs_ffksp:path+mod | 318969 | 26.58% |
| strict_vs_ffksp:path+mod+block | 113429 | 9.45% |
| strict_vs_ffksp:same | 435214 | 36.27% |
| strict_vs_ksp:block | 102899 | 8.57% |
| strict_vs_ksp:mod | 75694 | 6.31% |
| strict_vs_ksp:mod+block | 36788 | 3.07% |
| strict_vs_ksp:path | 91548 | 7.63% |
| strict_vs_ksp:path+block | 12480 | 1.04% |
| strict_vs_ksp:path+mod | 307981 | 25.67% |
| strict_vs_ksp:path+mod+block | 97721 | 8.14% |
| strict_vs_ksp:same | 474889 | 39.57% |

## Independent Closed-Loop Outcomes (Exact 5-Seed Pilot)

These outcomes align request identity after trajectories naturally diverge.

| Successful method set | Requests |
|---|---:|
| all_block | 11317 |
| ff_ksp_k50_hops | 6021 |
| ksp_ff_k50_hops | 7386 |
| ksp_ff_k50_hops+ff_ksp_k50_hops | 29912 |
| strict_v13 | 1795 |
| strict_v13+ff_ksp_k50_hops | 2821 |
| strict_v13+ksp_ff_k50_hops | 4575 |
| strict_v13+ksp_ff_k50_hops+ff_ksp_k50_hops | 236173 |

## First Action Divergence

```json
[
  {
    "topology": "cost239",
    "load_erlang": 600.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "cost239",
    "load_erlang": 600.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "cost239",
    "load_erlang": 600.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "cost239",
    "load_erlang": 600.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "cost239",
    "load_erlang": 600.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 300.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 300.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 300.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 300.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 300.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 375.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 375.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 375.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 375.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 375.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 475.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 475.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 475.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 475.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 475.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 650.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 650.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 650.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 650.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "jpn48",
    "load_erlang": 650.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "nsfnet",
    "load_erlang": 250.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "nsfnet",
    "load_erlang": 250.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "nsfnet",
    "load_erlang": 250.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "nsfnet",
    "load_erlang": 250.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "nsfnet",
    "load_erlang": 250.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 450.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 450.0,
    "seed": 5002,
    "first_action_divergence_request_index": 501
  },
  {
    "topology": "usnet",
    "load_erlang": 450.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 450.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 450.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 550.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 550.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 550.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 550.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 550.0,
    "seed": 5005,
    "first_action_divergence_request_index": 501
  },
  {
    "topology": "usnet",
    "load_erlang": 650.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 650.0,
    "seed": 5002,
    "first_action_divergence_request_index": 501
  },
  {
    "topology": "usnet",
    "load_erlang": 650.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 650.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 650.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 900.0,
    "seed": 5001,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 900.0,
    "seed": 5002,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 900.0,
    "seed": 5003,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 900.0,
    "seed": 5004,
    "first_action_divergence_request_index": 500
  },
  {
    "topology": "usnet",
    "load_erlang": 900.0,
    "seed": 5005,
    "first_action_divergence_request_index": 500
  }
]
```
