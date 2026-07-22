# Smoke Validation

```json
{
  "phase": "smoke",
  "cells": 24,
  "request_sync_audit": {
    "total_rows": 24,
    "request_id_mismatch_count_sum": 0,
    "final_event_queue_length_sum": 0,
    "selected_mask_false_count_sum": 0,
    "invalid_path_after_legal_action_sum": 0,
    "modulation_reach_after_legal_action_sum": 0,
    "no_suitable_block_after_legal_action_sum": 0,
    "action_observation_consistency_error_sum": 0,
    "all_gates_passed": true
  },
  "per_method": {
    "ppo_c+ppo_r_top1": {
      "topology": "xlron_nsfnet_deeprmsa",
      "seed": 3030,
      "evaluated_requests": 50,
      "blocked": 0,
      "blocking_rate": 0.0,
      "c_no_valid_action": 0,
      "r_no_valid_action": 0,
      "no_suitable_block": 0,
      "server_overload": 0,
      "deadline_failure": 0,
      "other_failure": 0
    },
    "ppo_c+ksp_ff_highest": {
      "topology": "xlron_nsfnet_deeprmsa",
      "seed": 3030,
      "evaluated_requests": 50,
      "blocked": 0,
      "blocking_rate": 0.0,
      "c_no_valid_action": 0,
      "r_no_valid_action": 0,
      "no_suitable_block": 0,
      "server_overload": 0,
      "deadline_failure": 0,
      "other_failure": 0
    },
    "ppo_c+strict_v13": {
      "topology": "xlron_nsfnet_deeprmsa",
      "seed": 3030,
      "evaluated_requests": 50,
      "blocked": 0,
      "blocking_rate": 0.0,
      "c_no_valid_action": 0,
      "r_no_valid_action": 0,
      "no_suitable_block": 0,
      "server_overload": 0,
      "deadline_failure": 0,
      "other_failure": 0
    },
    "df_c+ppo_r_top1": {
      "topology": "xlron_nsfnet_deeprmsa",
      "seed": 3030,
      "evaluated_requests": 50,
      "blocked": 0,
      "blocking_rate": 0.0,
      "c_no_valid_action": 0,
      "r_no_valid_action": 0,
      "no_suitable_block": 0,
      "server_overload": 0,
      "deadline_failure": 0,
      "other_failure": 0
    },
    "df_c+ksp_ff_highest": {
      "topology": "xlron_nsfnet_deeprmsa",
      "seed": 3030,
      "evaluated_requests": 50,
      "blocked": 0,
      "blocking_rate": 0.0,
      "c_no_valid_action": 0,
      "r_no_valid_action": 0,
      "no_suitable_block": 0,
      "server_overload": 0,
      "deadline_failure": 0,
      "other_failure": 0
    },
    "df_c+strict_v13": {
      "topology": "xlron_nsfnet_deeprmsa",
      "seed": 3030,
      "evaluated_requests": 50,
      "blocked": 0,
      "blocking_rate": 0.0,
      "c_no_valid_action": 0,
      "r_no_valid_action": 0,
      "no_suitable_block": 0,
      "server_overload": 0,
      "deadline_failure": 0,
      "other_failure": 0
    }
  }
}
```