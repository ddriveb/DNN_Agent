# Complex5_v2 Debug Notes

Date: 2026-06-06

## What Was Broken

The complex5_v2 line looked like it was repeatedly failing, but the concrete
issues are more specific:

- `train_agent_c_bc.py` imported a non-existent `save_checkpoint`.
- The BC script did not print tracebacks or useful data-collection diagnostics.
- The complex5_v2 hard setting produces many states with no valid Agent-C action.
- The complex5 evaluation script reported `Deadline Met: 0%` because it read a
  missing `deadline_met` key as `False`.

## Fixes Applied

- `train_agent_c_bc.py`
  - Uses `torch.save` directly.
  - Defaults to `--device cpu`.
  - Prints full traceback on failure.
  - Flushes training/collection logs.
  - Creates checkpoint parent directories.
  - Reports teacher-data diagnostics:
    - total requests
    - no-valid C-mask count
    - teacher-none count
    - R-action-none count
    - env success/block count
    - failure reason count
    - valid action min/mean/max
    - split/server distribution

- `eval_complex5_offloading.py`
  - Treats successful allocations as deadline-satisfied when the env info dict
    does not explicitly include `deadline_met`.

## Key Diagnostic Result

Command:

```bash
PYTHONPATH=/mnt/d/project/DNN_Agent/sa_hmarl \
.venv/bin/python -u -m sa_hmarl.training.train_agent_c_bc \
  --num_episodes 20 \
  --requests_per_episode 120 \
  --epochs 2 \
  --batch_size 64 \
  --device cpu \
  --teacher_method df \
  --num_slots 16 \
  --num_servers 4 \
  --arrival_interval 0.20 \
  --edge_cost_max 60 \
  --split_profile complex5_v2 \
  --warm_start_c sa_hmarl/checkpoints/agent_c_frozen_r_snap24_reach_best.pt \
  --output_ckpt /tmp/agent_c_bc_diag.pt
```

Observed:

```text
total_requests: 2400
samples_collected: 433
no_valid_c_mask: 1967
env_success: 392
env_block: 41
failure reasons: server_overload=25, no_suitable_block=16
valid_action_mean: 2.19
valid_action_min: 0
valid_action_max: 20
teacher split distribution: split4=413, split3=16, split2=2, split0=1, split1=1
```

## Interpretation

The hard complex5_v2 setting is much harder than expected. Most later requests
have no valid C action after resources are consumed, so BC collects only a small
number of usable teacher states. The collected DF teacher actions are also
heavily collapsed to `split4`, which is not a rich supervision signal.

The current full-eval file `experiments/complex5v2_bc_eval.md` shows the hard
scenario has roughly:

```text
Complex5-Default blocking ~= 0.389
DF blocking ~= 0.385
IWD blocking ~= 0.373
```

So the previously quoted baseline blocking of `0.13-0.15` does not match this
full evaluation setting.

## Recommendation

Do not continue long RL runs until the scenario is moderated or the teacher
dataset is improved. Good next steps:

1. Use a pressure setting where valid C actions remain available for most
   requests, e.g. target `valid_action_mean >= 5` and `no_valid_c_mask < 50%`.
2. Collect BC data from multiple pressure levels, not only the hardest setting.
3. Use a stronger/more diverse teacher than pure DF, e.g. min-cost among
   DF/IWD/Greedy per state, to avoid split4 collapse.
4. Only resume RL fine-tuning after BC evaluation is at least competitive with
   the best rule baseline.

