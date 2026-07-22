# Action and Reward Semantics Audit

## Action space

* K = 50 candidate paths (sorted by hops).
* M = 1 spectrum block choice per path.
* Total actions = 50.
* Action decode: `path_id = action_id // M`, `FS_id = action_id % M`.
* With M=1, FS_id is always 0 and selects the first available contiguous block on the chosen path.

## Execution semantics

* If `path_id` is out of range → block.
* If the path has no feasible modulation → block.
* If FS_id has no available block → block.
* No fallback to KSP, PPO-R, or action 0.
* No re-sampling from a masked distribution.

## Reward semantics

* Successful allocation: reward = +1.
* Any blocking outcome (including invalid policy choice): reward = -1.
* The transition is stored for the selected action_id even when the execution fails.

## Differences from masked adapter

The masked adapter (`deep_rmsa_agent.py`) restricts the policy to actions that decode to
a feasible SA-HMARL flat action and uses `_credit_forced_block` to fold no-valid-action
blocking into the previous transition.  The source-semantic port removes both behaviors.
