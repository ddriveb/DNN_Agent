# Inference-Time Top-K Rollout Reranking Findings

## Question

Can we widen the gap over DeepRMSA by adding short online sequence reasoning on top of the v1.2 R-ranker?

This probes the "H-step horizon ceiling" hypothesis:

> v1.2 may fail because its learned counterfactual score cannot distinguish actions whose differences appear after the training horizon.

If online rollout improves over v1.2, then a Bellman-augmented ranker or post-decision value network may be worth revisiting.

## Method

At each R decision:

1. Score all legal R actions with the frozen v1.2 ranker.
2. Keep Top-K actions.
3. For each Top-K action, simulate current allocation.
4. Roll out a short future horizon using frozen PPO-C + the same v1.2 R-ranker.
5. Select by lexicographic outcome:
   - current success
   - future blocked count
   - future no-suitable-block count
   - future server-overload count
   - v1.2 score as tie-breaker

No training or checkpoint changes are used.

## S80 Results

Scenario:

- slots = 80
- workload = `default3`
- arrival interval = `0.15s`
- holding = `4-10s`
- size = `5-40MB`

### Top-3, H=2

| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 | Rollout used | Changed vs v1.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 rank-only | 7.42% | 7.42% | 6.25% | 1.17% | 7.994/16.450 | 12.2/19.5 ms | 0.00% | 0.00% |
| rollout Top-3 H=2 | 7.42% | 7.42% | 6.25% | 1.17% | 7.994/16.450 | 88.4/148.6 ms | 91.67% | 0.00% |
| DeepRMSA | 7.75% | 7.75% | 6.67% | 1.08% | 9.906/18.621 | 9.8/13.3 ms | 0.00% | 0.00% |

### Top-5, H=3

| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 | Rollout used | Changed vs v1.2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 rank-only | 6.94% | 6.94% | 5.69% | 1.25% | 7.664/15.307 | 11.9/20.1 ms | 0.00% | 0.00% |
| rollout Top-5 H=3 | 6.94% | 6.94% | 5.69% | 1.25% | 7.664/15.307 | 187.0/340.2 ms | 92.50% | 0.00% |
| DeepRMSA | 7.22% | 7.22% | 6.11% | 1.11% | 9.500/17.742 | 10.1/13.9 ms | 0.00% | 0.00% |

## Interpretation

The rollout mechanism was actually invoked on more than 90% of requests, so this is not a coverage issue.  However, it never selected a different action from v1.2.

This means that within the current R action set:

1. The Top-K v1.2 actions have indistinguishable short-horizon rollout outcomes.
2. v1.2's own score already agrees with the short online rollout objective.
3. Adding online sequence search only increases decision time without changing behavior.

## Verdict

**STOP_BELLMAN_AUGMENTATION_FOR_CURRENT_ACTION_SPACE.**

The "H-step horizon ceiling" hypothesis is not supported by this diagnostic for the current action space.  A Bellman-augmented ranker is unlikely to improve v1.2 unless we first change the action space or candidate generation.

## Next Options

Further R-side gains require changing what actions are available, not just how current actions are scored:

1. Expanded path set: test `k=10` / congestion-aware extra paths.
2. Expanded slot-start placement: expose left/right/center-fit or explicit start-slot candidates.
3. System-level mechanisms: deadline-aware waiting, defragmentation, or admission control.

Without such changes, v1.2 remains the best current R-side method.
