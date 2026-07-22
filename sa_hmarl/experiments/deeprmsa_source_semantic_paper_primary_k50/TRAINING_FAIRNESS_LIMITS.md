# Training Fairness Limits

## Comparison regimes

1. **Evaluation-mechanics matched**: all methods are evaluated on the same
   PAPER_PRIMARY_K50 environment, request traces, warm-up, and metric definitions.
   This is satisfied by the current experiment.

2. **Training-regime matched**: Strict v1.3 was evaluated zero-shot from a ranker
   trained on the original coupled C/R task.  DeepRMSA-source-semantic was trained
   natively on optical-only data.  Therefore algorithmic superiority claims must be
   deferred until Strict v1.3 is also native-trained on the same optical-only data budget.

## Allowed statements

* "native-trained DeepRMSA-source-semantic vs zero-shot Strict v1.3 under
  PAPER_PRIMARY_K50 evaluation mechanics."
* "DeepRMSA-source-semantic converges / does not converge on PAPER_PRIMARY_K50."

## Disallowed statements

* "DeepRMSA is better than Strict v1.3" (training regimes differ).
* "DeepRMSA-source-semantic is Original DeepRMSA" (topology/K/training differ).
