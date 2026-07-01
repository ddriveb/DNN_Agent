"""Train the v1.2 planner-distilled amortized R-side controller.

This is a readability alias for ``train_r_counterfactual_ranking.py``.  The
training objective and implementation are shared with the original script:
offline H-step counterfactual planning targets are distilled into a lightweight
candidate-level RMSA scorer.
"""
from __future__ import annotations

import time

from sa_hmarl.training.train_r_counterfactual_ranking import build_parser, train


if __name__ == "__main__":
    args = build_parser().parse_args()
    args._start_time = time.time()
    report = train(args)
    print(
        f"Training complete. Best val top-1={report['best_val_top1']:.2%} "
        f"Test top-1={report['test_metrics']['top1_accuracy']:.2%}"
    )
