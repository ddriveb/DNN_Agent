# Source Code Organization

This directory contains the **paper-version** source code for the Predictor-Guided Top-K Correction Agent.

## Directory Structure

```
src/
├── README.md                          # This file
├── run_paper_eval.py                  # Unified evaluation script (reproduces all paper tables)
├── compile_paper_results.py           # Compile results from existing experiment JSONs
├── checkpoints/                       # Model checkpoints (copied from experiments/)
│   ├── pretrained_nsfnet_v2b.pt       # Predictor (from predictor_mvp)
│   ├── imitation_agent_enhanced.pt    # ImitationAgent (30D state)
│   └── correction_net.pt              # CorrectionNet (best model)
├── agents/                            # Our proposed agents
│   ├── __init__.py
│   ├── topk_selector_agent.py         # Predictor-guided Top-K selector
│   ├── correction_net.py              # Residual correction network
│   └── correction_agent.py            # Full CorrectionNet agent
├── baselines/                         # Baseline methods
│   ├── __init__.py
│   └── heuristic_agents.py            # YinLike, Random, ShortestPath, LoadBalanced
└── results/                           # Paper tables (JSON + markdown)
    ├── paper_summary.md
    ├── paper_table_1_main.json
    ├── paper_table_2_ablation.json
    ├── paper_table_3_load_sweep.json
    └── paper_table_4_interpretability.json
```

**Note**: Core environment code (network, mapper, encoder, predictor, traffic) lives in
`experiments/predictor_mvp/` and `experiments/agent_mvp/`. `src/` is a lightweight
paper-reproduction layer that imports from those locations via `sys.path`.

## Relationship to experiments/

The `experiments/` directory contains the **original research log**:
- `experiments/predictor_mvp/`: Predictor training, calibration, cross-topology tests
- `experiments/agent_mvp/`: All agent experiments, ablations, and analysis scripts
- `experiments/yin2024_sim/`: Yin 2024 protocol simulation (Phase F, in progress)

`src/` is a lightweight paper-reproduction layer:
- Reproducing paper results
- Understanding the method
- Extending the work

All original experimental scripts remain in `experiments/` for reference.

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run full paper evaluation (reproduces all tables)
cd src
python run_paper_eval.py

# 3. Results will be saved to src/results/
```

## Key Design Principles

1. **Baseline Alignment**: `CorrectionAgent` with `lambda_corr=0.0` is *exactly* equivalent to `TopKSelectorAgent`. This is verified by per-request comparison (0 mismatches / 2000 requests).

2. **Residual Correction**: RL does not replace the selector. It adds a correction term:
   ```
   final_score = predictor_score + λ · correction_score
   ```

3. **Constrained Action Space**: Only top-K candidates from the neural proposal are considered (K=2). This avoids the action-space explosion problem.

## Main Method Pipeline

```
Request → ImitationAgent(30D) → Top-2 candidates
            ↓
        Predictor scores each (P_success, delay, load)
            ↓
        CorrectionNet(10D) adds long-term residual
            ↓
        final_score = predictor_score + 0.2 · correction_score
            ↓
        Mapper executes KSP + first-fit
```

## Citation

If you use this code, please cite the paper (once published).
