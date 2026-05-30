# Online Residual DQN Framework

## Core Figure

```mermaid
flowchart TD
    A[Inference Request r_t] --> B[Top-K Candidate Generator<br/>ImitationAgent]
    B --> C[Candidate Actions a in Top-K<br/>(split, server)]
    C --> D[Predictor Scoring<br/>P_success, delay, load]
    D --> E[Base Score]
    C --> F[Action-Aware State Builder<br/>per-action features + global trends]
    F --> G[Online Residual Q Network]
    G --> H[Residual Q(s,a)]
    E --> I[Final Score = Base Score + λ * Q_residual]
    H --> I
    I --> J[Action Selection<br/>epsilon-greedy during training<br/>argmax during evaluation]
    J --> K[Mapper Execution<br/>KSP + First-Fit]
    K --> L[Real Environment Outcome<br/>success/blocking, delay, frag impact]
    L --> M[Reward / Transition]
    M --> N[Replay Buffer]
    N --> O[Online TD Update<br/>Q-network]
    O --> P[Target Network Sync]
    P --> G
```

## Figure Caption

**Figure X. OnlineResidualDQN framework.**  
The high-level action space is constrained to Top-K `(split, server)` candidates proposed by the imitation model. Each candidate is first scored by the predictor for short-term feasibility, then corrected by an online residual Q-network that learns long-term value through environment interaction. The final action is executed by a deterministic mapper, and real blocking/delay/resource outcomes are fed back into replay-based TD learning.

## Short Paper Description

This figure should support the following message:

- The RL module does **not** replace the optical execution stack.
- The RL module is inserted **after** candidate generation and **before** deterministic RMSA execution.
- The model receives **real reward feedback** from the environment, which differentiates it from the previous offline CorrectionNet line.
- This design keeps the action space manageable while still enabling true online exploration and policy improvement.
