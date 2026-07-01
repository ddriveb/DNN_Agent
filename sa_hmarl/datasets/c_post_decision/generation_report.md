# C-Side Post-Decision Dataset Generation

- Feature dimension: 50
- Label: `explicit_spectrum_only_phi_after`

## Split Quality

| Split | Groups | Candidates | Empty | Label std | Nonzero-range multi-groups | Top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| train | 2400 | 11094 | 46.08% | 1.5484 | 61.28% | 22.64% |
| val | 1600 | 7422 | 46.25% | 1.5005 | 64.40% | 17.21% |
| test | 1600 | 6973 | 52.94% | 1.6766 | 53.22% | 23.24% |

**Verdict: PROCEED_TO_POST_DECISION_TRAINING**

## Generation Timing

- train: 551.5s worker time; 18.38s/episode; 49.71ms/candidate
- val: 399.7s worker time; 19.99s/episode; 53.86ms/candidate
- test: 350.7s worker time; 17.53s/episode; 50.29ms/candidate
- Total worker time: 1302.0s