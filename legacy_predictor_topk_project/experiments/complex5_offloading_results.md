# Complex5 Offloading Evaluation Results

**Profile:** complex5 | **Splits:** 5 | **Servers:** 4 | **Slots:** 24
**Seeds:** 42,123,456,789,2024 | **Episodes/seed:** 20 | **Requests/episode:** 60

## Summary

| Method | Blocking | Delay (ms) | ObjScore | Split Distribution |
|--------|----------|------------|----------|-------------------|
| Complex5-Default | 0.000±0.000 | 7.7±0.6 | 0.0594±0.0045 | split0=58.6%, split1=7.4%, split2=5.9%, split3=7.6%, split4=20.5% |
| Complex5-DelayAware | 0.000±0.000 | 5.6±0.4 | 0.0434±0.0033 | split0=31.4%, split1=6.0%, split2=11.4%, split3=13.7%, split4=37.6% |
| GREEDY | 0.000±0.000 | 5.5±0.4 | 0.0426±0.0033 | split0=29.9%, split1=0.1%, split2=0.3%, split3=0.9%, split4=68.8% |
| DF | 0.000±0.000 | 5.2±0.4 | 0.0402±0.0032 | split0=29.2%, split1=0.2%, split2=0.9%, split3=1.7%, split4=68.0% |
| RF | 0.000±0.000 | 6.4±0.5 | 0.0493±0.0035 | split0=30.2%, split1=0.5%, split2=1.5%, split3=1.8%, split4=65.9% |
| WO | 0.000±0.000 | 6.7±0.5 | 0.0518±0.0038 | split0=30.6%, split1=0.6%, split2=1.4%, split3=2.1%, split4=65.4% |
| IWD | 0.000±0.000 | 6.3±0.5 | 0.0487±0.0042 | split0=36.9%, split1=8.6%, split2=11.5%, split3=16.4%, split4=26.6% |

## Detailed Metrics

### Complex5-Default
- Blocking: 0.000 ± 0.000
- Avg Delay: 7.7 ± 0.6 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0594 ± 0.0045
- Splits: split0=58.6%, split1=7.4%, split2=5.9%, split3=7.6%, split4=20.5%

### Complex5-DelayAware
- Blocking: 0.000 ± 0.000
- Avg Delay: 5.6 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0434 ± 0.0033
- Splits: split0=31.4%, split1=6.0%, split2=11.4%, split3=13.7%, split4=37.6%

### GREEDY
- Blocking: 0.000 ± 0.000
- Avg Delay: 5.5 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0426 ± 0.0033
- Splits: split0=29.9%, split1=0.1%, split2=0.3%, split3=0.9%, split4=68.8%

### DF
- Blocking: 0.000 ± 0.000
- Avg Delay: 5.2 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0402 ± 0.0032
- Splits: split0=29.2%, split1=0.2%, split2=0.9%, split3=1.7%, split4=68.0%

### RF
- Blocking: 0.000 ± 0.000
- Avg Delay: 6.4 ± 0.5 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0493 ± 0.0035
- Splits: split0=30.2%, split1=0.5%, split2=1.5%, split3=1.8%, split4=65.9%

### WO
- Blocking: 0.000 ± 0.000
- Avg Delay: 6.7 ± 0.5 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0518 ± 0.0038
- Splits: split0=30.6%, split1=0.6%, split2=1.4%, split3=2.1%, split4=65.4%

### IWD
- Blocking: 0.000 ± 0.000
- Avg Delay: 6.3 ± 0.5 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0487 ± 0.0042
- Splits: split0=36.9%, split1=8.6%, split2=11.5%, split3=16.4%, split4=26.6%
