# Complex5 Offloading Evaluation Results

**Profile:** complex5 | **Splits:** 5 | **Servers:** 4 | **Slots:** 12
**Seeds:** 42,123,456,789,2024 | **Episodes/seed:** 20 | **Requests/episode:** 60

## Summary

| Method | Blocking | Delay (ms) | ObjScore | Split Distribution |
|--------|----------|------------|----------|-------------------|
| Complex5-Default | 0.000±0.000 | 4.8±0.5 | 0.0370±0.0040 | split0=80.0%, split1=2.6%, split2=2.2%, split3=4.6%, split4=10.6% |
| Complex5-DelayAware | 0.000±0.000 | 3.5±0.3 | 0.0268±0.0026 | split0=53.7%, split1=3.2%, split2=4.3%, split3=7.4%, split4=31.4% |
| GREEDY | 0.000±0.000 | 3.6±0.4 | 0.0277±0.0029 | split0=52.9%, split1=0.1%, split2=0.9%, split3=1.2%, split4=44.9% |
| DF | 0.000±0.000 | 3.5±0.4 | 0.0269±0.0030 | split0=53.0%, split1=0.3%, split2=0.9%, split3=1.5%, split4=44.2% |
| RF | 0.000±0.000 | 4.1±0.4 | 0.0318±0.0032 | split0=54.0%, split1=0.6%, split2=1.4%, split3=1.8%, split4=42.2% |
| WO | 0.000±0.000 | 4.2±0.4 | 0.0324±0.0033 | split0=54.5%, split1=0.6%, split2=1.5%, split3=2.0%, split4=41.4% |
| IWD | 0.000±0.000 | 4.2±0.4 | 0.0321±0.0033 | split0=60.3%, split1=7.0%, split2=8.4%, split3=10.2%, split4=14.0% |

## Detailed Metrics

### Complex5-Default
- Blocking: 0.000 ± 0.000
- Avg Delay: 4.8 ± 0.5 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0370 ± 0.0040
- Splits: split0=80.0%, split1=2.6%, split2=2.2%, split3=4.6%, split4=10.6%

### Complex5-DelayAware
- Blocking: 0.000 ± 0.000
- Avg Delay: 3.5 ± 0.3 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0268 ± 0.0026
- Splits: split0=53.7%, split1=3.2%, split2=4.3%, split3=7.4%, split4=31.4%

### GREEDY
- Blocking: 0.000 ± 0.000
- Avg Delay: 3.6 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0277 ± 0.0029
- Splits: split0=52.9%, split1=0.1%, split2=0.9%, split3=1.2%, split4=44.9%

### DF
- Blocking: 0.000 ± 0.000
- Avg Delay: 3.5 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0269 ± 0.0030
- Splits: split0=53.0%, split1=0.3%, split2=0.9%, split3=1.5%, split4=44.2%

### RF
- Blocking: 0.000 ± 0.000
- Avg Delay: 4.1 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0318 ± 0.0032
- Splits: split0=54.0%, split1=0.6%, split2=1.4%, split3=1.8%, split4=42.2%

### WO
- Blocking: 0.000 ± 0.000
- Avg Delay: 4.2 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0324 ± 0.0033
- Splits: split0=54.5%, split1=0.6%, split2=1.5%, split3=2.0%, split4=41.4%

### IWD
- Blocking: 0.000 ± 0.000
- Avg Delay: 4.2 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0321 ± 0.0033
- Splits: split0=60.3%, split1=7.0%, split2=8.4%, split3=10.2%, split4=14.0%
