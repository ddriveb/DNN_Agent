# Complex5 Offloading Evaluation Results

**Profile:** complex5 | **Splits:** 5 | **Servers:** 4 | **Slots:** 24
**Seeds:** 42,123,456,789,2024 | **Episodes/seed:** 20 | **Requests/episode:** 120

## Summary

| Method | Blocking | Delay (ms) | ObjScore | Split Distribution |
|--------|----------|------------|----------|-------------------|
| Complex5-Default | 0.000±0.000 | 8.1±0.7 | 0.0625±0.0055 | split0=58.0%, split1=9.9%, split2=5.7%, split3=6.3%, split4=20.0% |
| Complex5-DelayAware | 0.000±0.000 | 5.7±0.4 | 0.0440±0.0028 | split0=32.2%, split1=6.2%, split2=11.3%, split3=14.1%, split4=36.1% |
| GREEDY | 0.000±0.000 | 5.5±0.3 | 0.0419±0.0023 | split0=29.7%, split1=0.1%, split2=0.6%, split3=1.2%, split4=68.4% |
| DF | 0.000±0.000 | 5.4±0.4 | 0.0417±0.0032 | split0=28.9%, split1=0.3%, split2=1.2%, split3=2.4%, split4=67.2% |
| RF | 0.000±0.000 | 6.4±0.4 | 0.0489±0.0027 | split0=30.5%, split1=0.6%, split2=1.6%, split3=2.5%, split4=64.8% |
| WO | 0.000±0.000 | 6.7±0.4 | 0.0512±0.0032 | split0=30.7%, split1=0.8%, split2=2.0%, split3=2.7%, split4=63.9% |
| IWD | 0.000±0.000 | 6.7±0.5 | 0.0513±0.0041 | split0=36.5%, split1=8.6%, split2=11.2%, split3=16.2%, split4=27.5% |

## Detailed Metrics

### Complex5-Default
- Blocking: 0.000 ± 0.000
- Avg Delay: 8.1 ± 0.7 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0625 ± 0.0055
- Splits: split0=58.0%, split1=9.9%, split2=5.7%, split3=6.3%, split4=20.0%

### Complex5-DelayAware
- Blocking: 0.000 ± 0.000
- Avg Delay: 5.7 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0440 ± 0.0028
- Splits: split0=32.2%, split1=6.2%, split2=11.3%, split3=14.1%, split4=36.1%

### GREEDY
- Blocking: 0.000 ± 0.000
- Avg Delay: 5.5 ± 0.3 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0419 ± 0.0023
- Splits: split0=29.7%, split1=0.1%, split2=0.6%, split3=1.2%, split4=68.4%

### DF
- Blocking: 0.000 ± 0.000
- Avg Delay: 5.4 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0417 ± 0.0032
- Splits: split0=28.9%, split1=0.3%, split2=1.2%, split3=2.4%, split4=67.2%

### RF
- Blocking: 0.000 ± 0.000
- Avg Delay: 6.4 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0489 ± 0.0027
- Splits: split0=30.5%, split1=0.6%, split2=1.6%, split3=2.5%, split4=64.8%

### WO
- Blocking: 0.000 ± 0.000
- Avg Delay: 6.7 ± 0.4 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0512 ± 0.0032
- Splits: split0=30.7%, split1=0.8%, split2=2.0%, split3=2.7%, split4=63.9%

### IWD
- Blocking: 0.000 ± 0.000
- Avg Delay: 6.7 ± 0.5 ms
- Avg FS: 0.0 ± 0.0
- Avg Waste: 0.0 ± 0.0
- Avg Path: 0.0 ± 0.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.0513 ± 0.0041
- Splits: split0=36.5%, split1=8.6%, split2=11.2%, split3=16.2%, split4=27.5%
