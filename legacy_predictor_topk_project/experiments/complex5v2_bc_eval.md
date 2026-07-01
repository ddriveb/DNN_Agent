# Complex5 Offloading Evaluation Results

**Profile:** complex5_v2 | **Splits:** 5 | **Servers:** 4 | **Slots:** 16
**Seeds:** 42,123,456,789,2024 | **Episodes/seed:** 20 | **Requests/episode:** 120

## Summary

| Method | Blocking | Delay (ms) | ObjScore | Split Distribution |
|--------|----------|------------|----------|-------------------|
| Complex5-Default | 0.389±0.055 | 6.9±0.4 | 0.4425±0.0557 | split0=47.8%, split1=0.6%, split2=2.1%, split3=5.0%, split4=44.5% |
| GREEDY | 0.392±0.051 | 7.0±0.3 | 0.4458±0.0518 | split0=47.9%, split1=0.5%, split2=1.6%, split3=4.2%, split4=45.9% |
| DF | 0.385±0.053 | 7.2±0.2 | 0.4409±0.0511 | split0=47.6%, split1=0.8%, split2=2.4%, split3=5.8%, split4=43.4% |
| RF | 0.399±0.057 | 8.1±0.3 | 0.4619±0.0577 | split0=48.6%, split1=1.3%, split2=2.6%, split3=5.7%, split4=41.9% |
| WO | 0.399±0.057 | 8.6±0.2 | 0.4650±0.0575 | split0=48.8%, split1=1.5%, split2=3.0%, split3=6.0%, split4=40.6% |
| IWD | 0.373±0.051 | 9.1±0.3 | 0.4436±0.0486 | split0=53.4%, split1=7.7%, split2=10.7%, split3=13.7%, split4=14.6% |

## Detailed Metrics

### Complex5-Default
- Blocking: 0.389 ± 0.055
- Avg Delay: 6.9 ± 0.4 ms
- Avg FS: 2.3 ± 0.0
- Avg Waste: 0.4 ± 0.1
- Avg Path: 487.2 ± 37.8 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.4425 ± 0.0557
- Splits: split0=47.8%, split1=0.6%, split2=2.1%, split3=5.0%, split4=44.5%

### GREEDY
- Blocking: 0.392 ± 0.051
- Avg Delay: 7.0 ± 0.3 ms
- Avg FS: 2.3 ± 0.0
- Avg Waste: 0.4 ± 0.0
- Avg Path: 519.3 ± 16.9 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.4458 ± 0.0518
- Splits: split0=47.9%, split1=0.5%, split2=1.6%, split3=4.2%, split4=45.9%

### DF
- Blocking: 0.385 ± 0.053
- Avg Delay: 7.2 ± 0.2 ms
- Avg FS: 2.3 ± 0.0
- Avg Waste: 0.3 ± 0.0
- Avg Path: 512.3 ± 23.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.4409 ± 0.0511
- Splits: split0=47.6%, split1=0.8%, split2=2.4%, split3=5.8%, split4=43.4%

### RF
- Blocking: 0.399 ± 0.057
- Avg Delay: 8.1 ± 0.3 ms
- Avg FS: 2.4 ± 0.0
- Avg Waste: 0.3 ± 0.0
- Avg Path: 617.6 ± 21.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.4619 ± 0.0577
- Splits: split0=48.6%, split1=1.3%, split2=2.6%, split3=5.7%, split4=41.9%

### WO
- Blocking: 0.399 ± 0.057
- Avg Delay: 8.6 ± 0.2 ms
- Avg FS: 2.4 ± 0.0
- Avg Waste: 0.3 ± 0.0
- Avg Path: 645.1 ± 20.0 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.4650 ± 0.0575
- Splits: split0=48.8%, split1=1.5%, split2=3.0%, split3=6.0%, split4=40.6%

### IWD
- Blocking: 0.373 ± 0.051
- Avg Delay: 9.1 ± 0.3 ms
- Avg FS: 2.1 ± 0.0
- Avg Waste: 0.3 ± 0.0
- Avg Path: 569.7 ± 15.9 km
- Deadline Met: 0.0% ± 0.0%
- Objective Score: 0.4436 ± 0.0486
- Splits: split0=53.4%, split1=7.7%, split2=10.7%, split3=13.7%, split4=14.6%
