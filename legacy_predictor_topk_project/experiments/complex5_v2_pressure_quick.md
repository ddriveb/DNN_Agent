# Complex5_v2 Pressure Scan Results

**Profile:** complex5_v2 | **Splits:** 5 | **Servers:** 4
**Seeds:** 42, 123 | **Episodes/seed:** 5

## Grid

| Slots | Reqs/Ep | Interval | Greedy Blk | DF Blk | RF Blk | IWD Blk | Best Obj |
|-------|---------|----------|------------|--------|--------|---------|----------|
| 20 | 60 | 0.25 | 0.342 | 0.333 | 0.358 | 0.275 | 0.3577 |
| 20 | 60 | 0.2 | 0.342 | 0.333 | 0.358 | 0.292 | 0.3731 |
| 20 | 90 | 0.25 | 0.372 | 0.372 | 0.383 | 0.389 | 0.4232 |
| 20 | 90 | 0.2 | 0.367 | 0.367 | 0.389 | 0.389 | 0.4203 |
| 16 | 60 | 0.25 | 0.375 | 0.375 | 0.392 | 0.333 | 0.4100 |
| 16 | 60 | 0.2 | 0.375 | 0.375 | 0.392 | 0.492 | 0.4276 |
| 16 | 90 | 0.25 | 0.389 | 0.383 | 0.389 | 0.411 | 0.4394 |
| 16 | 90 | 0.2 | 0.389 | 0.378 | 0.383 | 0.683 | 0.4418 |
| 20 | 60 | 0.25 | 0.350 | 0.333 | 0.358 | 0.300 | 0.4008 |
| 20 | 60 | 0.2 | 0.350 | 0.333 | 0.358 | 0.500 | 0.4120 |
| 20 | 90 | 0.25 | 0.394 | 0.389 | 0.406 | 0.406 | 0.4586 |
| 20 | 90 | 0.2 | 0.400 | 0.394 | 0.406 | 0.700 | 0.4721 |
| 16 | 60 | 0.25 | 0.375 | 0.375 | 0.400 | 0.342 | 0.4350 |
| 16 | 60 | 0.2 | 0.375 | 0.375 | 0.400 | 0.533 | 0.4438 |
| 16 | 90 | 0.25 | 0.400 | 0.400 | 0.411 | 0.456 | 0.4647 |
| 16 | 90 | 0.2 | 0.406 | 0.406 | 0.467 | 0.711 | 0.4800 |

## Detailed Results

### slots=20, reqs=60, interval=0.25
- **GREEDY**: blk=0.342±0.000, delay=6.6±0.0ms, obj=0.3926±0.0000, fs=2.7±0.0
  - Splits: split4=65.8%
  - Reasons: no_valid_c_action=41
- **DF**: blk=0.333±0.000, delay=9.5±0.0ms, obj=0.4064±0.0000, fs=2.9±0.0
  - Splits: split3=2.5%, split4=64.2%
  - Reasons: no_valid_c_action=40
- **RF**: blk=0.358±0.000, delay=8.1±0.0ms, obj=0.4204±0.0000, fs=3.0±0.0
  - Splits: split0=0.8%, split3=0.8%, split4=62.5%
  - Reasons: no_valid_c_action=43
- **IWD**: blk=0.275±0.000, delay=10.8±0.0ms, obj=0.3577±0.0000, fs=2.4±0.0
  - Splits: split0=0.8%, split2=6.7%, split3=54.2%, split4=10.8%
  - Reasons: no_valid_c_action=33

### slots=20, reqs=60, interval=0.2
- **GREEDY**: blk=0.342±0.000, delay=7.1±0.0ms, obj=0.3961±0.0000, fs=2.8±0.0
  - Splits: split4=65.8%
  - Reasons: no_valid_c_action=41
- **DF**: blk=0.333±0.000, delay=9.1±0.0ms, obj=0.4031±0.0000, fs=2.8±0.0
  - Splits: split3=1.7%, split4=65.0%
  - Reasons: no_valid_c_action=40
- **RF**: blk=0.358±0.000, delay=8.0±0.0ms, obj=0.4203±0.0000, fs=2.9±0.0
  - Splits: split2=0.8%, split3=0.8%, split4=62.5%
  - Reasons: no_valid_c_action=43
- **IWD**: blk=0.292±0.000, delay=10.6±0.0ms, obj=0.3731±0.0000, fs=2.3±0.0
  - Splits: split2=6.7%, split3=57.5%, split4=6.7%
  - Reasons: no_valid_c_action=35

### slots=20, reqs=90, interval=0.25
- **GREEDY**: blk=0.372±0.000, delay=6.6±0.0ms, obj=0.4232±0.0000, fs=2.6±0.0
  - Splits: split4=62.8%
  - Reasons: no_valid_c_action=67
- **DF**: blk=0.372±0.000, delay=10.2±0.0ms, obj=0.4505±0.0000, fs=2.7±0.0
  - Splits: split1=0.6%, split2=0.6%, split3=2.2%, split4=59.4%
  - Reasons: no_valid_c_action=67
- **RF**: blk=0.383±0.000, delay=8.3±0.0ms, obj=0.4474±0.0000, fs=2.8±0.0
  - Splits: split0=0.6%, split4=62.2%
  - Reasons: no_valid_c_action=67, server_overload=2
- **IWD**: blk=0.389±0.000, delay=10.8±0.0ms, obj=0.4716±0.0000, fs=2.3±0.0
  - Splits: split1=0.6%, split2=3.3%, split3=45.6%, split4=11.7%
  - Reasons: no_valid_c_action=70

### slots=20, reqs=90, interval=0.2
- **GREEDY**: blk=0.367±0.000, delay=7.0±0.0ms, obj=0.4203±0.0000, fs=2.7±0.0
  - Splits: split4=63.9%
  - Reasons: no_valid_c_action=65, server_overload=1
- **DF**: blk=0.367±0.000, delay=9.1±0.0ms, obj=0.4367±0.0000, fs=2.7±0.0
  - Splits: split0=0.6%, split3=2.8%, split4=60.6%
  - Reasons: no_valid_c_action=65, server_overload=1
- **RF**: blk=0.389±0.000, delay=8.0±0.0ms, obj=0.4503±0.0000, fs=2.7±0.0
  - Splits: split2=0.6%, split3=1.7%, split4=58.9%
  - Reasons: no_valid_c_action=70
- **IWD**: blk=0.389±0.000, delay=10.9±0.0ms, obj=0.4728±0.0000, fs=2.3±0.0
  - Splits: split2=3.3%, split3=48.9%, split4=8.9%
  - Reasons: no_valid_c_action=70

### slots=16, reqs=60, interval=0.25
- **GREEDY**: blk=0.375±0.000, delay=6.5±0.0ms, obj=0.4251±0.0000, fs=2.7±0.0
  - Splits: split4=62.5%
  - Reasons: no_valid_c_action=45
- **DF**: blk=0.375±0.000, delay=8.7±0.0ms, obj=0.4419±0.0000, fs=2.7±0.0
  - Splits: split2=1.7%, split3=2.5%, split4=58.3%
  - Reasons: no_valid_c_action=45
- **RF**: blk=0.392±0.000, delay=8.0±0.0ms, obj=0.4536±0.0000, fs=2.8±0.0
  - Splits: split0=0.8%, split1=0.8%, split2=0.8%, split3=1.7%, split4=56.7%
  - Reasons: no_valid_c_action=47
- **IWD**: blk=0.333±0.000, delay=10.0±0.0ms, obj=0.4100±0.0000, fs=2.3±0.0
  - Splits: split2=7.5%, split3=50.8%, split4=8.3%
  - Reasons: no_valid_c_action=40

### slots=16, reqs=60, interval=0.2
- **GREEDY**: blk=0.375±0.000, delay=6.8±0.0ms, obj=0.4276±0.0000, fs=2.7±0.0
  - Splits: split4=62.5%
  - Reasons: no_valid_c_action=45
- **DF**: blk=0.375±0.000, delay=8.4±0.0ms, obj=0.4395±0.0000, fs=2.7±0.0
  - Splits: split2=2.5%, split3=1.7%, split4=58.3%
  - Reasons: no_valid_c_action=45
- **RF**: blk=0.392±0.000, delay=7.9±0.0ms, obj=0.4526±0.0000, fs=2.8±0.0
  - Splits: split2=0.8%, split3=2.5%, split4=57.5%
  - Reasons: no_valid_c_action=47
- **IWD**: blk=0.492±0.000, delay=9.9±0.0ms, obj=0.5675±0.0000, fs=2.4±0.0
  - Splits: split2=6.7%, split3=40.8%, split4=3.3%
  - Reasons: no_valid_c_action=59

### slots=16, reqs=90, interval=0.25
- **GREEDY**: blk=0.389±0.000, delay=6.6±0.0ms, obj=0.4394±0.0000, fs=2.6±0.0
  - Splits: split4=61.1%
  - Reasons: no_valid_c_action=70
- **DF**: blk=0.383±0.000, delay=8.8±0.0ms, obj=0.4514±0.0000, fs=2.6±0.0
  - Splits: split1=0.6%, split2=1.7%, split3=3.3%, split4=56.1%
  - Reasons: no_valid_c_action=69
- **RF**: blk=0.389±0.000, delay=8.2±0.0ms, obj=0.4516±0.0000, fs=2.7±0.0
  - Splits: split0=0.6%, split1=0.6%, split2=1.7%, split3=2.2%, split4=56.1%
  - Reasons: no_valid_c_action=70
- **IWD**: blk=0.411±0.000, delay=10.5±0.0ms, obj=0.4917±0.0000, fs=2.3±0.0
  - Splits: split0=0.6%, split1=0.6%, split2=6.7%, split3=40.6%, split4=10.6%
  - Reasons: no_valid_c_action=74

### slots=16, reqs=90, interval=0.2
- **GREEDY**: blk=0.389±0.000, delay=6.9±0.0ms, obj=0.4418±0.0000, fs=2.6±0.0
  - Splits: split4=61.1%
  - Reasons: no_valid_c_action=70
- **DF**: blk=0.378±0.000, delay=8.6±0.0ms, obj=0.4437±0.0000, fs=2.6±0.0
  - Splits: split2=2.2%, split3=3.9%, split4=57.2%
  - Reasons: no_valid_c_action=66, server_overload=2
- **RF**: blk=0.383±0.000, delay=8.1±0.0ms, obj=0.4459±0.0000, fs=2.6±0.0
  - Splits: split0=0.6%, split1=1.1%, split2=0.6%, split3=3.9%, split4=57.2%
  - Reasons: no_valid_c_action=66, server_overload=3
- **IWD**: blk=0.683±0.000, delay=9.5±0.0ms, obj=0.7561±0.0000, fs=2.3±0.0
  - Splits: split2=2.8%, split3=26.7%, split4=2.2%
  - Reasons: no_valid_c_action=123

### slots=20, reqs=60, interval=0.25
- **GREEDY**: blk=0.350±0.000, delay=8.1±0.0ms, obj=0.4124±0.0000, fs=2.9±0.0
  - Splits: split4=65.0%
  - Reasons: no_valid_c_action=42
- **DF**: blk=0.333±0.000, delay=11.4±0.0ms, obj=0.4214±0.0000, fs=2.9±0.0
  - Splits: split3=1.7%, split4=65.0%
  - Reasons: no_valid_c_action=40
- **RF**: blk=0.358±0.000, delay=9.1±0.0ms, obj=0.4285±0.0000, fs=2.9±0.0
  - Splits: split2=0.8%, split3=1.7%, split4=61.7%
  - Reasons: no_valid_c_action=43
- **IWD**: blk=0.300±0.000, delay=13.1±0.0ms, obj=0.4008±0.0000, fs=2.4±0.0
  - Splits: split2=5.0%, split3=51.7%, split4=13.3%
  - Reasons: no_valid_c_action=36

### slots=20, reqs=60, interval=0.2
- **GREEDY**: blk=0.350±0.000, delay=8.6±0.0ms, obj=0.4164±0.0000, fs=2.9±0.0
  - Splits: split4=65.0%
  - Reasons: no_valid_c_action=42
- **DF**: blk=0.333±0.000, delay=10.2±0.0ms, obj=0.4120±0.0000, fs=2.9±0.0
  - Splits: split3=3.3%, split4=63.3%
  - Reasons: no_valid_c_action=40
- **RF**: blk=0.358±0.000, delay=9.9±0.0ms, obj=0.4342±0.0000, fs=2.9±0.0
  - Splits: split2=1.7%, split3=2.5%, split4=60.0%
  - Reasons: no_valid_c_action=43
- **IWD**: blk=0.500±0.000, delay=12.8±0.0ms, obj=0.5981±0.0000, fs=2.6±0.0
  - Splits: split2=3.3%, split3=40.0%, split4=6.7%
  - Reasons: no_valid_c_action=60

### slots=20, reqs=90, interval=0.25
- **GREEDY**: blk=0.394±0.000, delay=8.3±0.0ms, obj=0.4586±0.0000, fs=2.7±0.0
  - Splits: split4=63.9%
  - Reasons: no_valid_c_action=65, server_overload=6
- **DF**: blk=0.389±0.000, delay=11.9±0.0ms, obj=0.4806±0.0000, fs=2.8±0.0
  - Splits: split3=2.2%, split4=58.9%
  - Reasons: no_valid_c_action=70
- **RF**: blk=0.406±0.000, delay=9.7±0.0ms, obj=0.4802±0.0000, fs=2.8±0.0
  - Splits: split0=0.6%, split1=0.6%, split3=0.6%, split4=57.8%
  - Reasons: no_valid_c_action=73
- **IWD**: blk=0.406±0.000, delay=13.1±0.0ms, obj=0.5066±0.0000, fs=2.4±0.0
  - Splits: split1=0.6%, split2=5.0%, split3=44.4%, split4=13.9%
  - Reasons: no_suitable_block=1, no_valid_c_action=65, server_overload=7

### slots=20, reqs=90, interval=0.2
- **GREEDY**: blk=0.400±0.000, delay=9.4±0.0ms, obj=0.4721±0.0000, fs=2.8±0.0
  - Splits: split3=0.6%, split4=60.0%
  - Reasons: no_valid_c_action=71, server_overload=1
- **DF**: blk=0.394±0.000, delay=10.6±0.0ms, obj=0.4763±0.0000, fs=2.7±0.0
  - Splits: split2=0.6%, split3=2.2%, split4=58.3%
  - Reasons: no_suitable_block=1, no_valid_c_action=70
- **RF**: blk=0.406±0.000, delay=10.6±0.0ms, obj=0.4872±0.0000, fs=2.7±0.0
  - Splits: split2=1.1%, split3=5.0%, split4=53.3%
  - Reasons: no_valid_c_action=73
- **IWD**: blk=0.700±0.000, delay=11.8±0.0ms, obj=0.7911±0.0000, fs=2.4±0.0
  - Splits: split2=2.2%, split3=24.4%, split4=4.4%
  - Reasons: no_valid_c_action=124, server_overload=2

### slots=16, reqs=60, interval=0.25
- **GREEDY**: blk=0.375±0.000, delay=8.0±0.0ms, obj=0.4367±0.0000, fs=2.8±0.0
  - Splits: split4=62.5%
  - Reasons: no_valid_c_action=45
- **DF**: blk=0.375±0.000, delay=9.6±0.0ms, obj=0.4485±0.0000, fs=2.8±0.0
  - Splits: split3=0.8%, split4=61.7%
  - Reasons: no_valid_c_action=45
- **RF**: blk=0.400±0.000, delay=9.2±0.0ms, obj=0.4710±0.0000, fs=2.9±0.0
  - Splits: split1=0.8%, split3=2.5%, split4=56.7%
  - Reasons: no_valid_c_action=48
- **IWD**: blk=0.342±0.000, delay=12.1±0.0ms, obj=0.4350±0.0000, fs=2.4±0.0
  - Splits: split1=0.8%, split2=7.5%, split3=45.8%, split4=11.7%
  - Reasons: no_valid_c_action=41

### slots=16, reqs=60, interval=0.2
- **GREEDY**: blk=0.375±0.000, delay=8.9±0.0ms, obj=0.4438±0.0000, fs=2.8±0.0
  - Splits: split2=0.8%, split3=0.8%, split4=60.8%
  - Reasons: no_valid_c_action=45
- **DF**: blk=0.375±0.000, delay=9.8±0.0ms, obj=0.4504±0.0000, fs=2.9±0.0
  - Splits: split3=1.7%, split4=60.8%
  - Reasons: no_valid_c_action=45
- **RF**: blk=0.400±0.000, delay=10.3±0.0ms, obj=0.4788±0.0000, fs=2.8±0.0
  - Splits: split0=0.8%, split2=2.5%, split3=1.7%, split4=55.0%
  - Reasons: no_valid_c_action=48
- **IWD**: blk=0.533±0.000, delay=12.1±0.0ms, obj=0.6263±0.0000, fs=2.4±0.0
  - Splits: split1=0.8%, split2=5.8%, split3=32.5%, split4=7.5%
  - Reasons: no_valid_c_action=64

### slots=16, reqs=90, interval=0.25
- **GREEDY**: blk=0.400±0.000, delay=8.4±0.0ms, obj=0.4647±0.0000, fs=2.7±0.0
  - Splits: split4=63.9%
  - Reasons: no_valid_c_action=65, server_overload=7
- **DF**: blk=0.400±0.000, delay=10.2±0.0ms, obj=0.4782±0.0000, fs=2.7±0.0
  - Splits: split1=0.6%, split3=1.1%, split4=62.2%
  - Reasons: no_valid_c_action=65, server_overload=7
- **RF**: blk=0.411±0.000, delay=10.3±0.0ms, obj=0.4902±0.0000, fs=2.7±0.0
  - Splits: split1=1.7%, split2=1.7%, split3=4.4%, split4=51.1%
  - Reasons: no_valid_c_action=74
- **IWD**: blk=0.456±0.000, delay=12.5±0.0ms, obj=0.5521±0.0000, fs=2.4±0.0
  - Splits: split2=7.8%, split3=36.7%, split4=12.8%
  - Reasons: no_valid_c_action=77, server_overload=2, no_suitable_block=3

### slots=16, reqs=90, interval=0.2
- **GREEDY**: blk=0.406±0.000, delay=9.7±0.0ms, obj=0.4800±0.0000, fs=2.7±0.0
  - Splits: split2=0.6%, split3=1.1%, split4=58.9%
  - Reasons: no_valid_c_action=71, server_overload=2
- **DF**: blk=0.406±0.000, delay=10.6±0.0ms, obj=0.4872±0.0000, fs=2.7±0.0
  - Splits: split3=3.3%, split4=57.2%
  - Reasons: no_valid_c_action=71, server_overload=2
- **RF**: blk=0.467±0.000, delay=10.7±0.0ms, obj=0.5488±0.0000, fs=2.6±0.0
  - Splits: split0=0.6%, split1=1.7%, split2=1.7%, split3=2.8%, split4=51.7%
  - Reasons: no_valid_c_action=75, no_suitable_block=7, server_overload=2
- **IWD**: blk=0.711±0.000, delay=11.6±0.0ms, obj=0.8006±0.0000, fs=2.3±0.0
  - Splits: split2=3.9%, split3=20.6%, split4=4.4%
  - Reasons: no_valid_c_action=128
