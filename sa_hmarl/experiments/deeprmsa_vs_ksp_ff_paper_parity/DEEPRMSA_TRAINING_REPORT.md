# DEEPRMSA TRAINING REPORT — Stage D

Local source-semantic DeepRMSA: K=5 km-ordered paths, M=1, 5-dim unmasked
policy, 5×128 ELU actor/critic, upstream overlapping-window A2C
(batch 200, window 399, γ=0.95, entropy=0.01).

**Training config (disclosed deviations, see PREFLIGHT_AUDIT.md D6/D7):** lr=0.0001 (upstream 1e-5), grad-clip 5.0 (upstream 40), raw advantages (normalize_advantages=False), 1000000 requests per run. Deviations were driven by measured evidence on the worst-behaved seed: lr=1e-5 stalls at the SP-FF collapse at this compute budget; advantage normalization prevents escape; tighter grad-clip restores stable escape. DeepRMSA training was seed-unstable throughout — one NSFNET seed (43) sat at the SP-FF collapse for 525k requests, diverged to 32% blocking around 625k, then recovered and reached its best validation at 1M. This instability is consistent with the paper's thesis and is reported per-seed below.

## NSFNET

### Training seed 42

* Requests: 1000000, gradient updates: 4984, final ε: 0.950
* Train blocking (post-warmup stream): 8.5862% (invalid unmasked choices: 85604)
* Best validation blocking: 5.9222% at 675000 requests
* Test (10 seeds, argmax): 6.3890% ± 0.3536%
* Validation curve (requests:val_blocking): 25k:12.411%, 50k:12.300%, 75k:30.756%, 100k:9.944%, 125k:10.100%, 150k:11.189%, 175k:9.900%, 200k:10.900%, 225k:9.567%, 250k:9.489%, 275k:7.533%, 300k:6.933%, 325k:7.411%, 350k:7.744%, 375k:7.922%, 400k:8.989%, 425k:7.100%, 450k:6.700%, 475k:7.178%, 500k:6.778%, 525k:6.889%, 550k:7.033%, 575k:6.833%, 600k:6.100%, 625k:6.467%, 650k:6.122%, 675k:5.922%, 700k:6.300%, 725k:6.556%, 750k:6.144%, 775k:6.322%, 800k:6.300%, 825k:6.178%, 850k:6.522%, 875k:6.511%, 900k:6.856%, 925k:6.333%, 950k:6.756%, 975k:6.600%, 1000k:7.022%

### Training seed 43

* Requests: 1000000, gradient updates: 4984, final ε: 0.950
* Train blocking (post-warmup stream): 15.2346% (invalid unmasked choices: 151889)
* Best validation blocking: 8.6889% at 1000000 requests
* Test (10 seeds, argmax): 9.3400% ± 0.3681%
* Validation curve (requests:val_blocking): 25k:12.378%, 50k:12.378%, 75k:12.378%, 100k:12.378%, 125k:12.378%, 150k:12.378%, 175k:12.378%, 200k:12.378%, 225k:12.378%, 250k:12.378%, 275k:12.378%, 300k:12.378%, 325k:12.378%, 350k:12.378%, 375k:12.378%, 400k:12.378%, 425k:12.378%, 450k:12.378%, 475k:12.378%, 500k:12.378%, 525k:12.378%, 550k:12.378%, 575k:12.378%, 600k:12.378%, 625k:32.422%, 650k:32.389%, 675k:32.456%, 700k:32.167%, 725k:31.622%, 750k:30.722%, 775k:10.378%, 800k:9.756%, 825k:14.611%, 850k:9.178%, 875k:9.467%, 900k:10.244%, 925k:10.167%, 950k:8.833%, 975k:9.489%, 1000k:8.689%

### Training seed 44

* Requests: 1000000, gradient updates: 4984, final ε: 0.950
* Train blocking (post-warmup stream): 8.7349% (invalid unmasked choices: 87087)
* Best validation blocking: 5.5222% at 850000 requests
* Test (10 seeds, argmax): 5.8990% ± 0.4434%
* Validation curve (requests:val_blocking): 25k:12.378%, 50k:12.378%, 75k:12.378%, 100k:12.378%, 125k:11.678%, 150k:11.267%, 175k:9.711%, 200k:9.289%, 225k:9.033%, 250k:9.200%, 275k:8.156%, 300k:9.111%, 325k:8.989%, 350k:7.533%, 375k:7.200%, 400k:7.433%, 425k:7.056%, 450k:6.233%, 475k:6.511%, 500k:8.156%, 525k:6.022%, 550k:6.589%, 575k:6.022%, 600k:6.344%, 625k:6.344%, 650k:7.733%, 675k:6.500%, 700k:6.889%, 725k:6.456%, 750k:6.078%, 775k:5.544%, 800k:7.956%, 825k:6.300%, 850k:5.522%, 875k:5.633%, 900k:7.644%, 925k:8.089%, 950k:7.189%, 975k:7.267%, 1000k:7.022%

**NSFNET summary:** grand mean 7.2093%; training-seed variance (std) 1.8614%; test-seed variance (pooled std) 1.5910%; paper published 4.0000%.

## COST239

### Training seed 42

* Requests: 1000000, gradient updates: 4984, final ε: 0.950
* Train blocking (post-warmup stream): 14.0082% (invalid unmasked choices: 139662)
* Best validation blocking: 10.9556% at 950000 requests
* Test (10 seeds, argmax): 11.9520% ± 0.6050%
* Validation curve (requests:val_blocking): 25k:14.822%, 50k:24.633%, 75k:15.644%, 100k:14.767%, 125k:15.322%, 150k:14.789%, 175k:14.400%, 200k:13.422%, 225k:14.067%, 250k:12.211%, 275k:12.633%, 300k:12.411%, 325k:13.667%, 350k:14.200%, 375k:14.144%, 400k:13.289%, 425k:12.133%, 450k:12.689%, 475k:12.311%, 500k:12.889%, 525k:11.489%, 550k:11.333%, 575k:11.767%, 600k:11.700%, 625k:11.611%, 650k:11.811%, 675k:11.511%, 700k:12.656%, 725k:11.500%, 750k:11.511%, 775k:11.400%, 800k:11.089%, 825k:11.200%, 850k:11.100%, 875k:11.289%, 900k:11.489%, 925k:11.011%, 950k:10.956%, 975k:12.111%, 1000k:11.700%

### Training seed 43

* Requests: 1000000, gradient updates: 4984, final ε: 0.950
* Train blocking (post-warmup stream): 10.6809% (invalid unmasked choices: 106489)
* Best validation blocking: 8.4667% at 750000 requests
* Test (10 seeds, argmax): 9.2820% ± 0.5627%
* Validation curve (requests:val_blocking): 25k:13.533%, 50k:11.922%, 75k:10.744%, 100k:11.000%, 125k:12.933%, 150k:9.656%, 175k:9.578%, 200k:9.544%, 225k:9.067%, 250k:9.200%, 275k:9.022%, 300k:9.078%, 325k:8.922%, 350k:9.022%, 375k:10.067%, 400k:9.478%, 425k:9.200%, 450k:8.811%, 475k:9.267%, 500k:10.356%, 525k:8.633%, 550k:8.556%, 575k:8.989%, 600k:9.178%, 625k:8.756%, 650k:8.633%, 675k:9.100%, 700k:9.033%, 725k:8.767%, 750k:8.467%, 775k:8.733%, 800k:8.756%, 825k:8.711%, 850k:9.189%, 875k:9.133%, 900k:9.300%, 925k:9.011%, 950k:9.422%, 975k:9.289%, 1000k:10.156%

### Training seed 44

* Requests: 1000000, gradient updates: 4984, final ε: 0.950
* Train blocking (post-warmup stream): 10.6264% (invalid unmasked choices: 105945)
* Best validation blocking: 6.7889% at 875000 requests
* Test (10 seeds, argmax): 7.4610% ± 0.4384%
* Validation curve (requests:val_blocking): 25k:14.822%, 50k:14.822%, 75k:14.822%, 100k:14.822%, 125k:14.822%, 150k:14.822%, 175k:14.822%, 200k:14.822%, 225k:14.211%, 250k:10.722%, 275k:9.900%, 300k:9.544%, 325k:9.144%, 350k:9.456%, 375k:9.189%, 400k:10.389%, 425k:8.700%, 450k:8.778%, 475k:8.633%, 500k:9.211%, 525k:9.022%, 550k:9.111%, 575k:8.444%, 600k:8.844%, 625k:8.167%, 650k:7.789%, 675k:7.378%, 700k:7.422%, 725k:7.333%, 750k:7.411%, 775k:7.067%, 800k:7.267%, 825k:7.189%, 850k:7.067%, 875k:6.789%, 900k:6.900%, 925k:7.378%, 950k:6.800%, 975k:7.222%, 1000k:7.411%

**COST239 summary:** grand mean 9.5650%; training-seed variance (std) 2.2588%; test-seed variance (pooled std) 1.9469%; paper published 5.7500%.
