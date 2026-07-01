# DeepRMSA S24 Forced Transfer to S100

This diagnostic bypasses the official slot-mismatch guard and loads the S24 DeepRMSA weights into a S100 DeepRMSA agent. It is a transfer probe, not the default fair DeepRMSA baseline.

| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 |
|---|---:|---:|---:|---:|---:|
| DeepRMSA S24 forced-transfer on S100 | 1.11% | 1.11% | 1.04% | 0.07% | 8.882/17.494 ms |