# Calibration/workload alignment audit

Calibration prefixes are deterministic record-order prefixes. Changing their length changes both sample count and class mixture.

| Calibration queries | Unique calibration classes | Queries in test classes | Test classes observed |
|---:|---:|---:|---:|
| 400 | 400 | 400 | 400/400 |
| 1000 | 986 | 409 | 400/400 |
| 5000 | 4435 | 461 | 400/400 |
