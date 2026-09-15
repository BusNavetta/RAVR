# Unseen items and calibration coverage -- Deep1M (Deep1B base prefix, 96-d)

Gallery 1,000,000 items; 5,000 calibration queries now; 5,000 usable queries exist in the manifest at all.

A query touches **156.2 items** on average (the mined hard set is bounded by 160 regardless of gallery size), so coverage is now **0.4173**.

## Can more calibration queries fill the cap

| Target | Coverage `ravr` needs | Queries that would reach it | Reachable at all |
|---:|---:|---:|---|
| 10 | 0.25 | 2,550 | yes |
| 12 | 0.50 | 10,000 | yes |
| 14 | 0.75 | 50,000 | yes |

Estimated asymptotic coverage, with unlimited queries drawn from the same distribution: **0.7841**.

Extrapolator check: fitted on a random half of the workload (2,500 queries) and asked to predict the coverage at the full workload, whose true value is known exactly, the mean relative error is **0.0067**.

## Coverage against calibration workload

| Calibration queries | Mean coverage | Source |
|---:|---:|---|
| 100 | 0.0149 | measured (rarefaction) |
| 304 | 0.0428 | measured (rarefaction) |
| 508 | 0.0683 | measured (rarefaction) |
| 712 | 0.0923 | measured (rarefaction) |
| 916 | 0.1148 | measured (rarefaction) |
| 1,120 | 0.1363 | measured (rarefaction) |
| 1,325 | 0.1568 | measured (rarefaction) |
| 1,529 | 0.1763 | measured (rarefaction) |
| 1,733 | 0.1949 | measured (rarefaction) |
| 1,937 | 0.2129 | measured (rarefaction) |
| 2,141 | 0.2301 | measured (rarefaction) |
| 2,345 | 0.2466 | measured (rarefaction) |
| 2,550 | 0.2626 | measured (rarefaction) |
| 2,754 | 0.2780 | measured (rarefaction) |
| 2,958 | 0.2928 | measured (rarefaction) |
| 3,162 | 0.3071 | measured (rarefaction) |
| 3,366 | 0.3210 | measured (rarefaction) |
| 3,570 | 0.3344 | measured (rarefaction) |
| 3,775 | 0.3474 | measured (rarefaction) |
| 3,979 | 0.3600 | measured (rarefaction) |
| 4,183 | 0.3721 | measured (rarefaction) |
| 4,387 | 0.3839 | measured (rarefaction) |
| 4,591 | 0.3954 | measured (rarefaction) |
| 4,795 | 0.4065 | measured (rarefaction) |
| 5,000 | 0.4173 | measured (rarefaction) |
| 6,000 | 0.4659 | Chao extrapolation |
| 10,000 | 0.6038 | Chao extrapolation |
| 20,000 | 0.7405 | Chao extrapolation |
| 50,000 | 0.7835 | Chao extrapolation |
| 105,000 | 0.7841 | Chao extrapolation |
| 305,000 | 0.7841 | Chao extrapolation |
| 1,005,000 | 0.7841 | Chao extrapolation |

Rarefaction below the observed workload is exact: it is the expected coverage of a random subsample of the queries actually run. Above it, the Chao term estimates how many further items are reachable at all and is an estimate, not a measurement.

