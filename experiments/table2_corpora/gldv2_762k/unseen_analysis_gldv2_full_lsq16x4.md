# Unseen items and calibration coverage -- Google Landmarks Dataset v2.1, full index

Gallery 761,757 items; 5,000 calibration queries now; 5,000 usable queries exist in the manifest at all.

A query touches **155.3 items** on average (the mined hard set is bounded by 160 regardless of gallery size), so coverage is now **0.3877**.

## Can more calibration queries fill the cap

| Target | Coverage `ravr` needs | Queries that would reach it | Reachable at all |
|---:|---:|---:|---|
| 10 | 0.25 | 2,345 | yes |
| 12 | 0.50 | 10,000 | yes |
| 14 | 0.75 | never | NO |

Estimated asymptotic coverage, with unlimited queries drawn from the same distribution: **0.6478**.

Extrapolator check: fitted on a random half of the workload (2,500 queries) and asked to predict the coverage at the full workload, whose true value is known exactly, the mean relative error is **0.0126**.

## Coverage against calibration workload

| Calibration queries | Mean coverage | Source |
|---:|---:|---|
| 100 | 0.0191 | measured (rarefaction) |
| 304 | 0.0525 | measured (rarefaction) |
| 508 | 0.0813 | measured (rarefaction) |
| 712 | 0.1068 | measured (rarefaction) |
| 916 | 0.1298 | measured (rarefaction) |
| 1,120 | 0.1509 | measured (rarefaction) |
| 1,325 | 0.1704 | measured (rarefaction) |
| 1,529 | 0.1885 | measured (rarefaction) |
| 1,733 | 0.2053 | measured (rarefaction) |
| 1,937 | 0.2212 | measured (rarefaction) |
| 2,141 | 0.2362 | measured (rarefaction) |
| 2,345 | 0.2503 | measured (rarefaction) |
| 2,550 | 0.2639 | measured (rarefaction) |
| 2,754 | 0.2767 | measured (rarefaction) |
| 2,958 | 0.2889 | measured (rarefaction) |
| 3,162 | 0.3006 | measured (rarefaction) |
| 3,366 | 0.3118 | measured (rarefaction) |
| 3,570 | 0.3225 | measured (rarefaction) |
| 3,775 | 0.3329 | measured (rarefaction) |
| 3,979 | 0.3429 | measured (rarefaction) |
| 4,183 | 0.3524 | measured (rarefaction) |
| 4,387 | 0.3617 | measured (rarefaction) |
| 4,591 | 0.3706 | measured (rarefaction) |
| 4,795 | 0.3793 | measured (rarefaction) |
| 5,000 | 0.3877 | measured (rarefaction) |
| 6,000 | 0.4250 | Chao extrapolation |
| 10,000 | 0.5280 | Chao extrapolation |
| 20,000 | 0.6224 | Chao extrapolation |
| 50,000 | 0.6476 | Chao extrapolation |
| 105,000 | 0.6478 | Chao extrapolation |
| 305,000 | 0.6478 | Chao extrapolation |
| 1,005,000 | 0.6478 | Chao extrapolation |

Rarefaction below the observed workload is exact: it is the expected coverage of a random subsample of the queries actually run. Above it, the Chao term estimates how many further items are reachable at all and is an estimate, not a measurement.

