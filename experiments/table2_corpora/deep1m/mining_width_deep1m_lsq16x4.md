# Mining width and coverage -- Deep1M (Deep1B base prefix, 96-d)

Gallery 1,000,000 items, 5,000 calibration queries, seed 17, LSQ16x4.

Reproduction check: this script gives 0.3933 at the depth the panel used (50); the panel recorded 0.4170.

| `hard_negative_depth` | Mined set per query | Coverage | Calibration cost |
|---:|---:|---:|---:|
| 50 | 160 | 0.3933 | 1.0x |
| 100 | 310 | 0.5915 | 1.9x |
| 200 | 610 | 0.7912 | 3.8x |
| 400 | 1210 | 0.9298 | 7.6x |
| 800 | 2410 | 0.9872 | 15.1x |
| 1600 | 4810 | 0.9990 | 30.1x |
| 3200 | 9610 | 1.0000 | 60.1x |

| Target | Coverage needed | Depth that reaches it | Mined set |
|---:|---:|---:|---:|
| 10 | 0.25 | 50 | 160 |
| 12 | 0.50 | 100 | 310 |
| 14 | 0.75 | 200 | 610 |

The mined set is what the Ordinal-KL accumulation runs over, so calibration cost grows with it roughly linearly. Widening the mining is the only knob that acts on the constant behind `coverage = O(Q * width / N)`; more calibration queries cannot, because coverage saturates well below 1 (see `analyze_unseen.py`).

