# Mining width and coverage -- Google Landmarks Dataset v2.1, full index

Gallery 761,757 items, 5,000 calibration queries, seed 17, LSQ16x4.

Reproduction check: this script gives 0.3684 at the depth the panel used (50); the panel recorded 0.3840.

| `hard_negative_depth` | Mined set per query | Coverage | Calibration cost |
|---:|---:|---:|---:|
| 50 | 160 | 0.3684 | 1.0x |
| 100 | 310 | 0.5306 | 1.9x |
| 200 | 610 | 0.7080 | 3.8x |
| 400 | 1210 | 0.8611 | 7.6x |
| 800 | 2410 | 0.9548 | 15.1x |
| 1600 | 4810 | 0.9914 | 30.1x |
| 3200 | 9610 | 0.9992 | 60.1x |

| Target | Coverage needed | Depth that reaches it | Mined set |
|---:|---:|---:|---:|
| 10 | 0.25 | 50 | 160 |
| 12 | 0.50 | 100 | 310 |
| 14 | 0.75 | 400 | 1210 |

The mined set is what the Ordinal-KL accumulation runs over, so calibration cost grows with it roughly linearly. Widening the mining is the only knob that acts on the constant behind `coverage = O(Q * width / N)`; more calibration queries cannot, because coverage saturates well below 1 (see `analyze_unseen.py`).

