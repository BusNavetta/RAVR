# The allocator gain with the mining depth scaled to the gallery

LSQ16x4, seed 17, test split, 5,000 calibration queries on every corpus. The plotted strategy is `ravr_unseen_rd`, which fills the cap exactly, so every gain below is at equal bytes. The scaling rule is `d*(N) = ((k + 3*50) * N / 100000 - k) / 3`.

| Corpus | Items | Rule wants | Depth used | Coverage | Cap | uniform | RAVR + unseen RD | Gain | Gain at d=50 | Change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOP | 100,000 | 50 | 50 | 0.9675 | 10 | 0.2090 | 0.2855 | +0.0765 | +0.0765 | +0.0000 |
| | | | | | 12 | 0.2865 | 0.3688 | +0.0823 | +0.0823 | +0.0000 |
| | | | | | 14 | 0.3393 | 0.4088 | +0.0695 | +0.0695 | +0.0000 |
| GLDv2 100K | 100,000 | 50 | 50 | 0.8813 | 10 | 0.1595 | 0.2898 | +0.1303 | +0.1303 | +0.0000 |
| | | | | | 12 | 0.2145 | 0.3125 | +0.0980 | +0.0980 | +0.0000 |
| | | | | | 14 | 0.2690 | 0.3210 | +0.0520 | +0.0520 | +0.0000 |
| GLDv2 full | 761,757 | 403 | 400 | 0.8652 | 10 | 0.0857 | 0.1848 | +0.0990 | +0.0985 | +0.0005 |
| | | | | | 12 | 0.1298 | 0.2073 | +0.0775 | +0.0728 | +0.0048 |
| | | | | | 14 | 0.1787 | 0.2220 | +0.0433 | +0.0368 | +0.0065 |
| Deep1M | 1,000,000 | 530 | 400 | 0.9329 | 10 | 0.0847 | 0.1670 | +0.0823 | +0.0960 | -0.0137 |
| | | | | | 12 | 0.1328 | 0.2398 | +0.1070 | +0.1002 | +0.0068 |
| | | | | | 14 | 0.2112 | 0.2863 | +0.0750 | +0.0625 | +0.0125 |

Reading it: at 100K the rule returns the default depth, so those two corpora are unchanged and act as the reference. The two large corpora move, and the question the figure answers is whether the gain still falls with gallery size once every corpus is calibrated at a comparable coverage.

## The other knob, where it was available

Scaling the *queries* instead of the mining, on Deep1M: Q=9,200 at the default depth, coverage 0.5867 against 0.9329 for the depth-scaled cell. Fewer items seen, and it still wins where the budget is not the binding constraint.

| Cap | depth scaled (d=400) | queries scaled (Q=9,200) | difference |
|---:|---:|---:|---:|
| 10 | +0.0823 | +0.1025 | +0.0202 |
| 12 | +0.1070 | +0.1202 | +0.0133 |
| 14 | +0.0750 | +0.0703 | -0.0047 |

