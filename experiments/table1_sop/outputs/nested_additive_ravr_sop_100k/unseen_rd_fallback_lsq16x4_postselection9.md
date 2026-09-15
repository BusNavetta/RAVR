# Sparse-calibration unseen-item fallback

seed 17 exploratory; listed seeds are post-selection replication only
Compute backend: `cuda` (all rows).

| Target | Metric | Fallback minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | uniform | +0.10303 | [+0.08856, +0.11744] | [+0.08808, +0.11797] | 9/9 |
| 10 | teacher_recall_at_10 | ravr | +0.00000 | [+0.00000, +0.00000] | [-0.01494, +0.01494] | 0/9 |
| 10 | teacher_ndcg_at_10 | uniform | +0.06041 | [+0.05441, +0.06646] | [+0.04547, +0.07536] | 9/9 |
| 10 | teacher_ndcg_at_10 | ravr | +0.00000 | [+0.00000, +0.00000] | [-0.01494, +0.01494] | 0/9 |
| 12 | teacher_recall_at_10 | uniform | +0.07589 | [+0.06453, +0.08736] | [+0.06094, +0.09083] | 9/9 |
| 12 | teacher_recall_at_10 | ravr | +0.03314 | [+0.02700, +0.03953] | [+0.01819, +0.04808] | 9/9 |
| 12 | teacher_ndcg_at_10 | uniform | +0.03425 | [+0.03041, +0.03822] | [+0.01931, +0.04920] | 9/9 |
| 12 | teacher_ndcg_at_10 | ravr | +0.01726 | [+0.01431, +0.02047] | [+0.00232, +0.03220] | 9/9 |
| 14 | teacher_recall_at_10 | uniform | +0.03544 | [+0.02756, +0.04333] | [+0.02050, +0.05039] | 9/9 |
| 14 | teacher_recall_at_10 | ravr | +0.06975 | [+0.05967, +0.08006] | [+0.05481, +0.08469] | 9/9 |
| 14 | teacher_ndcg_at_10 | uniform | +0.01275 | [+0.01061, +0.01494] | [-0.00220, +0.02769] | 9/9 |
| 14 | teacher_ndcg_at_10 | ravr | +0.03369 | [+0.02948, +0.03807] | [+0.01874, +0.04863] | 9/9 |

| Target | Strategy | Mean B/item | Range |
|---:|---|---:|---:|
| 10 | uniform | 14.97824 | [14.97824, 14.97824] |
| 10 | ravr | 14.97824 | [14.97824, 14.97824] |
| 10 | ravr_unseen_rd | 14.97824 | [14.97824, 14.97824] |
| 12 | uniform | 15.97824 | [15.97824, 15.97824] |
| 12 | ravr | 15.21401 | [15.20167, 15.22509] |
| 12 | ravr_unseen_rd | 15.97824 | [15.97824, 15.97824] |
| 14 | uniform | 16.97824 | [16.97824, 16.97824] |
| 14 | ravr | 15.21401 | [15.20167, 15.22509] |
| 14 | ravr_unseen_rd | 16.97824 | [16.97824, 16.97824] |
