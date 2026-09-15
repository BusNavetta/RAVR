# Random allocation range -- Google Landmarks Dataset v2.1

Every row is scored at the identical complete-file cap. The random range is the distribution over legal per-item prefix assignments at that cap; the percentile says what fraction of random allocations a strategy beats.

| Cap | Draws | Random min | Random median | Random max | Strategy | R@10 | Percentile |
|---:|---:|---:|---:|---:|---|---:|---:|
| 10 | 1,536,960 | 0.1263 | 0.1575 | 0.1895 | Uniform prefix | 0.1595 | 61.50% |
| | | | | | RAVR histogram, shuffled | 0.1670 | 92.89% |
| | | | | | Reconstruction RD | 0.1820 | 99.99% |
| | | | | | Query-IP MSE | 0.1863 | 100.00% |
| | | | | | Ordinal-KL per exposure | 0.2727 | 100.00% |
| | | | | | Candidate frequency | 0.2840 | 100.00% |
| | | | | | RAVR + unseen RD | 0.2898 | 100.00% |
| | | | | | RAVR (Ordinal-KL) | 0.2898 | 100.00% |
| 12 | 1,699,360 | 0.1778 | 0.2080 | 0.2573 | Uniform prefix | 0.2145 | 79.62% |
| | | | | | RAVR histogram, shuffled | 0.2233 | 96.42% |
| | | | | | Reconstruction RD | 0.2510 | 100.00% |
| | | | | | Query-IP MSE | 0.2525 | 100.00% |
| | | | | | Ordinal-KL per exposure | 0.3013 | 100.00% |
| | | | | | Candidate frequency | 0.3113 | 100.00% |
| | | | | | RAVR + unseen RD | 0.3125 | 100.00% |
| | | | | | RAVR (Ordinal-KL) | 0.3125 | 100.00% |
| 14 | 1,539,024 | 0.2380 | 0.2660 | 0.3030 | Uniform prefix | 0.2690 | 65.73% |
| | | | | | RAVR histogram, shuffled | 0.2787 | 95.96% |
| | | | | | Reconstruction RD | 0.2938 | 99.98% |
| | | | | | Query-IP MSE | 0.2853 | 99.44% |
| | | | | | Ordinal-KL per exposure | 0.3210 | 100.00% |
| | | | | | Candidate frequency | 0.3218 | 100.00% |
| | | | | | RAVR + unseen RD | 0.3210 | 100.00% |
| | | | | | RAVR (Ordinal-KL) | 0.3210 | 100.00% |

A percentile of 100 means no random allocation matched it in this sweep; it is a lower bound on how extreme the strategy is, limited by the number of draws.

