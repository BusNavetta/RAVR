# Allocator comparison -- gldv2_full same frozen LSQ16x4 within each seed and rate

10 codec seeds. `B/item` is the complete serialized file, so a strategy that reports **less than the uniform row at the same target has not spent its budget**.

| Target | Strategy | B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 11.52227 | 0.0877 | 0.6973 |
| 10 | Random same histogram | 11.52227 | 0.1098 | 0.7558 |
| 10 | Reconstruction RD | 11.52227 | 0.0987 | 0.7244 |
| 10 | QAQ-style query-IP MSE | 11.52227 | 0.1060 | 0.7419 |
| 10 | Ordinal-KL per exposure | 11.52227 | 0.1873 | 0.8079 |
| 10 | RAVR + unseen-item RD | 11.52227 | 0.1878 | 0.8088 |
| 10 | RAVR (Ordinal-KL) | 11.52227 | 0.1878 | 0.8088 |
| 12 | Uniform prefix | 12.52227 | 0.1306 | 0.7515 |
| 12 | Random same histogram | 11.82230 (under) | 0.1298 | 0.7722 |
| 12 | Reconstruction RD | 12.52227 | 0.1468 | 0.7743 |
| 12 | QAQ-style query-IP MSE | 12.52227 | 0.1497 | 0.7820 |
| 12 | Ordinal-KL per exposure | 11.82230 (under) | 0.1957 | 0.8142 |
| 12 | RAVR + unseen-item RD | 12.52227 | 0.2029 | 0.8179 |
| 12 | RAVR (Ordinal-KL) | 11.82230 (under) | 0.1957 | 0.8142 |
| 14 | Uniform prefix | 13.52227 | 0.1790 | 0.7961 |
| 14 | Random same histogram | 11.82230 (under) | 0.1298 | 0.7722 |
| 14 | Reconstruction RD | 13.52227 | 0.1921 | 0.8111 |
| 14 | QAQ-style query-IP MSE | 13.52227 | 0.1846 | 0.8072 |
| 14 | Ordinal-KL per exposure | 11.82230 (under) | 0.1957 | 0.8142 |
| 14 | RAVR + unseen-item RD | 13.52227 | 0.2139 | 0.8238 |
| 14 | RAVR (Ordinal-KL) | 11.82230 (under) | 0.1957 | 0.8142 |

`(under)` marks a row that did not spend its cap.

