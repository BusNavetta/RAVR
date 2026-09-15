# Allocator comparison -- deep1m same frozen LSQ16x4 within each seed and rate

10 codec seeds. `B/item` is the complete serialized file, so a strategy that reports **less than the uniform row at the same target has not spent its budget**.

| Target | Strategy | B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 11.10291 | 0.0873 | 0.8220 |
| 10 | Random same histogram | 11.10291 | 0.1354 | 0.8711 |
| 10 | Reconstruction RD | 11.10291 | 0.1211 | 0.8575 |
| 10 | QAQ-style query-IP MSE | 11.10291 | 0.1193 | 0.8581 |
| 10 | Ordinal-KL per exposure | 11.10291 | 0.1809 | 0.8851 |
| 10 | RAVR + unseen-item RD | 11.10291 | 0.1811 | 0.8850 |
| 10 | RAVR (Ordinal-KL) | 11.10291 | 0.1811 | 0.8850 |
| 12 | Uniform prefix | 12.10291 | 0.1447 | 0.8689 |
| 12 | Random same histogram | 11.52697 (under) | 0.1738 | 0.8884 |
| 12 | Reconstruction RD | 12.10291 | 0.1876 | 0.8953 |
| 12 | QAQ-style query-IP MSE | 12.10291 | 0.1940 | 0.8978 |
| 12 | Ordinal-KL per exposure | 11.52697 (under) | 0.2097 | 0.8990 |
| 12 | RAVR + unseen-item RD | 12.10291 | 0.2263 | 0.9067 |
| 12 | RAVR (Ordinal-KL) | 11.52697 (under) | 0.2097 | 0.8990 |
| 14 | Uniform prefix | 13.10291 | 0.2207 | 0.9070 |
| 14 | Random same histogram | 11.52697 (under) | 0.1738 | 0.8884 |
| 14 | Reconstruction RD | 13.10291 | 0.2571 | 0.9214 |
| 14 | QAQ-style query-IP MSE | 13.10291 | 0.2644 | 0.9229 |
| 14 | Ordinal-KL per exposure | 11.52697 (under) | 0.2097 | 0.8990 |
| 14 | RAVR + unseen-item RD | 13.10291 | 0.2676 | 0.9230 |
| 14 | RAVR (Ordinal-KL) | 11.52697 (under) | 0.2097 | 0.8990 |

`(under)` marks a row that did not spend its cap.

