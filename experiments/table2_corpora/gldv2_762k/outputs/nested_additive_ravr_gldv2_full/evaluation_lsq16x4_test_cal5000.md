# Frozen-LSQ16x4 allocator comparison

Every row uses the same frozen LSQ16x4 within each seed and rate, source item codes, exact packed-code cap, fp32 shared codebooks, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 11.52227 | 0.0877 | 0.6973 |
| 10 | RAVR (Ordinal-KL) | 11.52227 | 0.1878 | 0.8088 |
| 10 | RAVR + unseen-item RD | 11.52227 | 0.1878 | 0.8088 |
| 10 | Ordinal-KL per exposure | 11.52227 | 0.1873 | 0.8079 |
| 10 | QAQ-style query-IP MSE | 11.52227 | 0.1060 | 0.7419 |
| 10 | Random same histogram | 11.52227 | 0.1098 | 0.7558 |
| 10 | Reconstruction RD | 11.52227 | 0.0987 | 0.7244 |
| 12 | Uniform prefix | 12.52227 | 0.1306 | 0.7515 |
| 12 | RAVR (Ordinal-KL) | 11.82230 | 0.1957 | 0.8142 |
| 12 | RAVR + unseen-item RD | 12.52227 | 0.2029 | 0.8179 |
| 12 | Ordinal-KL per exposure | 11.82230 | 0.1957 | 0.8142 |
| 12 | QAQ-style query-IP MSE | 12.52227 | 0.1497 | 0.7820 |
| 12 | Random same histogram | 11.82230 | 0.1298 | 0.7722 |
| 12 | Reconstruction RD | 12.52227 | 0.1468 | 0.7743 |
| 14 | Uniform prefix | 13.52227 | 0.1790 | 0.7961 |
| 14 | RAVR (Ordinal-KL) | 11.82230 | 0.1957 | 0.8142 |
| 14 | RAVR + unseen-item RD | 13.52227 | 0.2139 | 0.8238 |
| 14 | Ordinal-KL per exposure | 11.82230 | 0.1957 | 0.8142 |
| 14 | QAQ-style query-IP MSE | 13.52227 | 0.1846 | 0.8072 |
| 14 | Random same histogram | 11.82230 | 0.1298 | 0.7722 |
| 14 | Reconstruction RD | 13.52227 | 0.1921 | 0.8111 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.10015 | [+0.09045, +0.10990] | [+0.08688, +0.11343] | 10/10 |
| 10 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00048 | [-0.00088, +0.00185] | [-0.01280, +0.01375] | 7/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.08187 | [+0.07228, +0.09163] | [+0.06860, +0.09515] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.08917 | [+0.07790, +0.10068] | [+0.07590, +0.10245] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.07800 | [+0.06790, +0.08820] | [+0.06472, +0.09127] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.11147 | [+0.10371, +0.11955] | [+0.09819, +0.12474] | 10/10 |
| 10 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00087 | [+0.00007, +0.00171] | [-0.01241, +0.01414] | 9/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.06691 | [+0.06000, +0.07419] | [+0.05364, +0.08019] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.08443 | [+0.07534, +0.09396] | [+0.07115, +0.09770] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.05297 | [+0.04660, +0.05960] | [+0.03969, +0.06624] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.06505 | [+0.05645, +0.07355] | [+0.05177, +0.07832] | 10/10 |
| 12 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01328, +0.01328] | 0/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.04602 | [+0.03747, +0.05455] | [+0.03275, +0.05930] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.04890 | [+0.03840, +0.05960] | [+0.03562, +0.06217] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.06582 | [+0.05643, +0.07518] | [+0.05255, +0.07910] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.06266 | [+0.05689, +0.06867] | [+0.04939, +0.07594] | 10/10 |
| 12 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01328, +0.01328] | 0/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.03213 | [+0.02700, +0.03755] | [+0.01885, +0.04540] | 10/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.03983 | [+0.03283, +0.04734] | [+0.02655, +0.05310] | 10/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.04200 | [+0.03677, +0.04744] | [+0.02873, +0.05528] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | +0.01665 | [+0.00895, +0.02422] | [+0.00337, +0.02993] | 10/10 |
| 14 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01328, +0.01328] | 0/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.01105 | [+0.00218, +0.01975] | [-0.00223, +0.02433] | 10/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | +0.00355 | [-0.00592, +0.01300] | [-0.00973, +0.01683] | 6/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.06582 | [+0.05643, +0.07518] | [+0.05255, +0.07910] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | +0.01802 | [+0.01397, +0.02205] | [+0.00475, +0.03130] | 10/10 |
| 14 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01328, +0.01328] | 0/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00693 | [+0.00261, +0.01136] | [-0.00635, +0.02020] | 10/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | +0.00306 | [-0.00129, +0.00746] | [-0.01021, +0.01634] | 8/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.04200 | [+0.03677, +0.04744] | [+0.02873, +0.05528] | 10/10 |
