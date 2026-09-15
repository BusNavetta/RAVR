# Frozen-LSQ16x4 allocator comparison

Every row uses the same frozen LSQ16x4 within each seed and rate, source item codes, exact packed-code cap, fp32 shared codebooks, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 11.10291 | 0.0873 | 0.8220 |
| 10 | RAVR (Ordinal-KL) | 11.10291 | 0.1811 | 0.8850 |
| 10 | RAVR + unseen-item RD | 11.10291 | 0.1811 | 0.8850 |
| 10 | Ordinal-KL per exposure | 11.10291 | 0.1809 | 0.8851 |
| 10 | QAQ-style query-IP MSE | 11.10291 | 0.1193 | 0.8581 |
| 10 | Random same histogram | 11.10291 | 0.1354 | 0.8711 |
| 10 | Reconstruction RD | 11.10291 | 0.1211 | 0.8575 |
| 12 | Uniform prefix | 12.10291 | 0.1447 | 0.8689 |
| 12 | RAVR (Ordinal-KL) | 11.52697 | 0.2097 | 0.8990 |
| 12 | RAVR + unseen-item RD | 12.10291 | 0.2263 | 0.9067 |
| 12 | Ordinal-KL per exposure | 11.52697 | 0.2097 | 0.8990 |
| 12 | QAQ-style query-IP MSE | 12.10291 | 0.1940 | 0.8978 |
| 12 | Random same histogram | 11.52697 | 0.1738 | 0.8884 |
| 12 | Reconstruction RD | 12.10291 | 0.1876 | 0.8953 |
| 14 | Uniform prefix | 13.10291 | 0.2207 | 0.9070 |
| 14 | RAVR (Ordinal-KL) | 11.52697 | 0.2097 | 0.8990 |
| 14 | RAVR + unseen-item RD | 13.10291 | 0.2676 | 0.9230 |
| 14 | Ordinal-KL per exposure | 11.52697 | 0.2097 | 0.8990 |
| 14 | QAQ-style query-IP MSE | 13.10291 | 0.2644 | 0.9229 |
| 14 | Random same histogram | 11.52697 | 0.1738 | 0.8884 |
| 14 | Reconstruction RD | 13.10291 | 0.2571 | 0.9214 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.09383 | [+0.08348, +0.10433] | [+0.07920, +0.10845] | 10/10 |
| 10 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00025 | [-0.00125, +0.00173] | [-0.01437, +0.01487] | 6/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.06182 | [+0.05057, +0.07305] | [+0.04720, +0.07645] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.06000 | [+0.04870, +0.07135] | [+0.04537, +0.07462] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.04572 | [+0.03477, +0.05673] | [+0.03110, +0.06035] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.06299 | [+0.05786, +0.06809] | [+0.04836, +0.07761] | 10/10 |
| 10 | teacher_ndcg_at_10 | Ordinal-KL per exposure | -0.00009 | [-0.00060, +0.00042] | [-0.01472, +0.01453] | 2/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.02695 | [+0.02331, +0.03055] | [+0.01232, +0.04157] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.02751 | [+0.02341, +0.03150] | [+0.01288, +0.04213] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.01395 | [+0.01071, +0.01714] | [-0.00068, +0.02857] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.06497 | [+0.05390, +0.07603] | [+0.05035, +0.07960] | 10/10 |
| 12 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01462, +0.01462] | 0/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.01568 | [+0.00530, +0.02617] | [+0.00105, +0.03030] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.02207 | [+0.01070, +0.03352] | [+0.00745, +0.03670] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.03590 | [+0.02625, +0.04565] | [+0.02128, +0.05052] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.03011 | [+0.02613, +0.03418] | [+0.01549, +0.04474] | 10/10 |
| 12 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01462, +0.01462] | 0/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00115 | [-0.00183, +0.00406] | [-0.01347, +0.01578] | 7/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.00368 | [+0.00015, +0.00711] | [-0.01095, +0.01830] | 9/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.01052 | [+0.00801, +0.01297] | [-0.00410, +0.02515] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | -0.01095 | [-0.02252, +0.00102] | [-0.02557, +0.00367] | 2/10 |
| 14 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01462, +0.01462] | 0/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | -0.05467 | [-0.06478, -0.04450] | [-0.06930, -0.04005] | 0/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | -0.04742 | [-0.05880, -0.03592] | [-0.06205, -0.03280] | 0/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.03590 | [+0.02625, +0.04565] | [+0.02128, +0.05052] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | -0.00799 | [-0.01165, -0.00433] | [-0.02262, +0.00663] | 0/10 |
| 14 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00000 | [+0.00000, +0.00000] | [-0.01462, +0.01462] | 0/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | -0.02391 | [-0.02712, -0.02083] | [-0.03853, -0.00928] | 0/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | -0.02247 | [-0.02597, -0.01903] | [-0.03709, -0.00784] | 0/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.01052 | [+0.00801, +0.01297] | [-0.00410, +0.02515] | 10/10 |
