# Frozen-LSQ16x4 allocator comparison

Every row uses the same frozen LSQ16x4 within each seed and rate, source item codes, exact packed-code cap, fp32 shared codebooks, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 14.97824 | 0.1633 | 0.7139 |
| 10 | RAVR (Ordinal-KL) | 14.97824 | 0.2849 | 0.8111 |
| 10 | RAVR + unseen-item RD | 14.97824 | 0.2849 | 0.8111 |
| 10 | Ordinal-KL per exposure | 14.97824 | 0.2665 | 0.7986 |
| 10 | Candidate frequency | 14.97824 | 0.2795 | 0.8121 |
| 10 | QAQ-style query-IP MSE | 14.97824 | 0.1834 | 0.7487 |
| 10 | Random same histogram | 14.97824 | 0.1620 | 0.7428 |
| 10 | Reconstruction RD | 14.97824 | 0.1790 | 0.7406 |
| 12 | Uniform prefix | 15.97824 | 0.2180 | 0.7642 |
| 12 | RAVR (Ordinal-KL) | 15.97824 | 0.3125 | 0.8288 |
| 12 | RAVR + unseen-item RD | 15.97824 | 0.3125 | 0.8288 |
| 12 | Ordinal-KL per exposure | 15.97824 | 0.3072 | 0.8247 |
| 12 | Candidate frequency | 15.97824 | 0.3112 | 0.8298 |
| 12 | QAQ-style query-IP MSE | 15.97824 | 0.2383 | 0.7888 |
| 12 | Random same histogram | 15.97824 | 0.2244 | 0.7844 |
| 12 | Reconstruction RD | 15.97824 | 0.2392 | 0.7875 |
| 14 | Uniform prefix | 16.97824 | 0.2761 | 0.8053 |
| 14 | RAVR (Ordinal-KL) | 16.95917 | 0.3236 | 0.8349 |
| 14 | RAVR + unseen-item RD | 16.97824 | 0.3236 | 0.8350 |
| 14 | Ordinal-KL per exposure | 16.95917 | 0.3236 | 0.8349 |
| 14 | Candidate frequency | 16.97824 | 0.3251 | 0.8364 |
| 14 | QAQ-style query-IP MSE | 16.97824 | 0.2803 | 0.8132 |
| 14 | Random same histogram | 16.95917 | 0.2795 | 0.8138 |
| 14 | Reconstruction RD | 16.97824 | 0.2914 | 0.8199 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.12160 | [+0.11000, +0.13315] | [+0.10718, +0.13603] | 10/10 |
| 10 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.01848 | [+0.01260, +0.02445] | [+0.00405, +0.03290] | 10/10 |
| 10 | teacher_recall_at_10 | Candidate frequency | +0.00543 | [-0.00160, +0.01270] | [-0.00900, +0.01985] | 10/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.10160 | [+0.08968, +0.11338] | [+0.08717, +0.11603] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.10590 | [+0.09345, +0.11855] | [+0.09147, +0.12033] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.12295 | [+0.11173, +0.13428] | [+0.10853, +0.13738] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.09719 | [+0.08982, +0.10464] | [+0.08277, +0.11162] | 10/10 |
| 10 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.01252 | [+0.00961, +0.01564] | [-0.00191, +0.02694] | 10/10 |
| 10 | teacher_ndcg_at_10 | Candidate frequency | -0.00103 | [-0.00458, +0.00249] | [-0.01545, +0.01340] | 2/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.06241 | [+0.05614, +0.06873] | [+0.04798, +0.07683] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.07048 | [+0.06293, +0.07836] | [+0.05605, +0.08490] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.06830 | [+0.06226, +0.07446] | [+0.05388, +0.08273] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.09450 | [+0.08625, +0.10268] | [+0.08008, +0.10893] | 10/10 |
| 12 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00530 | [+0.00195, +0.00875] | [-0.00913, +0.01973] | 10/10 |
| 12 | teacher_recall_at_10 | Candidate frequency | +0.00122 | [-0.00368, +0.00635] | [-0.01320, +0.01565] | 6/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.07422 | [+0.06485, +0.08375] | [+0.05980, +0.08865] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.07330 | [+0.06368, +0.08310] | [+0.05887, +0.08773] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.08810 | [+0.07968, +0.09660] | [+0.07367, +0.10253] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.06466 | [+0.06013, +0.06928] | [+0.05023, +0.07908] | 10/10 |
| 12 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00416 | [+0.00265, +0.00579] | [-0.01026, +0.01859] | 10/10 |
| 12 | teacher_ndcg_at_10 | Candidate frequency | -0.00094 | [-0.00305, +0.00118] | [-0.01536, +0.01349] | 3/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.04008 | [+0.03507, +0.04522] | [+0.02566, +0.05451] | 10/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.04139 | [+0.03624, +0.04690] | [+0.02696, +0.05581] | 10/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.04440 | [+0.04039, +0.04855] | [+0.02997, +0.05882] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | +0.04750 | [+0.04042, +0.05480] | [+0.03307, +0.06192] | 10/10 |
| 14 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00005 | [-0.00010, +0.00025] | [-0.01438, +0.01448] | 3/10 |
| 14 | teacher_recall_at_10 | Candidate frequency | -0.00148 | [-0.00532, +0.00243] | [-0.01590, +0.01295] | 2/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.04327 | [+0.03580, +0.05098] | [+0.02885, +0.05770] | 10/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | +0.03225 | [+0.02527, +0.03935] | [+0.01782, +0.04668] | 10/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.04410 | [+0.03827, +0.04995] | [+0.02967, +0.05852] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | +0.02963 | [+0.02622, +0.03308] | [+0.01521, +0.04406] | 10/10 |
| 14 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00002 | [-0.00005, +0.00011] | [-0.01441, +0.01444] | 3/10 |
| 14 | teacher_ndcg_at_10 | Candidate frequency | -0.00148 | [-0.00301, -0.00002] | [-0.01590, +0.01295] | 2/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.02168 | [+0.01819, +0.02552] | [+0.00725, +0.03610] | 10/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | +0.01499 | [+0.01235, +0.01777] | [+0.00057, +0.02942] | 10/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.02116 | [+0.01859, +0.02380] | [+0.00673, +0.03558] | 10/10 |
