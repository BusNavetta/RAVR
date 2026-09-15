# Frozen-LSQ16x4 allocator comparison

Every row uses the same frozen LSQ16x4 within each seed and rate, source item codes, exact packed-code cap, fp32 shared codebooks, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 14.97824 | 0.2231 | 0.8099 |
| 10 | RAVR (Ordinal-KL) | 14.97824 | 0.3012 | 0.8585 |
| 10 | RAVR + unseen-item RD | 14.97824 | 0.3012 | 0.8585 |
| 10 | Ordinal-KL per exposure | 14.97824 | 0.2915 | 0.8545 |
| 10 | Candidate frequency | 14.97824 | 0.2530 | 0.8429 |
| 10 | QAQ-style query-IP MSE | 14.97824 | 0.2333 | 0.8311 |
| 10 | Random same histogram | 14.97824 | 0.2118 | 0.8274 |
| 10 | Reconstruction RD | 14.97824 | 0.2422 | 0.8330 |
| 12 | Uniform prefix | 15.97824 | 0.2961 | 0.8612 |
| 12 | RAVR (Ordinal-KL) | 15.97824 | 0.3771 | 0.8919 |
| 12 | RAVR + unseen-item RD | 15.97824 | 0.3771 | 0.8919 |
| 12 | Ordinal-KL per exposure | 15.97824 | 0.3707 | 0.8899 |
| 12 | Candidate frequency | 15.97824 | 0.3491 | 0.8832 |
| 12 | QAQ-style query-IP MSE | 15.97824 | 0.3155 | 0.8765 |
| 12 | Random same histogram | 15.97824 | 0.2930 | 0.8683 |
| 12 | Reconstruction RD | 15.97824 | 0.3185 | 0.8778 |
| 14 | Uniform prefix | 16.97824 | 0.3707 | 0.8988 |
| 14 | RAVR (Ordinal-KL) | 16.97824 | 0.4174 | 0.9125 |
| 14 | RAVR + unseen-item RD | 16.97824 | 0.4174 | 0.9125 |
| 14 | Ordinal-KL per exposure | 16.97824 | 0.4167 | 0.9120 |
| 14 | Candidate frequency | 16.97824 | 0.4078 | 0.9089 |
| 14 | QAQ-style query-IP MSE | 16.97824 | 0.3874 | 0.9063 |
| 14 | Random same histogram | 16.97824 | 0.3711 | 0.9007 |
| 14 | Reconstruction RD | 16.97824 | 0.3892 | 0.9077 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.07803 | [+0.06603, +0.09000] | [+0.06110, +0.09495] | 10/10 |
| 10 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00965 | [+0.00510, +0.01430] | [-0.00728, +0.02658] | 10/10 |
| 10 | teacher_recall_at_10 | Candidate frequency | +0.04810 | [+0.03520, +0.06100] | [+0.03117, +0.06503] | 10/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.06782 | [+0.05508, +0.08063] | [+0.05090, +0.08475] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.05897 | [+0.04635, +0.07155] | [+0.04205, +0.07590] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.08937 | [+0.07562, +0.10313] | [+0.07245, +0.10630] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.04863 | [+0.04315, +0.05407] | [+0.03171, +0.06556] | 10/10 |
| 10 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00403 | [+0.00252, +0.00559] | [-0.01289, +0.02096] | 10/10 |
| 10 | teacher_ndcg_at_10 | Candidate frequency | +0.01561 | [+0.00998, +0.02126] | [-0.00132, +0.03253] | 10/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.02743 | [+0.02215, +0.03261] | [+0.01051, +0.04436] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.02545 | [+0.01999, +0.03078] | [+0.00853, +0.04238] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.03111 | [+0.02565, +0.03661] | [+0.01418, +0.04803] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.08100 | [+0.06970, +0.09240] | [+0.06407, +0.09793] | 10/10 |
| 12 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00638 | [+0.00225, +0.01052] | [-0.01055, +0.02330] | 10/10 |
| 12 | teacher_recall_at_10 | Candidate frequency | +0.02805 | [+0.01745, +0.03917] | [+0.01112, +0.04498] | 10/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.06165 | [+0.04975, +0.07343] | [+0.04472, +0.07857] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.05862 | [+0.04655, +0.07055] | [+0.04170, +0.07555] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.08407 | [+0.07168, +0.09663] | [+0.06715, +0.10100] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.03073 | [+0.02605, +0.03535] | [+0.01380, +0.04765] | 10/10 |
| 12 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00202 | [+0.00031, +0.00382] | [-0.01490, +0.01895] | 9/10 |
| 12 | teacher_ndcg_at_10 | Candidate frequency | +0.00871 | [+0.00441, +0.01319] | [-0.00822, +0.02563] | 10/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.01541 | [+0.01138, +0.01933] | [-0.00152, +0.03233] | 10/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.01408 | [+0.00958, +0.01843] | [-0.00285, +0.03100] | 10/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.02358 | [+0.01910, +0.02805] | [+0.00666, +0.04051] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | +0.04665 | [+0.03815, +0.05555] | [+0.02972, +0.06357] | 10/10 |
| 14 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00068 | [-0.00178, +0.00318] | [-0.01625, +0.01760] | 5/10 |
| 14 | teacher_recall_at_10 | Candidate frequency | +0.00958 | [+0.00293, +0.01637] | [-0.00735, +0.02650] | 10/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.03000 | [+0.02232, +0.03767] | [+0.01307, +0.04692] | 10/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | +0.02817 | [+0.02045, +0.03585] | [+0.01125, +0.04510] | 10/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.04622 | [+0.03832, +0.05400] | [+0.02930, +0.06315] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | +0.01376 | [+0.01093, +0.01667] | [-0.00317, +0.03068] | 10/10 |
| 14 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00054 | [-0.00045, +0.00162] | [-0.01639, +0.01746] | 6/10 |
| 14 | teacher_ndcg_at_10 | Candidate frequency | +0.00360 | [+0.00115, +0.00615] | [-0.01333, +0.02052] | 10/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00623 | [+0.00371, +0.00870] | [-0.01070, +0.02315] | 10/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | +0.00480 | [+0.00204, +0.00745] | [-0.01213, +0.02172] | 10/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.01182 | [+0.00907, +0.01456] | [-0.00511, +0.02874] | 10/10 |
