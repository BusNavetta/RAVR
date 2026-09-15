# Frozen-RQ16x4 allocator comparison

Every row uses the same frozen RQ16x4 within each seed and rate, source item codes, exact packed-code cap, fp16 shared codebooks, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 13.00896 | 0.2807 | 0.8464 |
| 10 | RAVR (Ordinal-KL) | 13.00896 | 0.3119 | 0.8603 |
| 10 | RAVR + unseen-item RD | 13.00896 | 0.3119 | 0.8603 |
| 10 | Ordinal-KL per exposure | 13.00896 | 0.3053 | 0.8565 |
| 10 | Candidate frequency | 13.00896 | 0.2819 | 0.8461 |
| 10 | QAQ-style query-IP MSE | 13.00896 | 0.2722 | 0.8484 |
| 10 | Random same histogram | 13.00896 | 0.2629 | 0.8383 |
| 10 | Reconstruction RD | 13.00896 | 0.2764 | 0.8516 |
| 12 | Uniform prefix | 14.00896 | 0.3303 | 0.8754 |
| 12 | RAVR (Ordinal-KL) | 14.00896 | 0.3574 | 0.8849 |
| 12 | RAVR + unseen-item RD | 14.00896 | 0.3574 | 0.8849 |
| 12 | Ordinal-KL per exposure | 14.00896 | 0.3538 | 0.8829 |
| 12 | Candidate frequency | 14.00896 | 0.3399 | 0.8752 |
| 12 | QAQ-style query-IP MSE | 14.00896 | 0.3196 | 0.8756 |
| 12 | Random same histogram | 14.00896 | 0.3142 | 0.8670 |
| 12 | Reconstruction RD | 14.00896 | 0.3245 | 0.8788 |
| 14 | Uniform prefix | 15.00896 | 0.3734 | 0.8964 |
| 14 | RAVR (Ordinal-KL) | 15.00896 | 0.3863 | 0.9005 |
| 14 | RAVR + unseen-item RD | 15.00896 | 0.3863 | 0.9005 |
| 14 | Ordinal-KL per exposure | 15.00896 | 0.3859 | 0.9003 |
| 14 | Candidate frequency | 15.00896 | 0.3813 | 0.8956 |
| 14 | QAQ-style query-IP MSE | 15.00896 | 0.3676 | 0.8973 |
| 14 | Random same histogram | 15.00896 | 0.3621 | 0.8903 |
| 14 | Reconstruction RD | 15.00896 | 0.3707 | 0.8992 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.03117 | [+0.02280, +0.03957] | [+0.01930, +0.04305] | 10/10 |
| 10 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00663 | [+0.00233, +0.01113] | [-0.00525, +0.01850] | 9/10 |
| 10 | teacher_recall_at_10 | Candidate frequency | +0.03002 | [+0.02202, +0.03807] | [+0.01815, +0.04190] | 10/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.03970 | [+0.03087, +0.04857] | [+0.02782, +0.05157] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.03550 | [+0.02602, +0.04498] | [+0.02362, +0.04737] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.04892 | [+0.04060, +0.05727] | [+0.03705, +0.06080] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.01387 | [+0.01037, +0.01741] | [+0.00199, +0.02574] | 10/10 |
| 10 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00377 | [+0.00206, +0.00565] | [-0.00811, +0.01564] | 10/10 |
| 10 | teacher_ndcg_at_10 | Candidate frequency | +0.01417 | [+0.01091, +0.01743] | [+0.00230, +0.02605] | 10/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.01186 | [+0.00814, +0.01552] | [-0.00002, +0.02373] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.00864 | [+0.00449, +0.01270] | [-0.00323, +0.02052] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.02201 | [+0.01850, +0.02558] | [+0.01013, +0.03388] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.02707 | [+0.01962, +0.03470] | [+0.01520, +0.03895] | 10/10 |
| 12 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00360 | [-0.00070, +0.00822] | [-0.00828, +0.01548] | 8/10 |
| 12 | teacher_recall_at_10 | Candidate frequency | +0.01753 | [+0.01123, +0.02380] | [+0.00565, +0.02940] | 10/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.03782 | [+0.02815, +0.04752] | [+0.02595, +0.04970] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.03285 | [+0.02352, +0.04227] | [+0.02097, +0.04473] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.04317 | [+0.03587, +0.05070] | [+0.03130, +0.05505] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.00951 | [+0.00669, +0.01231] | [-0.00236, +0.02139] | 10/10 |
| 12 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00209 | [+0.00053, +0.00383] | [-0.00979, +0.01396] | 10/10 |
| 12 | teacher_ndcg_at_10 | Candidate frequency | +0.00970 | [+0.00721, +0.01222] | [-0.00217, +0.02158] | 10/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00938 | [+0.00619, +0.01253] | [-0.00249, +0.02126] | 10/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.00612 | [+0.00260, +0.00960] | [-0.00576, +0.01799] | 10/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.01799 | [+0.01521, +0.02078] | [+0.00612, +0.02987] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | +0.01288 | [+0.00673, +0.01907] | [+0.00100, +0.02475] | 10/10 |
| 14 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00042 | [-0.00132, +0.00220] | [-0.01145, +0.01230] | 5/10 |
| 14 | teacher_recall_at_10 | Candidate frequency | +0.00498 | [-0.00035, +0.01033] | [-0.00690, +0.01685] | 9/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.01863 | [+0.01212, +0.02522] | [+0.00675, +0.03050] | 10/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | +0.01560 | [+0.00915, +0.02225] | [+0.00372, +0.02748] | 10/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.02412 | [+0.01825, +0.02995] | [+0.01225, +0.03600] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | +0.00413 | [+0.00247, +0.00584] | [-0.00774, +0.01601] | 10/10 |
| 14 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00018 | [-0.00031, +0.00072] | [-0.01170, +0.01205] | 6/10 |
| 14 | teacher_ndcg_at_10 | Candidate frequency | +0.00492 | [+0.00306, +0.00686] | [-0.00696, +0.01679] | 10/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00327 | [+0.00135, +0.00516] | [-0.00861, +0.01514] | 10/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | +0.00127 | [-0.00091, +0.00344] | [-0.01060, +0.01315] | 7/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.01018 | [+0.00828, +0.01211] | [-0.00170, +0.02205] | 10/10 |
