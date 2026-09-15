# Frozen-RQ16x4 allocator comparison

Every row uses the same frozen RQ16x4 within each seed and rate, source item codes, exact packed-code cap, fp32 shared codebooks, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 14.97760 | 0.2807 | 0.8464 |
| 10 | RAVR (Ordinal-KL) | 14.97760 | 0.3117 | 0.8603 |
| 10 | RAVR + unseen-item RD | 14.97760 | 0.3117 | 0.8603 |
| 10 | Ordinal-KL per exposure | 14.97760 | 0.3052 | 0.8565 |
| 10 | Candidate frequency | 14.97760 | 0.2819 | 0.8461 |
| 10 | QAQ-style query-IP MSE | 14.97760 | 0.2722 | 0.8484 |
| 10 | Random same histogram | 14.97760 | 0.2630 | 0.8383 |
| 10 | Reconstruction RD | 14.97760 | 0.2764 | 0.8516 |
| 12 | Uniform prefix | 15.97760 | 0.3302 | 0.8754 |
| 12 | RAVR (Ordinal-KL) | 15.97760 | 0.3574 | 0.8850 |
| 12 | RAVR + unseen-item RD | 15.97760 | 0.3574 | 0.8850 |
| 12 | Ordinal-KL per exposure | 15.97760 | 0.3539 | 0.8829 |
| 12 | Candidate frequency | 15.97760 | 0.3398 | 0.8752 |
| 12 | QAQ-style query-IP MSE | 15.97760 | 0.3196 | 0.8756 |
| 12 | Random same histogram | 15.97760 | 0.3141 | 0.8670 |
| 12 | Reconstruction RD | 15.97760 | 0.3246 | 0.8788 |
| 14 | Uniform prefix | 16.97760 | 0.3734 | 0.8964 |
| 14 | RAVR (Ordinal-KL) | 16.97760 | 0.3864 | 0.9005 |
| 14 | RAVR + unseen-item RD | 16.97760 | 0.3864 | 0.9005 |
| 14 | Ordinal-KL per exposure | 16.97760 | 0.3859 | 0.9003 |
| 14 | Candidate frequency | 16.97760 | 0.3814 | 0.8956 |
| 14 | QAQ-style query-IP MSE | 16.97760 | 0.3677 | 0.8973 |
| 14 | Random same histogram | 16.97760 | 0.3621 | 0.8904 |
| 14 | Reconstruction RD | 16.97760 | 0.3706 | 0.8993 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.03095 | [+0.02260, +0.03932] | [+0.01910, +0.04280] | 10/10 |
| 10 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00653 | [+0.00220, +0.01105] | [-0.00533, +0.01838] | 9/10 |
| 10 | teacher_recall_at_10 | Candidate frequency | +0.02977 | [+0.02180, +0.03782] | [+0.01792, +0.04163] | 10/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.03950 | [+0.03070, +0.04838] | [+0.02765, +0.05135] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.03530 | [+0.02582, +0.04480] | [+0.02345, +0.04715] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.04875 | [+0.04047, +0.05710] | [+0.03690, +0.06060] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.01386 | [+0.01036, +0.01739] | [+0.00200, +0.02571] | 10/10 |
| 10 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00378 | [+0.00207, +0.00567] | [-0.00807, +0.01563] | 10/10 |
| 10 | teacher_ndcg_at_10 | Candidate frequency | +0.01417 | [+0.01091, +0.01743] | [+0.00231, +0.02602] | 10/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.01183 | [+0.00811, +0.01549] | [-0.00002, +0.02368] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.00864 | [+0.00447, +0.01271] | [-0.00322, +0.02049] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.02202 | [+0.01850, +0.02559] | [+0.01016, +0.03387] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.02722 | [+0.01975, +0.03492] | [+0.01537, +0.03908] | 10/10 |
| 12 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00352 | [-0.00083, +0.00818] | [-0.00833, +0.01538] | 8/10 |
| 12 | teacher_recall_at_10 | Candidate frequency | +0.01760 | [+0.01130, +0.02392] | [+0.00575, +0.02945] | 10/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.03780 | [+0.02817, +0.04747] | [+0.02595, +0.04965] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.03285 | [+0.02357, +0.04225] | [+0.02100, +0.04470] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.04330 | [+0.03597, +0.05087] | [+0.03145, +0.05515] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.00952 | [+0.00672, +0.01231] | [-0.00233, +0.02137] | 10/10 |
| 12 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00209 | [+0.00052, +0.00382] | [-0.00977, +0.01394] | 10/10 |
| 12 | teacher_ndcg_at_10 | Candidate frequency | +0.00971 | [+0.00721, +0.01222] | [-0.00214, +0.02156] | 10/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00939 | [+0.00621, +0.01253] | [-0.00246, +0.02124] | 10/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.00614 | [+0.00262, +0.00962] | [-0.00572, +0.01799] | 10/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.01799 | [+0.01522, +0.02076] | [+0.00614, +0.02984] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | +0.01295 | [+0.00682, +0.01917] | [+0.00110, +0.02480] | 10/10 |
| 14 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00048 | [-0.00128, +0.00228] | [-0.01138, +0.01233] | 6/10 |
| 14 | teacher_recall_at_10 | Candidate frequency | +0.00498 | [-0.00035, +0.01035] | [-0.00688, +0.01683] | 9/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.01868 | [+0.01213, +0.02530] | [+0.00682, +0.03053] | 10/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | +0.01573 | [+0.00928, +0.02237] | [+0.00387, +0.02758] | 10/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.02425 | [+0.01847, +0.03000] | [+0.01240, +0.03610] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | +0.00410 | [+0.00244, +0.00580] | [-0.00775, +0.01595] | 10/10 |
| 14 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00018 | [-0.00031, +0.00073] | [-0.01167, +0.01204] | 6/10 |
| 14 | teacher_ndcg_at_10 | Candidate frequency | +0.00486 | [+0.00301, +0.00681] | [-0.00699, +0.01672] | 10/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00323 | [+0.00130, +0.00513] | [-0.00862, +0.01508] | 10/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | +0.00124 | [-0.00094, +0.00341] | [-0.01061, +0.01309] | 7/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.01012 | [+0.00824, +0.01204] | [-0.00173, +0.02197] | 10/10 |
