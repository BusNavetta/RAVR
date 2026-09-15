# Frozen-LSQ allocator comparison

Every row uses the same trained LSQ16x4 codec, source item codes, exact packed-code cap, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 14.97824 | 0.2226 | 0.8085 |
| 10 | RAVR (Ordinal-KL) | 14.97824 | 0.2800 | 0.8532 |
| 10 | QAQ-style query-IP MSE | 14.97824 | 0.2365 | 0.8308 |
| 10 | Random same histogram | 14.97824 | 0.2100 | 0.8267 |
| 10 | Reconstruction RD | 14.97824 | 0.2442 | 0.8322 |
| 12 | Uniform prefix | 15.97824 | 0.2953 | 0.8576 |
| 12 | RAVR (Ordinal-KL) | 15.97824 | 0.3653 | 0.8883 |
| 12 | QAQ-style query-IP MSE | 15.97824 | 0.3162 | 0.8753 |
| 12 | Random same histogram | 15.97824 | 0.2922 | 0.8665 |
| 12 | Reconstruction RD | 15.97824 | 0.3257 | 0.8780 |
| 14 | Uniform prefix | 16.97824 | 0.3708 | 0.8959 |
| 14 | RAVR (Ordinal-KL) | 16.97824 | 0.4143 | 0.9102 |
| 14 | QAQ-style query-IP MSE | 16.97824 | 0.3903 | 0.9039 |
| 14 | Random same histogram | 16.97824 | 0.3711 | 0.8981 |
| 14 | Reconstruction RD | 16.97824 | 0.3926 | 0.9066 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.05737 | [+0.04430, +0.07037] | [+0.04130, +0.07345] | 10/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.04352 | [+0.03135, +0.05570] | [+0.02745, +0.05960] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.03577 | [+0.02342, +0.04800] | [+0.01970, +0.05185] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.07000 | [+0.05663, +0.08348] | [+0.05393, +0.08607] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.04470 | [+0.03896, +0.05036] | [+0.02862, +0.06077] | 10/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.02231 | [+0.01685, +0.02782] | [+0.00623, +0.03838] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.02096 | [+0.01530, +0.02654] | [+0.00488, +0.03703] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.02643 | [+0.02126, +0.03171] | [+0.01035, +0.04250] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.07002 | [+0.05898, +0.08108] | [+0.05395, +0.08610] | 10/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.04910 | [+0.03777, +0.06043] | [+0.03303, +0.06517] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.03955 | [+0.02860, +0.05060] | [+0.02348, +0.05562] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.07310 | [+0.06165, +0.08455] | [+0.05702, +0.08917] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.03071 | [+0.02637, +0.03494] | [+0.01463, +0.04678] | 10/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.01294 | [+0.00852, +0.01725] | [-0.00313, +0.02902] | 10/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.01032 | [+0.00574, +0.01476] | [-0.00575, +0.02640] | 10/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.02175 | [+0.01779, +0.02562] | [+0.00567, +0.03782] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | +0.04350 | [+0.03595, +0.05120] | [+0.02742, +0.05957] | 10/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.02395 | [+0.01618, +0.03165] | [+0.00788, +0.04002] | 10/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | +0.02168 | [+0.01375, +0.02960] | [+0.00560, +0.03775] | 10/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.04315 | [+0.03555, +0.05087] | [+0.02707, +0.05922] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | +0.01437 | [+0.01182, +0.01690] | [-0.00171, +0.03044] | 10/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00637 | [+0.00355, +0.00929] | [-0.00971, +0.02244] | 10/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | +0.00361 | [+0.00086, +0.00627] | [-0.01246, +0.01969] | 10/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.01214 | [+0.00957, +0.01470] | [-0.00393, +0.02822] | 10/10 |
