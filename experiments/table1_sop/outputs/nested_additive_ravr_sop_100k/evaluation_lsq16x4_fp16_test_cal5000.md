# Frozen-LSQ16x4 allocator comparison

Every row uses the same frozen LSQ16x4 within each seed and rate, source item codes, exact packed-code cap, fp16 shared codebooks, uint32 IDs, float16 assigned-prefix norms, and bucket metadata.

| Target | Strategy | Full B/item | R@10 | NDCG@10 |
|---:|---|---:|---:|---:|
| 10 | Uniform prefix | 13.00896 | 0.2232 | 0.8099 |
| 10 | RAVR (Ordinal-KL) | 13.00896 | 0.3012 | 0.8585 |
| 10 | RAVR + unseen-item RD | 13.00896 | 0.3012 | 0.8585 |
| 10 | Ordinal-KL per exposure | 13.00896 | 0.2915 | 0.8545 |
| 10 | Candidate frequency | 13.00896 | 0.2531 | 0.8429 |
| 10 | QAQ-style query-IP MSE | 13.00896 | 0.2334 | 0.8311 |
| 10 | Random same histogram | 13.00896 | 0.2118 | 0.8274 |
| 10 | Reconstruction RD | 13.00896 | 0.2422 | 0.8330 |
| 12 | Uniform prefix | 14.00896 | 0.2962 | 0.8612 |
| 12 | RAVR (Ordinal-KL) | 14.00896 | 0.3771 | 0.8919 |
| 12 | RAVR + unseen-item RD | 14.00896 | 0.3771 | 0.8919 |
| 12 | Ordinal-KL per exposure | 14.00896 | 0.3708 | 0.8899 |
| 12 | Candidate frequency | 14.00896 | 0.3491 | 0.8832 |
| 12 | QAQ-style query-IP MSE | 14.00896 | 0.3156 | 0.8765 |
| 12 | Random same histogram | 14.00896 | 0.2931 | 0.8683 |
| 12 | Reconstruction RD | 14.00896 | 0.3185 | 0.8778 |
| 14 | Uniform prefix | 15.00896 | 0.3708 | 0.8988 |
| 14 | RAVR (Ordinal-KL) | 15.00896 | 0.4174 | 0.9125 |
| 14 | RAVR + unseen-item RD | 15.00896 | 0.4174 | 0.9125 |
| 14 | Ordinal-KL per exposure | 15.00896 | 0.4167 | 0.9120 |
| 14 | Candidate frequency | 15.00896 | 0.4077 | 0.9089 |
| 14 | QAQ-style query-IP MSE | 15.00896 | 0.3873 | 0.9063 |
| 14 | Random same histogram | 15.00896 | 0.3712 | 0.9007 |
| 14 | Reconstruction RD | 15.00896 | 0.3892 | 0.9077 |

| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |
|---:|---|---|---:|---:|---:|---:|
| 10 | teacher_recall_at_10 | Uniform prefix | +0.07805 | [+0.06605, +0.09000] | [+0.06115, +0.09495] | 10/10 |
| 10 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00975 | [+0.00523, +0.01440] | [-0.00715, +0.02665] | 10/10 |
| 10 | teacher_recall_at_10 | Candidate frequency | +0.04815 | [+0.03528, +0.06103] | [+0.03125, +0.06505] | 10/10 |
| 10 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.06780 | [+0.05515, +0.08052] | [+0.05090, +0.08470] | 10/10 |
| 10 | teacher_recall_at_10 | Reconstruction RD | +0.05897 | [+0.04640, +0.07153] | [+0.04207, +0.07587] | 10/10 |
| 10 | teacher_recall_at_10 | Random same histogram | +0.08940 | [+0.07568, +0.10310] | [+0.07250, +0.10630] | 10/10 |
| 10 | teacher_ndcg_at_10 | Uniform prefix | +0.04864 | [+0.04316, +0.05408] | [+0.03174, +0.06554] | 10/10 |
| 10 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00405 | [+0.00253, +0.00561] | [-0.01285, +0.02095] | 10/10 |
| 10 | teacher_ndcg_at_10 | Candidate frequency | +0.01561 | [+0.00998, +0.02127] | [-0.00129, +0.03251] | 10/10 |
| 10 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.02745 | [+0.02218, +0.03262] | [+0.01055, +0.04435] | 10/10 |
| 10 | teacher_ndcg_at_10 | Reconstruction RD | +0.02548 | [+0.02001, +0.03080] | [+0.00858, +0.04238] | 10/10 |
| 10 | teacher_ndcg_at_10 | Random same histogram | +0.03111 | [+0.02564, +0.03662] | [+0.01421, +0.04801] | 10/10 |
| 12 | teacher_recall_at_10 | Uniform prefix | +0.08095 | [+0.06958, +0.09238] | [+0.06405, +0.09785] | 10/10 |
| 12 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00635 | [+0.00222, +0.01052] | [-0.01055, +0.02325] | 10/10 |
| 12 | teacher_recall_at_10 | Candidate frequency | +0.02803 | [+0.01743, +0.03915] | [+0.01113, +0.04493] | 10/10 |
| 12 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.06150 | [+0.04963, +0.07325] | [+0.04460, +0.07840] | 10/10 |
| 12 | teacher_recall_at_10 | Reconstruction RD | +0.05867 | [+0.04658, +0.07060] | [+0.04177, +0.07557] | 10/10 |
| 12 | teacher_recall_at_10 | Random same histogram | +0.08408 | [+0.07172, +0.09660] | [+0.06718, +0.10097] | 10/10 |
| 12 | teacher_ndcg_at_10 | Uniform prefix | +0.03071 | [+0.02605, +0.03534] | [+0.01381, +0.04761] | 10/10 |
| 12 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00203 | [+0.00031, +0.00384] | [-0.01487, +0.01893] | 9/10 |
| 12 | teacher_ndcg_at_10 | Candidate frequency | +0.00871 | [+0.00442, +0.01318] | [-0.00819, +0.02561] | 10/10 |
| 12 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.01539 | [+0.01137, +0.01931] | [-0.00151, +0.03229] | 10/10 |
| 12 | teacher_ndcg_at_10 | Reconstruction RD | +0.01407 | [+0.00956, +0.01842] | [-0.00283, +0.03097] | 10/10 |
| 12 | teacher_ndcg_at_10 | Random same histogram | +0.02359 | [+0.01913, +0.02806] | [+0.00669, +0.04049] | 10/10 |
| 14 | teacher_recall_at_10 | Uniform prefix | +0.04657 | [+0.03802, +0.05555] | [+0.02967, +0.06347] | 10/10 |
| 14 | teacher_recall_at_10 | Ordinal-KL per exposure | +0.00063 | [-0.00180, +0.00310] | [-0.01627, +0.01752] | 5/10 |
| 14 | teacher_recall_at_10 | Candidate frequency | +0.00960 | [+0.00295, +0.01637] | [-0.00730, +0.02650] | 10/10 |
| 14 | teacher_recall_at_10 | QAQ-style query-IP MSE | +0.03000 | [+0.02232, +0.03765] | [+0.01310, +0.04690] | 10/10 |
| 14 | teacher_recall_at_10 | Reconstruction RD | +0.02817 | [+0.02045, +0.03582] | [+0.01127, +0.04507] | 10/10 |
| 14 | teacher_recall_at_10 | Random same histogram | +0.04615 | [+0.03822, +0.05393] | [+0.02925, +0.06305] | 10/10 |
| 14 | teacher_ndcg_at_10 | Uniform prefix | +0.01377 | [+0.01094, +0.01669] | [-0.00313, +0.03067] | 10/10 |
| 14 | teacher_ndcg_at_10 | Ordinal-KL per exposure | +0.00051 | [-0.00047, +0.00159] | [-0.01639, +0.01741] | 5/10 |
| 14 | teacher_ndcg_at_10 | Candidate frequency | +0.00360 | [+0.00116, +0.00615] | [-0.01330, +0.02050] | 10/10 |
| 14 | teacher_ndcg_at_10 | QAQ-style query-IP MSE | +0.00623 | [+0.00371, +0.00870] | [-0.01067, +0.02313] | 10/10 |
| 14 | teacher_ndcg_at_10 | Reconstruction RD | +0.00480 | [+0.00204, +0.00747] | [-0.01210, +0.02170] | 10/10 |
| 14 | teacher_ndcg_at_10 | Random same histogram | +0.01181 | [+0.00907, +0.01455] | [-0.00509, +0.02871] | 10/10 |
