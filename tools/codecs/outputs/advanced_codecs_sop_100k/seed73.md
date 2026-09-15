# Stanford Online Products advanced additive-codec comparison

All rows use the same exact-cosine teacher, query panel, codec-training sample, explicit IDs, and float16 inverse reconstruction norms.

| Codec | Nominal bits | Packed B/item | Full B/item | R@1 | R@10 | NDCG@10 | Recon cosine | Train s | Encode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSQ12x4 | 48 | 6.0000 | 14.9933 | 0.3250 | 0.3603 | 0.8924 | 0.7408 | 8.6 | 5.0 |
| RQ16x4 | 64 | 8.0000 | 17.9757 | 0.4125 | 0.4220 | 0.9131 | 0.7317 | 9.1 | 3.4 |
| LSQ16x4 | 64 | 8.0000 | 17.9763 | 0.4450 | 0.4338 | 0.9253 | 0.7670 | 13.4 | 11.5 |
