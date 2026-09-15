# Stanford Online Products advanced additive-codec comparison

All rows use the same exact-cosine teacher, query panel, codec-training sample, explicit IDs, and float16 inverse reconstruction norms.

| Codec | Nominal bits | Packed B/item | Full B/item | R@1 | R@10 | NDCG@10 | Recon cosine | Train s | Encode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LSQ12x4 | 48 | 6.0000 | 14.9933 | 0.3125 | 0.3578 | 0.8950 | 0.7423 | 8.6 | 5.0 |
| RQ16x4 | 64 | 8.0000 | 17.9757 | 0.4025 | 0.4190 | 0.9142 | 0.7320 | 7.2 | 3.2 |
| LSQ16x4 | 64 | 8.0000 | 17.9763 | 0.4475 | 0.4547 | 0.9286 | 0.7678 | 12.6 | 7.8 |
