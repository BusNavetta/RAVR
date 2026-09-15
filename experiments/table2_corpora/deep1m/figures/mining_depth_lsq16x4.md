# RAVR against mining depth -- deep1m LSQ16x4

Every depth is averaged over the same seeds (17), so the curves are like for like.

A curve that stops before the rightmost gridline did not spend its cap: the allocator ran out of items it has any signal for. `RAVR + unseen RD` fills that slack by reconstruction distortion, so the last column asks the question that matters: does a wider mining beat the fallback at the default depth?

| Mining depth | Seeds | Target | Uniform | RAVR | RAVR B/item | Spent | RAVR + unseen RD | Widening minus fallback |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 50 | 1 | 10 | 0.0847 | 0.1808 | 11.10291 | yes | 0.1808 | +0.0000 |
| 50 | 1 | 12 | 0.1328 | 0.2148 | 11.53214 | NO | 0.2330 | -0.0182 |
| 50 | 1 | 14 | 0.2112 | 0.2148 | 11.53214 | NO | 0.2737 | -0.0590 |
| 200 | 1 | 10 | 0.0847 | 0.1700 | 11.10291 | yes | 0.1700 | -0.0108 |
| 200 | 1 | 12 | 0.1328 | 0.2445 | 12.10291 | yes | 0.2445 | +0.0115 |
| 200 | 1 | 14 | 0.2112 | 0.2787 | 12.92140 | NO | 0.2828 | +0.0050 |
| 400 | 1 | 10 | 0.0847 | 0.1670 | 11.10291 | yes | 0.1670 | -0.0137 |
| 400 | 1 | 12 | 0.1328 | 0.2398 | 12.10291 | yes | 0.2398 | +0.0068 |
| 400 | 1 | 14 | 0.2112 | 0.2863 | 13.10291 | yes | 0.2863 | +0.0125 |
