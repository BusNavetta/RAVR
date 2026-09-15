# Allocator gain against gallery size -- LSQ16x4

Absolute B/item are **not** comparable across these rows: the shared codebook is a fixed 393,216 bytes, so it costs 3.93 B/item at 100K and 0.52 B/item at 762K. The comparison is the gain at a matched uniform-equivalent prefix, where both arms pay the same overhead.

| Corpus | Gallery | Target | Uniform R@10 | RAVR R@10 | Gain | Family 95% | Positive seeds | B/item |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SOP 100K |  | 10 | 0.2231 | 0.3012 | +0.07803 | [+0.06110, +0.09495] | 10/10 | 14.97824 |
| SOP 100K |  | 12 | 0.2961 | 0.3771 | +0.08100 | [+0.06407, +0.09793] | 10/10 | 15.97824 |
| SOP 100K |  | 14 | 0.3707 | 0.4174 | +0.04665 | [+0.02972, +0.06357] | 10/10 | 16.97824 |
| GLDv2 100K |  | 10 | 0.1633 | 0.2849 | +0.12160 | [+0.10720, +0.13600] | 10/10 | 14.97824 |
| GLDv2 100K |  | 12 | 0.2180 | 0.3125 | +0.09450 | [+0.08010, +0.10890] | 10/10 | 15.97824 |
| GLDv2 100K |  | 14 | 0.2761 | 0.3236 | +0.04750 | [+0.03310, +0.06190] | 10/10 | 16.95917 |
| GLDv2 full index |  | 10 | 0.0877 | 0.1878 | +0.10015 | [+0.08690, +0.11340] | 10/10 | 11.52227 |
| GLDv2 full index |  | 12 | 0.1306 | 0.1957 | +0.06505 | [+0.05180, +0.07830] | 10/10 | 11.82230 |
| GLDv2 full index |  | 14 | 0.1790 | 0.1957 | +0.01665 | [+0.00340, +0.02990] | 10/10 | 11.82230 |

GLDv2 100K against the full index, same query workload, same encoder, same codec family:

| Target | 100K gain | Full-index gain | Change |
|---:|---:|---:|---:|
| 10 | +0.12160 | +0.10015 | -0.02145 |
| 12 | +0.09450 | +0.06505 | -0.02945 |
| 14 | +0.04750 | +0.01665 | -0.03085 |

The gain **shrinks** with gallery size.

