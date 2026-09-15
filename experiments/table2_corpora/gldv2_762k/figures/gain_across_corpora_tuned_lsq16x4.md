# Allocator gain across corpora, with a RAVR that spends the cap -- LSQ16x4

Seed 17, test split, 5,000 calibration queries on every corpus. Raw `ravr`, not the `ravr_unseen_rd` fallback. The mining depth per corpus is the smallest one at which raw `ravr` spends the whole cap at **every** target, so every gain below is at equal bytes.

| Corpus | Items | Depth | Coverage | Target | Uniform | RAVR | Gain | Gain at d=50 | Unspent at d=50 (B/item) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOP 100K | 100,000 | 50 | 0.9675 | 10 | 0.2090 | 0.2855 | +0.07650 | +0.07650 | 0.000 |
| | | | | 12 | 0.2865 | 0.3688 | +0.08225 | +0.08225 | 0.000 |
| | | | | 14 | 0.3393 | 0.4088 | +0.06950 | +0.06950 | 0.000 |
| GLDv2 100K | 100,000 | 50 | 0.8813 | 10 | 0.1595 | 0.2898 | +0.13025 | +0.13025 | 0.000 |
| | | | | 12 | 0.2145 | 0.3125 | +0.09800 | +0.09800 | 0.000 |
| | | | | 14 | 0.2690 | 0.3210 | +0.05200 | +0.05200 | 0.000 |
| GLDv2 full index | 761,757 | 800 | 0.9570 | 10 | 0.0857 | 0.1830 | +0.09725 | +0.09850 | 0.000 |
| | | | | 12 | 0.1298 | 0.2115 | +0.08175 | +0.06475 | 0.716 |
| | | | | 14 | 0.1787 | 0.2255 | +0.04675 | +0.01575 | 1.716 |
| Deep1M | 1,000,000 | 400 | 0.9329 | 10 | 0.0847 | 0.1670 | +0.08225 | +0.09600 | 0.000 |
| | | | | 12 | 0.1328 | 0.2398 | +0.10700 | +0.08200 | 0.571 |
| | | | | 14 | 0.2112 | 0.2863 | +0.07500 | +0.00350 | 1.571 |

The dotted lines in the figure are the same corpora at the default depth, which is what the shipped `gain_across_corpora` figure plots. Where a dotted line falls away at the 14-stage cap, RAVR was being scored on fewer bytes than uniform.

Widening the mining is not free -- it divides the Ordinal-KL weight on every negative and reorders the risk table. `UNSEEN_KNOBS.md` measures that cost, and shows that more calibration queries buy the same coverage without it, where more queries can be had.

