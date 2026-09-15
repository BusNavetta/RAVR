# Is the random-allocation range an artefact of the histogram bins?

GLDv2 100K, LSQ16x4, seed 17, cap 12 stages, 30,000 draws per configuration. RAVR is untouched throughout: only the sampler that produces the *random* baseline changes.

## Harness check

The shipped configuration re-run here against the published 1,699,360-draw sweep, subsampled to the same count:

| | mean | sd | p99 | max |
|---|---:|---:|---:|---:|
| published sampler | 0.20798 | 0.00820 | 0.22900 | 0.25075 |
| this module, same bins | 0.20796 | 0.00821 | 0.22850 | 0.24875 |

## Every configuration at cap 12

Sorted by how far the sampled maximum reaches. `holes` counts empty cells at the recall lattice inside the observed span; `histograms` counts distinct sampled histograms, so 1 means the grid left nothing to vary.

| configuration | bins | width (B) | support | min | median | p99.9 | max | sd | histograms | holes | best histogram | RAVR - max | RAVR z | flags |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 bins, 2 B, widened | 4 | 2 | 4-16 | 0.1840 | 0.2288 | 0.2705 | 0.2845 | 0.01583 | 29,327 | 46 | 0.2845 | +0.0280 | +5.4 | longer tail |
| 7 bins, 1 B, widened | 7 | 1 | 4-16 | 0.1800 | 0.2208 | 0.2628 | 0.2757 | 0.01272 | 30,000 | 50 | 0.2757 | +0.0368 | +7.3 | longer tail |
| 2 bins, 4 B | 2 | 4 | 8-16 | 0.2238 | 0.2440 | 0.2598 | 0.2635 | 0.00501 | 1 | 9 | 0.2635 | +0.0490 | +13.7 | degenerate, longer tail |
| shipped bins, two-point prior | 5 | 1 | 8-16 | 0.1815 | 0.2080 | 0.2473 | 0.2575 | 0.00912 | 26,420 | 25 | 0.2575 | +0.0550 | +11.4 | longer tail |
| shipped bins, spiked prior | 5 | 1 | 8-16 | 0.1805 | 0.2140 | 0.2403 | 0.2563 | 0.00712 | 17,391 | 53 | 0.2563 | +0.0562 | +14.3 | longer tail |
| 3 bins, 2 B | 3 | 2 | 8-16 | 0.1848 | 0.2152 | 0.2478 | 0.2545 | 0.01114 | 19,506 | 16 | 0.2545 | +0.0580 | +8.8 | longer tail |
| 6 bins, uneven (high) | 6 | mixed | 8-16 | 0.1863 | 0.2112 | 0.2350 | 0.2500 | 0.00686 | 29,982 | 46 | 0.2500 | +0.0625 | +14.8 | longer tail |
| shipped: 5 bins, 1 B | 5 | 1 | 8-16 | 0.1815 | 0.2080 | 0.2398 | 0.2488 | 0.00821 | 29,981 | 23 | 0.2488 | +0.0637 | +12.7 | -- |
| 6 bins, uneven (low) | 6 | mixed | 8-16 | 0.1823 | 0.2095 | 0.2393 | 0.2485 | 0.00952 | 29,976 | 17 | 0.2485 | +0.0640 | +10.9 | -- |
| shipped bins, flat prior | 5 | 1 | 8-16 | 0.1818 | 0.2067 | 0.2373 | 0.2482 | 0.00866 | 30,000 | 32 | 0.2482 | +0.0643 | +12.1 | -- |
| 9 bins, 0.5 B | 9 | 0.5 | 8-16 | 0.1818 | 0.2050 | 0.2328 | 0.2465 | 0.00735 | 30,000 | 41 | 0.2465 | +0.0660 | +14.6 | sub-byte |
| shipped bins, mixed prior | 5 | 1 | 8-16 | 0.1900 | 0.2105 | 0.2265 | 0.2300 | 0.00516 | 30,000 | 11 | 0.2475 | +0.0825 | +19.8 | short tail |
| 3 bins, 1 B, clipped | 3 | 1 | 10-14 | 0.1908 | 0.2083 | 0.2215 | 0.2265 | 0.00422 | 19,601 | 20 | 0.2278 | +0.0860 | +24.7 | clipped, short tail |
| 5 bins, 0.5 B, clipped | 5 | 0.5 | 10-14 | 0.1928 | 0.2083 | 0.2208 | 0.2235 | 0.00410 | 29,986 | 2 | 0.2235 | +0.0890 | +25.4 | clipped, sub-byte, short tail |

RAVR 0.3125, uniform 0.2145. Across every configuration the random maximum spans [0.2235, 0.2845] and the best histogram found spans [0.2235, 0.2845]; RAVR sits +0.0280 above the most generous of them.

### How much of RAVR's gain is the histogram at all

The oracle keeps the assignment random and optimises only the shape of the histogram, so the fraction below is the share of RAVR's lead over uniform that *any* per-item-blind allocation could reach. The panel's own `random_histogram` control -- RAVR's exact histogram with the assignment shuffled -- scores 0.2233, which is 8.9% of the same lead -- well under what the best histogram on the same grid reaches, so RAVR is not winning by picking a good shape. Read the two together: the shape has a ceiling, and what is left over is the assignment, which no grid can touch.

| configuration | best histogram - uniform | share of RAVR - uniform |
|---|---:|---:|
| 4 bins, 2 B, widened | +0.0700 | 71.4% |
| 7 bins, 1 B, widened | +0.0612 | 62.5% |
| 2 bins, 4 B | +0.0490 | 50.0% |
| shipped bins, two-point prior | +0.0430 | 43.9% |
| shipped bins, spiked prior | +0.0418 | 42.6% |
| 3 bins, 2 B | +0.0400 | 40.8% |
| 6 bins, uneven (high) | +0.0355 | 36.2% |
| shipped: 5 bins, 1 B | +0.0343 | 34.9% |
| 6 bins, uneven (low) | +0.0340 | 34.7% |
| shipped bins, flat prior | +0.0337 | 34.4% |
| 9 bins, 0.5 B | +0.0320 | 32.7% |
| shipped bins, mixed prior | +0.0330 | 33.7% |
| 3 bins, 1 B, clipped | +0.0132 | 13.5% |
| 5 bins, 0.5 B, clipped | +0.0090 | 9.2% |

### What a longer sweep could have reached

A generalised-Pareto fit to each configuration's top 1%, at every cap. `k > 0` means the fitted tail ends at a finite ceiling. Read the **probability** column first: the ceiling is the same fit divided by `k`, so it swings wildly whenever `k` lands near zero, while the probability at the target stays stable. Extrapolating a tail this far is an estimate, not a proof.

| cap | configuration | p99 | shape k | fitted ceiling | ceiling vs RAVR | P(draw >= RAVR) |
|---:|---|---:|---:|---:|---:|---:|
| 10 | 7 bins, 1 B, widened | 0.2042 | +0.027 | 0.4709 | +0.1812 | 6.1e-09 |
| 10 | 2 bins, 4 B | 0.1877 | +0.196 | 0.2003 | -0.0895 | 0 (beyond the ceiling) |
| 10 | 3 bins, 2 B | 0.1795 | +0.204 | 0.1895 | -0.1003 | 0 (beyond the ceiling) |
| 10 | 3 bins, 1 B, clipped | 0.1730 | +0.303 | 0.1833 | -0.1064 | 0 (beyond the ceiling) |
| 10 | shipped bins, spiked prior | 0.1713 | +0.264 | 0.1854 | -0.1044 | 0 (beyond the ceiling) |
| 10 | shipped: 5 bins, 1 B | 0.1725 | +0.204 | 0.1854 | -0.1044 | 0 (beyond the ceiling) |
| 10 | 9 bins, 0.5 B | 0.1723 | +0.161 | 0.1863 | -0.1035 | 0 (beyond the ceiling) |
| 12 | 4 bins, 2 B, widened | 0.2605 | +0.213 | 0.2891 | -0.0234 | 0 (beyond the ceiling) |
| 12 | 7 bins, 1 B, widened | 0.2508 | +0.119 | 0.3001 | -0.0124 | 0 (beyond the ceiling) |
| 12 | 2 bins, 4 B | 0.2558 | +0.231 | 0.2656 | -0.0469 | 0 (beyond the ceiling) |
| 12 | shipped bins, two-point prior | 0.2352 | +0.383 | 0.2566 | -0.0559 | 0 (beyond the ceiling) |
| 12 | shipped bins, spiked prior | 0.2268 | +0.083 | 0.3041 | -0.0084 | 0 (beyond the ceiling) |
| 12 | 3 bins, 2 B | 0.2397 | +0.177 | 0.2636 | -0.0489 | 0 (beyond the ceiling) |
| 12 | 6 bins, uneven (high) | 0.2278 | +0.106 | 0.2624 | -0.0501 | 0 (beyond the ceiling) |
| 12 | shipped: 5 bins, 1 B | 0.2285 | -0.062 | unbounded | -- | 2.7e-08 |
| 12 | 6 bins, uneven (low) | 0.2315 | +0.119 | 0.2644 | -0.0481 | 0 (beyond the ceiling) |
| 12 | shipped bins, flat prior | 0.2295 | +0.173 | 0.2555 | -0.0570 | 0 (beyond the ceiling) |
| 12 | 9 bins, 0.5 B | 0.2245 | +0.150 | 0.2552 | -0.0573 | 0 (beyond the ceiling) |
| 12 | shipped bins, mixed prior | 0.2227 | +0.475 | 0.2285 | -0.0840 | 0 (beyond the ceiling) |
| 12 | 3 bins, 1 B, clipped | 0.2175 | +0.200 | 0.2284 | -0.0841 | 0 (beyond the ceiling) |
| 12 | 5 bins, 0.5 B, clipped | 0.2175 | +0.221 | 0.2258 | -0.0867 | 0 (beyond the ceiling) |
| 14 | 2 bins, 4 B | 0.3010 | +0.356 | 0.3068 | -0.0142 | 0 (beyond the ceiling) |
| 14 | 7 bins, 1 B, widened | 0.2905 | +0.151 | 0.3171 | -0.0039 | 0 (beyond the ceiling) |
| 14 | 3 bins, 2 B | 0.2895 | +0.345 | 0.2995 | -0.0215 | 0 (beyond the ceiling) |
| 14 | shipped bins, spiked prior | 0.2812 | +0.324 | 0.2999 | -0.0211 | 0 (beyond the ceiling) |
| 14 | shipped: 5 bins, 1 B | 0.2833 | +0.069 | 0.3311 | +0.0101 | 1.5e-12 |
| 14 | 9 bins, 0.5 B | 0.2797 | +0.185 | 0.2987 | -0.0223 | 0 (beyond the ceiling) |
| 14 | 3 bins, 1 B, clipped | 0.2705 | +0.222 | 0.2816 | -0.0394 | 0 (beyond the ceiling) |

## The other caps

| cap | configuration | median | max | best histogram | RAVR | RAVR - max | RAVR z |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 7 bins, 1 B, widened | 0.1723 | 0.2365 | 0.2365 | 0.2898 | +0.0532 | +9.0 |
| 10 | 2 bins, 4 B | 0.1758 | 0.1968 | 0.1968 | 0.2898 | +0.0930 | +22.2 |
| 10 | 3 bins, 2 B | 0.1595 | 0.1878 | 0.1878 | 0.2898 | +0.1020 | +15.2 |
| 10 | 3 bins, 1 B, clipped | 0.1593 | 0.1863 | 0.1863 | 0.2898 | +0.1035 | +23.1 |
| 10 | shipped bins, spiked prior | 0.1593 | 0.1853 | 0.1853 | 0.2898 | +0.1045 | +22.9 |
| 10 | shipped: 5 bins, 1 B | 0.1575 | 0.1833 | 0.1833 | 0.2898 | +0.1065 | +19.7 |
| 10 | 9 bins, 0.5 B | 0.1585 | 0.1815 | 0.1815 | 0.2898 | +0.1083 | +20.8 |
| 12 | 4 bins, 2 B, widened | 0.2288 | 0.2845 | 0.2845 | 0.3125 | +0.0280 | +5.4 |
| 12 | 7 bins, 1 B, widened | 0.2208 | 0.2757 | 0.2757 | 0.3125 | +0.0368 | +7.3 |
| 12 | 2 bins, 4 B | 0.2440 | 0.2635 | 0.2635 | 0.3125 | +0.0490 | +13.7 |
| 12 | shipped bins, two-point prior | 0.2080 | 0.2575 | 0.2575 | 0.3125 | +0.0550 | +11.4 |
| 12 | shipped bins, spiked prior | 0.2140 | 0.2563 | 0.2563 | 0.3125 | +0.0562 | +14.3 |
| 12 | 3 bins, 2 B | 0.2152 | 0.2545 | 0.2545 | 0.3125 | +0.0580 | +8.8 |
| 12 | 6 bins, uneven (high) | 0.2112 | 0.2500 | 0.2500 | 0.3125 | +0.0625 | +14.8 |
| 12 | shipped: 5 bins, 1 B | 0.2080 | 0.2488 | 0.2488 | 0.3125 | +0.0637 | +12.7 |
| 12 | 6 bins, uneven (low) | 0.2095 | 0.2485 | 0.2485 | 0.3125 | +0.0640 | +10.9 |
| 12 | shipped bins, flat prior | 0.2067 | 0.2482 | 0.2482 | 0.3125 | +0.0643 | +12.1 |
| 12 | 9 bins, 0.5 B | 0.2050 | 0.2465 | 0.2465 | 0.3125 | +0.0660 | +14.6 |
| 12 | shipped bins, mixed prior | 0.2105 | 0.2300 | 0.2475 | 0.3125 | +0.0825 | +19.8 |
| 12 | 3 bins, 1 B, clipped | 0.2083 | 0.2265 | 0.2278 | 0.3125 | +0.0860 | +24.7 |
| 12 | 5 bins, 0.5 B, clipped | 0.2083 | 0.2235 | 0.2235 | 0.3125 | +0.0890 | +25.4 |
| 14 | 2 bins, 4 B | 0.2918 | 0.3083 | 0.3083 | 0.3210 | +0.0127 | +7.3 |
| 14 | 7 bins, 1 B, widened | 0.2715 | 0.3065 | 0.3065 | 0.3210 | +0.0145 | +5.8 |
| 14 | 3 bins, 2 B | 0.2732 | 0.3008 | 0.3008 | 0.3210 | +0.0202 | +7.3 |
| 14 | shipped bins, spiked prior | 0.2680 | 0.2970 | 0.2970 | 0.3210 | +0.0240 | +8.3 |
| 14 | shipped: 5 bins, 1 B | 0.2660 | 0.2955 | 0.2955 | 0.3210 | +0.0255 | +7.3 |
| 14 | 9 bins, 0.5 B | 0.2627 | 0.2938 | 0.2938 | 0.3210 | +0.0272 | +8.0 |
| 14 | 3 bins, 1 B, clipped | 0.2583 | 0.2795 | 0.2795 | 0.3210 | +0.0415 | +11.5 |
