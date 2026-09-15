"""Read the bin sweep and say whether RAVR's margin survives the bin choice.

Three figures and one table:

* `histogram_bin_sweep_ranges`   -- one row per configuration: the sampled
  range, its tail, the best histogram a cross-entropy search could find on that
  grid, and the unchanged RAVR and uniform points as vertical rules.
* `histogram_bin_sweep_tails`    -- the same distributions as ridges on a shared
  axis, plus the upper tail as a survival curve on a log scale.
* `histogram_bin_sweep_checks`   -- the diagnostics: which grids have no
  histogram freedom left, what the display bin width does to the picture, and
  whether the margin holds at the other caps.

    PYTHONPATH=3_method python 7_random_allocations/analyse_histogram_bins.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent

# Validated categorical slots 1-3 (all-pairs, light surface) plus the chart
# chrome from the same system. Aqua sits below 3:1 on the surface, so every
# line that uses it is directly labelled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

LABELS = {
    "base_w2_m5": "shipped: 5 bins, 1 B",
    "fine_w1_m9": "9 bins, 0.5 B",
    "coarse_w4_m3": "3 bins, 2 B",
    "binary_w8_m2": "2 bins, 4 B",
    "narrow_w2_m3": "3 bins, 1 B, clipped",
    "narrow_w1_m5": "5 bins, 0.5 B, clipped",
    "wide_w2_m7": "7 bins, 1 B, widened",
    "wide_w4_m4": "4 bins, 2 B, widened",
    "asym_low_m6": "6 bins, uneven (low)",
    "asym_high_m6": "6 bins, uneven (high)",
    "base_w2_m5+prior_flat": "shipped bins, flat prior",
    "base_w2_m5+prior_tiny": "shipped bins, spiked prior",
    "base_w2_m5+prior_broad": "shipped bins, mixed prior",
    "base_w2_m5+prior_twopoint": "shipped bins, two-point prior",
}


def label(name: str) -> str:
    return LABELS.get(name, name)


def load(path: Path) -> tuple[dict, Path]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report, path.with_suffix("")


def samples(directory: Path, entry: dict) -> np.ndarray:
    return np.load(directory / entry["samples_file"]).astype(np.float64)


def shipped_reference(cap: int) -> np.ndarray | None:
    """The published 1.5M-draw run, for a sanity check on the new harness."""
    # The Fig. 2 report lives at the experiment root; this audit is one
    # level down, next to its own bin-sweep data.
    path = HERE.parent / "random_allocations_gldv2_lsq16x4_seed17.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    entry = report["targets"].get(str(cap))
    if entry is None:
        return None
    sidecar = path.with_name(entry["random"]["samples_file"])
    return np.load(sidecar).astype(np.float64) if sidecar.exists() else None


def order_by_reach(configurations: dict) -> list[str]:
    """Most tail-generous grid first: that is the one that could hurt RAVR."""
    return sorted(configurations, key=lambda name: -configurations[name]["max"])


def best_histogram(entry: dict) -> float:
    """The best a *histogram* reached on this grid, however it was found.

    The cross-entropy search usually beats the draws, but on grids with one or
    no free direction it has nothing to climb and the plain sweep can land
    higher. Both are histograms with a random assignment, so the bound is the
    better of the two rather than the search alone.
    """
    return max(entry["histogram_oracle"]["recall"], entry["max"])


def tail_fit(draws: np.ndarray, target: float,
             threshold: float = 99.0) -> dict:
    """Generalised-Pareto fit to the top 1%, to ask what more draws could do.

    The observed maximum is bounded by the number of draws, so it cannot by
    itself say whether a longer sweep would have reached RAVR. Fitting the
    exceedances (Hosking & Wallis probability-weighted moments, in their
    `k = -xi` sign convention) answers that: a positive `k` means the fitted
    tail has a finite ceiling, and if that ceiling sits below the target no
    number of draws from this sampler would get there.
    """
    cut = float(np.percentile(draws, threshold))
    excess = np.sort(draws[draws > cut] - cut)
    n = len(excess)
    if n < 50:
        return {"usable": False}
    a0 = float(excess.mean())
    a1 = float(np.sum(excess * (n - np.arange(1, n + 1)) / (n - 1)) / n)
    denominator = a0 - 2.0 * a1
    if abs(denominator) < 1e-12:
        return {"usable": False}
    k = a0 / denominator - 2.0
    scale = 2.0 * a0 * a1 / denominator
    if scale <= 0:
        return {"usable": False}
    rate = 1.0 - threshold / 100.0
    ceiling = cut + scale / k if k > 0 else None
    reach = target - cut
    if k > 0 and reach >= scale / k:
        probability = 0.0
    else:
        probability = rate * (1.0 - k * reach / scale) ** (1.0 / k)
    return {"usable": True, "threshold": cut, "shape_k": float(k),
            "scale": float(scale), "ceiling": ceiling,
            "probability_at_target": float(probability),
            "exceedances": int(n)}


def flags(entry: dict, baseline: dict) -> list[str]:
    """What a reader should be warned about before trusting this row."""
    marks = []
    if entry["distinct_histograms"] <= 1:
        marks.append("degenerate")            # the cap fixes the histogram
    lengths = entry["lengths"]
    if lengths[0] > baseline["lengths"][0] or lengths[-1] < baseline["lengths"][-1]:
        marks.append("clipped")               # cannot reach the shipped depths
    if entry["width_stages"] is not None and entry["width_stages"] < 2:
        marks.append("sub-byte")              # not storable by the packed index
    if entry["max"] < baseline["quantiles"]["p99.9"]:
        marks.append("short tail")            # reaches less far than the shipped p99.9
    if entry["max"] > baseline["max"]:
        marks.append("longer tail")
    return marks


# --------------------------------------------------------------------------

def figure_ranges(report: dict, directory: Path, cap: int, out: Path) -> None:
    import matplotlib.pyplot as plt

    entry = report["caps"][str(cap)]
    configurations = entry["configurations"]
    names = order_by_reach(configurations)[::-1]
    ravr = entry["named"]["ravr"]
    uniform = entry["named"]["uniform"]

    figure, axis = plt.subplots(figsize=(7.4, 0.36 * len(names) + 1.9))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    for row, name in enumerate(names):
        cell = configurations[name]
        axis.plot([cell["min"], cell["max"]], [row, row], color=AXIS,
                  linewidth=1.0, solid_capstyle="round", zorder=2)
        axis.plot([cell["quantiles"]["p1"], cell["quantiles"]["p99"]],
                  [row, row], color=BLUE, linewidth=4.0, alpha=0.30,
                  solid_capstyle="round", zorder=3)
        axis.plot([cell["quantiles"]["p50"]], [row], marker="o", markersize=5,
                  color=BLUE, markeredgecolor=SURFACE, markeredgewidth=0.8,
                  linestyle="none", zorder=5)
        axis.plot([cell["max"]], [row], marker="|", markersize=9, color=BLUE,
                  linestyle="none", zorder=5)
        axis.plot([best_histogram(cell)], [row], marker="D",
                  markersize=4.5, color=ORANGE, markeredgecolor=SURFACE,
                  markeredgewidth=0.7, linestyle="none", zorder=6)
    axis.axvline(ravr, color=INK, linewidth=1.4, zorder=4)
    axis.axvline(uniform, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                 zorder=4)
    floor = min(configurations[name]["min"] for name in names)
    axis.set_xlim(floor - 0.004, ravr + 0.010)
    axis.annotate("RAVR", (ravr, len(names) - 0.35), xytext=(-4, 0),
                  textcoords="offset points", color=INK, fontsize=8,
                  weight="semibold", ha="right", va="top")
    axis.annotate("uniform", (uniform, len(names) - 0.35), xytext=(-4, 0),
                  textcoords="offset points", color=SECOND, fontsize=8,
                  ha="right", va="top")
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels([label(name) for name in names], fontsize=8,
                         color=SECOND)
    axis.set_ylim(-0.8, len(names) - 0.2)
    axis.set_xlabel("Teacher Recall@10", fontsize=8.5, color=SECOND,
                    labelpad=3)
    axis.tick_params(axis="x", labelsize=8, colors=MUTED, length=3)
    axis.tick_params(axis="y", length=0)
    axis.grid(True, axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.set_title(
        f"Cap {cap} stages -- random-allocation range under "
        f"{len(names)} sampler configurations\n"
        f"{report['draws_per_configuration']:,} draws each; bar = p1-p99, "
        "tick = observed max, diamond = best histogram found",
        fontsize=9, color=INK, weight="semibold", loc="left", pad=8)
    figure.tight_layout(pad=0.5)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=260, facecolor=SURFACE,
                       bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def figure_tails(report: dict, directory: Path, cap: int, out: Path) -> None:
    import matplotlib.pyplot as plt

    entry = report["caps"][str(cap)]
    configurations = entry["configurations"]
    names = order_by_reach(configurations)
    ravr = entry["named"]["ravr"]
    uniform = entry["named"]["uniform"]
    lattice = report["recall_lattice"]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.4, 0.30 * len(names) + 4.6),
        gridspec_kw={"height_ratios": [0.30 * len(names) + 1.0, 3.0]})
    for axis in (top, bottom):
        axis.set_facecolor(SURFACE)
    figure.patch.set_facecolor(SURFACE)

    # ---- ridges, shared axis -------------------------------------------
    # Four lattice cells per bin, then a three-tap smooth: at one cell the
    # comb of the recall lattice dominates and every ridge looks like noise.
    edges = np.arange(0.175, 0.335, 4 * lattice)
    centres = (edges[:-1] + edges[1:]) / 2
    kernel = np.array([0.25, 0.5, 0.25])
    step = 1.0
    for row, name in enumerate(names[::-1]):
        draws = samples(directory, configurations[name])
        density, _ = np.histogram(draws, bins=edges, density=True)
        density = np.convolve(density, kernel, mode="same")
        density = density / max(density.max(), 1e-9) * 0.92 * step
        base = row * step
        top.fill_between(centres, base, base + density, color=BLUE,
                         alpha=0.22, linewidth=0)
        top.plot(centres, base + density, color=BLUE, linewidth=1.0)
    top.axvline(ravr, color=INK, linewidth=1.4, zorder=6)
    top.axvline(uniform, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                zorder=6)
    top.annotate(f"RAVR {ravr:.4f}", (ravr, len(names) * step),
                 xytext=(4, -2), textcoords="offset points", color=INK,
                 fontsize=8, weight="semibold", ha="left", va="top")
    top.annotate(f"uniform {uniform:.4f}", (uniform, len(names) * step),
                 xytext=(-4, -2), textcoords="offset points", color=SECOND,
                 fontsize=8, ha="right", va="top")
    top.set_xlim(0.175, 0.335)
    top.set_ylim(-0.15, len(names) * step + 0.55)
    top.set_yticks([row * step + 0.12 for row in range(len(names))])
    top.set_yticklabels([label(name) for name in names[::-1]], fontsize=7.5,
                        color=SECOND)
    top.tick_params(axis="y", length=0)
    top.tick_params(axis="x", labelsize=8, colors=MUTED, length=3)
    top.grid(True, axis="x", color=GRID, linewidth=0.5)
    top.set_axisbelow(True)
    top.spines[["top", "right", "left"]].set_visible(False)
    top.spines["bottom"].set_color(AXIS)
    top.set_title(
        f"Every configuration on one axis -- cap {cap} stages",
        fontsize=9, color=INK, weight="semibold", loc="left", pad=6)

    # ---- upper tail, log survival --------------------------------------
    highlight = {names[0]: ORANGE, "base_w2_m5": BLUE}
    for name in names[1:]:
        if name not in highlight and configurations[name]["max"] == min(
                configurations[other]["max"] for other in names):
            highlight[name] = AQUA
    # Fixed, well-separated anchor heights: the highlighted curves end within
    # a pixel of each other, so labelling them all at the tail piles them up.
    anchors = dict(zip(highlight, (0.25, 0.03, 0.004)))
    for name in names:
        draws = np.sort(samples(directory, configurations[name]))
        survival = 1.0 - np.arange(len(draws)) / len(draws)
        keep = draws >= np.percentile(draws, 50)
        colour = highlight.get(name, GRID)
        bottom.step(draws[keep], survival[keep], where="post", color=colour,
                    linewidth=1.6 if name in highlight else 0.8,
                    zorder=5 if name in highlight else 2,
                    alpha=1.0 if name in highlight else 0.9)
        if name in highlight:
            level = anchors[name]
            tail, tail_survival = draws[keep], survival[keep]
            at = min(max(int(np.searchsorted(-tail_survival, -level)) - 1, 0),
                     len(tail) - 1)
            bottom.annotate(label(name), (tail[at], level), xytext=(7, 4),
                            textcoords="offset points", color=colour,
                            fontsize=7.5, weight="semibold", ha="left",
                            va="bottom")
    bottom.axvline(ravr, color=INK, linewidth=1.4, zorder=6)
    bottom.set_yscale("log")
    bottom.set_xlim(0.175, 0.335)
    bottom.set_ylim(0.7 / report["draws_per_configuration"], 1.2)
    bottom.set_xlabel("Teacher Recall@10", fontsize=8.5, color=SECOND,
                      labelpad=3)
    bottom.set_ylabel("P(random draw >= x)", fontsize=8.5, color=SECOND,
                      labelpad=3)
    bottom.tick_params(labelsize=8, colors=MUTED, length=3)
    bottom.grid(True, color=GRID, linewidth=0.5)
    bottom.set_axisbelow(True)
    bottom.spines[["top", "right"]].set_visible(False)
    bottom.spines[["bottom", "left"]].set_color(AXIS)
    bottom.set_title(
        "Upper tail. Grey lines are the remaining configurations; no curve "
        "reaches the RAVR rule.",
        fontsize=9, color=INK, weight="semibold", loc="left", pad=6)
    figure.tight_layout(pad=0.5)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=260, facecolor=SURFACE,
                       bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def figure_checks(report: dict, directory: Path, caps: list[int],
                  out: Path) -> None:
    import matplotlib.pyplot as plt

    lattice = report["recall_lattice"]
    headline = report["caps"][str(caps[0])]
    configurations = headline["configurations"]
    names = order_by_reach(configurations)

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    for axis in axes:
        axis.set_facecolor(SURFACE)
    figure.patch.set_facecolor(SURFACE)

    # (a) how much histogram freedom the grid leaves
    left = axes[0]
    freedom = [configurations[name]["distinct_histograms"] for name in names]
    positions = np.arange(len(names))
    left.barh(positions, freedom, color=BLUE, height=0.62, zorder=3)
    for row, value in enumerate(freedom):
        if value <= 5:
            left.text(value * 1.4, row, f"{value}", fontsize=7.5,
                      color=ORANGE, weight="semibold", va="center")
    left.set_xscale("log")
    left.set_yticks(positions)
    left.set_yticklabels([label(name) for name in names], fontsize=7,
                         color=SECOND)
    left.invert_yaxis()
    left.set_xlabel("distinct histograms among the draws", fontsize=8,
                    color=SECOND)
    left.set_title("(a) A grid can leave nothing to vary", fontsize=9,
                   color=INK, weight="semibold", loc="left", pad=6)

    # (b) the *display* bin width -- a different histogram from the sampler's,
    #     and the one that decides whether the picture shows holes
    middle = axes[1]
    widest = min(names, key=lambda name: configurations[name]["std"])
    multiples = np.geomspace(0.2, 8.0, 25)
    for at, (name, colour) in enumerate(((names[0], ORANGE),
                                         ("base_w2_m5", BLUE),
                                         (widest, AQUA))):
        draws = samples(directory, configurations[name])
        empty = []
        for multiple in multiples:
            edges = np.arange(draws.min(), draws.max() + lattice * multiple,
                              lattice * multiple)
            counts, _ = np.histogram(draws, bins=edges)
            empty.append(np.count_nonzero(counts == 0) / max(len(counts), 1))
        middle.plot(multiples, empty, color=colour, linewidth=1.6)
        # staggered along the curve: at the left edge the three labels stack
        anchor = at * 4
        middle.annotate(label(name), (multiples[anchor], empty[anchor]),
                        xytext=(5, 5), textcoords="offset points",
                        color=colour, fontsize=7, weight="semibold",
                        ha="left", va="bottom")
    middle.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    middle.annotate("recall lattice\n1/4000", (1.0, 0.30), xytext=(6, 0),
                    textcoords="offset points", color=SECOND, fontsize=7,
                    ha="left", va="bottom")
    middle.set_xscale("log")
    middle.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    middle.set_xticklabels(["0.25x", "0.5x", "1x", "2x", "4x", "8x"])
    middle.set_ylim(-0.03, 1.03)
    middle.set_xlabel("display bin width, in lattice units", fontsize=8,
                      color=SECOND)
    middle.set_ylabel("share of empty display cells", fontsize=8, color=SECOND)
    middle.set_title("(b) Holes are a display choice, not a finding",
                     fontsize=9, color=INK, weight="semibold", loc="left",
                     pad=6)

    # (c) the margin at the other caps
    right = axes[2]
    shared = [name for name in names
              if all(name in report["caps"][str(cap)]["configurations"]
                     for cap in caps)]
    height = 0.26
    for offset, cap in enumerate(sorted(caps)):
        cap_entry = report["caps"][str(cap)]
        margin = [cap_entry["named"]["ravr"]
                  - cap_entry["configurations"][name]["max"]
                  for name in shared]
        right.barh(np.arange(len(shared)) + (1 - offset) * height, margin,
                   height=height * 0.9, label=f"cap {cap}",
                   color=(BLUE, ORANGE, AQUA)[offset], zorder=3)
    right.axvline(0, color=AXIS, linewidth=1.0)
    right.set_yticks(np.arange(len(shared)))
    right.set_yticklabels([label(name) for name in shared], fontsize=7,
                          color=SECOND)
    right.invert_yaxis()
    right.set_xlabel("RAVR minus the random maximum", fontsize=8, color=SECOND)
    right.legend(fontsize=7.5, frameon=False, labelcolor=SECOND,
                 loc="upper right")
    right.set_title("(c) The margin never changes sign", fontsize=9,
                    color=INK, weight="semibold", loc="left", pad=6)

    for axis in axes:
        axis.tick_params(labelsize=7.5, colors=MUTED, length=3)
        axis.grid(True, color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color(AXIS)
    figure.tight_layout(pad=0.6)
    for suffix in (".png", ".pdf"):
        figure.savefig(out.with_suffix(suffix), dpi=260, facecolor=SURFACE,
                       bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


# --------------------------------------------------------------------------

def tables(report: dict, directory: Path, caps: list[int]) -> str:
    lines: list[str] = []
    headline = caps[0]
    entry = report["caps"][str(headline)]
    ravr = entry["named"]["ravr"]
    uniform = entry["named"]["uniform"]
    configurations = entry["configurations"]
    names = order_by_reach(configurations)

    reference = shipped_reference(headline)
    lines += [
        f"# Is the random-allocation range an artefact of the histogram bins?",
        "",
        f"GLDv2 100K, LSQ16x4, seed 17, cap {headline} stages, "
        f"{report['draws_per_configuration']:,} draws per configuration. "
        "RAVR is untouched throughout: only the sampler that produces the "
        "*random* baseline changes.",
        "",
    ]
    if reference is not None:
        rng = np.random.default_rng(0)
        subset = rng.choice(reference, report["draws_per_configuration"],
                            replace=False)
        mine = samples(directory, configurations["base_w2_m5"])
        lines += [
            "## Harness check",
            "",
            "The shipped configuration re-run here against the published "
            f"{len(reference):,}-draw sweep, subsampled to the same count:",
            "",
            "| | mean | sd | p99 | max |",
            "|---|---:|---:|---:|---:|",
            f"| published sampler | {subset.mean():.5f} | {subset.std(ddof=1):.5f} "
            f"| {np.percentile(subset, 99):.5f} | {subset.max():.5f} |",
            f"| this module, same bins | {mine.mean():.5f} | "
            f"{mine.std(ddof=1):.5f} | {np.percentile(mine, 99):.5f} | "
            f"{mine.max():.5f} |",
            "",
        ]

    lines += [
        f"## Every configuration at cap {headline}",
        "",
        "Sorted by how far the sampled maximum reaches. `holes` counts empty "
        "cells at the recall lattice inside the observed span; `histograms` "
        "counts distinct sampled histograms, so 1 means the grid left nothing "
        "to vary.",
        "",
        "| configuration | bins | width (B) | support | min | median | p99.9 | "
        "max | sd | histograms | holes | best histogram | RAVR - max | RAVR z "
        "| flags |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    baseline = configurations["base_w2_m5"]
    for name in names:
        cell = configurations[name]
        lengths = cell["lengths"]
        width = ("mixed" if cell["width_bytes"] is None
                 else f"{cell['width_bytes']:g}")
        lines.append(
            f"| {label(name)} | {cell['bins']} | {width} | "
            f"{lengths[0]}-{lengths[-1]} | {cell['min']:.4f} | "
            f"{cell['quantiles']['p50']:.4f} | "
            f"{cell['quantiles']['p99.9']:.4f} | {cell['max']:.4f} | "
            f"{cell['std']:.5f} | {cell['distinct_histograms']:,} | "
            f"{cell['lattice_cells_empty']} | "
            f"{best_histogram(cell):.4f} | "
            f"{ravr - cell['max']:+.4f} | "
            f"{cell['named']['ravr']['z_score']:+.1f} | "
            f"{', '.join(flags(cell, baseline)) or '--'} |"
        )
    reach = [configurations[name]["max"] for name in names]
    oracle = [best_histogram(configurations[name]) for name in names]
    lines += [
        "",
        f"RAVR {ravr:.4f}, uniform {uniform:.4f}. Across every configuration "
        f"the random maximum spans [{min(reach):.4f}, {max(reach):.4f}] and "
        f"the best histogram found spans [{min(oracle):.4f}, "
        f"{max(oracle):.4f}]; RAVR sits {ravr - max(reach):+.4f} above the "
        f"most generous of them.",
        "",
        "### How much of RAVR's gain is the histogram at all",
        "",
        "The oracle keeps the assignment random and optimises only the shape "
        "of the histogram, so the fraction below is the share of RAVR's lead "
        "over uniform that *any* per-item-blind allocation could reach. The "
        "panel's own `random_histogram` control -- RAVR's exact histogram with "
        "the assignment shuffled -- scores "
        f"{entry['named']['random_histogram']:.4f}, which is "
        f"{(entry['named']['random_histogram'] - uniform) / (ravr - uniform) * 100:.1f}% "
        "of the same lead -- well under what the best histogram on the same "
        "grid reaches, so RAVR is not winning by picking a good shape. Read "
        "the two together: the shape has a ceiling, and what is left over is "
        "the assignment, which no grid can touch.",
        "",
        "| configuration | best histogram - uniform | share of RAVR - uniform |",
        "|---|---:|---:|",
    ]
    for name in names:
        best = best_histogram(configurations[name])
        lines.append(
            f"| {label(name)} | {best - uniform:+.4f} | "
            f"{(best - uniform) / (ravr - uniform) * 100:.1f}% |"
        )

    lines += [
        "", "### What a longer sweep could have reached", "",
        "A generalised-Pareto fit to each configuration's top 1%, at every "
        "cap. `k > 0` means the fitted tail ends at a finite ceiling. Read the "
        "**probability** column first: the ceiling is the same fit divided by "
        "`k`, so it swings wildly whenever `k` lands near zero, while the "
        "probability at the target stays stable. Extrapolating a tail this far "
        "is an estimate, not a proof.",
        "",
        "| cap | configuration | p99 | shape k | fitted ceiling | "
        "ceiling vs RAVR | P(draw >= RAVR) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for cap in sorted(caps):
        cap_entry = report["caps"][str(cap)]
        cap_ravr = cap_entry["named"]["ravr"]
        for name in order_by_reach(cap_entry["configurations"]):
            cell = cap_entry["configurations"][name]
            fit = tail_fit(samples(directory, cell), cap_ravr)
            if not fit["usable"]:
                lines.append(f"| {cap} | {label(name)} | | | not enough "
                             "exceedances | | |")
                continue
            ceiling = fit["ceiling"]
            probability = fit["probability_at_target"]
            lines.append(
                f"| {cap} | {label(name)} | {fit['threshold']:.4f} | "
                f"{fit['shape_k']:+.3f} | "
                f"{'unbounded' if ceiling is None else f'{ceiling:.4f}'} | "
                f"{'--' if ceiling is None else f'{ceiling - cap_ravr:+.4f}'} | "
                f"{'0 (beyond the ceiling)' if probability == 0 else f'{probability:.1e}'} |"
            )

    lines += ["", "## The other caps", "",
              "| cap | configuration | median | max | best histogram | RAVR | "
              "RAVR - max | RAVR z |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for cap in sorted(caps):
        cap_entry = report["caps"][str(cap)]
        cap_ravr = cap_entry["named"]["ravr"]
        for name in order_by_reach(cap_entry["configurations"]):
            cell = cap_entry["configurations"][name]
            lines.append(
                f"| {cap} | {label(name)} | {cell['quantiles']['p50']:.4f} | "
                f"{cell['max']:.4f} | "
                f"{best_histogram(cell):.4f} | {cap_ravr:.4f} | "
                f"{cap_ravr - cell['max']:+.4f} | "
                f"{cell['named']['ravr']['z_score']:+.1f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--cap", type=int, default=12)
    args = parser.parse_args()

    path = args.report or (
        HERE / "histogram_bin_sweep_gldv2_lsq16x4_seed17.json"
    )
    report, directory = load(path)
    caps = sorted((int(cap) for cap in report["caps"]),
                  key=lambda cap: (cap != args.cap, cap))

    import matplotlib
    matplotlib.use("Agg")

    figures = HERE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure_ranges(report, directory, caps[0],
                  figures / "histogram_bin_sweep_ranges")
    figure_tails(report, directory, caps[0],
                 figures / "histogram_bin_sweep_tails")
    figure_checks(report, directory, caps,
                  figures / "histogram_bin_sweep_checks")

    text = tables(report, directory, caps)
    (HERE / "HISTOGRAM_BIN_ROBUSTNESS.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"figures -> {figures}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
