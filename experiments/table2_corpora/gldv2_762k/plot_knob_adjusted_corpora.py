from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent

from benchmark_nested_additive_ravr import panel_results_root  # noqa: E402

SLUG = "gldv2_full"
RETRIEVAL_K = 10
DEFAULT_DEPTH = 50
REFERENCE_ITEMS = 100_000
STRATEGY = "ravr_unseen_rd"
BASELINE = "uniform"

# Ordered by gallery size; the two 100K corpora are the reference the scaling
# rule is anchored on.
CORPORA = (
    ("sop_100k", "SOP"),
    ("gldv2_100k", "GLDv2 100K"),
    ("gldv2_full", "GLDv2 full"),
    ("deep1m", "Deep1M"),
)

# Validated categorical slots 1-3 (all-pairs, light surface): one per cap.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAP_COLOUR = {10: BLUE, 12: ORANGE, 14: AQUA}


def target_depth(items: int) -> float:
    """Mining depth whose window is the same share of the gallery as at 100K."""
    window = (RETRIEVAL_K + 3 * DEFAULT_DEPTH) * items / REFERENCE_ITEMS
    return (window - RETRIEVAL_K) / 3.0


def available(slug: str, factory: str, split: str, seed: int,
              cal: int) -> dict[int, dict]:
    """Every mining depth this corpus has a panel for, keyed by depth."""
    out: dict[int, dict] = {}
    root = panel_results_root(slug)
    for directory in sorted(root.glob(f"nested_additive_ravr_{slug}*")):
        match = re.fullmatch(rf"nested_additive_ravr_{slug}(?:_hnd(\d+))?",
                             directory.name)
        if not match:
            continue
        depth = int(match.group(1)) if match.group(1) else DEFAULT_DEPTH
        cells = {}
        for path in directory.glob(
            f"{factory.lower()}_seed{seed}_{split}_u*_cal{cal}.json"
        ):
            report = json.loads(path.read_text(encoding="utf-8"))
            rows = report["rows"]
            if STRATEGY not in rows or BASELINE not in rows:
                continue
            cells[int(report["target_uniform_length"])] = {
                "coverage": float(report["coverage_fraction"]),
                "items": int(next(iter(rows.values()))
                             ["compute_backend"]["gallery_items"]),
                "dataset": report["dataset"],
                "gain": (rows[STRATEGY]["teacher_recall_at_10"]
                         - rows[BASELINE]["teacher_recall_at_10"]),
                "strategy": rows[STRATEGY]["teacher_recall_at_10"],
                "baseline": rows[BASELINE]["teacher_recall_at_10"],
                "raw_ravr": rows["ravr"]["teacher_recall_at_10"],
                "raw_fills_cap": rows["ravr"]["code_cap_slack_bytes"] == 0,
                "bytes_per_item": rows[STRATEGY]["persistent_bytes_per_item"],
            }
        if cells:
            out[depth] = cells
    return out


def choose(depths: dict[int, dict], items: int) -> tuple[int, float]:
    """The panel depth nearest the scaling rule, and the rule's own value."""
    wanted = target_depth(items)
    return min(depths, key=lambda d: abs(d - wanted)), wanted


def query_scaled(factory: str, seed: int, caps: list[int]) -> dict | None:
    path = HERE / f"unseen_knobs_{SLUG}_{factory.lower()}_seed{seed}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    panel_queries = report["panel_calibration_queries"]
    best = None
    for key, cell in report["cells"].items():
        queries, depth = (int(x) for x in key.split(":"))
        if depth != DEFAULT_DEPTH or queries <= panel_queries:
            continue
        if best is None or queries > best[0]:
            best = (queries, cell)
    if best is None:
        return None
    queries, cell = best
    gains = {}
    for cap in caps:
        strategies = cell["caps"].get(str(cap), {}).get("strategies")
        if strategies:
            gains[cap] = (strategies[STRATEGY]["recall_at_10"]
                          - strategies[BASELINE]["recall_at_10"])
    return {"queries": queries, "coverage": cell["coverage_fraction"],
            "gains": gains}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"),
                        default="LSQ16x4")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", choices=("validation", "test"),
                        default="test")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--caps", nargs="+", type=int, default=[10, 12, 14])
    args = parser.parse_args()

    panels = {}
    for slug, label in CORPORA:
        depths = available(slug, args.factory, args.split, args.seed,
                           args.calibration_queries)
        if not depths:
            print(f"skipping {slug}: no seed-{args.seed} panel found",
                  flush=True)
            continue
        items = next(iter(next(iter(depths.values())).values()))["items"]
        picked, wanted = choose(depths, items)
        panels[slug] = {
            "label": label, "items": items, "depths": depths,
            "adjusted_depth": picked, "target_depth": wanted,
        }
        print(f"{label:<12} N={items:>9,}  rule wants d={wanted:.0f}, "
              f"using d={picked} of {sorted(depths)}", flush=True)
    if not panels:
        raise SystemExit("no panel output found for any corpus")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [slug for slug, _ in CORPORA if slug in panels]
    x = np.arange(len(order))
    figure, axis = plt.subplots(figsize=(7.0, 4.3))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    for cap in args.caps:
        adjusted, default = [], []
        for slug in order:
            entry = panels[slug]
            cells = entry["depths"][entry["adjusted_depth"]]
            adjusted.append(cells.get(cap, {}).get("gain", np.nan))
            base = entry["depths"].get(DEFAULT_DEPTH, {})
            default.append(base.get(cap, {}).get("gain", np.nan))
        colour = CAP_COLOUR.get(cap, MUTED)
        axis.plot(x, default, color=colour, linewidth=1.0,
                  linestyle=(0, (2, 2)), marker="o", markersize=5.5,
                  markerfacecolor=SURFACE, markeredgecolor=colour,
                  markeredgewidth=1.2, zorder=3)
        axis.plot(x, adjusted, color=colour, linewidth=1.8, marker="o",
                  markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.8,
                  zorder=5, label=f"cap {cap}")
        # the movement the knob produces, where it moves anything
        for xi, (low, high) in enumerate(zip(default, adjusted)):
            if np.isfinite(low) and np.isfinite(high) and abs(high - low) > 1e-6:
                axis.annotate("", xy=(xi, high), xytext=(xi, low),
                              arrowprops=dict(arrowstyle="-|>", color=colour,
                                              linewidth=1.0, shrinkA=4,
                                              shrinkB=4, alpha=0.8), zorder=4)

    extra = query_scaled(args.factory, args.seed, args.caps)
    if extra is not None and SLUG in panels:
        at = order.index(SLUG)
        for cap in args.caps:
            if cap in extra["gains"]:
                axis.plot([at], [extra["gains"][cap]], marker="*",
                          markersize=13, color=CAP_COLOUR.get(cap, MUTED),
                          markeredgecolor=SURFACE, markeredgewidth=0.9,
                          linestyle="none", zorder=7)
        print(f"query-scaled cell on {panels[SLUG]['label']}: Q="
              f"{extra['queries']:,}, coverage {extra['coverage']:.4f}",
              flush=True)

    ticks = []
    for slug in order:
        entry = panels[slug]
        cells = entry["depths"][entry["adjusted_depth"]]
        coverage = next(iter(cells.values()))["coverage"]
        ticks.append(f"{entry['label']}\n{entry['items']/1000:,.0f}K items\n"
                     f"d={entry['adjusted_depth']}, coverage {coverage:.2f}")
    axis.set_xticks(x)
    axis.set_xticklabels(ticks, fontsize=7.5, color=SECOND)
    axis.set_xlim(-0.45, len(order) - 0.55)
    axis.set_ylabel(f"Teacher Recall@10 gain over uniform\n({STRATEGY}, seed "
                    f"{args.seed})", fontsize=8.5, color=SECOND, labelpad=3)
    axis.tick_params(axis="y", labelsize=8, colors=MUTED, length=3)
    axis.tick_params(axis="x", length=0)
    axis.grid(True, axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=CAP_COLOUR.get(cap, MUTED), linewidth=1.8,
                      marker="o", markersize=6, markeredgecolor=SURFACE,
                      label=f"cap {cap}") for cap in args.caps]
    handles += [
        Line2D([0], [0], color=MUTED, linewidth=1.8, marker="o", markersize=6,
               markeredgecolor=SURFACE, label="mining depth scaled to N"),
        Line2D([0], [0], color=MUTED, linewidth=1.0, linestyle=(0, (2, 2)),
               marker="o", markersize=5.5, markerfacecolor=SURFACE,
               markeredgecolor=MUTED, label="panel default, d=50"),
    ]
    if extra is not None and SLUG in panels:
        handles.append(Line2D(
            [0], [0], color=MUTED, marker="*", markersize=11, linestyle="none",
            markeredgecolor=SURFACE,
            label=f"queries scaled instead: Q={extra['queries']:,}, d=50"))
    axis.legend(handles=handles, fontsize=7, frameon=False, ncol=2,
                labelcolor=SECOND, loc="lower left")
    axis.set_title(
        "The allocator gain across corpora, with the mining depth scaled to "
        "the gallery\n"
        "open markers are the panels' default depth; queries stay at 5,000 "
        "except where a star says otherwise",
        fontsize=9, color=INK, weight="semibold", loc="left", pad=8)

    stem = HERE / "figures" / f"knob_adjusted_corpora_{args.factory.lower()}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.5)
    for suffix in (".png", ".pdf"):
        figure.savefig(stem.with_suffix(suffix), dpi=260, facecolor=SURFACE,
                       bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)

    lines = [
        "# The allocator gain with the mining depth scaled to the gallery", "",
        f"{args.factory}, seed {args.seed}, {args.split} split, "
        f"{args.calibration_queries:,} calibration queries on every corpus. "
        f"The plotted strategy is `{STRATEGY}`, which fills the cap exactly, so "
        "every gain below is at equal bytes. The scaling rule is "
        "`d*(N) = ((k + 3*50) * N / 100000 - k) / 3`.", "",
        "| Corpus | Items | Rule wants | Depth used | Coverage | Cap | "
        "uniform | RAVR + unseen RD | Gain | Gain at d=50 | Change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slug in order:
        entry = panels[slug]
        cells = entry["depths"][entry["adjusted_depth"]]
        base = entry["depths"].get(DEFAULT_DEPTH, {})
        first = True
        for cap in args.caps:
            cell = cells.get(cap)
            if cell is None:
                continue
            was = base.get(cap, {}).get("gain")
            head = (f"| {entry['label']} | {entry['items']:,} | "
                    f"{entry['target_depth']:.0f} | {entry['adjusted_depth']} | "
                    f"{cell['coverage']:.4f} " if first else "| | | | | ")
            lines.append(
                f"{head}| {cap} | {cell['baseline']:.4f} | "
                f"{cell['strategy']:.4f} | {cell['gain']:+.4f} | "
                + (f"{was:+.4f} | {cell['gain'] - was:+.4f} |"
                   if was is not None else "-- | -- |"))
            first = False
    lines += [
        "",
        "Reading it: at 100K the rule returns the default depth, so those two "
        "corpora are unchanged and act as the reference. The two large "
        "corpora move, and the question the figure answers is whether the "
        "gain still falls with gallery size once every corpus is calibrated "
        "at a comparable coverage.", "",
    ]
    if extra is not None and SLUG in panels:
        reference = panels[SLUG]
        cells = reference["depths"][reference["adjusted_depth"]]
        lines += [
            "## The other knob, where it was available", "",
            "Scaling the *queries* instead of the mining, on "
            f"{reference['label']}: Q={extra['queries']:,} at the default "
            f"depth, coverage {extra['coverage']:.4f} against "
            f"{next(iter(cells.values()))['coverage']:.4f} for the "
            "depth-scaled cell. Fewer items seen, and it still wins where the "
            "budget is not the binding constraint.", "",
            "| Cap | depth scaled (d="
            f"{reference['adjusted_depth']}) | queries scaled "
            f"(Q={extra['queries']:,}) | difference |",
            "|---:|---:|---:|---:|",
        ]
        for cap in args.caps:
            if cap in extra["gains"] and cap in cells:
                depth_gain = cells[cap]["gain"]
                query_gain = extra["gains"][cap]
                lines.append(f"| {cap} | {depth_gain:+.4f} | "
                             f"{query_gain:+.4f} | "
                             f"{query_gain - depth_gain:+.4f} |")
        lines.append("")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"figure -> {stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
