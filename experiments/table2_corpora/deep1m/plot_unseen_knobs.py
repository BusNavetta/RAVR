from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SLUG = "deep1m"

# Categorical slots 1-3 of the validated palette (all-pairs, light surface),
# plus that system's chart chrome. Three caps, three slots.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAP_COLOUR = {10: BLUE, 12: ORANGE, 14: AQUA}
DEFAULT_DEPTH = 50


def panel_seconds(report: dict, queries: int, depth: int) -> float:
    from benchmark_nested_additive_ravr import panel_results_root

    tag = "" if int(depth) == DEFAULT_DEPTH else f"_hnd{int(depth)}"
    directory = (panel_results_root(SLUG)
                 / f"nested_additive_ravr_{SLUG}{tag}")
    pattern = (f"{report['factory'].lower()}_seed{report['seed']}_"
               f"{report['split']}_u*_cal{queries}.json")
    for path in sorted(directory.glob(pattern)):
        value = json.loads(path.read_text(encoding="utf-8")).get(
            "calibration_seconds", 0.0)
        if value:
            return float(value)
    return 0.0


def cells_of(report: dict) -> list[dict]:
    out = []
    for key, entry in report["cells"].items():
        queries, depth = (int(x) for x in key.split(":"))
        cell = {"key": key, "queries": queries, "depth": depth, **entry}
        if not cell["calibration_seconds"]:
            cell["calibration_seconds"] = panel_seconds(report, queries, depth)
            cell["calibration_seconds_from_panel"] = bool(
                cell["calibration_seconds"])
        out.append(cell)
    return sorted(out, key=lambda c: (c["depth"], c["queries"]))


def required_coverage(cap: int, nbits: int, allowed: list[int]) -> float:
    base = allowed[0] * nbits // 8
    top = allowed[-1] * nbits // 8 - base
    return (cap * nbits // 8 - base) / top


def strategy(cell: dict, cap: int, name: str) -> dict:
    return cell["caps"][str(cap)]["strategies"][name]


def style(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.tick_params(labelsize=7.5, colors=MUTED, length=3)
    axis.grid(True, color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color(AXIS)


def title(axis, text: str) -> None:
    axis.set_title(text, fontsize=9, color=INK, weight="semibold", loc="left",
                   pad=6)


def save(figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.6)
    for suffix in (".png", ".pdf"):
        figure.savefig(stem.with_suffix(suffix), dpi=260, facecolor=SURFACE,
                       bbox_inches="tight", pad_inches=0.04)


# --------------------------------------------------------------------------

def figure_budget_wall(report: dict, cells: list[dict], caps: list[int],
                       stem: Path) -> None:
    import matplotlib.pyplot as plt

    nbits, allowed = report["nbits"], report["allowed_lengths"]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    figure.patch.set_facecolor(SURFACE)

    left, middle, right = axes
    order = sorted(cells, key=lambda c: c["coverage_fraction"])
    coverage = [c["coverage_fraction"] for c in order]
    for cap in caps:
        spent = [strategy(c, cap, "ravr")["variable_bytes_spent"]
                 / c["caps"][str(cap)]["variable_budget_bytes"] for c in order]
        left.plot(coverage, spent, marker="o", markersize=4.5,
                  color=CAP_COLOUR[cap], linewidth=1.6,
                  markeredgecolor=SURFACE, markeredgewidth=0.7,
                  label=f"cap {cap}")
        need = required_coverage(cap, nbits, allowed)
        left.axvline(need, color=CAP_COLOUR[cap], linewidth=1.0,
                     linestyle=(0, (4, 3)), alpha=0.8)
        # along the top: the curves live in the lower half at these x, and the
        # legend needs the bottom-right corner
        left.annotate(f"(m-8)/8 = {need:.2f}", (need, 1.03), xytext=(4, 0),
                      textcoords="offset points", color=CAP_COLOUR[cap],
                      fontsize=6.5, rotation=90, ha="left", va="top")
    left.set_xlabel("calibration coverage", fontsize=8, color=SECOND)
    left.set_ylabel("share of the variable budget RAVR spends", fontsize=8,
                    color=SECOND)
    left.set_ylim(-0.03, 1.06)
    left.legend(fontsize=7.5, frameon=False, labelcolor=SECOND, loc="lower right")
    title(left, "(a) The derived bound is necessary, not sufficient")

    # what the cap-filling fallbacks recover, at the panel's own workload
    reference = next(c for c in cells if c["key"] == report["reference_cell"])
    names = ("uniform", "ravr", "ravr_unseen_rd", "ravr_unseen_qip")
    labels = ("uniform", "RAVR", "RAVR + unseen RD", "RAVR + unseen query-IP")
    width = 0.2
    for offset, (name, text) in enumerate(zip(names, labels)):
        values = [strategy(reference, cap, name)["recall_at_10"] for cap in caps]
        middle.bar(np.arange(len(caps)) + (offset - 1.5) * width, values,
                   width=width * 0.9, label=text, zorder=3,
                   color=(MUTED, BLUE, ORANGE, AQUA)[offset])
    middle.set_xticks(np.arange(len(caps)))
    middle.set_xticklabels([f"cap {cap}" for cap in caps])
    middle.set_ylabel("Teacher Recall@10", fontsize=8, color=SECOND)
    middle.legend(fontsize=7, frameon=False, labelcolor=SECOND, loc="upper left")
    title(middle, f"(b) Knob 3 at the panel's own workload "
                  f"(Q={reference['queries']}, d={reference['depth']})")

    # slack in bytes per item, every cell
    for cap in caps:
        slack = [strategy(c, cap, "ravr")["slack_bytes"]
                 / report["gallery_items"] for c in order]
        right.plot(coverage, slack, marker="o", markersize=4.5,
                   color=CAP_COLOUR[cap], linewidth=1.6,
                   markeredgecolor=SURFACE, markeredgewidth=0.7,
                   label=f"cap {cap}")
    right.axhline(0, color=AXIS, linewidth=1.0)
    right.set_xlabel("calibration coverage", fontsize=8, color=SECOND)
    right.set_ylabel("unspent budget (B/item)", fontsize=8, color=SECOND)
    right.legend(fontsize=7.5, frameon=False, labelcolor=SECOND)
    title(right, "(c) What raw RAVR leaves on the table")

    for axis in axes:
        style(axis)
    save(figure, stem)
    plt.close(figure)


def figure_tradeoff(report: dict, cells: list[dict], caps: list[int],
                    stem: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.4))
    figure.patch.set_facecolor(SURFACE)
    (iso10, iso12, iso14), (cov, dilute, concentrate) = axes

    for axis, cap in zip((iso10, iso12, iso14), caps):
        for cell in cells:
            colour = BLUE if cell["depth"] == DEFAULT_DEPTH else ORANGE
            marker = "o" if cell["depth"] == DEFAULT_DEPTH else "D"
            axis.plot([cell["coverage_fraction"]],
                      [strategy(cell, cap, "ravr_unseen_rd")["recall_at_10"]],
                      marker=marker, markersize=5.5, color=colour,
                      markeredgecolor=SURFACE, markeredgewidth=0.7,
                      linestyle="none", zorder=5)
            # alternate the offset: iso-coverage cells sit on top of each
            # other, and a single fixed offset overlaps their labels
            side = 1 if cell["depth"] <= 100 else -1
            axis.annotate(f"{cell['queries']}/{cell['depth']}",
                          (cell["coverage_fraction"],
                           strategy(cell, cap, "ravr_unseen_rd")["recall_at_10"]),
                          xytext=(5 * side, 3 * side),
                          textcoords="offset points", fontsize=5.8,
                          color=SECOND,
                          ha="left" if side > 0 else "right")
        axis.set_xlabel("calibration coverage", fontsize=8, color=SECOND)
        axis.set_ylabel("Teacher Recall@10 (RAVR + unseen RD)", fontsize=8,
                        color=SECOND)
        title(axis, f"cap {cap}")
    iso10.annotate("circles: default mining (d=50)\ndiamonds: widened mining",
                   (0.98, 0.03), xycoords="axes fraction", fontsize=6.8,
                   color=SECOND, va="bottom", ha="right")


    by_depth = sorted([c for c in cells if c["queries"]
                       == report["panel_calibration_queries"]],
                      key=lambda c: c["depth"])
    by_queries = sorted([c for c in cells if c["depth"] == DEFAULT_DEPTH],
                        key=lambda c: c["queries"])

    incidences = lambda cell: cell["queries"] * cell["mean_mined_set"]
    cov.plot([incidences(c) for c in by_depth],
             [c["coverage_fraction"] for c in by_depth], marker="D",
             markersize=5, color=ORANGE, linewidth=1.6,
             markeredgecolor=SURFACE, markeredgewidth=0.7,
             label=f"depth sweep, Q={report['panel_calibration_queries']}")
    cov.plot([incidences(c) for c in by_queries],
             [c["coverage_fraction"] for c in by_queries], marker="o",
             markersize=5, color=BLUE, linewidth=1.6,
             markeredgecolor=SURFACE, markeredgewidth=0.7,
             label=f"query sweep, d={DEFAULT_DEPTH}")
    rest = [c for c in cells if c not in by_depth and c not in by_queries]
    if rest:
        cov.plot([incidences(c) for c in rest],
                 [c["coverage_fraction"] for c in rest], marker="s",
                 markersize=4.5, color=AQUA, linestyle="none",
                 markeredgecolor=SURFACE, markeredgewidth=0.7,
                 label="mixed cells")
    cov.set_xscale("log")
    cov.set_xlabel("mined incidences, Q x (items mined per query)",
                   fontsize=8, color=SECOND)
    cov.set_ylabel("calibration coverage", fontsize=8, color=SECOND)
    cov.legend(fontsize=7, frameon=False, labelcolor=SECOND, loc="upper left")
    title(cov, "(d) For coverage the two knobs are one knob")

    dilute.plot([c["depth"] for c in by_depth],
                [c["mean_mined_set"] for c in by_depth], marker="D",
                markersize=5, color=ORANGE, linewidth=1.6,
                markeredgecolor=SURFACE, markeredgewidth=0.7,
                label="mined set per query")
    dilute.plot([c["depth"] for c in by_depth],
                [c["nominal_window"] for c in by_depth], color=MUTED,
                linewidth=1.0, linestyle=(0, (4, 3)), label="k + 3d")
    dilute.set_xscale("log")
    dilute.set_yscale("log")
    dilute.set_xlabel("hard-negative depth", fontsize=8, color=SECOND)
    dilute.set_ylabel("items mined per query", fontsize=8, color=SECOND)
    dilute.legend(fontsize=7, frameon=False, labelcolor=SECOND)
    title(dilute, "(e) Every extra negative divides the Ordinal-KL weight")

    # Same queries, only the window widens: any change in the risk ranking of
    # the items those queries all saw is dilution and nothing else.
    rho = [c.get("risk_spearman_vs_reference_on_shared_seen", 1.0)
           for c in by_depth]
    concentrate.plot([c["depth"] for c in by_depth], rho, marker="D",
                     markersize=5, color=ORANGE, linewidth=1.6,
                     markeredgecolor=SURFACE, markeredgewidth=0.7,
                     label="risk ranking vs d=50, on the items both saw (rho)")
    concentrate.plot([c["depth"] for c in by_depth],
                     [strategy(c, caps[0], "ravr")["recall_at_10"]
                      / strategy(by_depth[0], caps[0], "ravr")["recall_at_10"]
                      for c in by_depth], marker="s", markersize=4.5,
                     color=BLUE, linewidth=1.6, markeredgecolor=SURFACE,
                     markeredgewidth=0.7,
                     label=f"R@10 at cap {caps[0]}, relative to d={DEFAULT_DEPTH}")
    concentrate.set_xscale("log")
    concentrate.set_ylim(0.0, 1.06)
    concentrate.set_xlabel("hard-negative depth", fontsize=8, color=SECOND)
    concentrate.set_ylabel("ratio to the default depth", fontsize=8,
                           color=SECOND)
    concentrate.legend(fontsize=6.8, frameon=False, labelcolor=SECOND,
                       loc="lower left")
    title(concentrate, "(f) It reorders the items it already knew")

    for row in axes:
        for axis in row:
            style(axis)
    save(figure, stem)
    plt.close(figure)


def figure_cost(report: dict, cells: list[dict], caps: list[int],
                stem: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    figure.patch.set_facecolor(SURFACE)
    seconds, efficiency, frontier = axes

    timed = [c for c in cells if c["calibration_seconds"] > 0]
    panel_queries = report["panel_calibration_queries"]
    by_depth = sorted([c for c in timed if c["queries"] == panel_queries],
                      key=lambda c: c["depth"])
    by_queries = sorted([c for c in timed if c["depth"] == DEFAULT_DEPTH],
                        key=lambda c: c["queries"])

    def slope(points, knob):
        """log-log slope of seconds against the knob. 1 means linear, 0 free."""
        if len(points) < 2:
            return None
        x = np.log([c[knob] for c in points])
        y = np.log([c["calibration_seconds"] for c in points])
        return float(np.polyfit(x, y, 1)[0])

    # Both knobs on one dimensionless axis: multiples of the panel's default.
    # The slopes are the whole point -- one is 1, the other is 0.
    if by_queries:
        fit = slope(by_queries, "queries")
        seconds.plot([c["queries"] / panel_queries for c in by_queries],
                     [c["calibration_seconds"] for c in by_queries], marker="o",
                     markersize=5, color=BLUE, linewidth=1.6,
                     markeredgecolor=SURFACE, markeredgewidth=0.7,
                     label=("calibration queries"
                            + (f" (log-log slope {fit:.2f})" if fit else "")))
    if by_depth:
        fit = slope(by_depth, "depth")
        seconds.plot([c["depth"] / DEFAULT_DEPTH for c in by_depth],
                     [c["calibration_seconds"] for c in by_depth], marker="D",
                     markersize=5, color=ORANGE, linewidth=1.6,
                     markeredgecolor=SURFACE, markeredgewidth=0.7,
                     label=("hard-negative depth"
                            + (f" (log-log slope {fit:.2f})" if fit else "")))
    seconds.set_xscale("log")
    seconds.set_yscale("log")
    seconds.axvline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    seconds.set_xlabel("knob value, as a multiple of the panel default",
                       fontsize=8, color=SECOND)
    seconds.set_ylabel("calibration seconds", fontsize=8, color=SECOND)
    seconds.legend(fontsize=7, frameon=False, labelcolor=SECOND,
                   loc="upper left")
    title(seconds, "(a) Only one knob costs time")

    for cell in timed:
        colour = BLUE if cell["depth"] == DEFAULT_DEPTH else ORANGE
        efficiency.plot([cell["calibration_seconds"]],
                        [cell["coverage_fraction"]], marker="o" if
                        cell["depth"] == DEFAULT_DEPTH else "D", markersize=5.5,
                        color=colour, markeredgecolor=SURFACE,
                        markeredgewidth=0.7, linestyle="none")
        efficiency.annotate(f"{cell['queries']}/{cell['depth']}",
                            (cell["calibration_seconds"],
                             cell["coverage_fraction"]), xytext=(4, 3),
                            textcoords="offset points", fontsize=6,
                            color=SECOND)
    efficiency.set_xlabel("calibration seconds", fontsize=8, color=SECOND)
    efficiency.set_ylabel("calibration coverage", fontsize=8, color=SECOND)
    title(efficiency, "(b) Coverage per second bought")

    depth_sweep = sorted([c for c in cells if c["queries"] == panel_queries],
                         key=lambda c: c["depth"])
    # Depth is ordered, so it gets a sequential ramp, not categorical hues.
    # Steps 250-650 of the blue ramp: the lightest still clears 2:1 on this
    # surface, which is the ordinal floor.
    ramp = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
    for index, cell in enumerate(depth_sweep or cells[:1]):
        xs, ys = [], []
        for cap in caps:
            row = strategy(cell, cap, "ravr")
            xs.append(row["code_bytes_per_item"])
            ys.append(row["recall_at_10"])
        frontier.plot(xs, ys, marker="D", markersize=5, linewidth=1.6,
                      color=ramp[min(index, len(ramp) - 1)],
                      markeredgecolor=SURFACE, markeredgewidth=0.7,
                      label=f"RAVR, d={cell['depth']}")
    reference = next(c for c in cells if c["key"] == report["reference_cell"])
    for name, marker, colour, text in (
            ("uniform", "o", MUTED, "uniform"),
            ("ravr_unseen_rd", "X", INK, f"RAVR + unseen RD, d={DEFAULT_DEPTH}")):
        xs = [strategy(reference, cap, name)["code_bytes_per_item"] for cap in caps]
        ys = [strategy(reference, cap, name)["recall_at_10"] for cap in caps]
        frontier.plot(xs, ys, marker=marker, markersize=5, color=colour,
                      linewidth=1.4, linestyle=(0, (4, 3)),
                      markeredgecolor=SURFACE, markeredgewidth=0.7, label=text)
    frontier.set_xlabel("code bytes per item (excludes shared codebooks)",
                        fontsize=8, color=SECOND)
    frontier.set_ylabel("Teacher Recall@10", fontsize=8, color=SECOND)
    frontier.legend(fontsize=6.5, frameon=False, labelcolor=SECOND,
                    loc="lower right")
    title(frontier, f"(c) The frontier, one seed ({report['seed']}) throughout")

    for axis in axes:
        style(axis)
    save(figure, stem)
    plt.close(figure)


# --------------------------------------------------------------------------

def table(report: dict, cells: list[dict], caps: list[int]) -> str:
    nbits, allowed = report["nbits"], report["allowed_lengths"]
    reference = next(c for c in cells if c["key"] == report["reference_cell"])
    lines = [
        f"# The four knobs against unseen items -- {report['dataset']}", "",
        f"{report['gallery_items']:,} items, {report['factory']}, seed "
        f"{report['seed']}, {report['evaluation_queries']} {report['split']} "
        f"queries, candidate depth {report['rerank_candidates']:,}. Every cell "
        "is one calibration; RAVR itself is unchanged throughout.", "",
        "`ravr` can only reach the cap when `coverage >= (m-8)/8`: "
        + ", ".join(f"{required_coverage(cap, nbits, allowed):.2f} at cap {cap}"
                    for cap in caps) + ". That bound is **necessary, not "
        "sufficient** -- some seen items also carry a zero risk gap, and the "
        "bisection plus one-step repair does not reach the combinatorial "
        "optimum. What the sweep actually measured:",
        "",
        "| cap | bound | highest coverage that still underfills | "
        "lowest coverage that fills |", "|---:|---:|---:|---:|",
    ] + [
        (lambda under, over: f"| {cap} | "
         f"{required_coverage(cap, nbits, allowed):.2f} | "
         + (f"{max(under):.4f}" if under else "--") + " | "
         + (f"{min(over):.4f}" if over else "not reached in this sweep")
         + " |")(
            [c["coverage_fraction"] for c in cells
             if strategy(c, cap, "ravr")["slack_bytes"] > 0],
            [c["coverage_fraction"] for c in cells
             if strategy(c, cap, "ravr")["slack_bytes"] == 0])
        for cap in caps
    ] + [
        "",
        "## Every cell", "",
        "| Q | depth | coverage | unseen | mined/query | cal. s | "
        + " | ".join(f"spent@{cap} | R@10@{cap}" for cap in caps) + " |",
        "|---:|---:|---:|---:|---:|---:|" + "---:|---:|" * len(caps),
    ]
    for cell in sorted(cells, key=lambda c: (-c["coverage_fraction"],)):
        row = [f"| {cell['queries']}", f"{cell['depth']}",
               f"{cell['coverage_fraction']:.4f}",
               f"{cell['unseen_fraction']:.4f}",
               f"{cell['mean_mined_set']:.0f}",
               f"{cell['calibration_seconds']:.0f}"]
        for cap in caps:
            ravr = strategy(cell, cap, "ravr")
            row.append(f"{ravr['code_bytes_per_item']:.3f}")
            row.append(f"{ravr['recall_at_10']:.4f}")
        lines.append(" | ".join(row) + " |")

    lines += ["", "## Knob 3: the cap-filling fallbacks", "",
              "| Q | depth | cap | RAVR | + unseen RD | + unseen query-IP | "
              "uniform | best |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for cell in sorted(cells, key=lambda c: (c["depth"], c["queries"])):
        for cap in caps:
            values = {name: strategy(cell, cap, name)["recall_at_10"]
                      for name in ("ravr", "ravr_unseen_rd", "ravr_unseen_qip",
                                   "uniform")}
            best = max(values, key=values.get)
            lines.append(
                f"| {cell['queries']} | {cell['depth']} | {cap} | "
                + " | ".join(f"{values[name]:.4f}" for name in
                             ("ravr", "ravr_unseen_rd", "ravr_unseen_qip",
                              "uniform"))
                + f" | {best} |")

    lines += ["", "## Risk-table diagnostics", "",
              "`mined/query` is `coverage.sum() / Q`, so it is also the "
              "dilution factor: Ordinal-KL splits half the mass over the "
              "negatives of each cutoff, so a negative carries "
              "`0.5 / negatives`.", "",
              "| Q | depth | mined/query | weight per negative | "
              "top-1% risk share | effective items | Gini | "
              "rho vs the reference cell |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in sorted(cells, key=lambda c: (c["depth"], c["queries"])):
        rho = cell.get("risk_spearman_vs_reference_on_shared_seen")
        lines.append(
            f"| {cell['queries']} | {cell['depth']} | "
            f"{cell['mean_mined_set']:.0f} | "
            f"{cell['negative_weight_per_item']:.5f} | "
            f"{cell['risk_top1pct_share']:.4f} | "
            f"{cell['risk_effective_items']:,.0f} | {cell['risk_gini']:.4f} | "
            + (f"{rho:.4f} |" if rho is not None else "-- |"))
    lines += ["", f"Reference cell: `{report['reference_cell']}` "
                  f"(coverage {reference['coverage_fraction']:.4f}).", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--factory", default="lsq16x4")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    path = args.report or (
        HERE / f"unseen_knobs_{SLUG}_{args.factory}_seed{args.seed}.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    cells = cells_of(report)
    caps = sorted(int(cap) for cap in cells[0]["caps"])

    import matplotlib
    matplotlib.use("Agg")

    figures = HERE / "figures"
    figure_budget_wall(report, cells, caps,
                       figures / f"unseen_budget_wall_{args.factory}")
    figure_tradeoff(report, cells, caps,
                    figures / f"unseen_knob_tradeoff_{args.factory}")
    figure_cost(report, cells, caps,
                figures / f"unseen_knob_cost_{args.factory}")
    text = table(report, cells, caps)
    (HERE / f"unseen_knobs_{SLUG}_{args.factory}.md").write_text(
        text, encoding="utf-8")
    print(text)
    print(f"figures -> {figures}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
