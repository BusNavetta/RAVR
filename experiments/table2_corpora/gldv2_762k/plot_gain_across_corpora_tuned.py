from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent

from benchmark_nested_additive_ravr import panel_results_root  # noqa: E402

SLUG = "gldv2_full"
DEFAULT_DEPTH = 50
METRIC = "teacher_recall_at_10"

# Same corpora, colours and markers as the shipped `gain_across_corpora`
# figure, so the two can be read side by side.
CORPORA = (
    ("sop", "SOP 100K", "#111827", "*"),
    ("gldv2", "GLDv2 100K", "#2563eb", "D"),
    ("gldv2_full", "GLDv2 full index", "#228b22", "^"),
    ("deep1m", "Deep1M", "#e67e22", "o"),
)


def panels(slug: str, factory: str, split: str, seed: int,
           cal: int) -> dict[int, dict[int, dict]]:
    """`{depth: {target: row}}` for one corpus, one seed.

    `sop` and `gldv2` live in directories suffixed `_100k`; the others use the
    bare slug. The `_hndD` suffix carries the mining depth.
    """
    out: dict[int, dict[int, dict]] = {}
    for directory in sorted(panel_results_root(slug).glob(
            f"nested_additive_ravr_{slug}*")):
        match = re.fullmatch(
            rf"nested_additive_ravr_{slug}(?:_100k)?(?:_hnd(\d+))?",
            directory.name)
        if not match:
            continue
        depth = int(match.group(1)) if match.group(1) else DEFAULT_DEPTH
        cells = {}
        for path in directory.glob(
                f"{factory.lower()}_seed{seed}_{split}_u*_cal{cal}.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            rows = report["rows"]
            if "ravr" not in rows or "uniform" not in rows:
                continue
            cells[int(report["target_uniform_length"])] = {
                "gain": rows["ravr"][METRIC] - rows["uniform"][METRIC],
                "ravr": rows["ravr"][METRIC],
                "uniform": rows["uniform"][METRIC],
                "fills": rows["ravr"]["code_cap_slack_bytes"] == 0,
                "slack_per_item": (rows["ravr"]["code_cap_slack_bytes"]
                                   / max(rows["ravr"]["compute_backend"]
                                         ["gallery_items"], 1)),
                "coverage": float(report["coverage_fraction"]),
                "items": int(rows["ravr"]["compute_backend"]["gallery_items"]),
                "dataset": report["dataset"],
            }
        if cells:
            out[depth] = cells
    return out


def tuned_depth(depths: dict[int, dict[int, dict]],
                caps: list[int]) -> int | None:
    """Smallest depth whose raw `ravr` spends the cap at every target."""
    for depth in sorted(depths):
        cells = depths[depth]
        if all(cap in cells and cells[cap]["fills"] for cap in caps):
            return depth
    return None


def both_knobs(factory: str, seed: int, caps: list[int]) -> dict | None:
    path = HERE / f"unseen_knobs_{SLUG}_{factory.lower()}_seed{seed}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    panel_queries = report["panel_calibration_queries"]
    best = None
    for key, cell in report["cells"].items():
        queries, depth = (int(x) for x in key.split(":"))
        if queries <= panel_queries:
            continue
        rows = {cap: cell["caps"].get(str(cap), {}).get("strategies")
                for cap in caps}
        if not all(rows.values()):
            continue
        if not all(rows[cap]["ravr"]["slack_bytes"] == 0 for cap in caps):
            continue
        if best is None or queries > best["queries"] or (
                queries == best["queries"] and depth < best["depth"]):
            best = {
                "queries": queries, "depth": depth,
                "coverage": cell["coverage_fraction"],
                "mined": cell["mean_mined_set"],
                "gains": {cap: (rows[cap]["ravr"]["recall_at_10"]
                                - rows[cap]["uniform"]["recall_at_10"])
                          for cap in caps},
            }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"),
                        default="LSQ16x4")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", choices=("validation", "test"),
                        default="test")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--caps", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--show-default", action="store_true", default=True)
    args = parser.parse_args()

    found, unfixed = {}, []
    for slug, label, colour, marker in CORPORA:
        depths = panels(slug, args.factory, args.split, args.seed,
                        args.calibration_queries)
        if not depths:
            print(f"skipping {label}: no seed-{args.seed} panel", flush=True)
            continue
        picked = tuned_depth(depths, args.caps)
        if picked is None:
            deepest = max(depths)
            unfixed.append((label, deepest, depths[deepest]))
            print(f"{label:<18} NO depth in {sorted(depths)} fills every cap; "
                  f"drawing the deepest ({deepest}) and saying so", flush=True)
            picked = deepest
        else:
            print(f"{label:<18} depth {picked} of {sorted(depths)} "
                  f"(coverage "
                  f"{next(iter(depths[picked].values()))['coverage']:.4f})",
                  flush=True)
        found[slug] = {"label": label, "colour": colour, "marker": marker,
                       "depths": depths, "depth": picked}
    if not found:
        raise SystemExit("no panel output found")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.6, 3.1))
    for slug, entry in found.items():
        cells = entry["depths"][entry["depth"]]
        caps = [cap for cap in args.caps if cap in cells]
        if entry["depth"] != DEFAULT_DEPTH and DEFAULT_DEPTH in entry["depths"]:
            base = entry["depths"][DEFAULT_DEPTH]
            shipped = [cap for cap in args.caps if cap in base]
            axis.plot([cap for cap in shipped],
                      [base[cap]["gain"] for cap in shipped],
                      color=entry["colour"], linestyle=(0, (1.5, 1.5)),
                      linewidth=0.9, marker=entry["marker"], markersize=3.6,
                      markerfacecolor="white", markeredgewidth=0.7,
                      markeredgecolor=entry["colour"], alpha=0.75, zorder=2)
        axis.plot(caps, [cells[cap]["gain"] for cap in caps],
                  color=entry["colour"], linestyle="-", linewidth=1.3,
                  marker=entry["marker"],
                  markersize=7.0 if entry["marker"] == "*" else 5.2,
                  markeredgecolor="white", markeredgewidth=0.5, zorder=5,
                  label=f"{entry['label']}, d={entry['depth']}")
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle=":")
    axis.set_xticks(args.caps)
    axis.set_xlabel("Uniform-equivalent prefix (stages)", labelpad=2)
    axis.set_ylabel("RAVR minus uniform, R@10", labelpad=2)
    axis.set_title("Allocator gain across corpora" + chr(10) +
                   "mining depth tuned so RAVR spends the cap, seed 17",
                   pad=3.0, weight="semibold", fontsize=8.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="#9ca3af", linestyle=(0, (1.5, 1.5)),
                          linewidth=0.9, marker="o", markersize=3.6,
                          markerfacecolor="white", markeredgecolor="#9ca3af",
                          label="same corpus at the default d=50"))
    axis.legend(handles=handles, frameon=False, fontsize=6.3, loc="lower left")
    figure.tight_layout(pad=0.35)

    stem = HERE / "figures" / f"gain_across_corpora_tuned_{args.factory.lower()}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".png"):
        figure.savefig(stem.with_suffix(extension), dpi=260,
                       bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    lines = [
        f"# Allocator gain across corpora, with a RAVR that spends the cap "
        f"-- {args.factory}", "",
        f"Seed {args.seed}, {args.split} split, {args.calibration_queries:,} "
        "calibration queries on every corpus. Raw `ravr`, not the "
        "`ravr_unseen_rd` fallback. The mining depth per corpus is the "
        "smallest one at which raw `ravr` spends the whole cap at **every** "
        "target, so every gain below is at equal bytes.", "",
        "| Corpus | Items | Depth | Coverage | Target | Uniform | RAVR | Gain "
        "| Gain at d=50 | Unspent at d=50 (B/item) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, entry in found.items():
        cells = entry["depths"][entry["depth"]]
        base = entry["depths"].get(DEFAULT_DEPTH, {})
        first = True
        for cap in args.caps:
            cell = cells.get(cap)
            if cell is None:
                continue
            was = base.get(cap)
            head = (f"| {entry['label']} | {cell['items']:,} | "
                    f"{entry['depth']} | {cell['coverage']:.4f} "
                    if first else "| | | | ")
            lines.append(
                f"{head}| {cap} | {cell['uniform']:.4f} | {cell['ravr']:.4f} | "
                f"{cell['gain']:+.5f} | "
                + (f"{was['gain']:+.5f} | {was['slack_per_item']:.3f} |"
                   if was else "-- | -- |"))
            first = False
    if unfixed:
        lines += ["", "**Still short of the cap** at the deepest mining this "
                  "repository has run:", ""]
        for label, depth, cells in unfixed:
            short = [f"cap {cap} ({cells[cap]['slack_per_item']:.3f} B/item "
                     "unspent)" for cap in sorted(cells)
                     if not cells[cap]["fills"]]
            lines.append(f"- {label} at d={depth}: " + ", ".join(short))
    lines += [
        "",
        "The dotted lines in the figure are the same corpora at the default "
        "depth, which is what the shipped `gain_across_corpora` figure plots. "
        "Where a dotted line falls away at the 14-stage cap, RAVR was being "
        "scored on fewer bytes than uniform.", "",
        "Widening the mining is not free -- it divides the Ordinal-KL weight "
        "on every negative and reorders the risk table. `UNSEEN_KNOBS.md` "
        "measures that cost, and shows that more calibration queries buy the "
        "same coverage without it, where more queries can be had.", "",
    ]
    extra = both_knobs(args.factory, args.seed, args.caps)
    if extra is not None and SLUG in found:
        entry = found[SLUG]
        cells = entry["depths"][entry["depth"]]
        lines += [
            "## Both knobs, where the queries could be raised", "",
            f"{entry['label']} is the one corpus in this repository with "
            "official calibration queries left over, so it is the only place "
            "the workload knob can be turned at all. Raising it lets the "
            "mining stay narrower for the same coverage -- "
            f"d={extra['depth']} against d={entry['depth']}, "
            f"{extra['mined']:.0f} mined items per query -- so the "
            "Ordinal-KL weight on each negative is diluted half as "
            "much for the same reach. Both cells spend every cap.", "",
            f"| Target | depth only (Q=5,000, d={entry['depth']}, coverage "
            f"{next(iter(cells.values()))['coverage']:.4f}) | both knobs "
            f"(Q={extra['queries']:,}, d={extra['depth']}, coverage "
            f"{extra['coverage']:.4f}) | difference |",
            "|---:|---:|---:|---:|",
        ]
        for cap in args.caps:
            if cap in cells and cap in extra["gains"]:
                lines.append(
                    f"| {cap} | {cells[cap]['gain']:+.5f} | "
                    f"{extra['gains'][cap]:+.5f} | "
                    f"{extra['gains'][cap] - cells[cap]['gain']:+.5f} |")
        lines.append("")
    summary = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"figure -> {stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
