from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent

from benchmark_nested_additive_ravr import panel_results_root  # noqa: E402

SLUG = "deep1m"
DEFAULT_DEPTH = 50
METRIC = "teacher_recall_at_10"
DEPTH_COLOURS = ("#111827", "#2563eb", "#228b22", "#e67e22", "#9333ea")


def collect(slug: str, factory: str, split: str, cal: int) -> dict[int, dict]:
    """Per-seed rows for every mining depth that has been run."""
    root = panel_results_root(slug)
    out: dict[int, dict] = {}
    for directory in sorted(root.glob(f"nested_additive_ravr_{slug}*")):
        match = re.fullmatch(rf"nested_additive_ravr_{slug}(?:_hnd(\d+))?",
                             directory.name)
        if not match:
            continue
        depth = int(match.group(1)) if match.group(1) else DEFAULT_DEPTH
        cells: dict[int, dict[str, list]] = {}
        seeds = set()
        for path in directory.glob(
            f"{factory.lower()}_seed*_{split}_u*_cal{cal}.json"
        ):
            report = json.loads(path.read_text(encoding="utf-8"))
            target = report["target_uniform_length"]
            seeds.add(report["seed"])
            bucket = cells.setdefault(target, {})
            for name, row in report["rows"].items():
                bucket.setdefault(name, []).append(
                    (row["persistent_bytes_per_item"], row[METRIC])
                )
        if cells:
            out[depth] = {"cells": cells, "seeds": sorted(seeds)}
    return out


def restrict(slug: str, factory: str, split: str, cal: int,
             seeds: list[int]) -> dict[int, dict]:
    """`collect`, re-read with only these seeds. Keeps the curves comparable."""
    keep = set(int(s) for s in seeds)
    root = panel_results_root(slug)
    out: dict[int, dict] = {}
    for directory in sorted(root.glob(f"nested_additive_ravr_{slug}*")):
        match = re.fullmatch(rf"nested_additive_ravr_{slug}(?:_hnd(\d+))?",
                             directory.name)
        if not match:
            continue
        depth = int(match.group(1)) if match.group(1) else DEFAULT_DEPTH
        cells: dict[int, dict[str, list]] = {}
        used = set()
        for path in directory.glob(
            f"{factory.lower()}_seed*_{split}_u*_cal{cal}.json"
        ):
            report = json.loads(path.read_text(encoding="utf-8"))
            if int(report["seed"]) not in keep:
                continue
            used.add(int(report["seed"]))
            bucket = cells.setdefault(report["target_uniform_length"], {})
            for name, row in report["rows"].items():
                bucket.setdefault(name, []).append(
                    (row["persistent_bytes_per_item"], row[METRIC])
                )
        if cells:
            out[depth] = {"cells": cells, "seeds": sorted(used)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=SLUG)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument(
        "--seeds", choices=("match", "all"), default="match",
        help="match: restrict every depth to the seeds they all have, so the "
             "curves are like for like; all: average whatever each depth has",
    )
    args = parser.parse_args()

    data = collect(args.dataset, args.factory, args.split,
                   args.calibration_queries)
    if not data:
        raise SystemExit(f"no panel output found for {args.dataset}")
    shared = sorted(set.intersection(*(set(v["seeds"]) for v in data.values())))
    if args.seeds == "match" and len(data) > 1:
        if not shared:
            raise SystemExit("no seed is present at every mining depth; "
                             "re-run with --seeds all and say so in the caption")
        data = restrict(args.dataset, args.factory, args.split,
                        args.calibration_queries, shared)
        print(f"seeds  matched across depths: {shared}", flush=True)
    else:
        print(f"seeds  per depth as available: "
              f"{ {d: v['seeds'] for d, v in data.items()} }", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 3.3))

    def series(depth: int, strategy: str):
        cells = data[depth]["cells"]
        points = []
        for target in sorted(cells):
            rows = cells[target].get(strategy)
            if not rows:
                continue
            byte, score = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
            points.append((byte, score))
        return list(zip(*sorted(points))) if points else ([], [])

    # uniform is the same allocation at every depth: draw it once
    reference_depth = min(data)
    ux, uy = series(reference_depth, "uniform")
    axis.plot(ux, uy, color="#9ca3af", marker="o", linestyle="--", linewidth=1.0,
              markersize=4.0, label="Uniform prefix", zorder=2)

    for index, depth in enumerate(sorted(data)):
        colour = DEPTH_COLOURS[index % len(DEPTH_COLOURS)]
        seeds = len(data[depth]["seeds"])
        suffix = f"d={depth}, {seeds} seed{'s' if seeds != 1 else ''}"
        x, y = series(depth, "ravr")
        fx, fy = series(depth, "ravr_unseen_rd")
        # Where raw RAVR already fills every cap the fallback is a no-op and
        # the two curves coincide; say so instead of drawing one over the other.
        identical = (len(x) == len(fx)
                     and np.allclose(x, fx, atol=1e-6)
                     and np.allclose(y, fy, atol=1e-6))
        if fx and not identical:
            axis.plot(fx, fy, color=colour, marker="X", linestyle="-.",
                      linewidth=1.1, markersize=5.5, markeredgecolor="white",
                      markeredgewidth=0.4, alpha=0.9,
                      label=f"RAVR + unseen RD, {suffix}", zorder=4 + index)
        if x:
            axis.plot(x, y, color=colour, marker="*", linestyle="-",
                      linewidth=1.3, markersize=8.0, markeredgecolor="white",
                      markeredgewidth=0.5, zorder=5 + index,
                      label=f"RAVR, {suffix}"
                            + (" (= + unseen RD)" if identical else ""))

    # mark the caps, so a curve that stops short is unmistakable
    for cap in ux:
        axis.axvline(cap, color="#e5e7eb", linewidth=0.8, zorder=1)
    axis.set_xlabel("Complete serialized storage (B/item)", labelpad=2)
    axis.set_ylabel("Teacher Recall@10", labelpad=2)
    matched = "seed-matched" if args.seeds == "match" and len(data) > 1 else "as available"
    axis.set_title(f"Does a wider mining let RAVR spend the cap? ({matched})",
                   pad=4.0, weight="semibold", fontsize=9)
    axis.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    # the curves rise left to right, so the free corner is bottom right
    axis.legend(frameon=False, fontsize=6.8, loc="lower right")
    figure.tight_layout(pad=0.35)

    stem = HERE / "figures" / f"mining_depth_{args.factory.lower()}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".png"):
        figure.savefig(stem.with_suffix(extension), dpi=260,
                       bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    lines = [
        f"# RAVR against mining depth -- {args.dataset} {args.factory}", "",
        ("Every depth is averaged over the same seeds "
         f"({', '.join(str(s) for s in shared)}), so the curves are like for "
         "like." if args.seeds == "match" and len(data) > 1 else
         "Each depth is averaged over whatever seeds it has -- read the seed "
         "column before comparing rows."), "",
        "A curve that stops before the rightmost gridline did not spend its "
        "cap: the allocator ran out of items it has any signal for. "
        "`RAVR + unseen RD` fills that slack by reconstruction "
        "distortion, so the last column asks the question that "
        "matters: does a wider mining beat the fallback at the "
        "default depth?", "",
        "| Mining depth | Seeds | Target | Uniform | RAVR | RAVR B/item | "
        "Spent | RAVR + unseen RD | Widening minus fallback |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for depth in sorted(data):
        cells = data[depth]["cells"]
        seeds = len(data[depth]["seeds"])
        for target in sorted(cells):
            ravr = cells[target].get("ravr")
            uniform = cells[target].get("uniform")
            fallback = cells[target].get("ravr_unseen_rd")
            if not ravr or not uniform:
                continue
            rb = float(np.mean([r[0] for r in ravr]))
            cb = float(np.mean([r[0] for r in uniform]))
            rv = float(np.mean([r[1] for r in ravr]))
            fv = float(np.mean([r[1] for r in fallback])) if fallback else None
            # the honest comparison: this depth's RAVR against the DEFAULT
            # depth's fallback, since both reach the cap
            baseline = data.get(DEFAULT_DEPTH, {}).get("cells", {}).get(target, {})
            base_rows = baseline.get("ravr_unseen_rd")
            delta = (rv - float(np.mean([r[1] for r in base_rows]))
                     if base_rows else None)
            lines.append(
                f"| {depth} | {seeds} | {target} | "
                f"{np.mean([r[1] for r in uniform]):.4f} | {rv:.4f} | "
                f"{rb:.5f} | {'yes' if abs(rb - cb) < 1e-6 else 'NO'} | "
                f"{fv:.4f} | " if fv is not None else
                f"| {depth} | {seeds} | {target} | "
                f"{np.mean([r[1] for r in uniform]):.4f} | {rv:.4f} | "
                f"{rb:.5f} | {'yes' if abs(rb - cb) < 1e-6 else 'NO'} | - | "
            )
            lines[-1] += (f"{delta:+.4f} |" if delta is not None else "- |")
    summary = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"figure -> {stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
