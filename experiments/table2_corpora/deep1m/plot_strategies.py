from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

from aggregate_nested_additive_ravr import LABELS  # noqa: E402
from benchmark_nested_additive_ravr import panel_results_dir  # noqa: E402

SLUG = "deep1m"
METRICS = (
    ("teacher_recall_at_10", "Teacher Recall@10", "(a) Set fidelity"),
    ("teacher_ndcg_at_10", "Teacher NDCG@10", "(b) Ordered fidelity"),
)
STYLES = {
    "ravr": dict(color="#111827", marker="*", linestyle="-", linewidth=1.3,
                 markersize=7.5, zorder=6),
    "ravr_unseen_rd": dict(color="#dc2626", marker="X", linestyle="-.",
                           linewidth=1.1, markersize=5.5, zorder=5),
    "ordinal_per_exposure": dict(color="#7c3aed", marker="P", linestyle="--",
                                 linewidth=0.95, markersize=4.5, zorder=4),
    "candidate_frequency": dict(color="#0891b2", marker="v", linestyle="--",
                                linewidth=0.95, markersize=4.2, zorder=4),
    "query_ip_mse": dict(color="#2563eb", marker="D", linestyle="--",
                         linewidth=0.95, markersize=3.8, zorder=3),
    "reconstruction": dict(color="#228b22", marker="^", linestyle="--",
                           linewidth=0.95, markersize=4.2, zorder=3),
    "random_histogram": dict(color="#9ca3af", marker="s", linestyle=":",
                             linewidth=0.9, markersize=3.4, zorder=2),
    "uniform": dict(color="#e67e22", marker="o", linestyle="--",
                    linewidth=0.95, markersize=4.0, zorder=3),
}
ORDER = ("uniform", "random_histogram", "reconstruction", "query_ip_mse",
         "candidate_frequency", "ordinal_per_exposure", "ravr_unseen_rd", "ravr")


def report_path(slug: str, factory: str, split: str, cal: int) -> Path:
    return (
        panel_results_dir(slug)
        / f"evaluation_{factory.lower()}_{split}_cal{cal}.json"
    )


def plot(report: dict, stem: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frontier = {}
    for row in report["frontier"]:
        frontier.setdefault(row["strategy"], []).append(row)
    present = [name for name in ORDER if name in frontier]

    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    for axis, (metric, ylabel, panel_title) in zip(axes, METRICS):
        for name in present:
            rows = sorted(frontier[name],
                          key=lambda r: r["persistent_bytes_per_item"]["mean"])
            axis.errorbar(
                [r["persistent_bytes_per_item"]["mean"] for r in rows],
                [r[metric]["mean"] for r in rows],
                yerr=[r[metric]["std"] for r in rows],
                capsize=1.7, capthick=0.65, elinewidth=0.65,
                markeredgewidth=0.5,
                markeredgecolor="white" if name == "ravr" else STYLES[name]["color"],
                **STYLES[name],
            )
        axis.set_title(panel_title, pad=3.0, weight="semibold")
        axis.set_xlabel("Complete serialized storage (B/item)", labelpad=2)
        axis.set_ylabel(ylabel, labelpad=2)
        axis.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([0], [0], label=LABELS.get(name, name),
               **{k: v for k, v in STYLES[name].items() if k != "zorder"})
        for name in present
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13),
                  ncol=4, frameon=False, columnspacing=0.9, handlelength=2.0,
                  fontsize=7)
    figure.suptitle(title, y=1.20, fontsize=9, weight="semibold")
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.80,
                           wspace=0.26)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".png"):
        figure.savefig(stem.with_suffix(extension), dpi=260,
                       bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def render(report: dict, present: list[str]) -> str:
    frontier = {}
    for row in report["frontier"]:
        frontier[(row["strategy"], row["target_length"])] = row
    targets = sorted({row["target_length"] for row in report["frontier"]})
    lines = [
        f"# Allocator comparison -- {report['dataset']} {report['codec']}", "",
        f"{len(report['seeds'])} codec seeds. `B/item` is the complete "
        "serialized file, so a strategy that reports **less than the uniform "
        "row at the same target has not spent its budget**.", "",
        "| Target | Strategy | B/item | R@10 | NDCG@10 |",
        "|---:|---|---:|---:|---:|",
    ]
    for target in targets:
        cap = frontier[("uniform", target)]["persistent_bytes_per_item"]["mean"]
        for name in present:
            row = frontier.get((name, target))
            if row is None:
                continue
            bytes_per_item = row["persistent_bytes_per_item"]["mean"]
            flag = " (under)" if bytes_per_item < cap - 1e-6 else ""
            lines.append(
                f"| {target} | {LABELS.get(name, name)} | "
                f"{bytes_per_item:.5f}{flag} | "
                f"{row['teacher_recall_at_10']['mean']:.4f} | "
                f"{row['teacher_ndcg_at_10']['mean']:.4f} |"
            )
    lines += ["", "`(under)` marks a row that did not spend its cap.", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=SLUG)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    path = report_path(args.dataset, args.factory, args.split,
                       args.calibration_queries)
    if not path.exists():
        raise SystemExit(f"missing aggregate report: {path}\n"
                         "Run the panel and aggregate stages first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    frontier = {row["strategy"] for row in report["frontier"]}
    present = [name for name in ORDER if name in frontier]

    stem = HERE / "figures" / f"strategy_frontier_{args.factory.lower()}"
    plot(report, stem, args.title or report["dataset"])
    stem.with_suffix(".md").write_text(render(report, present), encoding="utf-8")
    print(render(report, present), flush=True)
    print(f"figure -> {stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
