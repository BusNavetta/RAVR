from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent

MARKERS = {
    "ravr": ("#111827", "*", 13, "RAVR (Ordinal-KL)"),
    "ravr_unseen_rd": ("#dc2626", "X", 8, "RAVR + unseen RD"),
    "candidate_frequency": ("#0891b2", "v", 7, "Candidate frequency"),
    "ordinal_per_exposure": ("#7c3aed", "P", 7, "Ordinal-KL per exposure"),
    "query_ip_mse": ("#2563eb", "D", 6, "Query-IP MSE"),
    "reconstruction": ("#228b22", "^", 7, "Reconstruction RD"),
    "random_histogram": ("#6b7280", "s", 6, "RAVR histogram, shuffled"),
    "uniform": ("#e67e22", "o", 7, "Uniform prefix"),
}
ORDER = ("uniform", "random_histogram", "reconstruction", "query_ip_mse",
         "ordinal_per_exposure", "candidate_frequency", "ravr_unseen_rd", "ravr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--factory", default="lsq16x4")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    path = args.report or (
        HERE / f"random_allocations_gldv2_{args.factory}_seed{args.seed}.json"
    )
    if not path.exists():
        raise SystemExit(f"missing report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    targets = sorted(int(t) for t in report["targets"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, len(targets), figsize=(2.5 * len(targets), 3.4),
                                sharey=False)
    if len(targets) == 1:
        axes = [axes]

    for axis, target in zip(axes, targets):
        entry = report["targets"][str(target)]
        # Raw draws live in a sidecar .npy; at a million per cap they are not
        # something to carry through the report.
        sidecar = path.with_name(entry["random"]["samples_file"])
        if not sidecar.exists():
            raise SystemExit(f"missing draws for target {target}: {sidecar}")
        samples = np.load(sidecar).astype(np.float64)
        # A violin of a million points is slow and no more informative than a
        # large random subset of them.
        if len(samples) > 200_000:
            samples = np.random.default_rng(0).choice(samples, 200_000,
                                                      replace=False)
        parts = axis.violinplot(samples, positions=[0], widths=0.85,
                                showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor("#d1d5db")
            body.set_edgecolor("#9ca3af")
            body.set_alpha(0.85)
        low, high = samples.min(), samples.max()
        axis.plot([0, 0], [low, high], color="#6b7280", linewidth=0.9, zorder=2)
        for level in (low, high):
            axis.plot([-0.12, 0.12], [level, level], color="#6b7280",
                      linewidth=0.9, zorder=2)

        named = entry["named"]
        for name in ORDER:
            if name not in named:
                continue
            colour, marker, size, _ = MARKERS[name]
            value = named[name]["recall_at_10"]
            axis.plot([0], [value], marker=marker, color=colour,
                      markersize=size, markeredgecolor="white",
                      markeredgewidth=0.6, linestyle="none", zorder=6)
        axis.set_xticks([])
        axis.set_xlim(-0.7, 0.7)
        axis.set_title(f"cap {target} stages\n{entry['draws_completed']:,} draws",
                       fontsize=8, pad=4.0, weight="semibold")
        axis.grid(True, axis="y", color="#e5e7eb", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "bottom"]].set_visible(False)
    axes[0].set_ylabel("Teacher Recall@10", labelpad=2)

    handles = [
        Line2D([0], [0], marker=MARKERS[name][1], color=MARKERS[name][0],
               markersize=MARKERS[name][2] * 0.7, linestyle="none",
               label=MARKERS[name][3])
        for name in ORDER if name in report["targets"][str(targets[0])]["named"]
    ]
    handles.append(Line2D([0], [0], color="#9ca3af", linewidth=6, alpha=0.85,
                          label="random legal allocations"))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.10),
                  ncol=3, frameon=False, fontsize=7, handlelength=1.6)
    figure.suptitle(
        f"{report.get('dataset', 'GLDv2 100K')} -- where the named allocators "
        "fall among all allocations at the same cap",
        y=1.20, fontsize=8.5, weight="semibold",
    )
    figure.tight_layout(pad=0.4)
    stem = HERE / "figures" / f"random_allocation_range_{args.factory}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".png"):
        figure.savefig(stem.with_suffix(extension), dpi=260,
                       bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)

    lines = [
        f"# Random allocation range -- {report.get('dataset', 'GLDv2 100K')}", "",
        "Every row is scored at the identical complete-file cap. The random "
        "range is the distribution over legal per-item prefix assignments at "
        "that cap; the percentile says what fraction of random allocations a "
        "strategy beats.", "",
        "| Cap | Draws | Random min | Random median | Random max | Strategy | "
        "R@10 | Percentile |",
        "|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for target in targets:
        entry = report["targets"][str(target)]
        r = entry["random"]
        first = True
        for name in ORDER:
            if name not in entry["named"]:
                continue
            cell = entry["named"][name]
            head = (f"| {target} | {entry['draws_completed']:,} | {r['min']:.4f} | "
                    f"{r['p50']:.4f} | {r['max']:.4f} " if first else "| | | | | ")
            beats = cell["percentile_in_random_range"]
            lines.append(
                f"{head}| {MARKERS[name][3]} | {cell['recall_at_10']:.4f} | "
                f"{beats:.2f}%{'' if cell['spends_the_cap'] else ' (under cap)'} |"
            )
            first = False
    lines += [
        "",
        "A percentile of 100 means no random allocation matched it in this "
        "sweep; it is a lower bound on how extreme the strategy is, limited by "
        "the number of draws.",
        "",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"figure -> {stem.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
