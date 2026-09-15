from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent

from aggregate_nested_additive_ravr import (  # noqa: E402
    LABELS, METRICS, aggregate,
)
from benchmark_faiss_fixed_rate import _load_dataset  # noqa: E402
from benchmark_nested_additive_ravr import (  # noqa: E402
    panel_results_dir, rerank_candidates_for, run_seed,
)
from compute_backend import backend_report, configure_compute_backend  # noqa: E402
from faiss_additive_codecs import AdditiveCodecConfig, train_additive_bundle  # noqa: E402
from faiss_fixed_rate import FaissCodecBundle  # noqa: E402

REQUIRED = (
    "gldv2_100k_manifest.npz",
    "gldv2_100k_index_dinov2_vits14.npy",
    "gldv2_test_queries_dinov2_vits14.npy",
)
DEFAULT_SEEDS = (17, 42, 73, 3067, 4294, 4996, 5423, 7520, 7937, 9794)
# Frontier figures show the set-fidelity metric; the ordered metric is secondary.
HEADLINE = METRICS[0]


def require_files(data_root: Path) -> Path:
    root = data_root / "gldv2"
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise SystemExit(
            f"missing GLDv2 inputs in {root}:\n"
            + "\n".join(f"  {name}" for name in missing)
            + "\n\nEither ask William for these three files, or build them with:"
            "\n  python 4_gldv2/download.py --root <images>"
            f"\n  python 4_gldv2/prepare.py --root <images> --data-root {data_root}"
        )
    return root


def stage_codecs(data_root: Path, factory: str, seeds, train_samples: int,
                 effort: str, force: bool) -> None:
    data = _load_dataset("gldv2", data_root)
    bundle_dir = data["root"] / "advanced_codec_bundles"
    bundle_dir.mkdir(exist_ok=True)
    family = "lsq" if factory.lower().startswith("lsq") else "rq"
    screen = {
        "rq_kmeans_iterations": 3, "rq_refine_iterations": 2, "rq_beam_size": 3,
        "lsq_train_iterations": 8, "lsq_train_ils_iterations": 3,
        "lsq_encode_ils_iterations": 6, "lsq_icm_iterations": 3,
        "lsq_perturbations": 2,
    }

    suffix = "" if effort == "full" else f"_{effort}"
    for seed in seeds:
        path = bundle_dir / f"{factory.lower()}_seed{seed}{suffix}.fcix"
        if path.exists() and not force:
            cached = FaissCodecBundle.load(path)
            recorded = cached.metadata.get("effort")
            cached.close()
            if recorded != effort:
                raise SystemExit(
                    f"{path.name} was trained with effort={recorded!r} but "
                    f"effort={effort!r} was requested. Retrain it with --force "
                    "or delete the file; reusing it would compare codecs of "
                    "different quality."
                )
            print(f"codec  {path.name}: cached (effort={recorded})", flush=True)
            continue
        config = AdditiveCodecConfig(
            family=family, stages=16, nbits=4, seed=seed,
            **(screen if effort == "screen" else {}),
        )
        rng = np.random.default_rng(seed)
        train_ids = rng.choice(
            data["codec_pool"], min(train_samples, len(data["codec_pool"])),
            replace=False,
        )
        started = time.perf_counter()
        print(
            f"codec  {factory} seed {seed}: training on {len(train_ids)} vectors "
            f"(effort={effort}, faiss runs on CPU)", flush=True,
        )
        bundle = train_additive_bundle(data["gallery"], train_ids, config)
        bundle.metadata.update({"dataset": data["dataset"], "effort": effort})
        bundle.save(path)
        bundle.close()
        print(f"       -> {path.name} in {time.perf_counter() - started:.1f}s",
              flush=True)
    for seed in seeds:
        bundle = FaissCodecBundle.load(
            bundle_dir / f"{factory.lower()}_seed{seed}{suffix}.fcix"
        )
        metadata = bundle.metadata
        bundle.close()
        if metadata.get("factory") != factory:
            raise AssertionError(f"seed {seed} bundle is not {factory}")
        if metadata.get("effort") != effort:
            raise AssertionError(
                f"seed {seed} bundle records effort={metadata.get('effort')!r}, "
                f"not {effort!r}"
            )


def stage_panel(data_root: Path, factory: str, seeds, targets, split: str,
                calibration_queries: int, force: bool) -> None:
    print(
        f"panel  candidate depth for gldv2: "
        f"{rerank_candidates_for('gldv2')} (SOP uses {rerank_candidates_for('sop')})",
        flush=True,
    )
    for seed in seeds:
        for target in targets:
            run_seed(
                data_root, seed, split=split,
                calibration_queries=calibration_queries, target_length=target,
                factory=factory, precision="fp32", force_calibration=force,
                dataset="gldv2",
            )


def stage_aggregate(factory: str, seeds, targets, split: str,
                    calibration_queries: int, resamples: int) -> dict:
    return aggregate(
        factory, "fp32", tuple(seeds), tuple(targets), split,
        calibration_queries, resamples, 20260827, dataset="gldv2",
    )


def _report_path(dataset: str, factory: str, split: str, cal: int) -> Path:
    return (
        panel_results_dir(dataset)
        / f"evaluation_{factory.lower()}_{split}_cal{cal}.json"
    )


def _frontier(report: dict, strategy: str, metric: str):
    rows = sorted(
        (row for row in report["frontier"] if row["strategy"] == strategy),
        key=lambda row: row["persistent_bytes_per_item"]["mean"],
    )
    return (
        [row["persistent_bytes_per_item"]["mean"] for row in rows],
        [row[metric]["mean"] for row in rows],
        [row[metric]["std"] for row in rows],
    )


def stage_compare(sop: Path, gldv2: Path, stem: Path) -> dict:
    """Side-by-side frontier and allocator-gain figures for the two corpora."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    reports = {}
    for name, path in (("SOP", sop), ("GLDv2", gldv2)):
        if not path.exists():
            raise SystemExit(
                f"missing aggregate report for {name}: {path}\n"
                "Run the panel and aggregate stages for that dataset first."
            )
        reports[name] = json.loads(path.read_text(encoding="utf-8"))

    styles = {
        "ravr": dict(color="#111827", marker="*", linestyle="-", linewidth=1.25,
                     markersize=7.0, zorder=5),
        "uniform": dict(color="#e67e22", marker="o", linestyle="--", linewidth=0.95,
                        markersize=3.8, zorder=3),
        "query_ip_mse": dict(color="#2563eb", marker="D", linestyle="--",
                             linewidth=0.95, markersize=3.5, zorder=3),
        "reconstruction": dict(color="#228b22", marker="^", linestyle="--",
                               linewidth=0.95, markersize=4.0, zorder=3),
        "random_histogram": dict(color="#9ca3af", marker="s", linestyle=":",
                                 linewidth=0.9, markersize=3.2, zorder=2),
    }
    ranges = []
    for report in reports.values():
        values = [
            row["persistent_bytes_per_item"]["mean"] for row in report["frontier"]
        ]
        ranges.append((min(values), max(values)))
    (low_a, high_a), (low_b, high_b) = ranges
    share_x = max(low_a, low_b) <= min(high_a, high_b)
    if not share_x:
        print(
            f"note: byte caps do not overlap "
            f"({low_a:.2f}-{high_a:.2f} vs {low_b:.2f}-{high_b:.2f} B/item); "
            "using per-panel x limits. Absolute frontiers are not comparable "
            "across these two runs -- read the gain figure instead.",
            flush=True,
        )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.55), sharex=share_x, sharey=True
    )
    for axis, (name, report) in zip(axes, reports.items()):
        for strategy, style in styles.items():
            x, y, error = _frontier(report, strategy, HEADLINE)
            if not x:
                continue
            axis.errorbar(
                x, y, yerr=error, capsize=1.7, capthick=0.65, elinewidth=0.65,
                markeredgewidth=0.55,
                markeredgecolor="white" if strategy == "ravr" else style["color"],
                **style,
            )
        seeds = len(report["seeds"])
        axis.set_title(f"({'ab'[list(reports).index(name)]}) {name}, {seeds} seeds",
                       pad=3.0, weight="semibold")
        axis.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Complete serialized storage (B/item)", labelpad=2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Teacher Recall@10", labelpad=2)
    handles = [
        Line2D([0], [0], label=LABELS[name],
               **{k: v for k, v in style.items() if k != "zorder"})
        for name, style in styles.items()
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.055),
               ncol=3, frameon=False, columnspacing=0.9, handlelength=1.9)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.67, wspace=0.24)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=260, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)

    # Second figure: the headline allocator gain, dataset against dataset.
    gains = {}
    for name, report in reports.items():
        rows = [
            row for row in report["comparisons"]
            if row["ravr_minus"] == "uniform" and row["metric"] == HEADLINE
        ]
        rows.sort(key=lambda row: row["target_length"])
        gains[name] = rows
    fig, axis = plt.subplots(figsize=(3.6, 2.55))
    for name, colour, marker in (("SOP", "#111827", "*"), ("GLDv2", "#2563eb", "D")):
        rows = gains[name]
        if not rows:
            continue
        x = [row["target_length"] for row in rows]
        y = [row["estimate"] for row in rows]
        low = [row["estimate"] - row["family_simultaneous_95_interval"][0]
               for row in rows]
        high = [row["family_simultaneous_95_interval"][1] - row["estimate"]
                for row in rows]
        axis.errorbar(x, y, yerr=[low, high], color=colour, marker=marker,
                      linestyle="-", linewidth=1.1, markersize=6.0, capsize=2.2,
                      capthick=0.7, elinewidth=0.7, label=name)
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Uniform-equivalent prefix (stages)", labelpad=2)
    axis.set_ylabel("RAVR minus uniform, R@10", labelpad=2)
    axis.set_title("Allocator gain, family-wise 95% bands", pad=3.0, weight="semibold")
    axis.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    gain_stem = stem.with_name(stem.name + "_gain")
    for suffix in (".pdf", ".png"):
        fig.savefig(gain_stem.with_suffix(suffix), dpi=260, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)

    lines = [
        "# SOP vs GLDv2, frozen-codec allocator", "",
        "| Dataset | Target | RAVR - uniform R@10 | Family 95% | Positive seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rows in gains.items():
        for row in rows:
            low, high = row["family_simultaneous_95_interval"]
            lines.append(
                f"| {name} | {row['target_length']} | {row['estimate']:+.5f} | "
                f"[{low:+.5f}, {high:+.5f}] | {row['positive_seeds']}"
                f"/{len(row['seed_differences'])} |"
            )
    summary = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"figures -> {stem.with_suffix('.png')}", flush=True)
    print(f"           {gain_stem.with_suffix('.png')}", flush=True)
    return gains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--targets", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--effort", choices=("screen", "full"), default="full")
    parser.add_argument("--resamples", type=int, default=100000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stages", nargs="+", default=["codecs", "panel", "aggregate", "compare"],
        choices=("codecs", "panel", "aggregate", "compare"),
    )
    parser.add_argument("--sop-report", type=Path, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    stages = set(args.stages)

    if "panel" in stages and args.effort != "full":
        raise SystemExit(
            f"--effort {args.effort} trains screening codecs, which the panel "
            "does not load. Use --stages codecs to screen, and rerun the panel "
            "with --effort full."
        )
    if stages - {"compare"}:
        require_files(data_root)
        resolved = configure_compute_backend(args.device)
        print(f"compute backend: {resolved} ({backend_report()})", flush=True)

    if "codecs" in stages:
        stage_codecs(data_root, args.factory, args.seeds, args.train_samples,
                     args.effort, args.force)
    if "panel" in stages:
        stage_panel(data_root, args.factory, args.seeds, args.targets, args.split,
                    args.calibration_queries, args.force)
    if "aggregate" in stages:
        stage_aggregate(args.factory, args.seeds, args.targets, args.split,
                        args.calibration_queries, args.resamples)
    if "compare" in stages:
        sop = args.sop_report or _report_path(
            "sop", args.factory, args.split, args.calibration_queries
        )
        gldv2 = _report_path("gldv2", args.factory, args.split,
                             args.calibration_queries)
        stage_compare(sop, gldv2, HERE / "figures" / "sop_vs_gldv2")


if __name__ == "__main__":
    sys.exit(main())
