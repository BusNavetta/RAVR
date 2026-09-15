"""Aggregate, bootstrap, and plot matched-codec allocator experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_baseline_claims import crossed_bootstrap_means, summarize_bootstrap
from benchmark_nested_additive_ravr import (
    DEFAULT_SEEDS,
    STRATEGIES,
    # `mining_depth.install` rebinds this name here as well as in the panel.
    # `panel_results_dir` reads it out of the panel's namespace, so that patch
    # still reaches this module; the import keeps the attribute it writes to.
    panel_output_dir,  # noqa: F401
    panel_results_dir,
)


DATASET_LABELS = {
    "sop": "Stanford Online Products, 100K gallery",
    "gldv2": "Google Landmarks Dataset v2.1, 100K gallery",
}


METRICS = ("teacher_recall_at_10", "teacher_ndcg_at_10")
COMPARATORS = (
    "uniform",
    "ordinal_per_exposure",
    "candidate_frequency",
    "query_ip_mse",
    "reconstruction",
    "random_histogram",
)
LABELS = {
    "uniform": "Uniform prefix",
    "ravr": "RAVR (Ordinal-KL)",
    "ravr_unseen_rd": "RAVR + unseen-item RD",
    "ordinal_per_exposure": "Ordinal-KL per exposure",
    "candidate_frequency": "Candidate frequency",
    "query_ip_mse": "QAQ-style query-IP MSE",
    "reconstruction": "Reconstruction RD",
    "random_histogram": "Random same histogram",
}


def _summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _validate_reports(
    reports: dict,
    factory: str,
    precision: str,
    seeds: tuple[int, ...],
    targets: tuple[int, ...],
    split: str,
    calibration_queries: int,
) -> str:
    """Reject stale or mixed-provenance artifacts before bootstrapping them."""
    backends = set()
    query_counts = set()
    expected_strategies = set(STRATEGIES)
    for seed in seeds:
        for target in targets:
            report = reports[(seed, target)]
            expected = {
                "seed": int(seed),
                "split": split,
                "frozen_codec": factory,
                "codebook_storage_precision": precision,
                "target_uniform_length": int(target),
                "calibration_queries": int(calibration_queries),
            }
            for field, value in expected.items():
                if report.get(field) != value:
                    raise ValueError(
                        f"artifact provenance mismatch for seed={seed}, target={target}: "
                        f"{field}={report.get(field)!r}, expected {value!r}"
                    )
            backend = report.get("compute_backend", {}).get("resolved")
            if backend not in {"cpu", "cuda"}:
                raise ValueError(
                    f"artifact lacks a resolved compute backend: seed={seed}, "
                    f"target={target}; rerun it with the current benchmark"
                )
            backends.add(backend)
            rows = report.get("rows", {})
            if set(rows) != expected_strategies:
                raise ValueError(
                    f"artifact strategy set mismatch for seed={seed}, target={target}"
                )
            if report.get("strategies") != list(STRATEGIES):
                raise ValueError(
                    f"artifact strategy order mismatch for seed={seed}, target={target}"
                )
            for strategy, row in rows.items():
                if row.get("codec_sha256") != report.get("codec_sha256"):
                    raise ValueError(
                        f"codec hash mismatch in {strategy}, seed={seed}, target={target}"
                    )
                row_backend = row.get("compute_backend", {}).get("resolved")
                if row_backend != backend:
                    raise ValueError(
                        f"backend mismatch in {strategy}, seed={seed}, target={target}: "
                        f"{row_backend!r} != {backend!r}"
                    )
                for metric in METRICS:
                    values = row.get("per_query", {}).get(metric)
                    if not isinstance(values, list) or not values:
                        raise ValueError(
                            f"missing per-query {metric} in {strategy}, seed={seed}, "
                            f"target={target}"
                        )
                    query_counts.add(len(values))
    if len(backends) != 1:
        raise ValueError(f"cannot aggregate mixed compute backends: {sorted(backends)}")
    if len(query_counts) != 1:
        raise ValueError(f"per-query result lengths differ: {sorted(query_counts)}")
    return next(iter(backends))


def aggregate(
    factory: str,
    precision: str,
    seeds: tuple[int, ...],
    targets: tuple[int, ...],
    split: str,
    calibration_queries: int,
    resamples: int,
    bootstrap_seed: int,
    dataset: str = "sop",
) -> dict:
    output = panel_results_dir(dataset)
    precision_tag = "" if precision == "fp32" else f"_{precision}"
    reports = {
        (seed, target): json.loads((
            output / f"{factory.lower()}{precision_tag}_seed{seed}_{split}_u{target}_cal{calibration_queries}.json"
        ).read_text(encoding="utf-8"))
        for seed in seeds for target in targets
    }
    compute_backend = _validate_reports(
        reports, factory, precision, seeds, targets, split, calibration_queries
    )
    cells, differences = [], []
    for target in targets:
        for metric in METRICS:
            for comparator in COMPARATORS:
                per_seed = []
                for seed in seeds:
                    rows = reports[(seed, target)]["rows"]
                    ravr = np.asarray(rows["ravr"]["per_query"][metric])
                    baseline = np.asarray(rows[comparator]["per_query"][metric])
                    per_seed.append(ravr - baseline)
                cells.append((target, metric, comparator))
                differences.append(np.stack(per_seed))
    cube = np.stack(differences, axis=2)
    bootstrap = crossed_bootstrap_means(
        cube, resamples=resamples, seed=bootstrap_seed
    )
    estimate, se, pointwise, simultaneous, critical = summarize_bootstrap(
        cube, bootstrap
    )

    comparisons = []
    for index, (target, metric, comparator) in enumerate(cells):
        seed_means = cube[:, :, index].mean(axis=1)
        comparisons.append({
            "target_length": target,
            "metric": metric,
            "ravr_minus": comparator,
            "estimate": float(estimate[index]),
            "bootstrap_se": float(se[index]),
            "pointwise_95_interval": [float(x) for x in pointwise[index]],
            "family_simultaneous_95_interval": [
                float(x) for x in simultaneous[index]
            ],
            "positive_seeds": int(np.sum(seed_means > 0)),
            "seed_differences": [float(x) for x in seed_means],
        })

    frontier = []
    for target in targets:
        first = reports[(seeds[0], target)]
        for strategy in STRATEGIES:
            rows = [reports[(seed, target)]["rows"][strategy] for seed in seeds]
            frontier.append({
                "target_length": target,
                "strategy": strategy,
                "persistent_bytes_per_item": _summary([
                    row["persistent_bytes_per_item"] for row in rows
                ]),
                "code_only_bytes_per_item": _summary([
                    row["code_only_bytes_per_item"] for row in rows
                ]),
                **{metric: _summary([row[metric] for row in rows]) for metric in METRICS},
                "mean_assigned_length": _summary([
                    row["mean_assigned_length"] for row in rows
                ]),
                "codec_sha256_by_seed": [row["codec_sha256"] for row in rows],
                "uniform_persistent_cap": int(
                    first.get("uniform_cap", first.get("exact_cap"))["persistent_bytes"]
                ),
            })

    result = {
        "dataset": DATASET_LABELS.get(dataset, dataset),
        "split": split,
        "seeds": list(seeds),
        "targets": list(targets),
        "calibration_queries": calibration_queries,
        "codec": f"same frozen {factory} within each seed and rate",
        "codebook_storage_precision": precision,
        "compute_backend": compute_backend,
        "strategies": list(STRATEGIES),
        "bootstrap": {
            "resamples": resamples,
            "seed": bootstrap_seed,
            "family": f"{len(cells)} target-metric-comparator cells",
            "simultaneous_half_width": float(critical),
        },
        "frontier": frontier,
        "comparisons": comparisons,
    }
    stem = (
        f"evaluation_{factory.lower()}{precision_tag}_{split}_cal{calibration_queries}"
    )
    (output / f"{stem}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / f"{stem}.md").write_text(render_markdown(result), encoding="utf-8")
    plot(result, output / f"{stem}_frontier")
    return result


def render_markdown(report: dict) -> str:
    lines = [
        f"# Frozen-{report['codec'].split()[2]} allocator comparison",
        "",
        f"Every row uses the {report['codec']}, source item codes, exact "
        f"packed-code cap, {report['codebook_storage_precision']} shared codebooks, "
        "uint32 IDs, float16 assigned-prefix norms, and bucket metadata.",
        "",
        "| Target | Strategy | Full B/item | R@10 | NDCG@10 |",
        "|---:|---|---:|---:|---:|",
    ]
    for target in report["targets"]:
        for strategy in STRATEGIES:
            row = next(value for value in report["frontier"] if (
                value["target_length"] == target and value["strategy"] == strategy
            ))
            lines.append(
                f"| {target} | {LABELS[strategy]} | "
                f"{row['persistent_bytes_per_item']['mean']:.5f} | "
                f"{row['teacher_recall_at_10']['mean']:.4f} | "
                f"{row['teacher_ndcg_at_10']['mean']:.4f} |"
            )
    lines.extend([
        "",
        "| Target | Metric | RAVR minus | Delta | Pointwise 95% | Family 95% | Positive seeds |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for row in report["comparisons"]:
        interval = row["pointwise_95_interval"]
        family = row["family_simultaneous_95_interval"]
        lines.append(
            f"| {row['target_length']} | {row['metric']} | {LABELS[row['ravr_minus']]} | "
            f"{row['estimate']:+.5f} | [{interval[0]:+.5f}, {interval[1]:+.5f}] | "
            f"[{family[0]:+.5f}, {family[1]:+.5f}] | "
            f"{row['positive_seeds']}/{len(report['seeds'])} |"
        )
    return "\n".join(lines) + "\n"


def plot(report: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = {
        "ravr": dict(color="#111827", marker="*", linestyle="-", linewidth=1.25, markersize=7.0, zorder=5),
        "uniform": dict(color="#e67e22", marker="o", linestyle="--", linewidth=0.95, markersize=3.8, zorder=3),
        "query_ip_mse": dict(color="#2563eb", marker="D", linestyle="--", linewidth=0.95, markersize=3.5, zorder=3),
        "reconstruction": dict(color="#228b22", marker="^", linestyle="--", linewidth=0.95, markersize=4.0, zorder=3),
        "random_histogram": dict(color="#9ca3af", marker="s", linestyle=":", linewidth=0.9, markersize=3.2, zorder=2),
        "ordinal_per_exposure": dict(color="#7c3aed", marker="P", linestyle="--", linewidth=0.9, markersize=3.5, zorder=3),
        "candidate_frequency": dict(color="#b45309", marker="v", linestyle=":", linewidth=0.9, markersize=3.7, zorder=3),
        "ravr_unseen_rd": dict(color="#dc2626", marker="X", linestyle="-.", linewidth=0.9, markersize=3.7, zorder=4),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), sharex=True)
    for axis, metric, title in zip(
        axes, METRICS, ("(a) Set fidelity", "(b) Ordered fidelity")
    ):
        for strategy in STRATEGIES:
            rows = sorted(
                (row for row in report["frontier"] if row["strategy"] == strategy),
                key=lambda row: row["persistent_bytes_per_item"]["mean"],
            )
            x = [row["persistent_bytes_per_item"]["mean"] for row in rows]
            y = [row[metric]["mean"] for row in rows]
            error = [row[metric]["std"] for row in rows]
            style = styles[strategy]
            axis.errorbar(
                x, y, yerr=error, capsize=1.7, capthick=0.65,
                elinewidth=0.65, markeredgewidth=0.55,
                markeredgecolor="white" if strategy == "ravr" else style["color"],
                **style,
            )
        axis.set_title(title, pad=3.0, weight="semibold")
        axis.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Complete serialized storage (B/item)", labelpad=2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Teacher Recall@10", labelpad=2)
    axes[1].set_ylabel("Teacher NDCG@10", labelpad=2)
    handles = [
        Line2D([0], [0], label=LABELS[name], **{
            key: value for key, value in styles[name].items() if key != "zorder"
        }) for name in STRATEGIES
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.055),
        ncol=4, frameon=False, columnspacing=0.9, handlelength=1.9,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.67, wspace=0.24)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sop")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--targets", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--resamples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    args = parser.parse_args()
    result = aggregate(
        args.factory, args.precision, tuple(args.seeds), tuple(args.targets), args.split,
        args.calibration_queries, args.resamples, args.bootstrap_seed,
        dataset=args.dataset,
    )
    print(render_markdown(result), flush=True)


if __name__ == "__main__":
    main()
