"""Crossed seed-query inference for the matched-size RAVR comparisons.

"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs" / "baseline_claim_audit"
DATASETS = ("sop", "gldv2")
METRICS = ("teacher_recall_at_10", "teacher_ndcg_at_10")
METRIC_LABELS = {
    "teacher_recall_at_10": "Teacher R@10",
    "teacher_ndcg_at_10": "Teacher NDCG@10",
}


@dataclass(frozen=True)
class Comparison:
    dataset: str
    baseline: str
    metric: str
    selected_depth: int

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.baseline}:{self.metric}"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_by_name(rows: Iterable[dict], name: str) -> dict:
    matches = [row for row in rows if row.get("display_name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one row named {name!r}, found {len(matches)}")
    return matches[0]


def _point_by_name(depth_report: dict, name: str) -> dict:
    matches = [
        point for point in depth_report["points"]
        if point.get("external_display_name") == name and point.get("feasible")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one feasible point named {name!r} at depth "
            f"{depth_report['depth']}, found {len(matches)}"
        )
    return matches[0]


def load_difference_cube(
    outputs: Path = HERE / "outputs",
    datasets: Iterable[str] = DATASETS,
) -> tuple[np.ndarray, list[Comparison], list[int]]:
    """Return paired RAVR-minus-baseline values as [seed, query, cell]."""
    cells: list[Comparison] = []
    columns: list[np.ndarray] = []
    reference_seeds: list[int] | None = None
    reference_queries: int | None = None

    for dataset in datasets:
        aggregate = _read(
            outputs / f"ravr_depth_ablation_{dataset}_100k" / "evaluation.json"
        )
        selected = {
            row["external_display_name"]: int(row["selected_depth"])
            for row in aggregate["budget_selected_rows"]
        }
        seeds = [int(seed) for seed in aggregate["seeds"]]
        if reference_seeds is None:
            reference_seeds = seeds
        elif reference_seeds != seeds:
            raise ValueError("datasets do not use the same ordered codec seeds")

        fixed_reports = {
            seed: _read(outputs / f"faiss_fixed_rate_{dataset}_100k" / f"seed{seed}.json")
            for seed in seeds
        }
        depth_reports = {
            seed: _read(outputs / f"ravr_depth_ablation_{dataset}_100k" / f"seed{seed}.json")
            for seed in seeds
        }

        for baseline, depth in selected.items():
            per_seed_ravr: list[dict] = []
            per_seed_external: list[dict] = []
            for seed in seeds:
                depth_matches = [
                    item for item in depth_reports[seed]["depths"]
                    if int(item["depth"]) == depth
                ]
                if len(depth_matches) != 1:
                    raise ValueError(f"missing depth {depth} for {dataset}, seed {seed}")
                per_seed_ravr.append(_point_by_name(depth_matches[0], baseline))
                per_seed_external.append(_row_by_name(fixed_reports[seed]["rows"], baseline))

            for metric in METRICS:
                paired = []
                for ravr, external in zip(per_seed_ravr, per_seed_external, strict=True):
                    a = np.asarray(ravr["per_query"][metric], dtype=np.float64)
                    b = np.asarray(external["per_query"][metric], dtype=np.float64)
                    if a.shape != b.shape or a.ndim != 1:
                        raise ValueError(f"unaligned per-query values for {dataset}/{baseline}")
                    paired.append(a - b)
                array = np.stack(paired)
                if reference_queries is None:
                    reference_queries = int(array.shape[1])
                elif reference_queries != int(array.shape[1]):
                    raise ValueError("comparisons do not use the same query count")
                columns.append(array)
                cells.append(Comparison(dataset, baseline, metric, depth))

    if reference_seeds is None or not columns:
        raise ValueError("no comparison cells found")
    return np.stack(columns, axis=2), cells, reference_seeds


def crossed_bootstrap_means(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Resample seed and query indices independently, preserving all pairings."""
    if differences.ndim != 3:
        raise ValueError("differences must have shape [seed, query, cell]")
    n_seeds, n_queries, n_cells = differences.shape
    if n_seeds < 2 or n_queries < 2 or n_cells < 1:
        raise ValueError("bootstrap requires >=2 seeds, >=2 queries, and >=1 cell")
    rng = np.random.default_rng(seed)
    result = np.empty((resamples, n_cells), dtype=np.float64)

    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        size = stop - start
        seed_draws = rng.integers(0, n_seeds, size=(size, n_seeds))
        query_draws = rng.integers(0, n_queries, size=(size, n_queries))
        seed_counts = np.stack(
            [(seed_draws == index).sum(axis=1) for index in range(n_seeds)], axis=1
        ).astype(np.float64)
        batch = np.zeros((size, n_cells), dtype=np.float64)
        for index in range(n_seeds):
            query_means = differences[index][query_draws].mean(axis=1)
            batch += query_means * (seed_counts[:, index] / n_seeds)[:, None]
        result[start:stop] = batch
    return result


def summarize_bootstrap(
    differences: np.ndarray,
    bootstrap: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return estimates, SEs, pointwise CIs, simultaneous bands, and half-width.

    The simultaneous band uses the 95th percentile of the largest absolute
    centered bootstrap error over the whole comparison family.  It therefore
    has one common half-width and exactly reproduces the inspected audit's
    simultaneous-band definition.
    """
    estimate = differences.mean(axis=(0, 1))
    standard_error = bootstrap.std(axis=0, ddof=1)
    alpha = 1.0 - confidence
    pointwise = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0).T
    critical = float(
        np.quantile(np.abs(bootstrap - estimate[None, :]).max(axis=1), confidence)
    )
    simultaneous = np.column_stack(
        [estimate - critical, estimate + critical]
    )
    return estimate, standard_error, pointwise, simultaneous, critical


def _fmt_interval(values: Iterable[float]) -> str:
    lo, hi = values
    return f"[{lo:+.5f}, {hi:+.5f}]"


def build_report(
    differences: np.ndarray,
    cells: list[Comparison],
    seeds: list[int],
    bootstrap: np.ndarray,
    *,
    confidence: float,
    bootstrap_seed: int,
) -> dict:
    estimate, se, pointwise, simultaneous, critical = summarize_bootstrap(
        differences, bootstrap, confidence=confidence
    )
    rows = []
    for index, cell in enumerate(cells):
        seed_means = differences[:, :, index].mean(axis=1)
        seed_half_width = float(
            student_t.ppf(0.5 + confidence / 2.0, len(seed_means) - 1)
            * seed_means.std(ddof=1) / np.sqrt(len(seed_means))
        )
        rows.append({
            "key": cell.key,
            "dataset": cell.dataset,
            "baseline": cell.baseline,
            "metric": cell.metric,
            "metric_label": METRIC_LABELS[cell.metric],
            "selected_depth": cell.selected_depth,
            "difference": float(estimate[index]),
            "bootstrap_se": float(se[index]),
            "pointwise_interval": [float(value) for value in pointwise[index]],
            "simultaneous_interval": [float(value) for value in simultaneous[index]],
            "pointwise_positive": bool(pointwise[index, 0] > 0),
            "simultaneous_positive": bool(simultaneous[index, 0] > 0),
            "positive_seed_means": int((seed_means > 0).sum()),
            "seed_means": [float(value) for value in seed_means],
            "codec_seed_t_interval": [
                float(estimate[index] - seed_half_width),
                float(estimate[index] + seed_half_width),
            ],
            "codec_seed_t_interval_scope": (
                "exploratory unadjusted t interval over three codec-seed means; "
                "queries treated as the fixed locked panel"
            ),
        })
    return {
        "protocol": {
            "estimand": "mean paired per-query RAVR-minus-baseline difference",
            "resampling": "crossed codec-seed and locked-query bootstrap",
            "family": "2 datasets x 9 baselines x 2 metrics",
            "simultaneous_method": (
                "single-step maximum absolute centered bootstrap deviation over all cells"
            ),
            "confidence": confidence,
            "bootstrap_resamples": int(bootstrap.shape[0]),
            "bootstrap_seed": bootstrap_seed,
            "codec_seeds": seeds,
            "queries_per_seed": int(differences.shape[1]),
            "comparison_cells": len(cells),
        },
        "simultaneous_half_width": critical,
        "counts": {
            "positive_pointwise": sum(row["pointwise_positive"] for row in rows),
            "positive_simultaneous": sum(row["simultaneous_positive"] for row in rows),
            "negative_pointwise": sum(row["pointwise_interval"][1] < 0 for row in rows),
            "negative_simultaneous": sum(row["simultaneous_interval"][1] < 0 for row in rows),
        },
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    protocol = report["protocol"]
    counts = report["counts"]
    lines = [
        "# Crossed seed-query baseline audit",
        "",
        "The estimand is the paired per-query RAVR-minus-baseline difference. "
        "Codec seeds and locked query IDs are independently resampled, while every "
        "method pairing is retained. The simultaneous intervals use the 95th "
        "percentile of the largest absolute centered bootstrap error across all 36 "
        "comparisons.",
        "",
        f"- Seeds: `{protocol['codec_seeds']}`; queries/seed: "
        f"`{protocol['queries_per_seed']}`.",
        f"- Bootstrap resamples: `{protocol['bootstrap_resamples']}`; simultaneous "
        f"half-width: `{report['simultaneous_half_width']:.7f}`.",
        f"- Positive pointwise / simultaneous cells: "
        f"`{counts['positive_pointwise']}` / `{counts['positive_simultaneous']}` of "
        f"`{protocol['comparison_cells']}`.",
        f"- Negative pointwise / simultaneous cells: "
        f"`{counts['negative_pointwise']}` / `{counts['negative_simultaneous']}` of "
        f"`{protocol['comparison_cells']}`.",
        "",
        "| Dataset | Baseline | Metric | T | Difference | Pointwise 95% CI | Simultaneous 95% band | Seed-only t CI | Seed signs |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['dataset'].upper()} | {row['baseline']} | {row['metric_label']} "
            f"| {row['selected_depth']} | {row['difference']:+.5f} "
            f"| {_fmt_interval(row['pointwise_interval'])} "
            f"| {_fmt_interval(row['simultaneous_interval'])} "
            f"| {_fmt_interval(row['codec_seed_t_interval'])} "
            f"| {row['positive_seed_means']}/3 |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=HERE / "outputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    differences, cells, seeds = load_difference_cube(args.outputs)
    bootstrap = crossed_bootstrap_means(
        differences,
        resamples=args.resamples,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    report = build_report(
        differences,
        cells,
        seeds,
        bootstrap,
        confidence=args.confidence,
        bootstrap_seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "crossed_seed_query_bootstrap.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "crossed_seed_query_bootstrap.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(render_markdown(report))


if __name__ == "__main__":
    main()
