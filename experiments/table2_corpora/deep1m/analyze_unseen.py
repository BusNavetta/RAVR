from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent

from benchmark_faiss_fixed_rate import _load_dataset  # noqa: E402
from benchmark_nested_additive_ravr import (  # noqa: E402
    ALLOWED_LENGTHS, _state_path,
)

SLUG = "deep1m"
DEFAULT_SEEDS = (17, 42, 73, 3067, 4294, 4996, 5423, 7520, 7937, 9794)
NBITS = 4


def required_coverage(target: int, nbits: int = NBITS) -> float:
    base = ALLOWED_LENGTHS[0] * nbits // 8
    costs = [length * nbits // 8 - base for length in ALLOWED_LENGTHS]
    return (target * nbits // 8 - base) / costs[-1]


def rarefaction(counts: np.ndarray, queries: int, points: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    values, multiplicity = np.unique(counts, return_counts=True)
    out = np.empty(len(points), dtype=np.float64)
    log_denominator = gammaln(queries + 1) - gammaln(queries - points + 1)
    for index, q in enumerate(points):
        # log P(miss) = log C(Q-c, q) - log C(Q, q), zero-count items always missed
        available = queries - values
        with np.errstate(invalid="ignore"):
            log_numerator = np.where(
                available >= q,
                gammaln(available + 1) - gammaln(available - q + 1),
                -np.inf,
            )
        miss = np.exp(log_numerator - log_denominator[index])
        miss = np.where(values == 0, 1.0, miss)
        out[index] = float(np.sum(multiplicity * (1.0 - miss)))
    return out


def chao_extrapolation(counts: np.ndarray, queries: int, extra: np.ndarray,
                       pool: int) -> tuple[np.ndarray, float]:
    observed = float(np.count_nonzero(counts))
    f1 = float(np.count_nonzero(counts == 1))
    f2 = float(np.count_nonzero(counts == 2))
    if f2 > 0:
        f0 = (queries - 1) / queries * f1 * f1 / (2.0 * f2)
    elif f1 > 0:
        f0 = (queries - 1) / queries * f1 * (f1 - 1) / 2.0
    else:
        f0 = 0.0
    f0 = min(f0, max(pool - observed, 0.0))
    asymptote = min(observed + f0, float(pool))
    if f0 <= 0 or f1 <= 0:
        return np.full(len(extra), observed), asymptote
    growth = 1.0 - np.power(1.0 - f1 / (queries * f0 + f1), extra)
    return observed + f0 * growth, asymptote


def validate_extrapolation(counts: np.ndarray, queries: int, pool: int,
                           fraction: float = 0.5, seed: int = 20260902) -> dict:
    rng = np.random.default_rng(seed)
    small = int(round(queries * fraction))
    drawn = rng.hypergeometric(
        np.maximum(counts, 0), np.maximum(queries - counts, 0), small
    )
    predicted, asymptote = chao_extrapolation(
        drawn, small, np.array([queries - small]), pool
    )
    truth = float(np.count_nonzero(counts))
    return {
        "fitted_on_queries": small,
        "predicted_coverage_at_full": float(predicted[0] / pool),
        "true_coverage_at_full": float(truth / pool),
        "relative_error": float(abs(predicted[0] - truth) / max(truth, 1.0)),
        "asymptote_from_half": float(asymptote / pool),
    }

def analyse(data_root: Path, factory: str, seeds: tuple[int, ...],
            calibration_queries: int, targets: tuple[int, ...],
            slug: str = SLUG) -> dict:
    data = _load_dataset(slug, data_root)
    gallery_items = len(data["gallery"])
    with np.load(data["manifest_path"], allow_pickle=False) as manifest:
        available_queries = int(len(manifest["query_labels"]))
        held_out = int(len(manifest["validation_ids"]) + len(manifest["test_ids"]))

    rare_points = np.unique(np.concatenate([
        np.linspace(100, calibration_queries, 25).astype(np.int64),
        np.array([calibration_queries], dtype=np.int64),
    ]))
    extra_points = np.array(
        [1_000, 5_000, 15_000, 45_000, 100_000, 300_000, 1_000_000], dtype=np.int64
    )

    per_seed = []
    for seed in seeds:
        path = _state_path(data, factory, seed, calibration_queries, "fp32")
        if not path.exists():
            raise SystemExit(
                f"missing calibration state {path}; run the panel for this seed"
            )
        with np.load(path, allow_pickle=False) as state:
            counts = np.asarray(state["coverage"], dtype=np.int64)
        covered = rarefaction(counts, calibration_queries, rare_points)
        extrapolated, asymptote = chao_extrapolation(
            counts, calibration_queries, extra_points, gallery_items
        )
        touched = counts.sum()
        per_seed.append({
            "seed": int(seed),
            "extrapolator_check": validate_extrapolation(
                counts, calibration_queries, gallery_items
            ),
            "coverage": float(np.count_nonzero(counts) / gallery_items),
            "unseen_fraction": float(np.mean(counts == 0)),
            "items_touched_per_query": float(touched / calibration_queries),
            "seen_once": int(np.count_nonzero(counts == 1)),
            "seen_twice": int(np.count_nonzero(counts == 2)),
            "median_exposure_of_seen": float(np.median(counts[counts > 0])),
            "max_exposure": int(counts.max()),
            "asymptotic_coverage": float(asymptote / gallery_items),
            "rarefaction": {int(q): float(c / gallery_items)
                            for q, c in zip(rare_points, covered)},
            "extrapolation": {int(calibration_queries + e): float(c / gallery_items)
                              for e, c in zip(extra_points, extrapolated)},
        })

    def mean_of(key):
        return float(np.mean([row[key] for row in per_seed]))

    asymptote = mean_of("asymptotic_coverage")
    requirements = {int(t): required_coverage(t) for t in targets}
    # Smallest workload whose mean projected coverage clears each requirement.
    curve = {}
    for row in per_seed:
        for q, c in {**row["rarefaction"], **row["extrapolation"]}.items():
            curve.setdefault(int(q), []).append(c)
    curve = {q: float(np.mean(v)) for q, v in sorted(curve.items())}
    needed = {}
    for target, requirement in requirements.items():
        reachable = [q for q, c in curve.items() if c >= requirement]
        needed[target] = {
            "required_coverage": requirement,
            "queries_needed": int(min(reachable)) if reachable else None,
            "reachable_at_all": bool(asymptote >= requirement),
        }

    return {
        "dataset": slug,
        "display_name": data["dataset"],
        "factory": factory,
        "gallery_items": gallery_items,
        "calibration_queries_now": int(calibration_queries),
        "queries_available_in_manifest": available_queries,
        "queries_usable_for_calibration": available_queries - held_out,
        "seeds": [int(s) for s in seeds],
        "mean_coverage_now": mean_of("coverage"),
        "mean_items_touched_per_query": mean_of("items_touched_per_query"),
        "mean_asymptotic_coverage": asymptote,
        "coverage_curve": curve,
        "targets": needed,
        "extrapolator_check": {
            "mean_relative_error": float(np.mean(
                [row["extrapolator_check"]["relative_error"] for row in per_seed]
            )),
            "fitted_on_queries": per_seed[0]["extrapolator_check"]["fitted_on_queries"],
        },
        "per_seed": per_seed,
    }


def render(report: dict) -> str:
    lines = [
        f"# Unseen items and calibration coverage -- {report['display_name']}",
        "",
        f"Gallery {report['gallery_items']:,} items; "
        f"{report['calibration_queries_now']:,} calibration queries now; "
        f"{report['queries_usable_for_calibration']:,} usable queries exist in "
        f"the manifest at all.",
        "",
        f"A query touches **{report['mean_items_touched_per_query']:.1f} items** "
        "on average (the mined hard set is bounded by "
        f"{10 + 3 * 50} regardless of gallery size), so coverage is now "
        f"**{report['mean_coverage_now']:.4f}**.",
        "",
        "## Can more calibration queries fill the cap",
        "",
        "| Target | Coverage `ravr` needs | Queries that would reach it | "
        "Reachable at all |",
        "|---:|---:|---:|---|",
    ]
    for target, entry in sorted(report["targets"].items()):
        needed = entry["queries_needed"]
        lines.append(
            f"| {target} | {entry['required_coverage']:.2f} | "
            f"{needed:,}" if needed else
            f"| {target} | {entry['required_coverage']:.2f} | never"
        )
        lines[-1] += (f" | {'yes' if entry['reachable_at_all'] else 'NO'} |")
    lines += [
        "",
        f"Estimated asymptotic coverage, with unlimited queries drawn from the "
        f"same distribution: **{report['mean_asymptotic_coverage']:.4f}**.",
        "",
        "Extrapolator check: fitted on a random half of the workload "
        f"({report['extrapolator_check']['fitted_on_queries']:,} queries) and "
        "asked to predict the coverage at the full workload, whose true value "
        "is known exactly, the mean relative error is "
        f"**{report['extrapolator_check']['mean_relative_error']:.4f}**.",
        "",
        "## Coverage against calibration workload",
        "",
        "| Calibration queries | Mean coverage | Source |",
        "|---:|---:|---|",
    ]
    now = report["calibration_queries_now"]
    for q, c in report["coverage_curve"].items():
        kind = "measured (rarefaction)" if q <= now else "Chao extrapolation"
        lines.append(f"| {q:,} | {c:.4f} | {kind} |")
    lines += [
        "",
        "Rarefaction below the observed workload is exact: it is the expected "
        "coverage of a random subsample of the queries actually run. Above it, "
        "the Chao term estimates how many further items are reachable at all "
        "and is an estimate, not a measurement.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", default=SLUG)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--targets", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--calibration-queries", type=int, default=5000)
    args = parser.parse_args()

    report = analyse(
        Path(args.data_root).expanduser().resolve(), args.factory,
        tuple(args.seeds), args.calibration_queries, tuple(args.targets),
        args.dataset,
    )
    stem = HERE / f"unseen_analysis_{args.dataset}_{args.factory.lower()}"
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    text = render(report)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
