"""Post-selection replication of the sparse-calibration unseen-item fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_baseline_claims import crossed_bootstrap_means, summarize_bootstrap
from benchmark_nested_additive_ravr import DEFAULT_TARGET_LENGTHS

# The panel this audit reads lives one level up, in the experiment folder.
HERE = Path(__file__).resolve().parents[1]


METRICS = ("teacher_recall_at_10", "teacher_ndcg_at_10")
COMPARATORS = ("uniform", "ravr")


def aggregate(
    factory: str,
    seeds: tuple[int, ...],
    calibration_queries: int,
    targets: tuple[int, ...],
    resamples: int,
    bootstrap_seed: int,
) -> dict:
    output = HERE / "outputs" / "nested_additive_ravr_sop_100k"
    reports = {
        (seed, target): json.loads((
            output
            / f"{factory.lower()}_seed{seed}_test_u{target}_cal{calibration_queries}.json"
        ).read_text(encoding="utf-8"))
        for seed in seeds for target in targets
    }
    backends = {
        reports[(seed, target)].get("compute_backend", {}).get("resolved")
        for seed in seeds for target in targets
    }
    if len(backends) != 1 or None in backends:
        raise ValueError(
            f"all replication rows must use one recorded compute backend; got {backends}"
        )
    backend = next(iter(backends))
    cells, differences = [], []
    for target in targets:
        for metric in METRICS:
            for comparator in COMPARATORS:
                per_seed = []
                for seed in seeds:
                    rows = reports[(seed, target)]["rows"]
                    fallback = np.asarray(
                        rows["ravr_unseen_rd"]["per_query"][metric]
                    )
                    baseline = np.asarray(rows[comparator]["per_query"][metric])
                    per_seed.append(fallback - baseline)
                cells.append((target, metric, comparator))
                differences.append(np.stack(per_seed))
    cube = np.stack(differences, axis=2)
    bootstrap = crossed_bootstrap_means(cube, resamples=resamples, seed=bootstrap_seed)
    estimate, se, pointwise, simultaneous, critical = summarize_bootstrap(cube, bootstrap)
    comparisons = []
    for index, (target, metric, comparator) in enumerate(cells):
        seed_means = cube[:, :, index].mean(axis=1)
        comparisons.append({
            "target_length": target,
            "metric": metric,
            "fallback_minus": comparator,
            "estimate": float(estimate[index]),
            "bootstrap_se": float(se[index]),
            "pointwise_95_interval": [float(x) for x in pointwise[index]],
            "family_simultaneous_95_interval": [float(x) for x in simultaneous[index]],
            "positive_seeds": int(np.sum(seed_means > 0)),
            "seed_differences": [float(x) for x in seed_means],
        })
    storage = []
    for target in targets:
        for strategy in ("uniform", "ravr", "ravr_unseen_rd"):
            values = [
                reports[(seed, target)]["rows"][strategy]["persistent_bytes_per_item"]
                for seed in seeds
            ]
            storage.append({
                "target_length": target,
                "strategy": strategy,
                "mean_persistent_bytes_per_item": float(np.mean(values)),
                "min_persistent_bytes_per_item": float(np.min(values)),
                "max_persistent_bytes_per_item": float(np.max(values)),
            })
    result = {
        "dataset": "Stanford Online Products, 100K gallery",
        "codec": factory,
        "calibration_queries": calibration_queries,
        "selection_protocol": (
            "seed 17 exploratory; listed seeds are post-selection replication only"
        ),
        "seeds": list(seeds),
        "targets": list(targets),
        "compute_backend": backend,
        "bootstrap": {
            "resamples": resamples,
            "seed": bootstrap_seed,
            "family": f"{len(cells)} rate-metric-comparator cells",
            "simultaneous_half_width": float(critical),
        },
        "storage": storage,
        "comparisons": comparisons,
    }
    stem = output / f"unseen_rd_fallback_{factory.lower()}_postselection{len(seeds)}"
    stem.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(report: dict) -> str:
    lines = [
        "# Sparse-calibration unseen-item fallback",
        "",
        report["selection_protocol"],
        f"Compute backend: `{report['compute_backend']}` (all rows).",
        "",
        "| Target | Metric | Fallback minus | Delta | Pointwise 95% | Family 95% | Positive seeds |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        pointwise = row["pointwise_95_interval"]
        family = row["family_simultaneous_95_interval"]
        lines.append(
            f"| {row['target_length']} | {row['metric']} | {row['fallback_minus']} | "
            f"{row['estimate']:+.5f} | [{pointwise[0]:+.5f}, {pointwise[1]:+.5f}] | "
            f"[{family[0]:+.5f}, {family[1]:+.5f}] | "
            f"{row['positive_seeds']}/{len(report['seeds'])} |"
        )
    lines.extend([
        "",
        "| Target | Strategy | Mean B/item | Range |",
        "|---:|---|---:|---:|",
    ])
    for row in report["storage"]:
        lines.append(
            f"| {row['target_length']} | {row['strategy']} | "
            f"{row['mean_persistent_bytes_per_item']:.5f} | "
            f"[{row['min_persistent_bytes_per_item']:.5f}, "
            f"{row['max_persistent_bytes_per_item']:.5f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 73, 3067, 4294, 4996, 5423, 7520, 7937, 9794],
    )
    parser.add_argument("--calibration-queries", type=int, default=400)
    parser.add_argument("--targets", nargs="+", type=int, default=list(DEFAULT_TARGET_LENGTHS))
    parser.add_argument("--resamples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()
    result = aggregate(
        args.factory,
        tuple(args.seeds),
        args.calibration_queries,
        tuple(args.targets),
        args.resamples,
        args.bootstrap_seed,
    )
    print(render_markdown(result), flush=True)


if __name__ == "__main__":
    main()
