"""Aggregate frozen-codec RAVR sample-efficiency experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_baseline_claims import crossed_bootstrap_means, summarize_bootstrap
from benchmark_nested_additive_ravr import DEFAULT_SEEDS, DEFAULT_TARGET_LENGTHS

# The panel this audit reads lives one level up, in the experiment folder.
HERE = Path(__file__).resolve().parents[1]


METRICS = ("teacher_recall_at_10", "teacher_ndcg_at_10")


def aggregate(
    factory: str,
    seeds: tuple[int, ...],
    calibration_sizes: tuple[int, ...],
    targets: tuple[int, ...],
    resamples: int,
    bootstrap_seed: int,
) -> dict:
    output = HERE / "outputs" / "nested_additive_ravr_sop_100k"
    reports = {
        (seed, calibration, target): json.loads((
            output
            / f"{factory.lower()}_seed{seed}_test_u{target}_cal{calibration}.json"
        ).read_text(encoding="utf-8"))
        for seed in seeds
        for calibration in calibration_sizes
        for target in targets
    }
    backend_by_calibration = {}
    for calibration in calibration_sizes:
        backends = {
            reports[(seed, calibration, target)].get("compute_backend", {}).get(
                "resolved", "legacy_numpy_cpu"
            )
            for seed in seeds for target in targets
        }
        if len(backends) != 1:
            raise ValueError(
                f"calibration size {calibration} mixes compute backends: {backends}"
            )
        backend_by_calibration[calibration] = next(iter(backends))
    cells, differences = [], []
    for calibration in calibration_sizes:
        for target in targets:
            for metric in METRICS:
                per_seed = []
                for seed in seeds:
                    rows = reports[(seed, calibration, target)]["rows"]
                    ravr = np.asarray(rows["ravr"]["per_query"][metric])
                    uniform = np.asarray(rows["uniform"]["per_query"][metric])
                    per_seed.append(ravr - uniform)
                cells.append((calibration, target, metric))
                differences.append(np.stack(per_seed))
    cube = np.stack(differences, axis=2)
    bootstrap = crossed_bootstrap_means(
        cube, resamples=resamples, seed=bootstrap_seed
    )
    estimate, se, pointwise, simultaneous, critical = summarize_bootstrap(
        cube, bootstrap
    )
    comparisons = []
    for index, (calibration, target, metric) in enumerate(cells):
        seed_means = cube[:, :, index].mean(axis=1)
        comparisons.append({
            "calibration_queries": calibration,
            "target_length": target,
            "metric": metric,
            "estimate": float(estimate[index]),
            "bootstrap_se": float(se[index]),
            "pointwise_95_interval": [float(x) for x in pointwise[index]],
            "family_simultaneous_95_interval": [
                float(x) for x in simultaneous[index]
            ],
            "positive_seeds": int(np.sum(seed_means > 0)),
            "seed_differences": [float(x) for x in seed_means],
        })
    result = {
        "dataset": "Stanford Online Products, 100K gallery",
        "codec": factory,
        "seeds": list(seeds),
        "calibration_sizes": list(calibration_sizes),
        "targets": list(targets),
        "compute_backend_by_calibration_size": {
            str(key): value for key, value in backend_by_calibration.items()
        },
        "coverage_and_storage": [
            {
                "calibration_queries": calibration,
                "target_length": target,
                "mean_coverage_fraction": float(np.mean([
                    reports[(seed, calibration, target)]["coverage_fraction"]
                    for seed in seeds
                ])),
                "mean_ravr_bytes_per_item": float(np.mean([
                    reports[(seed, calibration, target)]["rows"]["ravr"]
                    ["persistent_bytes_per_item"] for seed in seeds
                ])),
                "mean_uniform_cap_bytes_per_item": float(np.mean([
                    reports[(seed, calibration, target)]["rows"]["uniform"]
                    ["persistent_bytes_per_item"] for seed in seeds
                ])),
            }
            for calibration in calibration_sizes for target in targets
        ],
        "bootstrap": {
            "resamples": resamples,
            "seed": bootstrap_seed,
            "family": f"{len(cells)} calibration-rate-metric cells",
            "simultaneous_half_width": float(critical),
        },
        "comparisons": comparisons,
    }
    stem = output / f"calibration_scale_{factory.lower()}_test"
    stem.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(report: dict) -> str:
    lines = [
        f"# {report['codec']} calibration-size ablation",
        "",
        "RAVR minus uniform on the frozen SOP test queries. The family band covers "
        "every calibration-size, rate, and metric cell in this report.",
        "",
        "Backends by calibration size: " + ", ".join(
            f"{key}={value}" for key, value in
            report["compute_backend_by_calibration_size"].items()
        ) + ". Every seed within a cell uses one backend.",
        "",
        "| Calibration queries | Target | Metric | Delta | Pointwise 95% | Family 95% | Positive seeds |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        pointwise = row["pointwise_95_interval"]
        family = row["family_simultaneous_95_interval"]
        lines.append(
            f"| {row['calibration_queries']} | {row['target_length']} | "
            f"{row['metric']} | {row['estimate']:+.5f} | "
            f"[{pointwise[0]:+.5f}, {pointwise[1]:+.5f}] | "
            f"[{family[0]:+.5f}, {family[1]:+.5f}] | "
            f"{row['positive_seeds']}/{len(report['seeds'])} |"
        )
    lines.extend([
        "",
        "| Calibration queries | Target | Candidate coverage | RAVR B/item | Cap B/item |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in report["coverage_and_storage"]:
        lines.append(
            f"| {row['calibration_queries']} | {row['target_length']} | "
            f"{row['mean_coverage_fraction']:.4f} | "
            f"{row['mean_ravr_bytes_per_item']:.5f} | "
            f"{row['mean_uniform_cap_bytes_per_item']:.5f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--calibration-sizes", nargs="+", type=int, default=[400, 1000, 2500, 5000]
    )
    parser.add_argument(
        "--targets", nargs="+", type=int, default=list(DEFAULT_TARGET_LENGTHS)
    )
    parser.add_argument("--resamples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    args = parser.parse_args()
    result = aggregate(
        args.factory,
        tuple(args.seeds),
        tuple(args.calibration_sizes),
        tuple(args.targets),
        args.resamples,
        args.bootstrap_seed,
    )
    print(render_markdown(result), flush=True)


if __name__ == "__main__":
    main()
