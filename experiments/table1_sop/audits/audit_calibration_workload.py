"""Describe how SOP calibration prefixes align with the frozen test workload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]



def audit(manifest: Path, sizes: tuple[int, ...]) -> dict:
    with np.load(manifest, allow_pickle=False) as data:
        labels = np.asarray(data["labels"])
        calibration_ids = np.asarray(data["calibration_ids"], dtype=np.int64)
        test_ids = np.asarray(data["test_ids"], dtype=np.int64)
    target_labels = np.unique(labels[test_ids])
    rows = []
    for size in sizes:
        ids = calibration_ids[:size]
        query_labels = labels[ids]
        in_target = np.isin(query_labels, target_labels)
        rows.append({
            "calibration_queries": int(size),
            "unique_calibration_classes": int(len(np.unique(query_labels))),
            "queries_from_test_workload_classes": int(np.sum(in_target)),
            "unique_test_workload_classes_observed": int(
                len(np.intersect1d(np.unique(query_labels), target_labels))
            ),
            "test_workload_classes": int(len(target_labels)),
        })
    result = {
        "manifest": str(manifest.resolve()),
        "note": (
            "Calibration prefixes are deterministic record-order prefixes. "
            "Changing their length changes both sample count and class mixture."
        ),
        "rows": rows,
    }
    output = HERE / "outputs" / "nested_additive_ravr_sop_100k"
    stem = output / "calibration_workload_alignment"
    stem.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    return result


def render_markdown(report: dict) -> str:
    lines = [
        "# Calibration/workload alignment audit",
        "",
        report["note"],
        "",
        "| Calibration queries | Unique calibration classes | Queries in test classes | Test classes observed |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['calibration_queries']} | "
            f"{row['unique_calibration_classes']} | "
            f"{row['queries_from_test_workload_classes']} | "
            f"{row['unique_test_workload_classes_observed']}/"
            f"{row['test_workload_classes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[400, 1000, 5000])
    args = parser.parse_args()
    print(render_markdown(audit(args.manifest, tuple(args.sizes))))


if __name__ == "__main__":
    main()
