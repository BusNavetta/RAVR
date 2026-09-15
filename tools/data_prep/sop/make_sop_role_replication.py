"""Create a new SOP calibration/validation/test split on frozen embeddings."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--output-data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--query-classes", type=int, default=400)
    parser.add_argument("--calibration-queries", type=int, default=5000)
    args = parser.parse_args()

    source = args.source_data_root.resolve() / "sop"
    output = args.output_data_root.resolve() / "sop"
    output.mkdir(parents=True, exist_ok=True)
    source_manifest_path = source / "sop_100k_manifest.npz"
    source_manifest = np.load(source_manifest_path, allow_pickle=False)
    labels = np.asarray(source_manifest["labels"], dtype=np.int32)
    by_label: dict[int, list[int]] = defaultdict(list)
    for item, label in enumerate(labels):
        by_label[int(label)].append(item)
    eligible = np.asarray([
        label for label, items in by_label.items() if len(items) >= 4
    ], dtype=np.int32)
    rng = np.random.default_rng(args.seed)
    query_labels = rng.choice(eligible, args.query_classes, replace=False)
    calibration_ids, validation_ids, test_ids = [], [], []
    for label in query_labels:
        chosen = rng.choice(by_label[int(label)], 3, replace=False)
        calibration_ids.append(chosen[0])
        validation_ids.append(chosen[1])
        test_ids.append(chosen[2])
    calibration_ids = np.asarray(calibration_ids, dtype=np.int64)
    validation_ids = np.asarray(validation_ids, dtype=np.int64)
    test_ids = np.asarray(test_ids, dtype=np.int64)
    role_seed_ids = np.concatenate((calibration_ids, validation_ids, test_ids))
    extra = args.calibration_queries - len(calibration_ids)
    if extra < 0:
        raise ValueError("calibration queries cannot be below query classes")
    if extra:
        available = np.setdiff1d(
            np.arange(len(labels), dtype=np.int64), role_seed_ids,
            assume_unique=False,
        )
        calibration_ids = np.concatenate((
            calibration_ids, rng.choice(available, extra, replace=False)
        ))
    union = np.concatenate((calibration_ids, validation_ids, test_ids))
    if len(np.unique(union)) != len(union):
        raise AssertionError("query roles overlap")
    codec_train_ids = np.setdiff1d(
        np.arange(len(labels), dtype=np.int64), union, assume_unique=False
    )
    manifest_path = output / "sop_100k_manifest.npz"
    np.savez(
        manifest_path,
        paths=source_manifest["paths"],
        labels=labels,
        source_ids=source_manifest["source_ids"],
        calibration_ids=calibration_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        codec_train_ids=codec_train_ids,
        seed=np.asarray(args.seed),
        panel=np.asarray("TFDS gallery; independent query-role replication"),
    )
    feature_name = "sop_100k_dinov2_vits14.npy"
    source_feature = source / feature_name
    output_feature = output / feature_name
    if not output_feature.exists():
        os.link(source_feature, output_feature)
    elif not os.path.samefile(source_feature, output_feature):
        raise FileExistsError(f"unrelated feature file already exists: {output_feature}")
    report = {
        "dataset": "Stanford Online Products",
        "panel": "same frozen TFDS gallery; independent query-role replication",
        "role_seed": args.seed,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "feature_hardlink": str(output_feature),
        "gallery_items": len(labels),
        "calibration_queries": len(calibration_ids),
        "validation_queries": len(validation_ids),
        "test_queries": len(test_ids),
        "query_roles_disjoint": True,
    }
    (output / "sop_100k_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
