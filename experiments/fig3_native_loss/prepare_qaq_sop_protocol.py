"""Export the locked SOP roles for the official QAQ replication.

The exported arrays are deliberately explicit: QAQ must not sample its own
training/query split if it is to be compared with the existing AJigma runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_faiss_fixed_rate import _load_dataset, _sha256
from compute_backend import backend_report, configure_compute_backend, topk_inner_products
from ravr_depth_ablation import _roles


DEFAULT_DATA = Path(r"D:\Research\Data\AJigma\tfds_replication")
DEFAULT_OUTPUT = Path(r"D:\Research\Data\AJigma\qaq_official\sop")


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _save(path: Path, value: np.ndarray) -> dict:
    value = np.ascontiguousarray(value)
    np.save(path, value, allow_pickle=False)
    return {
        "path": str(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "array_sha256": _array_sha256(value),
        "file_sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--codec-seed", type=int, default=17)
    parser.add_argument("--train-samples", type=int, default=12_000)
    parser.add_argument("--compute-backend", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_compute_backend(args.compute_backend)
    data = _load_dataset("sop", args.data_root.resolve())
    roles = _roles(data)
    rng = np.random.default_rng(args.codec_seed)
    train_ids = rng.choice(
        roles["codec_train_ids"],
        min(args.train_samples, len(roles["codec_train_ids"])),
        replace=False,
    ).astype(np.int64, copy=False)
    manifest = np.load(data["manifest_path"], allow_pickle=False)
    calibration_ids = np.asarray(manifest["calibration_ids"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_ids"], dtype=np.int64)
    test_ids = np.asarray(manifest["test_ids"], dtype=np.int64)

    role_sets = [set(x.tolist()) for x in (train_ids, calibration_ids, validation_ids, test_ids)]
    for left in range(len(role_sets)):
        for right in range(left + 1, len(role_sets)):
            if role_sets[left] & role_sets[right]:
                raise AssertionError(f"roles {left} and {right} overlap")

    output = args.output_root.resolve() / f"seed{args.codec_seed}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "protocol.json"
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing["source_manifest_sha256"] == _sha256(data["manifest_path"])
            and existing["codec_seed"] == args.codec_seed
            and existing["train_samples"] == len(train_ids)
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
        raise FileExistsError(f"incompatible protocol already exists: {manifest_path}")

    arrays = {}
    for name, value in (
        ("train_ids", train_ids),
        ("calibration_ids", calibration_ids),
        ("validation_ids", validation_ids),
        ("test_ids", test_ids),
        ("train", data["gallery"][train_ids]),
        ("calibration", data["gallery"][calibration_ids]),
        ("validation", data["gallery"][validation_ids]),
        ("test", data["gallery"][test_ids]),
    ):
        arrays[name] = _save(output / f"{name}.npy", np.asarray(value))

    teacher_reports = {}
    for split, queries, exclude in (
        ("validation", data["gallery"][validation_ids], validation_ids),
        ("test", data["gallery"][test_ids], test_ids),
    ):
        ids, scores, report = topk_inner_products(
            queries, data["gallery"], 10, exclude, batch_size=128
        )
        arrays[f"{split}_teacher_ids"] = _save(output / f"{split}_teacher_ids.npy", ids)
        arrays[f"{split}_teacher_scores"] = _save(
            output / f"{split}_teacher_scores.npy", scores
        )
        teacher_reports[split] = report

    protocol = {
        "dataset": data["dataset"],
        "gallery_path": str(data["root"] / "sop_100k_dinov2_vits14.npy"),
        "gallery_shape": list(data["gallery"].shape),
        "gallery_dtype": str(data["gallery"].dtype),
        "gallery_file_sha256": _sha256(data["root"] / "sop_100k_dinov2_vits14.npy"),
        "source_manifest": str(data["manifest_path"]),
        "source_manifest_sha256": _sha256(data["manifest_path"]),
        "source_role_seed": int(manifest["seed"]),
        "codec_seed": args.codec_seed,
        "train_samples": len(train_ids),
        "roles_disjoint": True,
        "teacher": "exact FP32 inner-product scan on original L2-normalised gallery; self ID excluded",
        "compute_backend": backend_report(),
        "teacher_reports": teacher_reports,
        "arrays": arrays,
    }
    manifest_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
