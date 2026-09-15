"""Train/evaluate a complete QUIP-cov(z) codec on the locked SOP protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from benchmark_faiss_fixed_rate import _evaluate_reconstruction, _load_dataset, _sha256
from compute_backend import backend_report, configure_compute_backend
from qaq_pq import QAQPQIndex, codebook_sha256
from quip_pq import QUIPConfig, train_quip


# Results live next to this experiment, not next to the shared loader.
HERE = Path(__file__).resolve().parent

DEFAULT_PROTOCOL = Path(r"D:\Research\Data\AJigma\qaq_official\sop\seed17")
DEFAULT_RELEASED_BASELINE = Path(
    r"D:\Research\Data\AJigma\external\Query-Aware-Quantization-sparse"
)


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _split_data(base: dict, ids: np.ndarray) -> dict:
    result = dict(base)
    result["queries"] = base["gallery"][ids]
    result["exclude_ids"] = ids
    result["query_labels"] = base["gallery_labels"][ids]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--released-baseline", type=Path, default=DEFAULT_RELEASED_BASELINE)
    parser.add_argument("--M", type=int, default=6)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--encode-batch-size", type=int, default=4096)
    parser.add_argument("--identity-permutation", action="store_true")
    parser.add_argument("--compute-backend", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure_compute_backend(args.compute_backend)

    protocol_dir = args.protocol.resolve()
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    source_manifest = Path(protocol["source_manifest"])
    base = _load_dataset("sop", source_manifest.parent.parent)
    if _sha256(base["manifest_path"]) != protocol["source_manifest_sha256"]:
        raise ValueError("the SOP manifest differs from the locked protocol")
    gallery = base["gallery"]
    train = np.load(protocol_dir / "train.npy", mmap_mode="r")
    calibration = np.load(protocol_dir / "calibration.npy", mmap_mode="r")
    validation_ids = np.load(protocol_dir / "validation_ids.npy")
    test_ids = np.load(protocol_dir / "test_ids.npy")
    config = QUIPConfig(
        m=args.M,
        k=args.K,
        seed=int(protocol["codec_seed"]),
        max_iterations=args.max_iterations,
        relative_tolerance=args.relative_tolerance,
        encode_batch_size=args.encode_batch_size,
        random_permutation=not args.identity_permutation,
    )
    bits = int(round(np.log2(config.k)))
    permutation_tag = "randomperm" if config.random_permutation else "identityperm"
    stem = (
        f"quip_covz_m{config.m}_k{config.k}_seed{config.seed}_"
        f"it{config.max_iterations}_{permutation_tag}"
    )
    bundle_dir = protocol_dir / "bundles"
    bundle_dir.mkdir(exist_ok=True)
    base_bundle_path = bundle_dir / f"{stem}.quipix"
    bundle_path = bundle_dir / (
        f"{stem}.quipix" if args.precision == "fp32"
        else f"{stem}_fp16.quipix"
    )
    train_report_path = bundle_dir / f"{stem}_training.json"
    covariance_path = bundle_dir / f"{stem}_covariance.npy"

    if args.precision == "fp16":
        if not base_bundle_path.exists() or not train_report_path.exists():
            raise FileNotFoundError(
                "FP16 evaluation requires the frozen FP32 QUIP bundle and training report"
            )
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        if bundle_path.exists() and not args.force:
            index = QAQPQIndex.load(bundle_path)
        else:
            source = QAQPQIndex.load(base_bundle_path)
            index = source.to_precision("fp16")
            index.save(bundle_path)
            source.close()
        trained = False
    elif bundle_path.exists() and train_report_path.exists() and not args.force:
        index = QAQPQIndex.load(bundle_path)
        train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
        trained = False
    else:
        print(
            f"[{time.strftime('%H:%M:%S')}] training QUIP-cov(z) M={config.m}, "
            f"K={config.k} on {len(train)} items / {len(calibration)} calibration queries",
            flush=True,
        )
        codebooks, codes, covariance, permutation, train_report = train_quip(
            gallery, train, calibration, config
        )
        np.save(covariance_path, covariance)
        train_report["covariance_sha256"] = hashlib.sha256(
            np.ascontiguousarray(covariance).view(np.uint8)
        ).hexdigest()
        train_report["permutation"] = permutation.astype(int).tolist()
        train_report_path.write_text(
            json.dumps(train_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        released = args.released_baseline.resolve()
        index = QAQPQIndex.from_codes(
            codebooks,
            codes,
            {
                "format": "QUIP-cov(z) fixed-rate product quantizer",
                "algorithm": "QUIP-cov(z)",
                "config": {
                    "M": config.m,
                    "K": config.k,
                    "bits_per_symbol": bits,
                    "seed": config.seed,
                    "init_seed": config.init_seed,
                    "max_iterations": config.max_iterations,
                    "relative_tolerance": config.relative_tolerance,
                    "random_permutation": config.random_permutation,
                },
                "dimension_permutation": permutation.astype(int).tolist(),
                "protocol_sha256": _sha256(protocol_dir / "protocol.json"),
                "paper": "Guo et al., AISTATS 2016, Sec. 3.1, Eq. (2)-(3)",
                "released_baseline_commit": _git_commit(released),
                "released_baseline_model_sha256": _sha256(
                    released / "src" / "packages" / "QUIP" / "cov.py"
                ),
                "accelerated_port": (
                    "same non-centred held-out-query covariance, Mahalanobis "
                    "assignment, and Euclidean centroid update; batched on CUDA"
                ),
            },
        )
        index.save(bundle_path)
        trained = True

    reloaded = QAQPQIndex.load(bundle_path)
    raw = reloaded.decode_raw()
    cosine = reloaded.decode_normalized()
    evaluations, cosine_evaluations = {}, {}
    for split, split_ids in (("validation", validation_ids), ("test", test_ids)):
        teacher_ids = np.load(protocol_dir / f"{split}_teacher_ids.npy")
        teacher_scores = np.load(protocol_dir / f"{split}_teacher_scores.npy")
        metrics, per_query = _evaluate_reconstruction(
            raw, _split_data(base, split_ids), teacher_ids, teacher_scores
        )
        metrics["mean_reconstruction_cosine"] = float(
            np.mean(np.sum(base["gallery"] * cosine, axis=1))
        )
        metrics["scoring"] = "author-native raw query/codeword inner product"
        evaluations[split] = {"metrics": metrics, "per_query": per_query}
        metrics, per_query = _evaluate_reconstruction(
            cosine, _split_data(base, split_ids), teacher_ids, teacher_scores
        )
        metrics["scoring"] = "common cosine-normalised reconstruction audit"
        cosine_evaluations[split] = {"metrics": metrics, "per_query": per_query}

    released = args.released_baseline.resolve()
    report = {
        "method": "QUIP-cov(z)",
        "dataset": base["dataset"],
        "protocol": str(protocol_dir / "protocol.json"),
        "protocol_sha256": _sha256(protocol_dir / "protocol.json"),
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "codec_training_items": len(train),
        "calibration_queries": len(calibration),
        "validation_queries": len(validation_ids),
        "test_queries": len(test_ids),
        "config": reloaded.metadata["config"],
        "bundle": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "training_report": str(train_report_path),
        "covariance": str(covariance_path),
        "trained_this_run": trained,
        "codebook_storage_precision": args.precision,
        "codebook_sha256": codebook_sha256(reloaded.codebooks),
        "code_only_bytes_per_item": reloaded.code_only_bytes_per_item,
        "full_serialized_bytes_per_item": reloaded.persistent_bytes_per_item,
        "persistent_bytes": reloaded.persistent_bytes,
        "storage": reloaded.storage_breakdown(),
        "evaluation": evaluations,
        "evaluation_cosine_normalized": cosine_evaluations,
        "teacher": protocol["teacher"],
        "provenance": {
            "paper": "https://proceedings.mlr.press/v51/guo16a.html",
            "paper_variant": "QUIP-cov(z)",
            "paper_configuration": "C=256 (8-bit symbols), fixed random permutation",
            "released_baseline_repository": "https://github.com/jzhang-0/Query-Aware-Quantization",
            "released_baseline_local_path": str(released),
            "released_baseline_commit": _git_commit(released),
            "released_baseline_model_sha256": _sha256(
                released / "src" / "packages" / "QUIP" / "cov.py"
            ),
            "released_baseline_difference": (
                "the QAQ-author baseline implements the core covariance-weighted Lloyd "
                "updates but omits the paper's fixed random dimension permutation"
            ),
            "license_file_present": any(
                (released / name).exists() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "compute_backend": backend_report(),
        },
    }
    output = HERE / "outputs" / (
        "quip_official_sop_100k" if args.precision == "fp32"
        else "quip_official_fp16_sop_100k"
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"seed {config.seed}: raw={evaluations['test']['metrics']['teacher_recall_at_10']:.5f}, "
        f"cosine={cosine_evaluations['test']['metrics']['teacher_recall_at_10']:.5f}, "
        f"bytes={reloaded.persistent_bytes_per_item:.5f}",
        flush=True,
    )
    print(f"result: {path}", flush=True)
    reloaded.close()
    if index is not reloaded:
        index.close()


if __name__ == "__main__":
    main()
