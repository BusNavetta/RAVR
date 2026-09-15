from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import faiss
import numpy as np

from benchmark_faiss_fixed_rate import (
    _evaluate_reconstruction,
    _load_dataset,
    _sha256,
)
from compute_backend import backend_report, configure_compute_backend
from qaq_pq import QAQConfig, QAQPQIndex, codebook_sha256, train_qaq


# Results live next to this experiment, not next to the shared loader.
HERE = Path(__file__).resolve().parent

DEFAULT_PROTOCOL = Path(r"D:\Research\Data\AJigma\qaq_official\sop\seed17")
DEFAULT_OFFICIAL_REPO = Path(
    r"D:\Research\Data\AJigma\external\Query-Aware-Quantization-sparse"
)


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _split_data(base: dict, ids: np.ndarray) -> dict:
    data = dict(base)
    data["queries"] = base["gallery"][ids]
    data["exclude_ids"] = ids
    data["query_labels"] = base["gallery_labels"][ids]
    return data


def _render(report: dict) -> str:
    lines = [
        f"# QAQ {report['config']['M']}x{report['config']['bits_per_symbol']}-bit SOP result",
        "",
        "Author QAQ equations are trained on the locked 12K/5K roles. The final "
        f"physical index contains {report['codebook_storage_precision']} codebooks, bit-packed codes, explicit "
        "uint32 IDs, float16 inverse norms, and its fixed header/alignment.",
        "",
        "| Split | Full B/item | Code B/item | Teacher R@1 | Teacher R@10 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        metrics = report["evaluation"][split]["metrics"]
        lines.append(
            f"| {split} | {report['full_serialized_bytes_per_item']:.5f} | "
            f"{report['code_only_bytes_per_item']:.5f} | "
            f"{metrics['teacher_recall_at_1']:.5f} | "
            f"{metrics['teacher_recall_at_10']:.5f} | "
            f"{metrics['teacher_ndcg_at_10']:.5f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--kv", type=int, default=16)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--train-iterations", type=int, default=2)
    parser.add_argument("--encode-iterations", type=int, default=3)
    parser.add_argument("--query-batch-size", type=int, default=500)
    parser.add_argument("--sample-count", type=int, default=1)
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
        raise ValueError("the SOP manifest differs from the locked QAQ protocol")
    gallery = base["gallery"]
    train = np.load(protocol_dir / "train.npy", mmap_mode="r")
    train_ids = np.load(protocol_dir / "train_ids.npy")
    calibration = np.load(protocol_dir / "calibration.npy", mmap_mode="r")
    validation = np.load(protocol_dir / "validation.npy", mmap_mode="r")
    validation_ids = np.load(protocol_dir / "validation_ids.npy")
    test_ids = np.load(protocol_dir / "test_ids.npy")
    config = QAQConfig(
        m=args.M,
        k=args.K,
        kv=args.kv,
        seed=int(protocol["codec_seed"]),
        kmeans_iterations=args.kmeans_iterations,
        train_iterations=args.train_iterations,
        encode_iterations=args.encode_iterations,
        query_batch_size=args.query_batch_size,
        sample_count=args.sample_count,
    )
    bits = int(np.log2(config.k))
    stem = (
        f"qaq_m{config.m}_k{config.k}_kv{config.kv}_seed{config.seed}_"
        f"s{config.sample_count}_it{config.train_iterations}"
    )
    bundle_dir = protocol_dir / "bundles"
    bundle_dir.mkdir(exist_ok=True)
    base_bundle_path = bundle_dir / f"{stem}.qaqix"
    bundle_path = bundle_dir / (
        f"{stem}.qaqix" if args.precision == "fp32" else f"{stem}_fp16.qaqix"
    )
    train_report_path = bundle_dir / f"{stem}_training.json"

    if args.precision == "fp16":
        if not base_bundle_path.exists() or not train_report_path.exists():
            raise FileNotFoundError(
                "FP16 evaluation requires the frozen FP32 QAQ bundle and training report"
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
            f"[{time.strftime('%H:%M:%S')}] training QAQ M={config.m}, K={config.k}, "
            f"kv={config.kv} on {len(train)} items / {len(calibration)} calibration queries",
            flush=True,
        )
        codebooks, codes, cluster_ids, train_report = train_qaq(
            gallery, train, train_ids, calibration, config, validation
        )
        train_report["cluster_histogram"] = np.bincount(
            cluster_ids, minlength=config.kv
        ).tolist()
        train_report_path.write_text(
            json.dumps(train_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        index = QAQPQIndex.from_codes(
            codebooks,
            codes,
            {
                "algorithm": "QAQ / SSR_V2",
                "config": {
                    "M": config.m,
                    "K": config.k,
                    "bits_per_symbol": bits,
                    "kv": config.kv,
                    "seed": config.seed,
                    "kmeans_iterations": config.kmeans_iterations,
                    "train_iterations": config.train_iterations,
                    "encode_iterations": config.encode_iterations,
                    "query_batch_size": config.query_batch_size,
                    "sample_count": config.sample_count,
                },
                "protocol_sha256": _sha256(protocol_dir / "protocol.json"),
                "official_repo": str(args.official_repo.resolve()),
                "official_commit": _git_commit(args.official_repo.resolve()),
                "official_model_sha256": _sha256(
                    args.official_repo.resolve() / "src" / "model" / "saq_v2.jl"
                ),
                "accelerated_port": (
                    "same QAQ query matrices, coordinate descent, and regularised "
                    "normal equations; algebraically batched on CUDA"
                ),
            },
        )
        index.save(bundle_path)
        trained = True

    reloaded = QAQPQIndex.load(bundle_path)
    reconstruction_raw = reloaded.decode_raw()
    reconstruction_cosine = reloaded.decode_normalized()
    evaluations = {}
    cosine_evaluations = {}
    for split, split_ids in (("validation", validation_ids), ("test", test_ids)):
        teacher_ids = np.load(protocol_dir / f"{split}_teacher_ids.npy")
        teacher_scores = np.load(protocol_dir / f"{split}_teacher_scores.npy")
        metrics, per_query = _evaluate_reconstruction(
            reconstruction_raw, _split_data(base, split_ids), teacher_ids, teacher_scores
        )
        metrics["mean_reconstruction_cosine"] = float(np.mean(np.sum(
            base["gallery"] * reconstruction_cosine, axis=1
        )))
        metrics["scoring"] = "author-native raw query/codeword inner product"
        evaluations[split] = {"metrics": metrics, "per_query": per_query}
        cosine_metrics, cosine_per_query = _evaluate_reconstruction(
            reconstruction_cosine, _split_data(base, split_ids), teacher_ids, teacher_scores
        )
        cosine_metrics["scoring"] = "common cosine-normalised reconstruction audit"
        cosine_evaluations[split] = {
            "metrics": cosine_metrics, "per_query": cosine_per_query
        }
    official_repo = args.official_repo.resolve()
    report = {
        "method": "QAQ",
        "dataset": base["dataset"],
        "protocol": str(protocol_dir / "protocol.json"),
        "protocol_sha256": _sha256(protocol_dir / "protocol.json"),
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "codec_training_items": len(train),
        "calibration_queries": len(calibration),
        "validation_queries": len(validation_ids),
        "test_queries": len(test_ids),
        "config": index.metadata["config"],
        "bundle": str(bundle_path),
        "bundle_sha256": _sha256(bundle_path),
        "training_report": str(train_report_path),
        "trained_this_run": trained,
        "codebook_storage_precision": args.precision,
        "codebook_sha256": codebook_sha256(index.codebooks),
        "code_only_bytes_per_item": index.code_only_bytes_per_item,
        "full_serialized_bytes_per_item": index.persistent_bytes_per_item,
        "persistent_bytes": index.persistent_bytes,
        "storage": index.storage_breakdown(),
        "evaluation": evaluations,
        "evaluation_cosine_normalized": cosine_evaluations,
        "teacher": protocol["teacher"],
        "official_source": {
            "repository": "https://github.com/jzhang-0/Query-Aware-Quantization",
            "local_path": str(official_repo),
            "commit": _git_commit(official_repo),
            "model_sha256": _sha256(official_repo / "src" / "model" / "saq_v2.jl"),
            "license_file_present": any(
                (official_repo / name).exists()
                for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
            ),
        },
        "implementation_validation": {
            "tests": "test_qaq_pq.py",
            "checks": [
                "CUDA coordinate descent equals scalar official equation",
                "co-occurrence normal equations equal dense official construction",
                "bit packing and physical reload round-trip",
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "compute_backend": backend_report(),
        },
    }
    output_dir = HERE / "outputs" / (
        "qaq_official_sop_100k" if args.precision == "fp32"
        else "qaq_official_fp16_sop_100k"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{stem}.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = _render(report)
    (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(f"result: {result_path}", flush=True)
    reloaded.close()
    if index is not reloaded:
        index.close()


if __name__ == "__main__":
    main()
