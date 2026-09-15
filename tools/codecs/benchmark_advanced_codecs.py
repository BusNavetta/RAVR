"""Benchmark modern additive codecs on the locked 100K retrieval panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import faiss
import numpy as np

from benchmark_faiss_fixed_rate import (
    _evaluate_reconstruction,
    _load_dataset,
    _sha256,
    _teacher,
)
from faiss_additive_codecs import AdditiveCodecConfig, train_additive_bundle
from faiss_fixed_rate import FaissCodecBundle

# Reports live next to this tool.
HERE = Path(__file__).resolve().parent


DEFAULT_DATA = Path.home() / "Desktop" / "AJigma_data"
DEFAULT_CANDIDATES = (
    "rq:4:6",
    "rq:6:6",
    "rq:8:6",
    "lsq:4:6",
    "lsq:6:6",
    "prq:2:2:6",
    "prq:2:3:6",
    "plsq:2:2:6",
)


def parse_candidate(value: str, seed: int, effort: str) -> AdditiveCodecConfig:
    fields = value.lower().split(":")
    family = fields[0]
    if family in {"rq", "lsq"} and len(fields) == 3:
        stages, nbits = map(int, fields[1:])
        splits = 1
    elif family in {"prq", "plsq"} and len(fields) == 4:
        splits, stages, nbits = map(int, fields[1:])
    else:
        raise ValueError(
            f"invalid candidate {value!r}; expected rq:stages:bits or "
            "prq:splits:stages:bits"
        )
    kwargs = {}
    if effort == "screen":
        kwargs = {
            "rq_kmeans_iterations": 3,
            "rq_refine_iterations": 2,
            "rq_beam_size": 3,
            "lsq_train_iterations": 8,
            "lsq_train_ils_iterations": 3,
            "lsq_encode_ils_iterations": 6,
            "lsq_icm_iterations": 3,
            "lsq_perturbations": 2,
        }
    return AdditiveCodecConfig(
        family=family, stages=stages, nbits=nbits, splits=splits, seed=seed,
        **kwargs,
    )


def render_markdown(report: dict) -> str:
    lines = [
        f"# {report['dataset']} advanced additive-codec comparison",
        "",
        "All rows use the same exact-cosine teacher, query panel, codec-training "
        "sample, explicit IDs, and float16 inverse reconstruction norms.",
        "",
        "| Codec | Nominal bits | Packed B/item | Full B/item | R@1 | R@10 | NDCG@10 | Recon cosine | Train s | Encode s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        codec = row["codec"]
        lines.append(
            f"| {row['display_name']} | {codec['nominal_code_bits_per_item']} | "
            f"{row['code_only_bytes_per_item']:.4f} | "
            f"{row['full_serialized_bytes_per_item']:.4f} | "
            f"{row['teacher_recall_at_1']:.4f} | "
            f"{row['teacher_recall_at_10']:.4f} | "
            f"{row['teacher_ndcg_at_10']:.4f} | "
            f"{row['mean_reconstruction_cosine']:.4f} | "
            f"{codec['train_seconds']:.1f} | {codec['encode_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def run(args) -> dict:
    data_root = Path(args.data_root).expanduser().resolve()
    data = _load_dataset(args.dataset, data_root)
    rng = np.random.default_rng(args.seed)
    train_ids = rng.choice(
        data["codec_pool"], min(args.train_samples, len(data["codec_pool"])),
        replace=False,
    )
    teacher_ids, teacher_scores = _teacher(
        data["gallery"], data["queries"], data["exclude_ids"]
    )
    output_dir = HERE / "outputs" / f"advanced_codecs_{args.dataset}_100k"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = data["root"] / "advanced_codec_bundles"
    bundle_dir.mkdir(exist_ok=True)

    rows = []
    for candidate in args.candidates:
        config = parse_candidate(candidate, args.seed, args.effort)
        suffix = "" if args.effort == "full" else f"_{args.effort}"
        path = bundle_dir / f"{config.file_stem()}{suffix}.fcix"
        print(
            f"[{time.strftime('%H:%M:%S')}] {config.display_name}: "
            f"train={len(train_ids)}, effort={args.effort}", flush=True,
        )
        if path.exists() and not args.force:
            bundle = FaissCodecBundle.load(path)
            expected = config.factory
            if bundle.metadata.get("factory") != expected:
                raise ValueError(f"cached bundle {path} is not {expected}")
        else:
            bundle = train_additive_bundle(data["gallery"], train_ids, config)
            bundle.metadata.update({
                "dataset": data["dataset"],
                "manifest_sha256": _sha256(data["manifest_path"]),
                "effort": args.effort,
            })
            bundle.save(path)
        reconstruction = bundle.decode_normalized()
        metrics, per_query = _evaluate_reconstruction(
            reconstruction, data, teacher_ids, teacher_scores
        )
        row = {
            "method": config.family.upper(),
            "display_name": bundle.metadata["display_name"],
            "seed": args.seed,
            "code_only_bytes_per_item": bundle.code_only_bytes_per_item,
            "full_serialized_bytes_per_item": bundle.persistent_bytes_per_item,
            "persistent_bytes": bundle.persistent_bytes,
            "storage": bundle.storage_breakdown(),
            "codec": bundle.metadata,
            "bundle": str(path),
            "bundle_sha256": _sha256(path),
            **metrics,
            "per_query": per_query,
        }
        rows.append(row)
        (output_dir / f"{config.file_stem()}{suffix}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] {config.display_name}: "
            f"full={row['full_serialized_bytes_per_item']:.4f} B/item, "
            f"R@10={row['teacher_recall_at_10']:.4f}", flush=True,
        )
        del reconstruction, bundle

    # Merge all already-completed candidates for this seed/effort.  This makes
    # partial invocations and interrupted sweeps produce one cumulative table.
    suffix = "" if args.effort == "full" else f"_{args.effort}"
    completed_rows = {}
    for row_path in output_dir.glob(f"*_seed{args.seed}{suffix}.json"):
        if row_path.name.startswith("seed"):
            continue
        completed = json.loads(row_path.read_text(encoding="utf-8"))
        completed_rows[completed["display_name"]] = completed
    rows = sorted(
        completed_rows.values(),
        key=lambda value: (
            value["full_serialized_bytes_per_item"], value["display_name"]
        ),
    )
    report = {
        "dataset": data["dataset"],
        "dataset_slug": args.dataset,
        "seed": args.seed,
        "gallery_items": len(data["gallery"]),
        "dimension": data["gallery"].shape[1],
        "test_queries": len(data["queries"]),
        "manifest_sha256": _sha256(data["manifest_path"]),
        "protocol": {
            "teacher": "exact cosine on original L2-normalised float32 gallery",
            "scoring": "exact cosine on decoded and renormalised vectors",
            "codec_training_items": len(train_ids),
            "effort": args.effort,
            "physical_storage": (
                "actual reloadable bundle: trained codec, packed codes, uint32 "
                "IDs, float16 inverse norms, fixed header, and alignment"
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    stem = f"seed{args.seed}{'' if args.effort == 'full' else '_' + args.effort}"
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (output_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sop", "gldv2"), default="sop")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-samples", type=int, default=12_000)
    parser.add_argument("--effort", choices=("screen", "full"), default="full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--candidates", nargs="+", default=list(DEFAULT_CANDIDATES))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
