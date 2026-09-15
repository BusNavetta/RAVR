from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

DISPLAY_NAME = "Deep1M (Deep1B base prefix, 96-d)"
SLUG = "deep1m"


def read_fbin(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        count, dimension = np.frombuffer(handle.read(8), dtype="<u4")
        data = np.frombuffer(
            handle.read(int(count) * int(dimension) * 4), dtype="<f4"
        )
    return data.reshape(int(count), int(dimension))


def read_ibin(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        count, dimension = np.frombuffer(handle.read(8), dtype="<u4")
        data = np.frombuffer(
            handle.read(int(count) * int(dimension) * 4), dtype="<i4"
        )
    return data.reshape(int(count), int(dimension))


def deterministic_rank(count: int, seed: int) -> np.ndarray:
    keys = [hashlib.sha256(f"{seed}:{i}".encode()).hexdigest() for i in range(count)]
    return np.argsort(np.asarray(keys))


def normalise(vectors: np.ndarray, name: str) -> tuple[np.ndarray, dict]:
    norms = np.linalg.norm(vectors, axis=1)
    report = {
        "rows": int(len(vectors)),
        "min_norm_before": float(norms.min()),
        "max_norm_before": float(norms.max()),
        "max_abs_deviation_before": float(np.abs(norms - 1.0).max()),
        "zero_norm_rows": int((norms == 0).sum()),
    }
    out = vectors / np.maximum(norms, 1e-12)[:, None]
    after = np.linalg.norm(out, axis=1)
    report["max_abs_deviation_after"] = float(np.abs(after - 1.0).max())
    print(f"  {name}: {report['rows']} rows, norm deviation before "
          f"{report['max_abs_deviation_before']:.3e}, after "
          f"{report['max_abs_deviation_after']:.3e}", flush=True)
    return np.ascontiguousarray(out, dtype=np.float32), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--validation-queries", type=int, default=400)
    parser.add_argument("--test-queries", type=int, default=400)
    parser.add_argument("--subset-seed", type=int, default=20260902)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.data_root).expanduser().resolve() / SLUG
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = (args.calibration_queries + args.validation_queries
              + args.test_queries)

    print("reading", flush=True)
    gallery_raw = read_fbin(root / "deep1m_base.fbin")
    queries_raw = read_fbin(root / "query.public.10K.fbin")
    official = read_ibin(root / "groundtruth.public.10K.ibin")
    print(f"  gallery {gallery_raw.shape}, queries {queries_raw.shape}, "
          f"official ground truth {official.shape}", flush=True)
    if gallery_raw.shape[1] != queries_raw.shape[1]:
        raise SystemExit("gallery and query dimensions differ")
    if len(queries_raw) < needed:
        raise SystemExit(f"{len(queries_raw)} queries available, {needed} needed")
    if len(official) != len(queries_raw):
        raise SystemExit(
            f"ground truth covers {len(official)} queries but the query file "
            f"holds {len(queries_raw)}"
        )

    print("normalising once, at ingest", flush=True)
    gallery, gallery_report = normalise(gallery_raw, "gallery")
    queries_all, query_report = normalise(queries_raw, "queries")

    order = deterministic_rank(len(queries_all), args.subset_seed)[:needed]
    queries = np.ascontiguousarray(queries_all[order], dtype=np.float32)
    positions = np.arange(needed, dtype=np.int64)

    np.save(out_dir / f"{SLUG}_gallery.npy", gallery)
    np.save(out_dir / f"{SLUG}_queries.npy", queries)
    np.savez(
        out_dir / f"{SLUG}_manifest.npz",
        display_name=np.array(DISPLAY_NAME),
        # Deep1M carries no class labels; -1 disables the panel's semantic
        # metric without disabling the ranking-fidelity metrics that matter.
        labels=np.full(len(gallery), -1, dtype=np.int64),
        query_labels=np.full(needed, -1, dtype=np.int64),
        codec_train_ids=np.arange(len(gallery), dtype=np.int64),
        calibration_ids=positions[:args.calibration_queries],
        validation_ids=positions[
            args.calibration_queries:
            args.calibration_queries + args.validation_queries
        ],
        test_ids=positions[args.calibration_queries + args.validation_queries:],
        query_source_ids=order.astype(np.int64),
        subset_seed=np.array(args.subset_seed),
    )
    # The official list indexes the 1B base; keep it aligned to the queries we
    # actually use, so the teacher check needs no re-indexing later.
    np.savez(
        out_dir / f"{SLUG}_official_gt.npz",
        neighbours=official[order].astype(np.int64),
        query_source_ids=order.astype(np.int64),
        gallery_rows=np.array(len(gallery), dtype=np.int64),
        note=np.array(
            "published 100-NN against the full Deep1B base; this gallery is a "
            "verbatim prefix of that base, so official neighbours with id < "
            "gallery_rows are true neighbours here in the same relative order"
        ),
    )
    (out_dir / f"{SLUG}_manifest.json").write_text(
        json.dumps({
            "display_name": DISPLAY_NAME,
            "gallery_items": int(len(gallery)),
            "dimension": int(gallery.shape[1]),
            "queries": needed,
            "queries_available": int(len(queries_all)),
            "calibration_queries": args.calibration_queries,
            "validation_queries": args.validation_queries,
            "test_queries": args.test_queries,
            "codec_pool_size": int(len(gallery)),
            "subset_seed": args.subset_seed,
            "gallery_normalisation": gallery_report,
            "query_normalisation": query_report,
            "official_ground_truth_neighbours": int(official.shape[1]),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir}. next:", flush=True)
    print(f"  python 6_deep1m/run_deep1m.py --data-root "
          f"{Path(args.data_root).expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
