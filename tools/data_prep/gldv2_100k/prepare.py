from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

MODEL = "facebook/dinov2-small"
GALLERY_SIZE = 100_000
DIMENSION = 384


def image_path(images_root: Path, image_id: str) -> Path:
    """GLDv2 stores <id>.jpg under three levels of its own leading characters."""
    return images_root / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"


def deterministic_rank(ids: list[str], seed: int) -> np.ndarray:
    """Order ids by SHA-256 of "<seed>:<id>" -- stable across machines and runs."""
    keys = [hashlib.sha256(f"{seed}:{i}".encode()).hexdigest() for i in ids]
    return np.argsort(np.asarray(keys))


def read_csv_column(path: Path, key: str, value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(value)
            if raw and raw != "None":
                out[row[key]] = raw
    return out


def available_ids(images_root: Path) -> list[str]:
    return sorted(p.stem for p in images_root.rglob("*.jpg"))


def embed(paths: list[Path], batch_size: int, device: str,
          workers: int = 8) -> np.ndarray:
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image, ImageFile
    from transformers import AutoImageProcessor, AutoModel

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    out = np.empty((len(paths), DIMENSION), dtype=np.float32)
    started = time.perf_counter()

    def load(path: Path):
        with Image.open(path) as handle:
            return handle.convert("RGB")

    with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            images = list(pool.map(load, chunk))
            inputs = processor(images=images, return_tensors="pt").to(device)
            cls = model(**inputs).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=1)
            out[start:start + len(chunk)] = cls.float().cpu().numpy()
            done = start + len(chunk)
            if start % (batch_size * 20) == 0:
                rate = done / max(time.perf_counter() - started, 1e-9)
                remaining = (len(paths) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(paths)}  {rate:.0f} img/s  "
                      f"eta {remaining / 60:.1f} min", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Where download.py put images/ and metadata/.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Panel data root; files land in <data-root>/gldv2/.")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--validation-queries", type=int, default=400)
    parser.add_argument("--test-queries", type=int, default=400)
    parser.add_argument("--subset-seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Reuse an existing manifest; recompute embeddings only.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.data_root).expanduser().resolve() / "gldv2"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "gldv2_100k_manifest.npz"
    metadata = root / "metadata"

    index_ids = available_ids(root / "images" / "index")
    test_ids_all = available_ids(root / "images" / "test")
    print(f"extracted: {len(index_ids)} index, {len(test_ids_all)} test", flush=True)
    if len(index_ids) < GALLERY_SIZE:
        raise SystemExit(
            f"only {len(index_ids)} index images extracted; "
            f"{GALLERY_SIZE} needed. Re-run download.py."
        )

    needed = args.calibration_queries + args.validation_queries + args.test_queries
    if len(test_ids_all) < needed:
        raise SystemExit(
            f"only {len(test_ids_all)} test images extracted; {needed} needed"
        )

    if args.manifest_only and manifest_path.exists():
        manifest = np.load(manifest_path, allow_pickle=False)
        gallery_ids = [str(x) for x in manifest["gallery_ids"]]
        query_ids = [str(x) for x in manifest["query_ids"]]
        print(f"reusing manifest {manifest_path.name}", flush=True)
    else:
        order = deterministic_rank(index_ids, args.subset_seed)
        gallery_ids = [index_ids[i] for i in order[:GALLERY_SIZE]]
        query_order = deterministic_rank(test_ids_all, args.subset_seed)
        query_ids = [test_ids_all[i] for i in query_order[:needed]]

        landmark_of = read_csv_column(
            metadata / "index_image_to_landmark.csv", "id", "landmark_id"
        )
        recognition = read_csv_column(
            metadata / "recognition_solution_v2.1.csv", "id", "landmarks"
        )
        retrieval = read_csv_column(
            metadata / "retrieval_solution_v2.1.csv", "id", "images"
        )
        gallery_labels = np.array(
            [int(landmark_of.get(i, -1)) for i in gallery_ids], dtype=np.int64
        )
        query_labels = np.array(
            [int(str(recognition.get(i, "-1")).split()[0]) for i in query_ids],
            dtype=np.int64,
        )
        positions = np.arange(needed, dtype=np.int64)
        np.savez(
            manifest_path,
            gallery_ids=np.array(gallery_ids),
            query_ids=np.array(query_ids),
            gallery_labels=gallery_labels,
            query_labels=query_labels,
            query_relevant_gallery_ids=np.array(
                [str(retrieval.get(i, "")) for i in query_ids]
            ),
            calibration_ids=positions[:args.calibration_queries],
            validation_ids=positions[
                args.calibration_queries:args.calibration_queries
                + args.validation_queries
            ],
            test_ids=positions[args.calibration_queries + args.validation_queries:],
            subset_seed=np.array(args.subset_seed),
        )
        gallery_id_set = set(gallery_ids)
        covered = int(np.sum([
            bool(set(str(retrieval.get(i, "")).split()) & gallery_id_set)
            for i in query_ids
        ]))
        covered_anywhere = int(np.sum([bool(retrieval.get(i, "")) for i in query_ids]))
        (out_dir / "gldv2_100k_manifest.json").write_text(
            json.dumps({
                "gallery_items": GALLERY_SIZE,
                "extracted_index_images": len(index_ids),
                "queries": needed,
                "queries_with_official_relevant_in_gallery": covered,
                "queries_with_official_relevant_anywhere": covered_anywhere,
                "distinct_gallery_labels": int(len(np.unique(gallery_labels))),
                "subset_seed": args.subset_seed,
                "note": "SHA-256 rank subset regenerated by 4_gldv2/prepare.py; "
                        "not guaranteed identical to the original manifest",
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"manifest: {GALLERY_SIZE} gallery, {needed} queries, "
            f"{covered} with an official relevant item inside the gallery "
            f"({covered_anywhere} with one anywhere in GLDv2)", flush=True
        )

    print(f"embedding gallery on {args.device}", flush=True)
    gallery = embed(
        [image_path(root / "images" / "index", i) for i in gallery_ids],
        args.batch_size, args.device,
    )
    np.save(out_dir / "gldv2_100k_index_dinov2_vits14.npy", gallery)

    print(f"embedding queries on {args.device}", flush=True)
    queries = embed(
        [image_path(root / "images" / "test", i) for i in query_ids],
        args.batch_size, args.device,
    )
    np.save(out_dir / "gldv2_test_queries_dinov2_vits14.npy", queries)

    print(f"\nwrote {out_dir}. next:", flush=True)
    print(
        f"  python 4_gldv2/run_gldv2.py --data-root "
        f"{Path(args.data_root).expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
