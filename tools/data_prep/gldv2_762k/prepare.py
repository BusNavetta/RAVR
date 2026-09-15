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
DIMENSION = 384
DISPLAY_NAME = "Google Landmarks Dataset v2.1, full index"


def image_path(images_root: Path, image_id: str) -> Path:
    return images_root / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"


def deterministic_rank(ids: list[str], seed: int) -> np.ndarray:
    keys = [hashlib.sha256(f"{seed}:{i}".encode()).hexdigest() for i in ids]
    return np.argsort(np.asarray(keys))


def read_csv_column(path: Path, key: str, value: str) -> dict[str, str]:
    """Read one column, treating GLDv2's literal "None" as absent."""
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(value)
            if raw and raw != "None":
                out[row[key]] = raw
    return out


def available_ids(images_root: Path) -> list[str]:
    return sorted(p.stem for p in images_root.rglob("*.jpg"))


def embed(paths: list[Path], batch_size: int, device: str, out: np.ndarray,
          workers: int = 8, log_every: int = 20) -> None:
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image, ImageFile
    from transformers import AutoImageProcessor, AutoModel

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    started = time.perf_counter()

    def load(path: Path):
        with Image.open(path) as handle:
            return handle.convert("RGB")

    with ThreadPoolExecutor(max_workers=workers) as pool, torch.inference_mode():
        for index, start in enumerate(range(0, len(paths), batch_size)):
            chunk = paths[start:start + batch_size]
            images = list(pool.map(load, chunk))
            inputs = processor(images=images, return_tensors="pt").to(device)
            cls = model(**inputs).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=1)
            out[start:start + len(chunk)] = cls.float().cpu().numpy()
            if index % log_every == 0:
                done = start + len(chunk)
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(f"  {done}/{len(paths)}  {rate:.0f} img/s  "
                      f"eta {(len(paths) - done) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Where 5_gldv2_full/download.py put images/index.")
    parser.add_argument("--query-root", type=Path, required=True,
                        help="A root holding images/test; the 100K module's raw "
                             "root serves, since the query partition is shared.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Files land in <data-root>/gldv2_full/.")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--validation-queries", type=int, default=400)
    parser.add_argument("--test-queries", type=int, default=400)
    parser.add_argument("--subset-seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", nargs="+", default=["manifest", "gallery", "queries"],
                        choices=("manifest", "gallery", "queries"))
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    query_root = Path(args.query_root).expanduser().resolve()
    out_dir = Path(args.data_root).expanduser().resolve() / "gldv2_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "gldv2_full_manifest.npz"
    stages = set(args.stages)

    index_ids = available_ids(root / "images" / "index")
    test_ids_all = available_ids(query_root / "images" / "test")
    needed = args.calibration_queries + args.validation_queries + args.test_queries
    print(f"extracted: {len(index_ids)} index, {len(test_ids_all)} test",
          flush=True)
    if not index_ids:
        raise SystemExit(f"no index images under {root / 'images' / 'index'}")
    if len(test_ids_all) < needed:
        raise SystemExit(
            f"only {len(test_ids_all)} test images under {query_root}; "
            f"{needed} needed"
        )

    if "manifest" in stages or not manifest_path.exists():
        # Every extracted index image is a gallery row: no cap, no subset rule.
        gallery_ids = index_ids
        query_order = deterministic_rank(test_ids_all, args.subset_seed)
        query_ids = [test_ids_all[i] for i in query_order[:needed]]

        metadata = root / "metadata"
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
        # The generic panel branch reads codec_train_ids from the manifest, so
        # the pool is sized from the gallery here instead of the hard-coded
        # 100000 that _load_dataset applies to the `gldv2` slug.
        codec_train_ids = np.arange(len(gallery_ids), dtype=np.int64)
        np.savez(
            manifest_path,
            display_name=np.array(DISPLAY_NAME),
            gallery_ids=np.array(gallery_ids),
            query_ids=np.array(query_ids),
            labels=gallery_labels,
            query_labels=query_labels,
            query_relevant_gallery_ids=np.array(
                [str(retrieval.get(i, "")) for i in query_ids]
            ),
            codec_train_ids=codec_train_ids,
            calibration_ids=positions[:args.calibration_queries],
            validation_ids=positions[
                args.calibration_queries:
                args.calibration_queries + args.validation_queries
            ],
            test_ids=positions[args.calibration_queries + args.validation_queries:],
            subset_seed=np.array(args.subset_seed),
        )
        gallery_id_set = set(gallery_ids)
        covered = int(np.sum([
            bool(set(str(retrieval.get(i, "")).split()) & gallery_id_set)
            for i in query_ids
        ]))
        (out_dir / "gldv2_full_manifest.json").write_text(
            json.dumps({
                "display_name": DISPLAY_NAME,
                "gallery_items": len(gallery_ids),
                "queries": needed,
                "queries_with_official_relevant_in_gallery": covered,
                "distinct_gallery_labels": int(len(np.unique(gallery_labels))),
                "unlabelled_gallery_items": int((gallery_labels < 0).sum()),
                "codec_pool_size": int(len(codec_train_ids)),
                "subset_seed": args.subset_seed,
                "note": "full official index partition, no 100K cap",
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"manifest: {len(gallery_ids)} gallery, {needed} queries, "
              f"{covered} with an official relevant item inside the gallery",
              flush=True)
    else:
        with np.load(manifest_path, allow_pickle=False) as data:
            gallery_ids = [str(x) for x in data["gallery_ids"]]
            query_ids = [str(x) for x in data["query_ids"]]
        print(f"reusing manifest {manifest_path.name}", flush=True)

    if "gallery" in stages:
        print(f"embedding {len(gallery_ids)} gallery images on {args.device}",
              flush=True)
        path = out_dir / "gldv2_full_gallery.npy"
        gallery = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(gallery_ids), DIMENSION)
        )
        embed([image_path(root / "images" / "index", i) for i in gallery_ids],
              args.batch_size, args.device, gallery)
        gallery.flush()
        del gallery

    if "queries" in stages:
        print(f"embedding {len(query_ids)} queries on {args.device}", flush=True)
        path = out_dir / "gldv2_full_queries.npy"
        queries = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(query_ids), DIMENSION)
        )
        embed([image_path(query_root / "images" / "test", i) for i in query_ids],
              args.batch_size, args.device, queries)
        queries.flush()
        del queries

    print(f"\nwrote {out_dir}. next:", flush=True)
    print(f"  python 5_gldv2_full/run_gldv2_full.py --data-root "
          f"{Path(args.data_root).expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
