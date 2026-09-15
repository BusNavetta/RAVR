from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from tfrecord.reader import tfrecord_loader
from torch.utils.data import IterableDataset


FEATURE_NAME = "sop_100k_dinov2_vits14.npy"
MANIFEST_NAME = "sop_100k_manifest.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _source(args) -> tuple[Path, dict, list[tuple[Path, int, int]]]:
    root = Path(args.tfds_root).expanduser().resolve()
    info_path = root / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    splits = {item["name"]: item for item in info["splits"]}
    shards: list[tuple[Path, int, int]] = []
    start = 0
    for split in ("train", "test"):
        paths = sorted(root.glob(f"stanford_online_products-{split}.tfrecord-*"))
        lengths = [int(value) for value in splits[split]["shardLengths"]]
        if len(paths) != len(lengths):
            raise ValueError(f"{split}: {len(paths)} shards but {len(lengths)} lengths")
        for path, length in zip(paths, lengths, strict=True):
            shards.append((path, start, length))
            start += length
    expected = sum(int(splits[name]["statistics"]["numExamples"])
                   for name in ("train", "test"))
    if start != expected:
        raise AssertionError(f"shard ledger has {start}, expected {expected}")
    return info_path, info, shards


def prepare(args) -> None:
    info_path, info, shards = _source(args)
    labels = np.empty(sum(length for _, _, length in shards), dtype=np.int32)
    seen = 0
    started = time.perf_counter()
    for path, start, expected in shards:
        count = 0
        for count, record in enumerate(tfrecord_loader(str(path), None), 1):
            labels[start + count - 1] = int(record["class_id"][0])
        if count != expected:
            raise AssertionError(f"{path.name}: read {count}, expected {expected}")
        seen += count
        print(f"Scanned {seen:,}/{len(labels):,} TFRecords", flush=True)

    rng = np.random.default_rng(args.seed)
    selected_source = np.sort(rng.choice(len(labels), args.gallery_size, replace=False))
    selected_labels = labels[selected_source]
    by_label: dict[int, list[int]] = defaultdict(list)
    for item, label in enumerate(selected_labels):
        by_label[int(label)].append(item)
    eligible = np.asarray(
        [label for label, items in by_label.items() if len(items) >= 4], dtype=np.int32
    )
    if len(eligible) < args.query_classes:
        raise RuntimeError(f"only {len(eligible)} labels have at least four selected views")
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
    if args.calibration_queries < len(calibration_ids):
        raise ValueError("calibration_queries cannot be smaller than query_classes")
    role_seed_ids = np.concatenate((calibration_ids, validation_ids, test_ids))
    extra_count = args.calibration_queries - len(calibration_ids)
    if extra_count:
        available = np.setdiff1d(
            np.arange(args.gallery_size, dtype=np.int64), role_seed_ids,
            assume_unique=False,
        )
        calibration_ids = np.concatenate((
            calibration_ids, rng.choice(available, extra_count, replace=False)
        ))
    query_union = np.concatenate((calibration_ids, validation_ids, test_ids))
    if len(np.unique(query_union)) != len(query_union):
        raise AssertionError("query roles overlap")
    codec_train_ids = np.setdiff1d(
        np.arange(args.gallery_size, dtype=np.int64), query_union,
        assume_unique=False,
    )

    output = Path(args.data_root).expanduser().resolve() / "sop"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    np.savez(
        manifest_path,
        paths=np.asarray([f"tfds:{value}" for value in selected_source]),
        labels=selected_labels,
        source_ids=selected_source.astype(np.int64),
        calibration_ids=calibration_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        codec_train_ids=codec_train_ids,
        seed=np.asarray(args.seed),
        panel=np.asarray("TFDS-record-order replication"),
    )
    counts = np.asarray([len(items) for items in by_label.values()])
    report = {
        "dataset": "Stanford Online Products",
        "panel": "independent deterministic TFDS-record-order replication",
        "not_the_committed_locked_panel": True,
        "source": str(info_path.parent),
        "dataset_info_sha256": _sha256(info_path),
        "listed_images": len(labels),
        "gallery_items": len(selected_source),
        "classes": len(by_label),
        "class_size_min": int(counts.min()),
        "class_size_median": float(np.median(counts)),
        "class_size_max": int(counts.max()),
        "calibration_queries": len(calibration_ids),
        "validation_queries": len(validation_ids),
        "test_queries": len(test_ids),
        "query_roles_disjoint": True,
        "codec_training_excludes_all_query_ids": True,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "scan_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "sop_100k_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


class SelectedTFRecordDataset(IterableDataset):
    def __init__(self, shards, selected_source, completed, transform):
        self.shards = shards
        self.selected_source = np.asarray(selected_source, dtype=np.int64)
        self.completed = np.asarray(completed, dtype=bool)
        self.transform = transform
        self.position = {int(source): item for item, source in enumerate(selected_source)
                         if not self.completed[item]}

    def __len__(self):
        return len(self.position)

    def __iter__(self):
        from io import BytesIO
        from PIL import Image
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        for path, start, _ in self.shards[worker_id::worker_count]:
            for local, record in enumerate(tfrecord_loader(str(path), None)):
                output_id = self.position.get(start + local)
                if output_id is None:
                    continue
                with Image.open(BytesIO(record["image"])) as image:
                    value = self.transform(image.convert("RGB"))
                yield value, output_id


def embed(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    _, _, shards = _source(args)
    output_root = Path(args.data_root).expanduser().resolve() / "sop"
    manifest = np.load(output_root / MANIFEST_NAME, allow_pickle=False)
    selected = np.asarray(manifest["source_ids"], dtype=np.int64)
    feature_path = output_root / FEATURE_NAME
    completed_path = output_root / "sop_100k_embedding_completed.npy"
    if feature_path.exists():
        features = np.lib.format.open_memmap(feature_path, mode="r+")
        if features.shape != (len(selected), 384):
            raise ValueError("existing feature array has the wrong shape")
    else:
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(len(selected), 384)
        )
        features[:] = np.nan
        features.flush()
    completed = (
        np.load(completed_path).astype(bool)
        if completed_path.exists()
        else np.zeros(len(selected), dtype=bool)
    )
    if completed.shape != (len(selected),):
        raise ValueError("completion mask has the wrong shape")

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
    ).eval().to(device)
    dataset = SelectedTFRecordDataset(shards, selected, completed, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )
    started = time.perf_counter()
    initial = int(completed.sum())
    processed = initial
    for batch_number, (images, ids) in enumerate(loader, 1):
        with torch.inference_mode():
            value = model(images.to(device, non_blocking=device.type == "cuda")).float()
            value = torch.nn.functional.normalize(value, dim=1)
        ids_numpy = ids.numpy()
        features[ids_numpy] = value.cpu().numpy()
        completed[ids_numpy] = True
        processed += len(ids_numpy)
        if batch_number % 25 == 0 or processed == len(selected):
            features.flush()
            np.save(completed_path, completed)
            elapsed = time.perf_counter() - started
            print(
                f"{processed:,}/{len(selected):,} "
                f"({(processed-initial)/max(elapsed, 1e-9):.1f} images/s)",
                flush=True,
            )
    if not np.all(completed):
        raise RuntimeError(f"only {int(completed.sum())}/{len(completed)} selected records found")
    norms = np.linalg.norm(features, axis=1)
    if not np.all(np.isfinite(features)) or float(np.max(np.abs(norms - 1))) > 2e-5:
        raise RuntimeError("feature validation failed")
    report = {
        "model": "facebookresearch/dinov2:dinov2_vits14",
        "dimension": 384,
        "items": len(selected),
        "dtype": "float32",
        "l2_normalized": True,
        "max_norm_error": float(np.max(np.abs(norms - 1))),
        "features": str(feature_path),
        "features_sha256": _sha256(feature_path),
        "device": str(device),
        "images_per_second": (len(selected) - initial) / max(time.perf_counter() - started, 1e-9),
    }
    _atomic_json(output_root / "sop_100k_embeddings.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare", prepare), ("embed", embed)):
        command = subparsers.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--tfds-root", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--gallery-size", type=int, default=100000)
        command.add_argument("--query-classes", type=int, default=400)
        command.add_argument("--calibration-queries", type=int, default=5000)
        if name == "embed":
            command.add_argument("--batch-size", type=int, default=64)
            command.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
