from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://s3.amazonaws.com/google-landmark"
# The official index partition is shards 000..099 inclusive; 100 does not exist.
INDEX_SHARDS = tuple(f"{shard:03d}" for shard in range(100))
METADATA = (
    "metadata/index.csv",
    "metadata/index_image_to_landmark.csv",
    "metadata/test.csv",
    "ground_truth/recognition_solution_v2.1.csv",
    "ground_truth/retrieval_solution_v2.1.csv",
)
# Measured from the bucket: every index shard is ~0.88 GB.
SHARD_BYTES = 0.88e9
EXTRACT_FACTOR = 1.0  # unpacked JPEGs weigh about the same as the archive


def _get(url: str, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                temporary = destination.with_suffix(destination.suffix + ".part")
                with open(temporary, "wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
                temporary.replace(destination)
            return
        except Exception as error:  # network flake; retry
            last = error
            print(f"  retry {attempt}/{attempts}: {destination.name}: {error}",
                  flush=True)
    raise RuntimeError(f"could not download {url}") from last


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def check_disk(root: Path, shards: int, drop_tars: bool) -> None:
    """Refuse to start a multi-hour download that cannot possibly fit."""
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    needed = shards * SHARD_BYTES * (1.0 if drop_tars else 1.0 + EXTRACT_FACTOR)
    already = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    print(
        f"disk: {free / 1e9:.1f} GB free at {root}, "
        f"{needed / 1e9:.1f} GB required for {shards} shards"
        f"{' (archives dropped after extraction)' if drop_tars else ''}, "
        f"{already / 1e9:.1f} GB already present",
        flush=True,
    )
    if free < needed:
        raise SystemExit(
            f"not enough space: {free / 1e9:.1f} GB free, "
            f"{needed / 1e9:.1f} GB needed. Either free space, point --root at "
            "another drive, or pass --drop-tars to halve the requirement."
        )


def fetch_shard(root: Path, split: str, shard: str) -> Path | None:
    """Download and verify one archive; returns None if it is already unpacked."""
    archive = root / "tars" / split / f"images_{shard}.tar"
    marker = root / "unpacked" / split / f"images_{shard}.done"
    if marker.exists() and not archive.exists():
        print(f"  done    {split}/{shard} (archive dropped)", flush=True)
        return None
    checksum_path = root / "md5" / split / f"md5.images_{shard}.txt"
    if not checksum_path.exists():
        _get(f"{BASE}/md5sum/{split}/md5.images_{shard}.txt", checksum_path)
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    if archive.exists() and _md5(archive) == expected:
        print(f"  cached  {split}/{shard}", flush=True)
        return archive
    _get(f"{BASE}/{split}/images_{shard}.tar", archive)
    actual = _md5(archive)
    if actual != expected:
        raise RuntimeError(
            f"MD5 mismatch for {archive}: expected {expected}, got {actual}"
        )
    print(f"  ok      {split}/{shard}  {archive.stat().st_size / 1e9:.2f} GB",
          flush=True)
    return archive


def extract(root: Path, split: str, archive: Path, drop_tars: bool) -> None:
    target = root / "images" / split
    target.mkdir(parents=True, exist_ok=True)
    marker = root / "unpacked" / split / f"{archive.stem}.done"
    if marker.exists():
        return
    print(f"  unpack  {archive.name}", flush=True)
    with tarfile.open(archive) as handle:
        handle.extractall(target, filter="data")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok", encoding="utf-8")
    if drop_tars:
        archive.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--jobs", type=int, default=5,
        help="Concurrent downloads. S3 throttles a single stream hard; five "
             "streams measured ~43 MB/s against ~2.4 MB/s for one.",
    )
    parser.add_argument(
        "--shards", type=int, default=len(INDEX_SHARDS),
        help="Take only the first N index shards. The full partition is 100; "
             "a smaller number is for smoke-testing this script, not for the "
             "experiment.",
    )
    parser.add_argument(
        "--drop-tars", action="store_true",
        help="Delete each archive once verified and unpacked.",
    )
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    shards = INDEX_SHARDS[: args.shards]
    if args.shards != len(INDEX_SHARDS):
        print(
            f"WARNING: taking {args.shards} of {len(INDEX_SHARDS)} index shards. "
            "This is not the full index partition.", flush=True,
        )
    check_disk(root, len(shards), args.drop_tars)

    print("metadata", flush=True)
    for name in METADATA:
        destination = root / "metadata" / Path(name).name
        if not destination.exists():
            _get(f"{BASE}/{name}", destination)
        print(f"  ok      {Path(name).name}", flush=True)

    print(f"index: {len(shards)} shards", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        archives = list(pool.map(lambda s: fetch_shard(root, "index", s), shards))
    if not args.skip_extract:
        for archive in archives:
            if archive is not None:
                extract(root, "index", archive, args.drop_tars)

    count = sum(1 for _ in (root / "images" / "index").rglob("*.jpg"))
    print(f"\nindex images extracted: {count}", flush=True)
    print("next:", flush=True)
    print(f"  python 5_gldv2_full/prepare.py --root {root} "
          "--query-root <the 100K module's raw root> --data-root <DATA_ROOT>",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
