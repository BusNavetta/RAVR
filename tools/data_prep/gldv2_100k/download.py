from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tarfile
import urllib.request

BASE = "https://s3.amazonaws.com/google-landmark"
INDEX_SHARDS = (
    "090", "007", "073", "098", "013", "026", "000",
    "077", "058", "092", "040", "033", "042", "086",
)
TEST_SHARDS = ("012", "007")
METADATA = (
    "metadata/index.csv",
    "metadata/index_image_to_landmark.csv",
    "metadata/test.csv",
    "ground_truth/recognition_solution_v2.1.csv",
    "ground_truth/retrieval_solution_v2.1.csv",
)


def _get(url: str, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                temporary = destination.with_suffix(destination.suffix + ".part")
                with open(temporary, "wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
                temporary.replace(destination)
            return
        except Exception as error:  # network flake; retry
            last = error
            print(f"  retry {attempt}/{attempts}: {error}", flush=True)
    raise RuntimeError(f"could not download {url}") from last


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_shard(root: Path, split: str, shard: str) -> Path:
    archive = root / "tars" / split / f"images_{shard}.tar"
    checksum_path = root / "md5" / split / f"md5.images_{shard}.txt"
    _get(f"{BASE}/md5sum/{split}/md5.images_{shard}.txt", checksum_path)
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    if archive.exists() and _md5(archive) == expected:
        print(f"  cached  {split}/{shard}", flush=True)
        return archive
    print(f"  fetch   {split}/{shard}", flush=True)
    _get(f"{BASE}/{split}/images_{shard}.tar", archive)
    actual = _md5(archive)
    if actual != expected:
        raise RuntimeError(
            f"MD5 mismatch for {archive}: expected {expected}, got {actual}"
        )
    print(f"  ok      {split}/{shard}", flush=True)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Verify and keep the archives without unpacking them.",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    print("metadata", flush=True)
    for name in METADATA:
        _get(f"{BASE}/{name}", root / "metadata" / Path(name).name)
        print(f"  ok      {Path(name).name}", flush=True)

    for split, shards in (("index", INDEX_SHARDS), ("test", TEST_SHARDS)):
        print(f"{split}: {len(shards)} shards", flush=True)
        archives = [fetch_shard(root, split, shard) for shard in shards]
        if args.skip_extract:
            continue
        target = root / "images" / split
        target.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            print(f"  unpack  {archive.name}", flush=True)
            with tarfile.open(archive) as handle:
                handle.extractall(target, filter="data")

    print("\ndownload complete. next:", flush=True)
    print(f"  python 4_gldv2/prepare.py --root {root}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
