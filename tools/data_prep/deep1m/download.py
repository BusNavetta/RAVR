from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np

BASE = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP"
GALLERY_ROWS = 1_000_000
HEADER_BYTES = 8


def _read_header(url: str) -> tuple[int, int]:
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        count, dimension = np.frombuffer(response.read(HEADER_BYTES), dtype="<u4")
    return int(count), int(dimension)


def _fetch(url: str, destination: Path, start: int | None = None,
           stop: int | None = None, attempts: int = 5) -> None:
    """Download a whole file, or a byte range, with retries."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if start is not None:
        headers["Range"] = f"bytes={start}-{stop}"
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=300) as response:
                temporary = destination.with_suffix(destination.suffix + ".part")
                written = 0
                with open(temporary, "wb") as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
                        written += len(chunk)
                        if written % (1 << 26) < (1 << 20):
                            print(f"    {written / 1e6:.0f} MB", flush=True)
                temporary.replace(destination)
            return
        except Exception as error:
            last = error
            print(f"  retry {attempt}/{attempts}: {destination.name}: {error}",
                  flush=True)
    raise RuntimeError(f"could not download {url}") from last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=GALLERY_ROWS,
                        help="Gallery prefix length. The experiment is 1,000,000; "
                             "a smaller value is for smoke-testing this script.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.rows != GALLERY_ROWS:
        print(f"WARNING: taking {args.rows} base rows, not the Deep1M "
              f"{GALLERY_ROWS}.", flush=True)

    count, dimension = _read_header(f"{BASE}/base.1B.fbin")
    print(f"base.1B.fbin: {count} x {dimension}", flush=True)
    if args.rows > count:
        raise SystemExit(f"asked for {args.rows} rows, file holds {count}")

    payload = args.rows * dimension * 4
    needed = payload + 8_000_000  # queries and ground truth are small
    free = shutil.disk_usage(root).free
    print(f"disk: {free / 1e9:.1f} GB free at {root}, "
          f"{needed / 1e9:.2f} GB required", flush=True)
    if free < needed:
        raise SystemExit(
            f"not enough space: {free / 1e9:.1f} GB free, "
            f"{needed / 1e9:.2f} GB needed."
        )

    gallery = root / "deep1m_base.fbin"
    if gallery.exists() and gallery.stat().st_size == HEADER_BYTES + payload:
        print("  cached  base prefix", flush=True)
    else:
        print(f"  fetch   base prefix ({payload / 1e6:.0f} MB)", flush=True)
        _fetch(f"{BASE}/base.1B.fbin", root / "deep1m_base.body",
               HEADER_BYTES, HEADER_BYTES + payload - 1)
        # Rewrite a valid fbin header for the prefix so the file is
        # self-describing rather than a bare byte range.
        with open(gallery, "wb") as handle:
            handle.write(np.asarray([args.rows, dimension], dtype="<u4").tobytes())
            with open(root / "deep1m_base.body", "rb") as body:
                while chunk := body.read(1 << 20):
                    handle.write(chunk)
        (root / "deep1m_base.body").unlink()
        print(f"  ok      {gallery.name}", flush=True)

    for name in ("query.public.10K.fbin", "groundtruth.public.10K.ibin"):
        destination = root / name
        if destination.exists():
            print(f"  cached  {name}", flush=True)
            continue
        print(f"  fetch   {name}", flush=True)
        _fetch(f"{BASE}/{name}", destination)
        print(f"  ok      {name}", flush=True)

    print("\ndownload complete. next:", flush=True)
    print(f"  python 6_deep1m/prepare.py --root {root} --data-root <DATA_ROOT>",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
