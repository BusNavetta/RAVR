from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prepare import (  # noqa: E402
    DISPLAY_NAME, SLUG, deterministic_rank, normalise, read_fbin,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="the raw download root holding query.public.10K.fbin")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subset-seed", type=int, default=20260902,
                        help="must match the seed prepare.py used")
    parser.add_argument("--count", type=int, default=None,
                        help="how many extra queries; default: all that remain")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = Path(args.data_root).expanduser().resolve() / SLUG
    manifest_path = out_dir / f"{SLUG}_manifest.npz"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}; run prepare.py first")
    manifest = np.load(manifest_path, allow_pickle=False)
    used = np.asarray(manifest["query_source_ids"], dtype=np.int64)

    raw = read_fbin(Path(args.root).expanduser().resolve()
                    / "query.public.10K.fbin")
    queries_all, report = normalise(raw, "queries")
    order = deterministic_rank(len(queries_all), args.subset_seed)

    # The manifest's own ids must be the head of this ranking, or the seed is
    # wrong and everything below would silently overlap the evaluation split.
    if not np.array_equal(order[:len(used)], used):
        raise SystemExit(
            f"--subset-seed {args.subset_seed} does not reproduce the "
            f"manifest's query ranking; the first {len(used)} ids differ, so "
            "the extra queries cannot be shown to be disjoint from validation "
            "and test"
        )
    remaining = order[len(used):]
    if args.count is not None:
        remaining = remaining[: int(args.count)]
    if not len(remaining):
        raise SystemExit("no official query is left over")

    extra = np.ascontiguousarray(queries_all[remaining], dtype=np.float32)
    out = args.out or (out_dir / f"{SLUG}_extra_calibration.npy")
    np.save(out, extra)
    out.with_suffix(".json").write_text(json.dumps({
        "dataset": DISPLAY_NAME,
        "slug": SLUG,
        "subset_seed": int(args.subset_seed),
        "already_used_queries": int(len(used)),
        "extra_queries": int(len(extra)),
        "source_ids": [int(x) for x in remaining],
        "query_normalisation": report,
        "note": "continuation of prepare.py's deterministic ranking, so these "
                "are disjoint from the validation and test splits by "
                "construction",
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{len(extra):,} extra official queries -> {out}", flush=True)
    print(f"calibration pool can now reach {len(used) - 800 + len(extra):,} "
          "(the manifest's 5,000 plus these)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
