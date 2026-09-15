from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent

from benchmark_faiss_fixed_rate import _load_dataset  # noqa: E402
from benchmark_nested_additive_ravr import (  # noqa: E402
    ALLOWED_LENGTHS, _state_path,
)
from faiss_fixed_rate import FaissCodecBundle  # noqa: E402
from nested_additive_ravr import (  # noqa: E402
    faiss_additive_codebooks, unpack_additive_codes,
)
from ravr_depth_ablation import _roles  # noqa: E402

SLUG = "deep1m"
RETRIEVAL_K = 10
PANEL_DEPTH = 50           # hard_negative_depth the panels ran with
DEPTH_LADDER = (50, 100, 200, 400, 800, 1600, 3200)
NBITS = 4


def required_coverage(target: int, nbits: int = NBITS) -> float:
    base = ALLOWED_LENGTHS[0] * nbits // 8
    costs = [length * nbits // 8 - base for length in ALLOWED_LENGTHS]
    return (target * nbits // 8 - base) / costs[-1]


def measure(data_root: Path, factory: str, seed: int, calibration_queries: int,
            depths: tuple[int, ...], device: str, batch: int,
            slug: str) -> dict:
    import torch

    data = _load_dataset(slug, data_root)
    roles = _roles(data)
    gallery = data["gallery"]
    queries = np.asarray(roles["calibration"][:calibration_queries],
                         dtype=np.float32)
    bundle = FaissCodecBundle.load(
        data["root"] / "advanced_codec_bundles" / f"{factory.lower()}_seed{seed}.fcix"
    )
    try:
        codebooks, nbits = faiss_additive_codebooks(bundle)
        codes = unpack_additive_codes(bundle.codes, codebooks.shape[0], nbits)
    finally:
        bundle.close()

    torch_device = torch.device(
        device if device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    base = ALLOWED_LENGTHS[0]
    book = torch.as_tensor(np.ascontiguousarray(codebooks[:base]), device=torch_device)
    reconstruction = torch.zeros(
        (len(gallery), book.shape[-1]), dtype=torch.float32, device=torch_device
    )
    for stage in range(base):
        index = torch.as_tensor(codes[:, stage].astype(np.int64), device=torch_device)
        reconstruction += book[stage].index_select(0, index)
    inverse = 1.0 / reconstruction.norm(dim=1).clamp_min(1e-9)
    gallery_t = torch.as_tensor(np.ascontiguousarray(gallery), device=torch_device)

    largest = max(depths)
    covered = {d: torch.zeros(len(gallery), dtype=torch.bool, device=torch_device)
               for d in depths}
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(queries), batch):
            block = torch.as_tensor(
                np.ascontiguousarray(queries[start:start + batch]),
                device=torch_device,
            )
            exact = block @ gallery_t.T
            approximate = (block @ reconstruction.T) * inverse
            # tau is the midpoint between the k-th and (k+1)-th teacher scores
            top = torch.topk(exact, RETRIEVAL_K + 1, dim=1)
            positives = top.indices[:, :RETRIEVAL_K]
            tau = 0.5 * (top.values[:, RETRIEVAL_K - 1] + top.values[:, RETRIEVAL_K])
            boundary = torch.topk(
                -(exact - tau[:, None]).abs(), largest, dim=1
            ).indices
            intruder = torch.topk(approximate - exact, largest, dim=1).indices
            high_base = torch.topk(approximate, largest, dim=1).indices
            for d in depths:
                mined = torch.cat(
                    (positives, boundary[:, :d], intruder[:, :d], high_base[:, :d]),
                    dim=1,
                ).reshape(-1)
                covered[d][mined] = True
            if start % (batch * 100) == 0:
                done = start + len(block)
                rate = done / max(time.perf_counter() - started, 1e-9)
                print(f"  {done}/{len(queries)} queries  {rate:.0f} q/s  "
                      f"eta {(len(queries) - done) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)

    result = {int(d): float(covered[d].float().mean().item()) for d in depths}
    del gallery_t, reconstruction, covered
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "coverage_by_depth": result,
        "seconds": float(time.perf_counter() - started),
        "gallery_items": int(len(gallery)),
        "calibration_queries": int(len(queries)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", default=SLUG)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEPTH_LADDER))
    parser.add_argument("--targets", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    depths = tuple(sorted(set(args.depths) | {PANEL_DEPTH}))
    measured = measure(data_root, args.factory, args.seed,
                       args.calibration_queries, depths, args.device,
                       args.batch, args.dataset)
    coverage = measured["coverage_by_depth"]


    data = _load_dataset(args.dataset, data_root)
    state_file = _state_path(data, args.factory, args.seed,
                             args.calibration_queries, "fp32")
    recorded = None
    if state_file.exists():
        with np.load(state_file, allow_pickle=False) as state:
            recorded = float(np.mean(np.asarray(state["coverage"]) > 0))

    requirements = {int(t): required_coverage(t) for t in args.targets}
    needed = {}
    for target, requirement in requirements.items():
        reachable = [d for d in sorted(coverage) if coverage[d] >= requirement]
        needed[target] = {
            "required_coverage": requirement,
            "depth_needed": int(min(reachable)) if reachable else None,
            "mined_set_size": (RETRIEVAL_K + 3 * int(min(reachable))
                               if reachable else None),
        }

    report = {
        "dataset": args.dataset,
        "factory": args.factory,
        "seed": args.seed,
        **measured,
        "panel_hard_negative_depth": PANEL_DEPTH,
        "panel_recorded_coverage": recorded,
        "reproduced_coverage_at_panel_depth": coverage.get(PANEL_DEPTH),
        "targets": needed,
    }
    stem = HERE / f"mining_width_{args.dataset}_{args.factory.lower()}"
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        f"# Mining width and coverage -- {data['dataset']}", "",
        f"Gallery {measured['gallery_items']:,} items, "
        f"{measured['calibration_queries']:,} calibration queries, seed "
        f"{args.seed}, {args.factory}.", "",
    ]
    if recorded is not None:
        lines += [
            f"Reproduction check: this script gives "
            f"{coverage.get(PANEL_DEPTH, float('nan')):.4f} at the depth the "
            f"panel used ({PANEL_DEPTH}); the panel recorded {recorded:.4f}.",
            "",
        ]
    lines += [
        "| `hard_negative_depth` | Mined set per query | Coverage | "
        "Calibration cost |",
        "|---:|---:|---:|---:|",
    ]
    for d in sorted(coverage):
        size = RETRIEVAL_K + 3 * d
        lines.append(f"| {d} | {size} | {coverage[d]:.4f} | "
                     f"{size / (RETRIEVAL_K + 3 * PANEL_DEPTH):.1f}x |")
    lines += [
        "",
        "| Target | Coverage needed | Depth that reaches it | Mined set |",
        "|---:|---:|---:|---:|",
    ]
    for target, entry in sorted(needed.items()):
        depth = entry["depth_needed"]
        lines.append(
            f"| {target} | {entry['required_coverage']:.2f} | "
            f"{depth if depth else 'beyond the ladder'} | "
            f"{entry['mined_set_size'] if depth else '-'} |"
        )
    lines += [
        "",
        "The mined set is what the Ordinal-KL accumulation runs over, so "
        "calibration cost grows with it roughly linearly. Widening the mining "
        "is the only knob that acts on the constant behind "
        "`coverage = O(Q * width / N)`; more calibration queries cannot, "
        "because coverage saturates well below 1 (see `analyze_unseen.py`).",
        "",
    ]
    text = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    sys.exit(main())
