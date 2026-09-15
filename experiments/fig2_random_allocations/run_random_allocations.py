from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = Path(__file__).resolve().parent

from benchmark_faiss_fixed_rate import _load_dataset, _teacher  # noqa: E402
from benchmark_nested_additive_ravr import (  # noqa: E402
    ALLOWED_LENGTHS, _evaluation_data, _load_or_calibrate, _selections,
    panel_results_dir,
)
from compute_backend import backend_report, configure_compute_backend  # noqa: E402
from faiss_fixed_rate import FaissCodecBundle  # noqa: E402
from nested_additive_ravr import (  # noqa: E402
    faiss_additive_codebooks, unpack_additive_codes,
)
from ravr_depth_ablation import _roles  # noqa: E402

SLUG = "gldv2"                 # GLDv2, 100K items
RETRIEVAL_K = 10
NBITS = 4
CONCENTRATIONS = (0.15, 0.5, 1.0, 3.0, 10.0)


def samples_path(stem: Path, target: int) -> Path:
    """Raw draws live beside the report, not inside it."""
    return stem.with_name(f"{stem.name}_target{target}_samples.npy")


def option_costs(nbits: int = NBITS) -> np.ndarray:
    base = ALLOWED_LENGTHS[0] * nbits // 8
    return np.array([length * nbits // 8 - base for length in ALLOWED_LENGTHS],
                    dtype=np.int64)


def build_score_table(bundle, gallery: np.ndarray, queries: np.ndarray,
                      device) -> "object":
    """`[queries, items, options]` cosine scores, one per prefix option.

    Inverse norms are rounded to float16 to match what the packed index
    actually stores, so this reproduces the panel's scoring rather than an
    idealised version of it.
    """
    import torch

    codebooks, nbits = faiss_additive_codebooks(bundle)
    codes = unpack_additive_codes(bundle.codes, codebooks.shape[0], nbits)
    book = torch.as_tensor(np.ascontiguousarray(codebooks), device=device)
    code_index = torch.as_tensor(codes.astype(np.int64), device=device)
    q = torch.as_tensor(np.ascontiguousarray(queries), device=device)

    reconstruction = torch.zeros(
        (len(gallery), book.shape[-1]), dtype=torch.float32, device=device
    )
    table = torch.empty(
        (len(queries), len(gallery), len(ALLOWED_LENGTHS)),
        dtype=torch.float32, device=device,
    )
    wanted = {int(length): option for option, length in enumerate(ALLOWED_LENGTHS)}
    for stage in range(book.shape[0]):
        reconstruction += book[stage].index_select(0, code_index[:, stage])
        length = stage + 1
        if length not in wanted:
            continue
        inverse = (1.0 / reconstruction.norm(dim=1).clamp_min(1e-9))
        inverse = inverse.to(torch.float16).to(torch.float32)
        table[:, :, wanted[length]] = (q @ reconstruction.T) * inverse
    del reconstruction, book, code_index, q
    return table


def recall_at_10(table, allocations, teacher, batch: int) -> np.ndarray:
    """Teacher Recall@10 for a batch of allocations, on the GPU."""
    import torch

    out = np.empty(len(allocations), dtype=np.float64)
    queries, items, _ = table.shape
    for start in range(0, len(allocations), batch):
        block = allocations[start:start + batch]
        # allocations are stored as int8 to keep the sweep in memory;
        # gather needs an integer index tensor of at least int32
        index = torch.as_tensor(
            np.ascontiguousarray(block), device=table.device
        ).long()
        # [B, queries, items] gathered from the option axis
        gathered = torch.gather(
            table.unsqueeze(0).expand(len(block), -1, -1, -1),
            3,
            index[:, None, :, None].expand(-1, queries, -1, 1),
        ).squeeze(3)
        top = torch.topk(gathered, RETRIEVAL_K, dim=2).indices
        hits = (top.unsqueeze(3) == teacher[None, :, None, :]).any(3).sum(2)
        out[start:start + len(block)] = (
            hits.to(torch.float64).mean(1) / RETRIEVAL_K
        ).cpu().numpy()
        del gathered, top, hits, index
    return out


def sample_allocation(index: int, items: int, budget: int, options: int,
                      rng: np.random.Generator) -> np.ndarray:
    """One random legal allocation: random histogram, random assignment, exact cap.

    Adjacent options differ by one byte, so the cap is met exactly when the
    option indices sum to `budget`. The draw is repaired by shifting randomly
    chosen eligible items one step at a time, which keeps the repair unbiased
    across items rather than always taking from the same end.
    """
    alpha = CONCENTRATIONS[index % len(CONCENTRATIONS)]
    probabilities = rng.dirichlet(np.full(options, alpha))
    allocation = rng.choice(options, size=items, p=probabilities).astype(np.int8)
    deficit = budget - int(allocation.sum())
    while deficit != 0:
        step = 1 if deficit > 0 else -1
        eligible = np.flatnonzero(
            allocation < options - 1 if step > 0 else allocation > 0
        )
        if not len(eligible):
            raise RuntimeError("no legal repair move remains")
        picked = rng.choice(eligible, size=min(abs(deficit), len(eligible)),
                            replace=False)
        allocation[picked] += step
        deficit = budget - int(allocation.sum())
    return allocation


def sample_chunk(start: int, count: int, items: int, budget: int, options: int,
                 rng: np.random.Generator) -> np.ndarray:
    """A block of draws. Never materialises the whole sweep: at 100K items an
    int8 allocation is 100 KB, but ten thousand of them would be a gigabyte."""
    return np.stack([
        sample_allocation(start + row, items, budget, options, rng)
        for row in range(count)
    ])


def stop_reason(deadline: float | None, stop_file: Path | None) -> str | None:
    """Why the sweep should stop, if it should. Checked only between blocks."""
    if stop_file is not None and stop_file.exists():
        return f"stop file present ({stop_file})"
    if deadline is not None and time.monotonic() >= deadline:
        return "time budget exhausted"
    return None


def run(data_root: Path, factory: str, seed: int, targets: tuple[int, ...],
        draws: int, batch: int, calibration_queries: int, split: str,
        device: str, rng_seed: int, deadline: float | None = None,
        stop_file: Path | None = None, on_target=None,
        stem: Path | None = None) -> dict:
    import torch

    data = _load_dataset(SLUG, data_root)
    roles = _roles(data)
    evaluation = _evaluation_data(data, roles, split)
    gallery = data["gallery"]
    queries = np.asarray(evaluation["queries"], dtype=np.float32)
    teacher_ids, _ = _teacher(gallery, queries, evaluation["exclude_ids"])

    bundle = FaissCodecBundle.load(
        data["root"] / "advanced_codec_bundles" / f"{factory.lower()}_seed{seed}.fcix"
    )
    try:
        state, _, _, _ = _load_or_calibrate(
            data, roles, bundle, factory, seed, calibration_queries, "fp32", False
        )
        nbits = int(bundle.metadata["config"]["nbits"])
        torch_device = torch.device(
            device if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        table = build_score_table(bundle, gallery, queries, torch_device)
    finally:
        bundle.close()

    teacher = torch.as_tensor(np.ascontiguousarray(teacher_ids),
                              device=table.device)
    costs = option_costs(nbits)
    rng = np.random.default_rng(rng_seed)
    results = {}

    remaining_targets = list(targets)
    for target in targets:
        # Split what is left of the budget evenly over the caps still to do,
        # otherwise the first cap consumes the whole allowance and the others
        # never run. A cap that finishes early hands its slack to the rest.
        slice_deadline = None
        if deadline is not None:
            left = max(deadline - time.monotonic(), 0.0)
            slice_deadline = time.monotonic() + left / max(len(remaining_targets), 1)
        remaining_targets.remove(target)
        budget = len(gallery) * (target * nbits // 8
                                 - ALLOWED_LENGTHS[0] * nbits // 8)
        named = _selections(state, len(gallery), seed, target, nbits)
        named_scores = {}
        for name, allocation in named.items():
            spent = int(costs[allocation].sum())
            named_scores[name] = {
                "recall_at_10": float(recall_at_10(
                    table, np.asarray(allocation)[None, :], teacher, 1
                )[0]),
                "variable_bytes": spent,
                "spends_the_cap": bool(spent == budget),
            }

        started = time.perf_counter()
        collected: list[np.ndarray] = []
        done = 0
        halted = None
        unlimited = draws <= 0
        with ThreadPoolExecutor(max_workers=1) as pool:
            block = batch if unlimited else min(batch, draws)
            pending = pool.submit(sample_chunk, done, block, len(gallery),
                                  budget, len(costs), rng)
            while unlimited or done < draws:
                drawn = pending.result()
                done += len(drawn)
                following = (batch if unlimited
                             else min(batch, max(draws - done, 0)))
                pending = (pool.submit(sample_chunk, done, following,
                                       len(gallery), budget, len(costs), rng)
                           if following else None)
                collected.append(recall_at_10(table, drawn, teacher, batch))
                del drawn
                # Checked between blocks, never inside one: the configuration
                # being measured is always finished and recorded first.
                halted = stop_reason(deadline, stop_file)
                if halted:
                    print(f"  stopping after {done:,} draws at target "
                          f"{target}: {halted}", flush=True)
                    break
                if (slice_deadline is not None
                        and time.monotonic() >= slice_deadline):
                    print(f"  target {target} used its share of the budget "
                          f"after {done:,} draws; moving to the next cap",
                          flush=True)
                    break
                if pending is None:
                    break
                if done % (batch * 500) == 0:
                    rate = done / max(time.perf_counter() - started, 1e-9)
                    left = ("" if slice_deadline is None else
                            f", {max(slice_deadline - time.monotonic(), 0) / 60:.0f}"
                            " min left on this cap")
                    print(f"  {done:,} draws  {rate:.1f}/s{left}", flush=True)
            if pending is not None:
                pending.cancel()
        sampled = np.concatenate(collected)
        np.save(samples_path(stem, target), sampled.astype(np.float32))
        elapsed = time.perf_counter() - started

        def percentile_of(value: float) -> float:
            return float(np.mean(sampled < value) * 100.0)

        results[int(target)] = {
            "variable_budget_bytes": int(budget),
            "draws_requested": int(draws),
            "draws_completed": int(len(sampled)),
            "halted_because": halted,
            "seconds": float(elapsed),
            "random": {
                "min": float(sampled.min()), "max": float(sampled.max()),
                "mean": float(sampled.mean()), "std": float(sampled.std(ddof=1)),
                "quantiles": {
                    f"p{q:g}": float(np.percentile(sampled, q))
                    for q in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
                },
                "p01": float(np.percentile(sampled, 1)),
                "p50": float(np.percentile(sampled, 50)),
                "p99": float(np.percentile(sampled, 99)),
                # Raw draws go to a sidecar .npy: at a million per cap they
                # would make the report unreadable and slow to rewrite at
                # every checkpoint.
                "samples_file": samples_path(stem, target).name,
            },
            "named": {
                name: {**entry, "percentile_in_random_range":
                       percentile_of(entry["recall_at_10"])}
                for name, entry in named_scores.items()
            },
        }
        print(f"target {target}: {len(sampled)} random allocations in "
              f"{elapsed:.1f}s; range [{sampled.min():.4f}, "
              f"{sampled.max():.4f}], "
              f"uniform {named_scores['uniform']['recall_at_10']:.4f}, "
              f"ravr {named_scores['ravr']['recall_at_10']:.4f}", flush=True)
        if on_target is not None:
            on_target(dict(results))
        if halted:
            print(f"target {target} completed and recorded; not starting "
                  "another target", flush=True)
            break

    del table, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset": data["dataset"],
        "slug": SLUG,
        "gallery_items": int(len(gallery)),
        "queries": int(len(queries)),
        "factory": factory,
        "seed": int(seed),
        "split": split,
        "calibration_queries": int(calibration_queries),
        "rng_seed": int(rng_seed),
        "compute_backend": backend_report(),
        "targets": results,
    }


def panel_reference(factory: str, split: str, cal: int, seed: int) -> dict:
    """The panel's own numbers, to check the fast scoring path."""
    out = {}
    directory = panel_results_dir(SLUG)
    for path in directory.glob(f"{factory.lower()}_seed{seed}_{split}_u*_cal{cal}.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        out[int(report["target_uniform_length"])] = {
            name: row["teacher_recall_at_10"]
            for name, row in report["rows"].items()
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--targets", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rng-seed", type=int, default=20260903)
    parser.add_argument(
        "--time-budget-hours", type=float, default=9.0,
        help="Wall-clock budget. Checked only between blocks, so the "
             "configuration in flight is always finished and recorded; 0 "
             "disables the limit.",
    )
    parser.add_argument(
        "--stop-file", type=Path, default=None,
        help="Create this file to stop early. The current block finishes, its "
             "target is completed and written, and no further target starts. "
             "Defaults to STOP next to this script.",
    )
    args = parser.parse_args()

    resolved = configure_compute_backend(args.device)
    print(f"compute backend: {resolved} ({backend_report()})", flush=True)

    stop_file = args.stop_file if args.stop_file is not None else HERE / "STOP"
    if stop_file.exists():
        raise SystemExit(
            f"{stop_file} already exists, so the run would stop immediately. "
            "Remove it first."
        )
    deadline = (time.monotonic() + args.time_budget_hours * 3600.0
                if args.time_budget_hours > 0 else None)
    print(f"time budget: {args.time_budget_hours:g} h" if deadline
          else "time budget: none", flush=True)
    print(f"stop file  : create {stop_file} to finish the current "
          "configuration and stop", flush=True)

    stem = HERE / f"random_allocations_{SLUG}_{args.factory.lower()}_seed{args.seed}"
    reference = panel_reference(
        args.factory, args.split, args.calibration_queries, args.seed
    )

    def checkpoint(partial: dict) -> None:
        """Write after every finished target, so an early stop loses nothing."""
        stem.with_suffix(".json").write_text(
            json.dumps({"targets": partial, "panel_reference": reference,
                        "partial": True}, indent=1, sort_keys=True),
            encoding="utf-8",
        )

    report = run(
        Path(args.data_root).expanduser().resolve(), args.factory, args.seed,
        tuple(args.targets), args.draws, args.batch, args.calibration_queries,
        args.split, args.device, args.rng_seed, deadline, stop_file,
        checkpoint, stem,
    )
    report["panel_reference"] = reference
    report["time_budget_hours"] = args.time_budget_hours
    report["partial"] = any(
        entry.get("halted_because") for entry in report["targets"].values()
    )
    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(f"\nreport -> {stem.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
