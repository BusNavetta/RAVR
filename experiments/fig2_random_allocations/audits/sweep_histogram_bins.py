"""Robustness check: does the random-allocation range depend on the bins?

`run_random_allocations.py` draws a random legal allocation in two steps: a
random **histogram** over the allowed prefix options, then a random assignment
of that histogram to items, repaired to hit the byte cap exactly. The bins of
that histogram are the allowed prefix lengths -- `(8, 10, 12, 14, 16)` stages,
one byte apart at 4 bits per stage. That choice is a free parameter of the
*baseline*, not of RAVR, and it decides which allocations the sweep can even
reach. This module varies it and leaves everything else alone.

Nothing here touches RAVR. The named strategies are read from the same
calibration state and scored exactly as `run_random_allocations.py` scores
them; only the sampler's bins and the histogram prior move.

Two axes are swept:

* **the bin grid** -- how many prefix options the sampler may use and how far
  apart they sit. Finer bins (half-byte steps) let a draw mix depths the packed
  index cannot store; coarser bins (2 or 4 bytes) force the mass to the ends;
  narrow grids clip the support around the cap; wide grids open depths below
  the base length.
* **the histogram prior** -- the Dirichlet concentration set. Small alpha makes
  near-degenerate histograms, large alpha makes fully mixed ones, and a
  two-point sampler puts all the mass on a random pair of bins, which is the
  highest-variance histogram a grid admits.

For every configuration the script reports the sampled distribution, its tail,
and where the *unchanged* named strategies fall in it. It also runs a
cross-entropy search over the histogram simplex for each grid: the best
histogram random assignment can reach, which upper-bounds what any bin choice
could have produced given more draws.

    PYTHONPATH=3_method python 7_random_allocations/sweep_histogram_bins.py
        --data-root C:/AJigma/data --draws 20000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = Path(__file__).resolve().parent

from benchmark_faiss_fixed_rate import _load_dataset, _teacher  # noqa: E402
from benchmark_nested_additive_ravr import (  # noqa: E402
    ALLOWED_LENGTHS, _evaluation_data, _load_or_calibrate, _selections,
)
from compute_backend import backend_report, configure_compute_backend  # noqa: E402
from faiss_fixed_rate import FaissCodecBundle  # noqa: E402
from nested_additive_ravr import (  # noqa: E402
    faiss_additive_codebooks, unpack_additive_codes,
)
from ravr_depth_ablation import _roles  # noqa: E402

SLUG = "gldv2"
RETRIEVAL_K = 10
BASELINE_CONCENTRATIONS = (0.15, 0.5, 1.0, 3.0, 10.0)
# Every grid the sweep can ask for is a subset of these lengths, so one score
# table serves the whole run. 8 is the shortest prefix the panel allows; 4 and
# 6 are only reached by the deliberately over-wide grids.
TABLE_LENGTHS = tuple(range(4, 17))


def grid_catalogue(cap: int) -> dict[str, dict]:
    """Bin grids, in stages. `width` is the step; mixed grids report None.

    Grids named `narrow_*` are defined relative to the cap: a fixed window such
    as (10, 12, 14) would be infeasible at cap 10, where it could only be met
    by giving every item exactly 10.
    """
    catalogue = {
        # the sampler as shipped: five bins, one byte apart
        "base_w2_m5": dict(lengths=(8, 10, 12, 14, 16), width=2),
        # finer than the packed index can store: half-byte steps
        "fine_w1_m9": dict(lengths=tuple(range(8, 17)), width=1),
        # coarser: two bytes and four bytes per step
        "coarse_w4_m3": dict(lengths=(8, 12, 16), width=4),
        "binary_w8_m2": dict(lengths=(8, 16), width=8),
        # support clipped around the cap
        "narrow_w2_m3": dict(lengths=(cap - 2, cap, cap + 2), width=2),
        "narrow_w1_m5": dict(lengths=tuple(range(cap - 2, cap + 3)), width=1),
        # support opened below the panel's base length
        "wide_w2_m7": dict(lengths=(4, 6, 8, 10, 12, 14, 16), width=2),
        "wide_w4_m4": dict(lengths=(4, 8, 12, 16), width=4),
        # unequal bin widths: dense on one side of the cap, coarse on the other
        "asym_low_m6": dict(lengths=(8, 9, 10, 11, 12, 16), width=None),
        "asym_high_m6": dict(lengths=(8, 12, 13, 14, 15, 16), width=None),
    }
    return {
        name: spec for name, spec in catalogue.items()
        if min(spec["lengths"]) <= cap <= max(spec["lengths"])
        and len(set(spec["lengths"])) == len(spec["lengths"])
        and all(4 <= length <= 16 for length in spec["lengths"])
    }


def prior_catalogue() -> dict[str, dict]:
    """Histogram priors, all on the shipped grid so only the prior moves."""
    return {
        "prior_baseline": dict(kind="dirichlet",
                               concentrations=BASELINE_CONCENTRATIONS),
        "prior_flat": dict(kind="dirichlet", concentrations=(1.0,)),
        "prior_tiny": dict(kind="dirichlet", concentrations=(0.01, 0.03, 0.1)),
        "prior_broad": dict(kind="dirichlet",
                            concentrations=(30.0, 100.0, 1000.0)),
        "prior_twopoint": dict(kind="two_point", concentrations=()),
    }


# --------------------------------------------------------------------------
# scoring -- identical to run_random_allocations.py, only the option axis is
# wider so that every grid in the sweep is a slice of the same table
# --------------------------------------------------------------------------

def build_score_table(codebooks, codes, gallery_size: int, queries: np.ndarray,
                      lengths: tuple[int, ...], device):
    """`[queries, items, len(lengths)]` cosine scores, one per prefix length.

    Inverse norms are rounded to float16 before use, because that is what the
    packed index stores. Copied from the shipped runner on purpose.
    """
    import torch

    book = torch.as_tensor(np.ascontiguousarray(codebooks), device=device)
    code_index = torch.as_tensor(codes.astype(np.int64), device=device)
    q = torch.as_tensor(np.ascontiguousarray(queries), device=device)
    reconstruction = torch.zeros((gallery_size, book.shape[-1]),
                                 dtype=torch.float32, device=device)
    table = torch.empty((len(queries), gallery_size, len(lengths)),
                        dtype=torch.float32, device=device)
    wanted = {int(length): column for column, length in enumerate(lengths)}
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
    """Teacher Recall@10 for a batch of allocations, on the GPU.

    `allocations` index the *table's* option axis, not a grid's.
    """
    import torch

    out = np.empty(len(allocations), dtype=np.float64)
    queries, items, _ = table.shape
    for start in range(0, len(allocations), batch):
        block = allocations[start:start + batch]
        index = torch.as_tensor(np.ascontiguousarray(block),
                                device=table.device).long()
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


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def repair(allocation: np.ndarray, costs: np.ndarray, budget: int,
           rng: np.random.Generator) -> np.ndarray:
    """Shift randomly chosen eligible items one bin at a time until the cap is met.

    The shipped sampler works in option indices because its bins are one byte
    apart, so an index step *is* a byte, and one round moves
    `min(|deficit|, len(eligible))` randomly chosen items. Here bins can have
    any width, so the deficit is tracked in stages and every candidate move
    carries its own cost. On an evenly spaced grid the branch below reduces to
    the shipped procedure exactly: the count is `|deficit| / width`, drawn
    without replacement from the eligible items.
    """
    options = len(costs)
    steps_up = costs[1:] - costs[:-1]
    even = bool(np.all(steps_up == steps_up[0])) if len(steps_up) else True
    total = int(np.bincount(allocation, minlength=options) @ costs)
    for _ in range(100_000):
        deficit = budget - total
        if deficit == 0:
            return allocation
        step = 1 if deficit > 0 else -1
        eligible = np.flatnonzero(
            allocation < options - 1 if step > 0 else allocation > 0
        )
        if not len(eligible):
            raise RuntimeError("no legal repair move remains")
        want = abs(deficit)
        if even:
            width = int(steps_up[0])
            take = min(want // width, len(eligible))
            if take == 0:                     # deficit smaller than one move
                take = 1
            picked = eligible[rng.choice(len(eligible), size=take,
                                         replace=False)]
            total += step * width * take
        else:
            delta = (steps_up[allocation[eligible]] if step > 0
                     else steps_up[allocation[eligible] - 1])
            fits = delta <= want
            if not fits.any():
                # only overshooting moves remain; take the cheapest and let the
                # next round push back from the other side
                cheapest = int(np.argmin(delta))
                picked = eligible[cheapest:cheapest + 1]
                total += step * int(delta[cheapest])
            else:
                usable = eligible[fits]
                order = rng.permutation(len(usable))
                ordered = delta[fits][order]
                take = max(int(np.searchsorted(np.cumsum(ordered), want,
                                               side="right")), 1)
                picked = usable[order[:take]]
                total += step * int(ordered[:take].sum())
        allocation[picked] += step
    raise RuntimeError("repair did not converge")


def draw_histogram(rng: np.random.Generator, options: int, kind: str,
                   concentration: float) -> np.ndarray:
    if kind == "dirichlet":
        return rng.dirichlet(np.full(options, concentration))
    if kind == "two_point":
        probabilities = np.zeros(options)
        if options == 1:
            probabilities[0] = 1.0
            return probabilities
        pair = rng.choice(options, size=2, replace=False)
        share = rng.uniform()
        probabilities[pair[0]] = share
        probabilities[pair[1]] = 1.0 - share
        return probabilities
    raise ValueError(f"unknown sampler kind: {kind}")


def sample_allocation(index: int, items: int, budget: int, costs: np.ndarray,
                      kind: str, concentrations: tuple[float, ...],
                      rng: np.random.Generator) -> np.ndarray:
    concentration = (concentrations[index % len(concentrations)]
                     if concentrations else 1.0)
    probabilities = draw_histogram(rng, len(costs), kind, concentration)
    allocation = rng.choice(len(costs), size=items,
                            p=probabilities).astype(np.int16)
    return repair(allocation, costs, budget, rng)


def sample_chunk(start: int, count: int, items: int, budget: int,
                 costs: np.ndarray, kind: str,
                 concentrations: tuple[float, ...],
                 seed: tuple[int, ...]) -> np.ndarray:
    """A block of draws, from a generator keyed by the block's own offset.

    Blocks are produced by several threads at once, so each gets its own
    generator rather than sharing one: the draws stay reproducible whatever
    order the threads finish in, and the generator lock stops being a queue.
    """
    rng = np.random.default_rng(list(seed) + [start])
    return np.stack([
        sample_allocation(start + row, items, budget, costs, kind,
                          concentrations, rng)
        for row in range(count)
    ])


# --------------------------------------------------------------------------
# per-configuration run
# --------------------------------------------------------------------------

def run_configuration(score, items: int, lengths: tuple[int, ...], cap: int,
                      kind: str, concentrations: tuple[float, ...], draws: int,
                      batch: int, seed: tuple[int, ...],
                      workers: int = 3) -> dict:
    """`score` maps a `[B, items]` array of *grid* indices to Recall@10.

    Drawing costs about as much as scoring, so several blocks are prepared
    ahead while the current one is on the GPU. The queue is bounded, otherwise
    the producers run away and a sweep's worth of int16 allocations -- 200 KB
    each at this gallery size -- would not fit in memory.
    """
    costs = np.asarray(lengths, dtype=np.int64)
    budget = items * cap
    started = time.perf_counter()
    collected: list[np.ndarray] = []
    histograms: list[np.ndarray] = []
    offsets = list(range(0, draws, batch))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit(offset: int):
            return pool.submit(sample_chunk, offset,
                               min(batch, draws - offset), items, budget,
                               costs, kind, concentrations, seed)

        queued = [submit(offset) for offset in offsets[:workers + 1]]
        following = workers + 1
        while queued:
            drawn = queued.pop(0).result()
            if following < len(offsets):
                queued.append(submit(offsets[following]))
                following += 1
            histograms.append(np.stack([
                np.bincount(row, minlength=len(costs)) for row in drawn
            ]))
            collected.append(score(drawn))
            del drawn
    sampled = np.concatenate(collected)
    shares = np.concatenate(histograms) / items
    return {
        "recall": sampled,
        "shares": shares,
        "seconds": float(time.perf_counter() - started),
    }


def histogram_oracle(score, items: int, lengths: tuple[int, ...], cap: int,
                     iterations: int, population: int,
                     rng: np.random.Generator) -> dict:
    """Best histogram a random assignment can reach, by cross-entropy search.

    The sampled range answers "did any of these draws reach RAVR". This answers
    the stronger question the bin sweep is really about: could *any* histogram
    on this grid have reached it. The assignment stays random throughout, so
    what is being maximised is only the shape of the histogram.
    """
    costs = np.asarray(lengths, dtype=np.int64)
    budget = items * cap
    options = len(costs)
    mean = np.full(options, 1.0 / options)
    concentration = 20.0
    best_value, best_shares = -np.inf, None
    for _ in range(iterations):
        allocations, shares = [], []
        for _ in range(population):
            probabilities = rng.dirichlet(np.maximum(mean * concentration, 1e-3))
            allocation = rng.choice(options, size=items,
                                    p=probabilities).astype(np.int16)
            allocation = repair(allocation, costs, budget, rng)
            allocations.append(allocation)
            shares.append(np.bincount(allocation, minlength=options) / items)
        values = score(np.stack(allocations))
        order = np.argsort(-values)
        if values[order[0]] > best_value:
            best_value = float(values[order[0]])
            best_shares = shares[order[0]]
        elite = max(population // 4, 2)
        mean = np.mean([shares[i] for i in order[:elite]], axis=0)
        concentration *= 1.15
    return {"recall": best_value,
            "shares": [float(x) for x in best_shares],
            "evaluations": iterations * population}


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------

def summarise(sampled: np.ndarray, shares: np.ndarray,
              named: dict[str, float], lattice: float) -> dict:
    quantiles = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
    unique_shares = np.unique(np.round(shares, 6), axis=0)
    # A recall can only be a multiple of 1/(queries*k); a display histogram
    # finer than that shows comb artefacts that are not in the data. Count the
    # empty cells at exactly the lattice spacing instead.
    edges = np.arange(sampled.min() - lattice / 2,
                      sampled.max() + 1.5 * lattice, lattice)
    counts, _ = np.histogram(sampled, bins=edges)
    return {
        "draws": int(len(sampled)),
        "min": float(sampled.min()), "max": float(sampled.max()),
        "mean": float(sampled.mean()),
        "std": float(sampled.std(ddof=1)),
        "quantiles": {f"p{q:g}": float(np.percentile(sampled, q))
                      for q in quantiles},
        "distinct_recall_values": int(len(np.unique(sampled))),
        "distinct_histograms": int(len(unique_shares)),
        "lattice_cells_spanned": int(len(counts)),
        "lattice_cells_empty": int(np.count_nonzero(counts == 0)),
        "share_std_per_bin": [float(x) for x in shares.std(axis=0)],
        "mean_shares": [float(x) for x in shares.mean(axis=0)],
        "named": {
            name: {
                "recall_at_10": float(value),
                "percentile_in_random_range":
                    float(np.mean(sampled < value) * 100.0),
                "z_score": float((value - sampled.mean())
                                 / max(sampled.std(ddof=1), 1e-12)),
                "margin_over_random_max": float(value - sampled.max()),
            }
            for name, value in named.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"),
                        default="LSQ16x4")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--caps", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--full-cap", type=int, default=12,
                        help="Cap that gets every grid and every prior; the "
                             "others get the reduced set.")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--oracle-iterations", type=int, default=60)
    parser.add_argument("--oracle-population", type=int, default=24)
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--split", choices=("validation", "test"),
                        default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--rng-seed", type=int, default=20260905)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    resolved = configure_compute_backend(args.device)
    print(f"compute backend: {resolved} ({backend_report()})", flush=True)
    import torch

    data = _load_dataset(SLUG, Path(args.data_root).expanduser().resolve())
    roles = _roles(data)
    evaluation = _evaluation_data(data, roles, args.split)
    gallery = data["gallery"]
    queries = np.asarray(evaluation["queries"], dtype=np.float32)
    teacher_ids, _ = _teacher(gallery, queries, evaluation["exclude_ids"])

    bundle = FaissCodecBundle.load(
        data["root"] / "advanced_codec_bundles"
        / f"{args.factory.lower()}_seed{args.seed}.fcix"
    )
    try:
        state, _, _, _ = _load_or_calibrate(
            data, roles, bundle, args.factory, args.seed,
            args.calibration_queries, "fp32", False
        )
        nbits = int(bundle.metadata["config"]["nbits"])
        codebooks, packed_bits = faiss_additive_codebooks(bundle)
        codes = unpack_additive_codes(bundle.codes, codebooks.shape[0],
                                      packed_bits)
    finally:
        bundle.close()

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    table = build_score_table(codebooks, codes, len(gallery), queries,
                              TABLE_LENGTHS, device)
    teacher = torch.as_tensor(np.ascontiguousarray(teacher_ids), device=device)
    column_of = {length: column for column, length in enumerate(TABLE_LENGTHS)}
    items = len(gallery)
    lattice = 1.0 / (len(queries) * RETRIEVAL_K)
    print(f"score table {tuple(table.shape)}  "
          f"{table.element_size() * table.nelement() / 2**30:.2f} GiB",
          flush=True)

    report = {
        "dataset": data["dataset"], "slug": SLUG, "gallery_items": items,
        "queries": int(len(queries)), "factory": args.factory,
        "seed": int(args.seed), "split": args.split, "nbits": nbits,
        "shipped_lengths": list(ALLOWED_LENGTHS),
        "table_lengths": list(TABLE_LENGTHS),
        "draws_per_configuration": int(args.draws),
        "rng_seed": int(args.rng_seed),
        "recall_lattice": lattice,
        "compute_backend": backend_report(),
        "caps": {},
    }
    out = args.out or (
        HERE / f"histogram_bin_sweep_{SLUG}_{args.factory.lower()}"
               f"_seed{args.seed}.json"
    )
    samples_directory = out.with_suffix("")
    samples_directory.mkdir(parents=True, exist_ok=True)

    reduced = ("base_w2_m5", "fine_w1_m9", "coarse_w4_m3", "binary_w8_m2",
               "wide_w2_m7", "narrow_w2_m3")
    rng = np.random.default_rng(args.rng_seed)

    for cap in args.caps:
        grids = grid_catalogue(cap)
        selections = _selections(state, items, args.seed, cap, nbits)
        # The named allocations are produced on the shipped grid and are not
        # touched: only their column in the wider score table is looked up.
        shipped_columns = np.array([column_of[length]
                                    for length in ALLOWED_LENGTHS])
        named = {
            name: float(recall_at_10(
                table, shipped_columns[allocation][None, :].astype(np.int16),
                teacher, 1
            )[0])
            for name, allocation in selections.items()
        }
        print(f"\ncap {cap}: ravr {named['ravr']:.4f}  "
              f"uniform {named['uniform']:.4f}", flush=True)

        wanted_grids = grids if cap == args.full_cap else {
            name: spec for name, spec in grids.items() if name in reduced
        }
        priors = prior_catalogue()
        wanted_priors = (priors if cap == args.full_cap else
                         {"prior_baseline": priors["prior_baseline"],
                          "prior_tiny": priors["prior_tiny"]})

        configurations = []
        for grid_name, spec in wanted_grids.items():
            configurations.append((grid_name, spec["lengths"], spec["width"],
                                   "dirichlet", BASELINE_CONCENTRATIONS,
                                   "grid"))
        for prior_name, prior in wanted_priors.items():
            if prior_name == "prior_baseline":
                continue                      # identical to base_w2_m5
            configurations.append((f"base_w2_m5+{prior_name}", ALLOWED_LENGTHS,
                                   2, prior["kind"], prior["concentrations"],
                                   "prior"))

        cap_entry = {"named": named, "configurations": {}}
        report["caps"][str(cap)] = cap_entry
        ravr_lengths = set(int(length) for length in
                           np.unique(np.asarray(ALLOWED_LENGTHS)
                                     [selections["ravr"]]))
        for name, lengths, width, kind, concentrations, axis in configurations:
            columns = np.array([column_of[length] for length in lengths])

            def score(block, columns=columns):
                return recall_at_10(table, columns[block].astype(np.int16),
                                    teacher, args.batch)

            result = run_configuration(
                score, items, lengths, cap, kind, concentrations, args.draws,
                args.batch,
                # str.__hash__ is salted per process; crc32 is not, and the
                # seed has to survive a restart
                (args.rng_seed, cap, zlib.crc32(name.encode())),
            )
            oracle = histogram_oracle(score, items, lengths, cap,
                                      args.oracle_iterations,
                                      args.oracle_population, rng)
            entry = summarise(result["recall"], result["shares"], named,
                              lattice)
            entry |= {
                "axis": axis, "lengths": list(lengths), "bins": len(lengths),
                "width_stages": width,
                "width_bytes": None if width is None else width * nbits / 8.0,
                "sampler": kind,
                "concentrations": list(concentrations),
                "seconds": result["seconds"],
                "histogram_oracle": oracle,
                "ravr_over_oracle": float(named["ravr"] - oracle["recall"]),
                "ravr_representable": bool(ravr_lengths <= set(lengths)),
                "samples_file": f"{cap}_{name}.npy",
            }
            np.save(samples_directory / entry["samples_file"],
                    result["recall"].astype(np.float32))
            cap_entry["configurations"][name] = entry
            print(f"  {name:<26} bins={len(lengths):<2} "
                  f"range [{entry['min']:.4f}, {entry['max']:.4f}] "
                  f"sd {entry['std']:.5f}  oracle {oracle['recall']:.4f}  "
                  f"ravr z {entry['named']['ravr']['z_score']:+.1f}  "
                  f"({result['seconds']:.0f}s)", flush=True)
            out.write_text(json.dumps(report, indent=1, sort_keys=True),
                           encoding="utf-8")

    print(f"\nreport -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
