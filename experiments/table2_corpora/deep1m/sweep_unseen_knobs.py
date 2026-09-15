from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark_nested_additive_ravr as panel  # noqa: E402
from benchmark_faiss_fixed_rate import _load_dataset, _teacher  # noqa: E402
from benchmark_nested_additive_ravr import (  # noqa: E402
    ALLOWED_LENGTHS, _evaluation_data, _load_or_calibrate,
)
from compute_backend import backend_report, configure_compute_backend  # noqa: E402
from faiss_fixed_rate import FaissCodecBundle  # noqa: E402
from nested_additive_ravr import (  # noqa: E402
    faiss_additive_codebooks, unpack_additive_codes,
)
from ravr_rq import solve_lagrangian  # noqa: E402
from ravr_depth_ablation import _roles  # noqa: E402

import lazy_selections  # noqa: E402
import mining_depth  # noqa: E402

SLUG = "deep1m"
RETRIEVAL_K = 10
DEFAULT_DEPTH = mining_depth.DEFAULT_DEPTH
STRATEGIES = ("uniform", "ravr", "ravr_unseen_rd", "ravr_unseen_qip",
              "ordinal_per_exposure", "reconstruction", "query_ip_mse")


def default_grid(panel_queries: int) -> list[tuple[int, int]]:

    q = panel_queries

    query_sweep = [(q, 50), (q // 2, 50), (q // 4, 50), (q // 8, 50)]
    depth_sweep = [(q, 100), (q, 200), (q, 400), (q, 800)]
    # Q * (k + 3d) held at ~8e5 and ~3e6 mined incidences
    iso_low = [(q // 4, 200), (q // 2, 100), (q // 8, 400)]
    iso_high = [(q // 2, 400), (q // 4, 800)]
    cells = query_sweep + depth_sweep + iso_low + iso_high
    seen, ordered = set(), []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            ordered.append(cell)
    return ordered



def set_depth(depth: int) -> str:
    if int(depth) != DEFAULT_DEPTH:
        return mining_depth.install(int(depth))
    for name in ("calibrate_frozen_additive_codec", "_state_path",
                 "panel_output_dir"):
        current = getattr(panel, name)
        original = getattr(current, "_original", None)
        if original is not None:
            setattr(panel, name, original)
    import aggregate_nested_additive_ravr as agg
    agg.panel_output_dir = panel.panel_output_dir
    return ""


class Evaluator:

    def __init__(self, codebooks, codes, queries, teacher_ids, device,
                 chunk: int = 200_000):
        import torch

        self.torch = torch
        self.device = device
        self.chunk = int(chunk)
        self.codes = codes
        self.n_items = len(codes)
        self.book = torch.as_tensor(np.ascontiguousarray(codebooks), device=device)
        self.q = torch.as_tensor(np.ascontiguousarray(queries), device=device)
        self.teacher = torch.as_tensor(np.ascontiguousarray(teacher_ids),
                                       device=device)
        self.column_of = {length: column
                          for column, length in enumerate(ALLOWED_LENGTHS)}

    def __call__(self, allocations: list[np.ndarray]) -> list[float]:
        torch = self.torch
        m, nq = len(allocations), len(self.q)
        best_value = torch.full((m, nq, RETRIEVAL_K), -np.inf, device=self.device)
        best_index = torch.zeros((m, nq, RETRIEVAL_K), dtype=torch.long,
                                 device=self.device)
        allocs = [torch.as_tensor(np.ascontiguousarray(a), device=self.device).long()
                  for a in allocations]
        for start in range(0, self.n_items, self.chunk):
            stop = min(start + self.chunk, self.n_items)
            code_index = torch.as_tensor(
                self.codes[start:stop].astype(np.int64), device=self.device)
            reconstruction = torch.zeros((stop - start, self.book.shape[-1]),
                                         dtype=torch.float32, device=self.device)
            table = torch.empty((nq, stop - start, len(ALLOWED_LENGTHS)),
                                dtype=torch.float32, device=self.device)
            for stage in range(self.book.shape[0]):
                reconstruction += self.book[stage].index_select(
                    0, code_index[:, stage])
                length = stage + 1
                if length not in self.column_of:
                    continue
                inverse = (1.0 / reconstruction.norm(dim=1).clamp_min(1e-9))
                inverse = inverse.to(torch.float16).to(torch.float32)
                table[:, :, self.column_of[length]] = (
                    self.q @ reconstruction.T) * inverse
            del reconstruction, code_index
            for j, allocation in enumerate(allocs):
                index = allocation[start:stop]
                gathered = torch.gather(
                    table, 2, index[None, :, None].expand(nq, -1, 1)).squeeze(2)
                value, position = torch.topk(gathered, RETRIEVAL_K, dim=1)
                merged_value = torch.cat([best_value[j], value], 1)
                merged_index = torch.cat([best_index[j], position + start], 1)
                order = torch.topk(merged_value, RETRIEVAL_K, dim=1).indices
                best_value[j] = torch.gather(merged_value, 1, order)
                best_index[j] = torch.gather(merged_index, 1, order)
                del gathered, value, position, merged_value, merged_index, order
            del table
        out = []
        for j in range(m):
            hits = (best_index[j].unsqueeze(2)
                    == self.teacher[:, None, :]).any(2).sum(1)
            out.append(float((hits.to(torch.float64) / RETRIEVAL_K).mean()))
        del best_value, best_index
        return out



def option_costs(nbits: int) -> np.ndarray:
    base = ALLOWED_LENGTHS[0] * nbits // 8
    return np.array([length * nbits // 8 - base for length in ALLOWED_LENGTHS],
                    dtype=np.int64)


def unseen_fallback(ravr: np.ndarray, table: np.ndarray, coverage: np.ndarray,
                    costs: np.ndarray, budget: int) -> np.ndarray:
    out = np.asarray(ravr).copy()
    unseen = np.flatnonzero(coverage == 0)
    if not len(unseen):
        return out
    seen = np.flatnonzero(coverage > 0)
    seen_cost = int(costs[out[seen]].sum())
    slack = budget - seen_cost
    if slack < 0:
        raise AssertionError("covered RAVR allocation exceeded the cap")
    out[unseen] = solve_lagrangian(np.asarray(table)[unseen], costs, slack)
    return out


def allocations_for(state: dict, n_items: int, seed: int, cap: int,
                    nbits: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    selections = panel._selections(state, n_items, seed, cap, nbits)
    costs = option_costs(nbits)
    budget = n_items * (cap * nbits // 8 - ALLOWED_LENGTHS[0] * nbits // 8)
    out, timings = {}, {}
    for name in STRATEGIES:
        started = time.perf_counter()
        if name == "ravr_unseen_qip":
            out[name] = unseen_fallback(
                out.get("ravr", selections["ravr"]),
                state["query_ip_distortion"], state["coverage"], costs, budget)
        else:
            out[name] = np.asarray(selections[name])
        timings[name] = float(time.perf_counter() - started)
    return out, timings



def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    total = values.sum()
    if total <= 0:
        return 0.0
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * values)) / (n * total) - (n + 1.0) / n)


def risk_diagnostics(state: dict, queries: int, depth: int,
                     reference: dict | None) -> dict:

    coverage = np.asarray(state["coverage"], dtype=np.int64)
    distortion = np.asarray(state["distortion"], dtype=np.float64)
    seen = coverage > 0
    gap = distortion[:, 0] - distortion[:, -1]
    gap = np.maximum(gap, 0.0)
    positive = gap[gap > 0]
    total = float(gap.sum())
    share = np.sort(gap)[::-1]
    top1 = float(share[:max(len(share) // 100, 1)].sum() / total) if total else 0.0
    if total > 0:
        p = gap / total
        nonzero = p[p > 0]
        entropy = float(-np.sum(nonzero * np.log(nonzero)))
    else:
        entropy = 0.0
    mined = float(coverage.sum()) / max(queries, 1)
    window = RETRIEVAL_K + 3 * depth
    out = {
        "coverage_fraction": float(np.mean(seen)),
        "unseen_fraction": float(np.mean(~seen)),
        "seen_items": int(seen.sum()),
        "mean_mined_set": mined,
        "nominal_window": int(window),
        "within_query_overlap": float(1.0 - mined / window),
        "mined_incidences": int(coverage.sum()),
        "expected_coverage_no_overlap": float(
            1.0 - np.exp(-mined * queries / len(coverage))),
        "negative_weight_per_item": float(0.5 / max(mined - 1.0, 1.0)),
        "risk_top1pct_share": top1,
        "risk_effective_items": float(np.exp(entropy)),
        "risk_gini": gini(gap),
        "risk_zero_among_seen": float(np.mean(gap[seen] == 0.0)) if seen.any() else 0.0,
        "risk_mean_on_seen": float(gap[seen].mean()) if seen.any() else 0.0,
        "risk_distinct_values": int(len(np.unique(positive))),
    }
    if reference is not None:
        from scipy.stats import spearmanr

        reference_seen = np.asarray(reference["seen_mask"])
        reference_gap = np.asarray(reference["gap"])
        both = reference_seen & seen
        if both.sum() > 2:
            rho = spearmanr(reference_gap[both], gap[both]).statistic
            out["risk_spearman_vs_reference_on_shared_seen"] = float(rho)
        out["shared_seen_items"] = int(both.sum())
    return out


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"),
                        default="LSQ16x4")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--caps", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--calibration-queries", type=int, default=5000,
                        help="the panel's own workload; the sweep divides it")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="explicit Q:depth cells, overriding the grid")
    parser.add_argument("--rerank-candidates", type=int, default=None,
                        help="defaults to this module's measured depth")
    parser.add_argument("--split", choices=("validation", "test"),
                        default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--chunk", type=int, default=200_000)
    parser.add_argument(
        "--extra-queries", type=Path, default=None,
        help="a .npy of further calibration queries, appended to the "
             "manifest's own. The only way to sweep knob 1 upward: the "
             "manifest stores exactly the queries the panel used, so without "
             "this the knob can only be swept down.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    resolved = configure_compute_backend(args.device)
    print(f"compute backend: {resolved}", flush=True)
    import torch

    depth_file = HERE / f"rerank_depth_{SLUG}_{args.factory.lower()}.json"
    rerank = args.rerank_candidates
    if rerank is None and depth_file.exists():
        rerank = int(json.loads(depth_file.read_text())["selected_depth"])
    if rerank is not None:
        panel.RERANK_CANDIDATES[SLUG] = int(rerank)
    print(f"panel  candidate depth for {SLUG}: "
          f"{panel.rerank_candidates_for(SLUG)}", flush=True)
    lazy_selections.install()

    data = _load_dataset(SLUG, Path(args.data_root).expanduser().resolve())
    roles = _roles(data)
    evaluation = _evaluation_data(data, roles, args.split)
    if args.extra_queries is not None:
        extra = np.load(args.extra_queries).astype(np.float32)
        if extra.ndim != 2 or extra.shape[1] != data["gallery"].shape[1]:
            raise SystemExit(f"{args.extra_queries}: expected "
                             f"(n, {data['gallery'].shape[1]}) queries")
        roles = dict(roles)
        roles["calibration"] = np.concatenate(
            [np.asarray(roles["calibration"], dtype=np.float32), extra])
        if roles.get("calibration_exclude") is not None:
            raise SystemExit("extra queries need matching exclude ids on an "
                             "in-gallery corpus; this module's queries are "
                             "external, so none are expected")
        print(f"queries  calibration pool extended to "
              f"{len(roles['calibration']):,} "
              f"(+{len(extra):,} from {args.extra_queries.name})", flush=True)
    gallery = data["gallery"]
    queries = np.asarray(evaluation["queries"], dtype=np.float32)
    teacher_ids, _ = _teacher(gallery, queries, evaluation["exclude_ids"])
    bundle_path = (data["root"] / "advanced_codec_bundles"
                   / f"{args.factory.lower()}_seed{args.seed}.fcix")
    bundle = FaissCodecBundle.load(bundle_path)
    nbits = int(bundle.metadata["config"]["nbits"])
    codebooks, packed_bits = faiss_additive_codebooks(bundle)
    codes = unpack_additive_codes(bundle.codes, codebooks.shape[0], packed_bits)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    evaluate = Evaluator(codebooks, codes, queries, teacher_ids, device,
                         args.chunk)
    costs = option_costs(nbits)
    base_bytes = ALLOWED_LENGTHS[0] * nbits // 8
    items = len(gallery)
    print(f"{SLUG}: {items:,} items, {len(queries)} {args.split} queries, "
          f"nbits {nbits}", flush=True)

    cells = ([tuple(int(x) for x in cell.split(":")) for cell in args.cells]
             if args.cells else default_grid(args.calibration_queries))
    print(f"cells: {cells}", flush=True)

    out = args.out or (HERE / f"unseen_knobs_{SLUG}_{args.factory.lower()}"
                              f"_seed{args.seed}.json")
    report = {
        "reference_cell": f"{args.calibration_queries}:{DEFAULT_DEPTH}",
        "dataset": data["dataset"], "slug": SLUG, "gallery_items": items,
        "evaluation_queries": int(len(queries)), "split": args.split,
        "factory": args.factory, "seed": int(args.seed), "nbits": nbits,
        "allowed_lengths": list(ALLOWED_LENGTHS),
        "option_costs": [int(c) for c in costs],
        "base_bytes": int(base_bytes),
        "rerank_candidates": int(panel.rerank_candidates_for(SLUG)),
        "panel_calibration_queries": int(args.calibration_queries),
        "compute_backend": backend_report(),
        "cells": {},
    }

    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        if previous.get("slug") == SLUG and previous.get("seed") == args.seed:
            report["cells"] = previous.get("cells", {})
            print(f"report  resuming from {len(report['cells'])} recorded "
                  f"cells in {out.name}", flush=True)
    reference = None

    try:
        for cell_queries, depth in cells:
            key = f"{cell_queries}:{depth}"
            if cell_queries > len(roles["calibration"]):
                print(f"cell Q={cell_queries} skipped: only "
                      f"{len(roles['calibration']):,} calibration queries "
                      "available (see --extra-queries)", flush=True)
                continue
            set_depth(depth)
            started = time.perf_counter()
            state, path, cached, seconds = _load_or_calibrate(
                data, roles, bundle, args.factory, args.seed, cell_queries,
                "fp32", False,
            )
            if cached:
                seconds = 0.0
            wall = time.perf_counter() - started
            entry = {
                "calibration_queries": int(cell_queries),
                "hard_negative_depth": int(depth),
                "state_file": path.name,
                "calibration_cached": bool(cached),
                "calibration_seconds": float(seconds),
                "load_or_calibrate_wall_seconds": float(wall),
                "candidate_recall_at_k": float(
                    state["calibration_report"]["candidate_recall_at_k"]),
            }
            entry.update(risk_diagnostics(state, cell_queries, depth, reference))
            is_reference = (cell_queries == args.calibration_queries
                            and depth == DEFAULT_DEPTH)
            if is_reference and reference is None:
                coverage = np.asarray(state["coverage"])
                distortion = np.asarray(state["distortion"], dtype=np.float64)
                reference = {
                    "seen_mask": coverage > 0,
                    "gap": np.maximum(distortion[:, 0] - distortion[:, -1], 0.0),
                }
            entry["caps"] = {}
            for cap in args.caps:
                budget = items * (cap * nbits // 8 - base_bytes)
                built, solve_seconds = allocations_for(
                    state, items, args.seed, cap, nbits)
                names = list(built)
                recalls = evaluate([built[name] for name in names])
                entry["caps"][str(cap)] = {
                    "variable_budget_bytes": int(budget),
                    "strategies": {
                        name: {
                            "recall_at_10": float(recall),
                            "variable_bytes_spent": int(costs[built[name]].sum()),
                            "slack_bytes": int(budget - costs[built[name]].sum()),
                            "code_bytes_per_item": float(
                                base_bytes + costs[built[name]].sum() / items),
                            "mean_assigned_length": float(
                                np.mean(np.asarray(ALLOWED_LENGTHS)[built[name]])),
                            "solve_seconds": solve_seconds[name],
                            "histogram": {
                                str(length): int(count) for length, count in
                                zip(ALLOWED_LENGTHS,
                                    np.bincount(built[name],
                                                minlength=len(ALLOWED_LENGTHS)))
                            },
                        }
                        for name, recall in zip(names, recalls)
                    },
                }
                spent = entry["caps"][str(cap)]["strategies"]["ravr"]
                print(f"  cap {cap}: ravr R@10 {spent['recall_at_10']:.4f} "
                      f"spent {spent['code_bytes_per_item']:.3f}/"
                      f"{base_bytes + budget / items:.3f} B/item, "
                      f"slack {spent['slack_bytes']:,}", flush=True)
            report["cells"][key] = entry
            out.write_text(json.dumps(report, indent=1, sort_keys=True),
                           encoding="utf-8")
            print(f"cell Q={cell_queries} d={depth}: coverage "
                  f"{entry['coverage_fraction']:.4f}, mined set "
                  f"{entry['mean_mined_set']:.0f}, calibration "
                  f"{entry['calibration_seconds']:.0f}s "
                  f"({'cached' if cached else 'fresh'})\n", flush=True)
    finally:
        bundle.close()
    print(f"report -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
