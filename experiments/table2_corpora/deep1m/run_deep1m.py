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
import lazy_selections  # noqa: E402
import mining_depth  # noqa: E402
from aggregate_nested_additive_ravr import METRICS, aggregate  # noqa: E402
from benchmark_faiss_fixed_rate import (  # noqa: E402
    _load_dataset, _teacher,
)
from benchmark_nested_additive_ravr import (  # noqa: E402
    ALLOWED_LENGTHS, _evaluation_data, panel_results_dir, run_seed,
)
from compute_backend import backend_report, configure_compute_backend  # noqa: E402
from faiss_additive_codecs import AdditiveCodecConfig, train_additive_bundle  # noqa: E402
from faiss_fixed_rate import FaissCodecBundle  # noqa: E402
from nested_additive_ravr import (  # noqa: E402
    faiss_additive_codebooks, unpack_additive_codes,
)
from ravr_depth_ablation import _roles  # noqa: E402

SLUG = "deep1m"
DEFAULT_SEEDS = (17, 42, 73, 3067, 4294, 4996, 5423, 7520, 7937, 9794)
HEADLINE = METRICS[0]
RECALL_TARGET = 0.99
RETRIEVAL_K = 10
DEPTH_LADDER = (10_000, 30_000, 60_000, 120_000, 250_000, 500_000)


def depth_path(factory: str) -> Path:
    return HERE / f"rerank_depth_{SLUG}_{factory.lower()}.json"


def teacher_path() -> Path:
    return HERE / f"teacher_validation_{SLUG}.json"


def stage_teacher(data_root: Path, split: str, sample: int) -> dict:

    data = _load_dataset(SLUG, data_root)
    roles = _roles(data)
    evaluation = _evaluation_data(data, roles, split)
    queries = np.asarray(evaluation["queries"], dtype=np.float32)
    gallery = data["gallery"]

    with np.load(data["root"] / f"{SLUG}_official_gt.npz", allow_pickle=False) as gt:
        neighbours = np.asarray(gt["neighbours"], dtype=np.int64)
        gallery_rows = int(gt["gallery_rows"])
    if gallery_rows != len(gallery):
        raise SystemExit(
            f"ground truth was aligned to {gallery_rows} gallery rows but the "
            f"gallery holds {len(gallery)}; re-run prepare.py"
        )
    with np.load(data["manifest_path"], allow_pickle=False) as manifest:
        role_ids = np.asarray(manifest[f"{split}_ids"], dtype=np.int64)

    started = time.perf_counter()
    teacher_ids, _ = _teacher(gallery, queries, evaluation["exclude_ids"])

    checked = mismatched = rank1_checked = rank1_agree = 0
    first_failure = None
    for row, query_position in enumerate(role_ids):
        official = neighbours[int(query_position)]
        inside = official[official < gallery_rows]
        if not len(inside):
            continue
        depth = min(len(inside), teacher_ids.shape[1])
        ours = teacher_ids[row, :depth]
        checked += 1
        rank1_checked += 1
        rank1_agree += int(ours[0] == inside[0])
        if not np.array_equal(ours, inside[:depth]):
            mismatched += 1
            if first_failure is None:
                first_failure = {
                    "evaluated_row": int(row),
                    "query_source_position": int(query_position),
                    "official_prefix": [int(x) for x in inside[:depth]],
                    "ours": [int(x) for x in ours],
                }

    # float64 recomputation on a sample, to tell a real defect from tie noise
    sample = min(sample, len(queries))
    exact64 = np.argsort(
        -(np.asarray(queries[:sample], dtype=np.float64)
          @ np.asarray(gallery, dtype=np.float64).T), axis=1
    )[:, :RETRIEVAL_K]
    float_agreement = float(np.mean([
        len(np.intersect1d(exact64[i], teacher_ids[i])) / RETRIEVAL_K
        for i in range(sample)
    ]))

    report = {
        "dataset": SLUG,
        "split": split,
        "evaluated_queries": int(len(queries)),
        "gallery_rows": int(len(gallery)),
        "queries_with_an_official_neighbour_in_gallery": checked,
        "queries_disagreeing": mismatched,
        "rank1_agreement": (rank1_agree / rank1_checked) if rank1_checked else None,
        "float64_top10_agreement_on_sample": float_agreement,
        "float64_sample": int(sample),
        "seconds": float(time.perf_counter() - started),
        "first_failure": first_failure,
    }
    teacher_path().write_text(json.dumps(report, indent=2, sort_keys=True),
                              encoding="utf-8")

    print(f"teacher  {checked} of {len(queries)} {split} queries have an "
          f"official neighbour inside the gallery", flush=True)
    print(f"teacher  order agreement: {checked - mismatched}/{checked}; "
          f"rank-1 agreement: {report['rank1_agreement']}", flush=True)
    print(f"teacher  float64 top-10 agreement on {sample} queries: "
          f"{float_agreement:.6f}", flush=True)

    if checked == 0:
        raise SystemExit(
            "no evaluated query has an official neighbour inside the gallery, "
            "so the ground truth cannot validate the evaluation path. Widen "
            "the split or check that the gallery really is a base prefix."
        )
    if mismatched or float_agreement < 1.0:
        raise SystemExit(
            "the exact evaluation path disagrees with the official Deep1B "
            "ground truth.\n"
            f"  queries checked        : {checked}\n"
            f"  queries disagreeing    : {mismatched}\n"
            f"  rank-1 agreement       : {report['rank1_agreement']}\n"
            f"  float64 agreement      : {float_agreement:.6f}\n"
            f"  first failure          : {json.dumps(first_failure)}\n"
            "Diagnose before running the panel, in this order: the metric "
            "(cosine vs L2 -- they coincide only for unit vectors), the ingest "
            "normalisation (prepare.py reports the norm deviation it found), "
            "row indexing (gallery row i must be Deep1B base row i), fbin/ibin "
            "parsing (little-endian uint32 header, then float32/int32), and "
            "the ground-truth interpretation (the published list ranks against "
            "the full 1B base, not against this prefix)."
        )
    print("teacher  OK: exact ranking matches the official ground truth",
          flush=True)
    return report



def measure_candidate_recall(bundle, gallery: np.ndarray, queries: np.ndarray,
                             depths: tuple[int, ...], device: str,
                             batch: int = 32) -> dict[int, float]:

    import torch

    codebooks, nbits = faiss_additive_codebooks(bundle)
    codes = unpack_additive_codes(bundle.codes, codebooks.shape[0], nbits)
    base = ALLOWED_LENGTHS[0]
    torch_device = torch.device(
        device if device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    book = torch.as_tensor(np.ascontiguousarray(codebooks[:base]), device=torch_device)
    reconstruction = torch.zeros(
        (len(gallery), book.shape[-1]), dtype=torch.float32, device=torch_device
    )
    for stage in range(base):
        index = torch.as_tensor(codes[:, stage].astype(np.int64), device=torch_device)
        reconstruction += book[stage].index_select(0, index)
    inverse = 1.0 / reconstruction.norm(dim=1).clamp_min(1e-9)
    gallery_t = torch.as_tensor(np.ascontiguousarray(gallery), device=torch_device)
    q = torch.as_tensor(np.ascontiguousarray(queries), device=torch_device)

    largest = max(depths)
    hits = {depth: 0 for depth in depths}
    with torch.inference_mode():
        for start in range(0, len(q), batch):
            block = q[start:start + batch]
            teacher = torch.topk(block @ gallery_t.T, RETRIEVAL_K, dim=1).indices
            approximate = (block @ reconstruction.T) * inverse
            ranked = torch.topk(
                approximate, min(largest, approximate.shape[1]), dim=1
            ).indices
            for depth in depths:
                prefix = ranked[:, :depth].cpu().numpy()
                truth = teacher.cpu().numpy()
                for row in range(len(block)):
                    hits[depth] += len(np.intersect1d(prefix[row], truth[row]))
    del gallery_t, reconstruction, q
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {d: hits[d] / (len(queries) * RETRIEVAL_K) for d in depths}


def stage_depth(data_root: Path, factory: str, seed: int, device: str,
                queries_used: int, force: bool) -> int:
    path = depth_path(factory)
    if path.exists() and not force:
        record = json.loads(path.read_text(encoding="utf-8"))
        print(f"depth  cached: {record['selected_depth']} "
              f"(CandidateRecall@10 {record['selected_recall']:.4f})", flush=True)
        return int(record["selected_depth"])

    data = _load_dataset(SLUG, data_root)
    roles = _roles(data)
    gallery = data["gallery"]
    validation = np.asarray(roles["validation"][:queries_used], dtype=np.float32)
    bundle = FaissCodecBundle.load(
        data["root"] / "advanced_codec_bundles" / f"{factory.lower()}_seed{seed}.fcix"
    )
    try:
        ladder = tuple(d for d in DEPTH_LADDER if d < len(gallery)) + (len(gallery),)
        started = time.perf_counter()
        recalls = measure_candidate_recall(bundle, gallery, validation, ladder, device)
    finally:
        bundle.close()
    for depth in ladder:
        print(f"depth  {depth:>8}: CandidateRecall@10 = {recalls[depth]:.4f}",
              flush=True)

    feasible = [d for d in ladder if recalls[d] >= RECALL_TARGET]
    if not feasible:
        best = max(ladder, key=lambda d: recalls[d])
        raise SystemExit(
            "candidate mining cannot reach the predeclared "
            f"CandidateRecall@10 >= {RECALL_TARGET} on this gallery.\n"
            f"  best measured recall : {recalls[best]:.4f}\n"
            f"  at candidate depth   : {best}\n"
            f"  gallery size         : {len(gallery)}\n"
            f"  validation queries   : {len(validation)} (seed {seed}, {factory})\n"
            "Calibration would mine a set that cannot see the teacher's top-10, "
            "truncating the allocation signal for a reason unrelated to the "
            "method. Raise DEPTH_LADDER or investigate the codec first."
        )
    selected = min(feasible)
    path.write_text(json.dumps({
        "dataset": SLUG,
        "factory": factory,
        "seed": seed,
        "gallery_size": int(len(gallery)),
        "dimension": int(gallery.shape[1]),
        "validation_queries": int(len(validation)),
        "retrieval_k": RETRIEVAL_K,
        "recall_target": RECALL_TARGET,
        "ladder": [int(d) for d in ladder],
        "recalls": {str(d): float(recalls[d]) for d in ladder},
        "selected_depth": int(selected),
        "selected_recall": float(recalls[selected]),
        "seconds": float(time.perf_counter() - started),
        "note": "measured on validation queries only; the test split is never "
                "read by this stage",
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"depth  selected {selected} at recall {recalls[selected]:.4f} "
          f"-> {path.name}", flush=True)
    return selected


def install_depth(depth: int) -> None:
    panel.RERANK_CANDIDATES[SLUG] = int(depth)
    if panel.rerank_candidates_for(SLUG) != int(depth):
        raise AssertionError("rerank depth injection failed")
    print(f"panel  candidate depth for {SLUG}: {depth} "
          f"(sop {panel.rerank_candidates_for('sop')}, "
          f"gldv2 {panel.rerank_candidates_for('gldv2')} -- both untouched)",
          flush=True)



def stage_codecs(data_root: Path, factory: str, seeds, train_samples: int,
                 effort: str, force: bool) -> None:
    data = _load_dataset(SLUG, data_root)
    bundle_dir = data["root"] / "advanced_codec_bundles"
    bundle_dir.mkdir(exist_ok=True)
    family = "lsq" if factory.lower().startswith("lsq") else "rq"
    screen = {
        "rq_kmeans_iterations": 3, "rq_refine_iterations": 2, "rq_beam_size": 3,
        "lsq_train_iterations": 8, "lsq_train_ils_iterations": 3,
        "lsq_encode_ils_iterations": 6, "lsq_icm_iterations": 3,
        "lsq_perturbations": 2,
    }
    suffix = "" if effort == "full" else f"_{effort}"
    for seed in seeds:
        path = bundle_dir / f"{factory.lower()}_seed{seed}{suffix}.fcix"
        if path.exists() and not force:
            cached = FaissCodecBundle.load(path)
            recorded = cached.metadata.get("effort")
            cached.close()
            if recorded != effort:
                raise SystemExit(
                    f"{path.name} was trained with effort={recorded!r} but "
                    f"effort={effort!r} was requested; retrain with --force."
                )
            print(f"codec  {path.name}: cached (effort={recorded})", flush=True)
            continue
        config = AdditiveCodecConfig(
            family=family, stages=16, nbits=4, seed=seed,
            **(screen if effort == "screen" else {}),
        )
        rng = np.random.default_rng(seed)
        pool = data["codec_pool"]
        train_ids = rng.choice(pool, min(train_samples, len(pool)), replace=False)
        started = time.perf_counter()
        print(f"codec  {factory} seed {seed}: training on {len(train_ids)} of "
              f"{len(pool)} pool vectors (effort={effort})", flush=True)
        bundle = train_additive_bundle(data["gallery"], train_ids, config)
        bundle.metadata.update({"dataset": data["dataset"], "effort": effort})
        bundle.save(path)
        bundle.close()
        print(f"       -> {path.name} in {time.perf_counter() - started:.1f}s",
              flush=True)


def stage_panel(data_root: Path, factory: str, seeds, targets, split: str,
                calibration_queries: int, force: bool,
                strategies: tuple[str, ...], lazy: bool = True) -> None:
    if "uniform" not in strategies:
        raise SystemExit("uniform must be included: it defines the byte cap "
                         "and every reported delta")
    if lazy:
        lazy_selections.install()
    if "candidate_frequency" not in strategies:
        print("panel  NOTE: candidate_frequency excluded. It is the control "
              "that decided the GLDv2 reading, so a panel without it cannot "
              "separate the Ordinal-KL objective from mined-candidate "
              "exposure.", flush=True)
    timings = []
    for seed in seeds:
        for target in targets:
            started = time.perf_counter()
            run_seed(
                data_root, seed, split=split,
                calibration_queries=calibration_queries, target_length=target,
                factory=factory, precision="fp32", force_calibration=force,
                strategies=strategies, dataset=SLUG,
            )
            elapsed = time.perf_counter() - started
            timings.append({"seed": int(seed), "target": int(target),
                            "seconds": float(elapsed)})
            print(f"panel  seed {seed} target {target}: {elapsed / 60:.1f} min",
                  flush=True)
    if timings:
        total = sum(row["seconds"] for row in timings)
        record = {
            "dataset": SLUG, "factory": factory, "split": split,
            "calibration_queries": int(calibration_queries),
            "strategies": list(strategies), "lazy_selections": bool(lazy),
            "cells": timings,
            "total_seconds": float(total),
            "total_hours": float(total / 3600.0),
            "mean_cell_minutes": float(total / len(timings) / 60.0),
        }
        (HERE / f"panel_timing_{SLUG}_{factory.lower()}_{split}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"panel  {len(timings)} cells in {total / 3600:.2f} h "
              f"({total / len(timings) / 60:.1f} min per cell)", flush=True)


def _report_path(dataset: str, factory: str, split: str, cal: int) -> Path:
    return (
        panel_results_dir(dataset)
        / f"evaluation_{factory.lower()}_{split}_cal{cal}.json"
    )


def stage_compare(factory: str, split: str, cal: int, stem: Path) -> None:
    """Gain against the other corpora, never absolute B/item."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wanted = {"SOP 100K": "sop", "GLDv2 100K": "gldv2",
              "GLDv2 full index": "gldv2_full", "Deep1M": SLUG}
    reports, missing = {}, []
    for name, slug in wanted.items():
        path = _report_path(slug, factory, split, cal)
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(f"{name}: {path}")
    if SLUG not in [wanted[n] for n in reports]:
        raise SystemExit("Deep1M has no aggregate report yet; run the panel "
                         "and aggregate stages first.\n" + "\n".join(missing))

    def gains(report):
        rows = [r for r in report["comparisons"]
                if r["ravr_minus"] == "uniform" and r["metric"] == HEADLINE]
        return sorted(rows, key=lambda r: r["target_length"])

    styles = {"SOP 100K": ("#111827", "*"), "GLDv2 100K": ("#2563eb", "D"),
              "GLDv2 full index": ("#228b22", "^"), "Deep1M": ("#e67e22", "o")}
    figure, axis = plt.subplots(figsize=(3.9, 2.7))
    for name, report in reports.items():
        rows = gains(report)
        if not rows:
            continue
        colour, marker = styles[name]
        axis.errorbar(
            [r["target_length"] for r in rows],
            [r["estimate"] for r in rows],
            yerr=[
                [r["estimate"] - r["family_simultaneous_95_interval"][0] for r in rows],
                [r["family_simultaneous_95_interval"][1] - r["estimate"] for r in rows],
            ],
            color=colour, marker=marker, linestyle="-", linewidth=1.1,
            markersize=6.5 if marker == "*" else 5.0, capsize=2.2,
            capthick=0.7, elinewidth=0.7, label=name,
        )
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Uniform-equivalent prefix (stages)", labelpad=2)
    axis.set_ylabel("RAVR minus uniform, R@10", labelpad=2)
    axis.set_title("Allocator gain across corpora", pad=3.0, weight="semibold")
    axis.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout(pad=0.35)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".pdf", ".png"):
        figure.savefig(stem.with_suffix(extension), dpi=260,
                       bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    lines = [
        f"# Allocator gain across corpora -- {factory}", "",
        "Absolute B/item are **not** comparable: the shared codebook is "
        "98,304 bytes at d=96 and 393,216 bytes at d=384, so an identical "
        "allocation looks four times cheaper per item on Deep1M. The "
        "comparison is the gain at a matched uniform-equivalent prefix.", "",
        "| Corpus | d | Target | Uniform R@10 | RAVR R@10 | Gain | "
        "Family 95% | Positive seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        frontier = {(r["strategy"], r["target_length"]): r
                    for r in report["frontier"]}
        for row in gains(report):
            target = row["target_length"]
            low, high = row["family_simultaneous_95_interval"]
            lines.append(
                f"| {name} | {96 if name == 'Deep1M' else 384} | {target} | "
                f"{frontier[('uniform', target)][HEADLINE]['mean']:.4f} | "
                f"{frontier[('ravr', target)][HEADLINE]['mean']:.4f} | "
                f"{row['estimate']:+.5f} | [{low:+.5f}, {high:+.5f}] | "
                f"{row['positive_seeds']}/{len(row['seed_differences'])} |"
            )
    if missing:
        lines += ["", "Not included (no aggregate report yet):", ""]
        lines += [f"- {entry}" for entry in missing]
    summary = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"figure -> {stem.with_suffix('.png')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--targets", nargs="+", type=int, default=[10, 12, 14])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--effort", choices=("screen", "full"), default="full")
    parser.add_argument("--resamples", type=int, default=100000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--depth-seed", type=int, default=17)
    parser.add_argument("--depth-queries", type=int, default=400)
    parser.add_argument("--teacher-float64-sample", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hard-negative-depth", type=int, default=mining_depth.DEFAULT_DEPTH,
        help="RAVR's own mining depth. The mined hard set is "
             "retrieval_k + 3*depth items per query and does not scale with "
             "the gallery, which is what starves the allocator at a million "
             "items. Raising it writes to suffixed states and outputs, so the "
             "default-depth run is left intact.",
    )
    parser.add_argument(
        "--no-lazy-selections", action="store_true",
        help="Use the shared eager _selections instead of this module's lazy "
             "override. Correct but, at million-item scale, dominated by "
             "allocators that are never evaluated.",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(panel.STRATEGIES),
        choices=tuple(panel.STRATEGIES),
        help="Allocators to EVALUATE. The default is all of them. Note this "
             "does not skip any allocator's solve_lagrangian call: "
             "_selections computes every table unconditionally before the "
             "filter is applied, so excluding candidate_frequency cuts packing "
             "and scoring but not the allocation wall it causes at "
             "million-item scale (measured: 375,532 bytes of slack closed one "
             "upgrade per O(N log N) pass, ~3.8 h for one seed and rate).",
    )
    parser.add_argument(
        "--stages", nargs="+",
        default=["teacher", "codecs", "depth", "panel", "aggregate", "compare"],
        choices=("teacher", "codecs", "depth", "panel", "aggregate", "compare"),
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    stages = set(args.stages)
    if "panel" in stages and args.effort != "full":
        raise SystemExit(
            f"--effort {args.effort} trains screening codecs, which the panel "
            "does not load. Screen with --stages codecs, then run the panel "
            "with --effort full."
        )
    mining_depth.install(args.hard_negative_depth)

    if stages - {"compare"}:
        resolved = configure_compute_backend(args.device)
        print(f"compute backend: {resolved} ({backend_report()})", flush=True)

    if "teacher" in stages:
        stage_teacher(data_root, args.split, args.teacher_float64_sample)
    if "codecs" in stages:
        stage_codecs(data_root, args.factory, args.seeds, args.train_samples,
                     args.effort, args.force)
    if {"depth", "panel"} & stages:
        depth = stage_depth(data_root, args.factory, args.depth_seed, args.device,
                            args.depth_queries, args.force and "depth" in stages)
        install_depth(depth)
    if "panel" in stages:
        stage_panel(data_root, args.factory, args.seeds, args.targets, args.split,
                    args.calibration_queries, args.force,
                    tuple(args.strategies), not args.no_lazy_selections)
    if "aggregate" in stages:
        # The shared aggregator demands the full strategy set and a comparator
        # row for each; a panel that legitimately skipped one would fail here
        # after the panel had already been paid for.
        lazy_selections.install_aggregate_subset(tuple(args.strategies))
        aggregate(args.factory, "fp32", tuple(args.seeds), tuple(args.targets),
                  args.split, args.calibration_queries, args.resamples,
                  20260902, dataset=SLUG)
    if "compare" in stages:
        stage_compare(args.factory, args.split, args.calibration_queries,
                      HERE / "figures" / f"gain_across_corpora_{args.factory.lower()}")


if __name__ == "__main__":
    sys.exit(main())
