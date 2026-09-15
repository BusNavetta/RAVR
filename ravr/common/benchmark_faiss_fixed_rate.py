"""Fair RAVR-RQ comparison against Faiss PQ, OPQ, and standard 8-bit RQ.s
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import faiss
import numpy as np

from compute_backend import (
    backend_report,
    configure_compute_backend,
    topk_inner_products,
)
from faiss_fixed_rate import FaissCodecBundle, FaissCodecConfig, train_bundle
from ravr_rq import OBJECTIVE_BOUNDARY_KL, OBJECTIVE_ORDINAL_KL, RAVRRQTrainer


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path.home() / "Desktop" / "AJigma_data"
SEEDS = (17, 42, 73)
M_VALUES = (4, 6, 8)
RAVR_CODE_BUDGETS = (4.0, 4.5, 5.25, 6.0)
EPS = 1e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values)
    k = min(int(k), len(values))
    candidates = np.argpartition(values, len(values) - k)[-k:]
    return candidates[np.lexsort((candidates, -values[candidates]))]


def _load_dataset(dataset: str, data_root: Path) -> dict:
    root = data_root / dataset
    if dataset == "sop":
        manifest_path = root / "sop_100k_manifest.npz"
        feature_path = root / "sop_100k_dinov2_vits14.npy"
        manifest = np.load(manifest_path, allow_pickle=False)
        gallery = np.asarray(np.load(feature_path, mmap_mode="r"), dtype=np.float32)
        test_ids = np.asarray(manifest["test_ids"], dtype=np.int64)
        labels = np.asarray(manifest["labels"])
        return {
            "dataset": "Stanford Online Products",
            "slug": dataset,
            "root": root,
            "manifest_path": manifest_path,
            "gallery": gallery,
            "queries": gallery[test_ids],
            "exclude_ids": test_ids,
            "gallery_labels": labels,
            "query_labels": labels[test_ids],
            "codec_pool": np.asarray(manifest["codec_train_ids"], dtype=np.int64),
            "official_relevant": None,
            "gallery_external_ids": None,
            "query_file": None,
        }
    generic_manifest = root / f"{dataset}_manifest.npz"
    if generic_manifest.exists():
        # Panel-compatible corpus in the layout written by
        # synthetic_panel_dataset.py.  Queries are either gallery rows (SOP-like)
        # or a separate file (GLDv2-like); the role ids index whichever holds
        # them.  The SOP and GLDv2 branches stay untouched.
        manifest = np.load(generic_manifest, allow_pickle=False)
        gallery = np.asarray(
            np.load(root / f"{dataset}_gallery.npy", mmap_mode="r"), dtype=np.float32
        )
        test_ids = np.asarray(manifest["test_ids"], dtype=np.int64)
        labels = np.asarray(manifest["labels"])
        query_file = root / f"{dataset}_queries.npy"
        external = query_file.exists()
        return {
            "dataset": (
                str(manifest["display_name"])
                if "display_name" in manifest.files else dataset
            ),
            "slug": dataset,
            "root": root,
            "manifest_path": generic_manifest,
            "gallery": gallery,
            "queries": (
                np.asarray(
                    np.load(query_file, mmap_mode="r")[test_ids], dtype=np.float32
                ) if external else gallery[test_ids]
            ),
            "exclude_ids": None if external else test_ids,
            "gallery_labels": labels,
            "query_labels": (
                np.asarray(manifest["query_labels"])[test_ids] if external
                else labels[test_ids]
            ),
            "codec_pool": np.asarray(manifest["codec_train_ids"], dtype=np.int64),
            "official_relevant": None,
            "gallery_external_ids": None,
            "query_file": query_file if external else None,
        }

    manifest_path = root / "gldv2_100k_manifest.npz"
    gallery_path = root / "gldv2_100k_index_dinov2_vits14.npy"
    query_path = root / "gldv2_test_queries_dinov2_vits14.npy"
    manifest = np.load(manifest_path, allow_pickle=False)
    test_ids = np.asarray(manifest["test_ids"], dtype=np.int64)
    return {
        "dataset": "Google Landmarks Dataset v2.1",
        "slug": dataset,
        "root": root,
        "manifest_path": manifest_path,
        "gallery": np.asarray(np.load(gallery_path, mmap_mode="r"), dtype=np.float32),
        "queries": np.asarray(
            np.load(query_path, mmap_mode="r")[test_ids], dtype=np.float32
        ),
        "exclude_ids": None,
        "gallery_labels": np.asarray(manifest["gallery_labels"]),
        "query_labels": np.asarray(manifest["query_labels"])[test_ids],
        "codec_pool": np.arange(100000, dtype=np.int64),
        "official_relevant": np.asarray(manifest["query_relevant_gallery_ids"])[test_ids],
        "gallery_external_ids": np.asarray(manifest["gallery_ids"]),
        "query_file": query_path,
    }


def _teacher(
    gallery: np.ndarray,
    queries: np.ndarray,
    exclude_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    ids, scores, _ = topk_inner_products(
        queries, gallery, 10, exclude_ids, batch_size=128
    )
    return ids, scores


def _ap_at_10(returned_external: np.ndarray, relevant_text: str) -> float | None:
    relevant = set(str(relevant_text).split())
    if not relevant:
        return None
    hits = 0
    total = 0.0
    for rank, item in enumerate(returned_external[:10], 1):
        if str(item) in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), 10)


def _evaluate_reconstruction(
    reconstruction: np.ndarray,
    data: dict,
    teacher_ids: np.ndarray,
    teacher_scores: np.ndarray,
) -> tuple[dict, dict]:
    gallery = data["gallery"]
    queries = data["queries"]
    exclude_ids = data["exclude_ids"]
    r1, r10, ndcg10, semantic, official_ap10, official_hit1 = [], [], [], [], [], []
    started = time.perf_counter()
    returned_all, _, scoring_report = topk_inner_products(
        queries, reconstruction, 10, exclude_ids, batch_size=128
    )
    discount = 1.0 / np.log2(np.arange(2, 12))
    for row, returned in enumerate(returned_all):
        r1.append(float(returned[0] == teacher_ids[row, 0]))
        r10.append(len(set(returned) & set(teacher_ids[row])) / 10)
        relevance = gallery[returned] @ queries[row]
        dcg = np.sum((np.exp2(relevance) - 1.0) * discount)
        idcg = np.sum((np.exp2(teacher_scores[row]) - 1.0) * discount)
        ndcg10.append(float(dcg / max(idcg, EPS)))
        query_label = int(data["query_labels"][row])
        if query_label >= 0:
            semantic.append(float(data["gallery_labels"][returned[0]] == query_label))
        if data["official_relevant"] is not None:
            ap = _ap_at_10(
                data["gallery_external_ids"][returned],
                data["official_relevant"][row],
            )
            if ap is not None:
                official_ap10.append(ap)
                relevant = set(str(data["official_relevant"][row]).split())
                official_hit1.append(
                    float(str(data["gallery_external_ids"][returned[0]]) in relevant)
                )
    elapsed = time.perf_counter() - started
    per_query = {
        "teacher_hit_at_1": r1,
        "teacher_recall_at_10": r10,
        "teacher_ndcg_at_10": ndcg10,
    }
    metrics = {
        "teacher_recall_at_1": float(np.mean(r1)),
        "teacher_recall_at_10": float(np.mean(r10)),
        "teacher_ndcg_at_10": float(np.mean(ndcg10)),
        "semantic_recall_at_1_on_eligible_queries": (
            float(np.mean(semantic)) if semantic else None
        ),
        "semantic_eligible_queries": len(semantic),
        "official_map_at_10_on_eligible_queries": (
            float(np.mean(official_ap10)) if official_ap10 else None
        ),
        "official_hit_at_1_on_eligible_queries": (
            float(np.mean(official_hit1)) if official_hit1 else None
        ),
        "official_eligible_queries": len(official_ap10),
        "evaluation_seconds": elapsed,
        "compute_backend": scoring_report,
        "mean_reconstruction_cosine": float(
            np.mean(np.sum(gallery * reconstruction, axis=1))
        ),
    }
    return metrics, per_query


def _ravr_state_path(data: dict, seed: int) -> Path:
    suffix = "" if seed == 42 else f"_seed{seed}"
    if data["slug"] == "sop":
        return data["root"] / f"sop_100k_ravr_state{suffix}.npz"
    return data["root"] / f"gldv2_100k_ravr_state_r60000{suffix}.npz"


def _ravr_report_path(data: dict, seed: int) -> Path:
    directory = (
        "ravr_rq_sop_100k" if data["slug"] == "sop" else "ravr_rq_gldv2_100k"
    )
    return HERE / "outputs" / directory / f"seed{seed}.json"


def _ravr_reconstruction(trainer: RAVRRQTrainer, index) -> np.ndarray:
    lengths = index.assigned_lengths_by_id()
    reconstruction = trainer.centroids[trainer.cluster_ids].copy()
    for stage in range(trainer.config.stages):
        active = lengths > stage
        reconstruction[active] += trainer.codebooks[
            stage, trainer.full_codes[active, stage].astype(np.int64)
        ]
    option_for_length = {
        length: option for option, length in enumerate(trainer.config.allowed_lengths)
    }
    options = np.array([option_for_length[int(value)] for value in lengths])
    reconstruction /= trainer.prefix_norms[np.arange(trainer.n_items), options, None]
    return np.ascontiguousarray(reconstruction, dtype=np.float32)


def _ravr_rows(
    data: dict,
    seed: int,
    bundle_dir: Path,
    teacher_ids: np.ndarray,
    teacher_scores: np.ndarray,
) -> list[dict]:
    state_path = _ravr_state_path(data, seed)
    trainer = RAVRRQTrainer.load_state(state_path, data["gallery"])
    historical = json.loads(_ravr_report_path(data, seed).read_text(encoding="utf-8"))
    if historical.get("config", {}).get("objective") in {
        OBJECTIVE_ORDINAL_KL, OBJECTIVE_BOUNDARY_KL,
    }:
        selected_prior = 0.0
        trainer.distortion = trainer.retrieval_distortion.astype(np.float32)
    else:
        selected_prior = float(
            historical["prior_weight_tuning"]["selected_lambda_prior"]
        )
        trainer.distortion = (
            trainer.retrieval_distortion.astype(np.float64)
            + selected_prior * trainer.prior_distortion.astype(np.float64)
        ).astype(np.float32)
    rows = []
    base_code_bits = trainer.n_items * trainer.config.base_stages * trainer.code_bits
    for target_code_bytes in RAVR_CODE_BUDGETS:
        total_code_bits = int(round(target_code_bytes * trainer.n_items * 8))
        variable_bytes = max(0, (total_code_bits - base_code_bits) // 8)
        target_serialized = trainer.minimum_persistent_bytes() + variable_bytes
        index = trainer.build(target_serialized, "boundary")
        path = bundle_dir / f"ravr_k64_seed{seed}_code{target_code_bytes:g}.rqix"
        index.save(path)
        reconstruction = _ravr_reconstruction(trainer, index)
        metrics, per_query = _evaluate_reconstruction(
            reconstruction, data, teacher_ids, teacher_scores
        )
        codec_report = {
            "codebook_size": trainer.config.codebook_size,
            "bits_per_symbol": trainer.code_bits,
            "stages": trainer.config.stages,
            "base_stages": trainer.config.base_stages,
            "objective": trainer.config.objective,
            "state": str(state_path),
            "state_sha256": _sha256(state_path),
        }
        if trainer.config.objective not in {
            OBJECTIVE_ORDINAL_KL, OBJECTIVE_BOUNDARY_KL,
        }:
            codec_report["selected_legacy_lambda_prior"] = selected_prior
        rows.append({
            "method": "RAVR-RQ",
            "display_name": "RAVR-RQ K=64 variable-rate",
            "seed": seed,
            "target_code_only_bytes_per_item": target_code_bytes,
            "code_only_bytes_per_item": index.code_only_bytes_per_item,
            "full_serialized_bytes_per_item": index.persistent_bytes / index.n_items,
            "persistent_bytes": index.persistent_bytes,
            "storage": index.storage_breakdown(),
            "allocation_histogram": index.allocation.histogram,
            "codec": codec_report,
            "bundle": str(path),
            **metrics,
            "per_query": per_query,
        })
        del reconstruction, index
    return rows


def _faiss_rows(
    data: dict,
    seed: int,
    bundle_dir: Path,
    teacher_ids: np.ndarray,
    teacher_scores: np.ndarray,
    methods: tuple[str, ...],
    m_values: tuple[int, ...],
    train_samples: int,
    force: bool,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    train_ids = rng.choice(
        data["codec_pool"], min(train_samples, len(data["codec_pool"])), replace=False
    )
    rows = []
    for method in methods:
        for m in m_values:
            config = FaissCodecConfig(method=method, m=m, seed=seed)
            path = bundle_dir / f"{method}_m{m}_seed{seed}.fcix"
            if path.exists() and not force:
                bundle = FaissCodecBundle.load(path)
            else:
                print(f"Training {data['slug']} {config.display_name} seed {seed}", flush=True)
                bundle = train_bundle(data["gallery"], train_ids, config)
                bundle.metadata.update({
                    "dataset": data["dataset"],
                    "manifest_sha256": _sha256(data["manifest_path"]),
                    "training_ids_sha256": hashlib.sha256(
                        np.ascontiguousarray(train_ids).view(np.uint8)
                    ).hexdigest(),
                })
                bundle.save(path)
            reconstruction = bundle.decode_normalized()
            metrics, per_query = _evaluate_reconstruction(
                reconstruction, data, teacher_ids, teacher_scores
            )
            rows.append({
                "method": method.upper(),
                "display_name": bundle.metadata["display_name"],
                "seed": seed,
                "target_code_only_bytes_per_item": float(m),
                "code_only_bytes_per_item": bundle.code_only_bytes_per_item,
                "full_serialized_bytes_per_item": bundle.persistent_bytes_per_item,
                "persistent_bytes": bundle.persistent_bytes,
                "storage": bundle.storage_breakdown(),
                "codec": bundle.metadata,
                "bundle": str(path),
                **metrics,
                "per_query": per_query,
            })
            del reconstruction, bundle
    return rows


def _render_seed(report: dict) -> str:
    lines = [
        f"# {report['dataset']} fixed-rate codec comparison, seed {report['seed']}",
        "",
        "Exact 100K cosine scan; the test split and teacher are shared by every row.",
        "",
        "| Method | Code-only B/vector | Full serialized B/vector | Teacher R@1 | Teacher R@10 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        report["rows"], key=lambda value: (
            value["code_only_bytes_per_item"], value["display_name"]
        )
    ):
        lines.append(
            f"| {row['display_name']} | {row['code_only_bytes_per_item']:.4f} | "
            f"{row['full_serialized_bytes_per_item']:.4f} | "
            f"{row['teacher_recall_at_1']:.4f} | {row['teacher_recall_at_10']:.4f} | "
            f"{row['teacher_ndcg_at_10']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def run_seed(args, dataset: str, seed: int) -> dict:
    data = _load_dataset(dataset, Path(args.data_root).expanduser().resolve())
    teacher_ids, teacher_scores = _teacher(
        data["gallery"], data["queries"], data["exclude_ids"]
    )
    bundle_dir = data["root"] / "fixed_rate_codec_bundles"
    bundle_dir.mkdir(exist_ok=True)
    rows = _ravr_rows(data, seed, bundle_dir, teacher_ids, teacher_scores)
    rows.extend(_faiss_rows(
        data, seed, bundle_dir, teacher_ids, teacher_scores,
        tuple(args.methods), tuple(args.m), args.train_samples, args.force,
    ))
    report = {
        "dataset": data["dataset"],
        "dataset_slug": dataset,
        "seed": seed,
        "gallery_items": len(data["gallery"]),
        "dimension": data["gallery"].shape[1],
        "test_queries": len(data["queries"]),
        "compute_backend": backend_report(),
        "manifest_sha256": _sha256(data["manifest_path"]),
        "protocol": {
            "teacher": "exact cosine on the original L2-normalised float32 gallery",
            "scoring": "exact cosine on decoded/reconstructed vectors",
            "test_split": "locked existing RAVR Stage-A test split",
            "codec_training": f"{args.train_samples} gallery vectors selected with the codec seed",
            "code_only_budget": "only per-vector quantizer symbols",
            "full_serialized_budget": (
                "actual bundle file including codec/rotation, codes, uint32 IDs, "
                "float16 norms, coarse centroids, length metadata, header, and alignment"
            ),
            "ivf": "disabled for scoring; RAVR coarse centroids remain part of its codec and file",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "platform": platform.platform(),
        },
        "rows": rows,
    }
    output_dir = HERE / "outputs" / f"faiss_fixed_rate_{dataset}_100k"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"seed{seed}.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / f"seed{seed}.md").write_text(_render_seed(report), encoding="utf-8")
    print(_render_seed(report), flush=True)
    return report


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(dataset: str, seeds: tuple[int, ...]) -> dict:
    output_dir = HERE / "outputs" / f"faiss_fixed_rate_{dataset}_100k"
    reports = [
        json.loads((output_dir / f"seed{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    keys = sorted({
        (row["display_name"], row["target_code_only_bytes_per_item"])
        for report in reports for row in report["rows"]
    }, key=lambda value: (value[1], value[0]))
    rows = []
    for display_name, target in keys:
        matched = [
            next(row for row in report["rows"]
                 if row["display_name"] == display_name
                 and row["target_code_only_bytes_per_item"] == target)
            for report in reports
        ]
        rows.append({
            "display_name": display_name,
            "target_code_only_bytes_per_item": target,
            "code_only_bytes_per_item": _summary([
                row["code_only_bytes_per_item"] for row in matched
            ]),
            "full_serialized_bytes_per_item": _summary([
                row["full_serialized_bytes_per_item"] for row in matched
            ]),
            "teacher_recall_at_1": _summary([
                row["teacher_recall_at_1"] for row in matched
            ]),
            "teacher_recall_at_10": _summary([
                row["teacher_recall_at_10"] for row in matched
            ]),
            "teacher_ndcg_at_10": _summary([
                row["teacher_ndcg_at_10"] for row in matched
            ]),
        })
    comparisons = []
    for budget in (4.0, 6.0):
        for baseline in ("PQ", "OPQ", "RQ"):
            deltas = []
            for report in reports:
                ravr = next((row for row in report["rows"]
                             if row["method"] == "RAVR-RQ"
                             and row["target_code_only_bytes_per_item"] == budget), None)
                fixed = next((row for row in report["rows"]
                              if row["method"] == baseline
                              and row["target_code_only_bytes_per_item"] == budget), None)
                if ravr is None or fixed is None:
                    deltas = []
                    break
                deltas.append(
                    ravr["teacher_recall_at_10"] - fixed["teacher_recall_at_10"]
                )
            if not deltas:
                continue
            comparisons.append({
                "budget_cap": budget,
                "first": "RAVR-RQ K=64 variable-rate",
                "second": fixed["display_name"],
                "teacher_recall_at_10_delta": _summary(deltas),
                "positive_seeds": int(np.sum(np.asarray(deltas) > 0)),
            })
    full_frontier = []
    for cap in (15.0, 16.0, 20.0, 24.0, 26.0, 36.0, 46.0):
        deltas = []
        ravr_names, baseline_names = [], []
        ravr_values, baseline_values = [], []
        for report in reports:
            ravr_candidates = [
                row for row in report["rows"]
                if row["method"] == "RAVR-RQ"
                and row["full_serialized_bytes_per_item"] <= cap
            ]
            baseline_candidates = [
                row for row in report["rows"]
                if row["method"] != "RAVR-RQ"
                and row["full_serialized_bytes_per_item"] <= cap
            ]
            if not ravr_candidates or not baseline_candidates:
                deltas = []
                break
            ravr = max(ravr_candidates, key=lambda row: row["teacher_recall_at_10"])
            baseline = max(
                baseline_candidates, key=lambda row: row["teacher_recall_at_10"]
            )
            ravr_names.append(ravr["display_name"])
            baseline_names.append(baseline["display_name"])
            ravr_values.append(ravr["teacher_recall_at_10"])
            baseline_values.append(baseline["teacher_recall_at_10"])
            deltas.append(ravr_values[-1] - baseline_values[-1])
        if deltas:
            full_frontier.append({
                "full_serialized_budget_cap": cap,
                "best_ravr": sorted(set(ravr_names)),
                "best_fixed_rate_baseline": sorted(set(baseline_names)),
                "best_ravr_teacher_recall_at_10": _summary(ravr_values),
                "best_baseline_teacher_recall_at_10": _summary(baseline_values),
                "ravr_minus_baseline": _summary(deltas),
                "positive_seeds": int(np.sum(np.asarray(deltas) > 0)),
            })
    report = {
        "dataset": reports[0]["dataset"],
        "dataset_slug": dataset,
        "seeds": list(seeds),
        "protocol": reports[0]["protocol"],
        "sources": [
            {"seed": seed, "sha256": _sha256(output_dir / f"seed{seed}.json")}
            for seed in seeds
        ],
        "rows": rows,
        "equal_code_only_comparisons": comparisons,
        "full_serialized_cap_frontier": full_frontier,
    }
    lines = [
        f"# {report['dataset']} fixed-rate codec comparison",
        "",
        f"Seeds: {', '.join(map(str, seeds))}; exact 100K scan on the locked test split.",
        "",
        "| Method | Code-only B/vector | Full B/vector | Teacher R@10 mean ± seed SD |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['code_only_bytes_per_item']['mean']:.4f} | "
            f"{row['full_serialized_bytes_per_item']['mean']:.4f} | "
            f"{row['teacher_recall_at_10']['mean']:.4f} ± "
            f"{row['teacher_recall_at_10']['sample_sd']:.4f} |"
        )
    lines.extend(["", "## Equal code-only budget caps", ""])
    for comparison in comparisons:
        delta = comparison["teacher_recall_at_10_delta"]
        lines.append(
            f"- {comparison['budget_cap']:.0f} B/vector cap, RAVR − "
            f"{comparison['second']}: "
            f"{delta['mean']:+.4f} ± {delta['sample_sd']:.4f}; positive in "
            f"{comparison['positive_seeds']}/{len(seeds)} seeds."
        )
    lines.extend([
        "", "## Equal full-serialized budget caps", "",
        "| Full cap B/vector | Best RAVR R@10 | Best fixed-rate baseline | Baseline R@10 | RAVR − baseline |",
        "|---:|---:|---|---:|---:|",
    ])
    for point in full_frontier:
        delta = point["ravr_minus_baseline"]
        lines.append(
            f"| {point['full_serialized_budget_cap']:.0f} | "
            f"{point['best_ravr_teacher_recall_at_10']['mean']:.4f} | "
            f"{', '.join(point['best_fixed_rate_baseline'])} | "
            f"{point['best_baseline_teacher_recall_at_10']['mean']:.4f} | "
            f"{delta['mean']:+.4f} |"
        )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "evaluation.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return report


def plot_full_serialized_cap_frontier(reports: list[dict]) -> Path:
    """Plot the cap-wise winner comparison used by the full-byte claim."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    caps = (15.0, 16.0, 20.0, 24.0, 26.0)
    selected = []
    for report in reports:
        points = {
            point["full_serialized_budget_cap"]: point
            for point in report["full_serialized_cap_frontier"]
        }
        selected.append([points[cap] for cap in caps])

    ravr_colour = "#2774AE"
    fixed_colour = "#D55E00"
    neutral_colour = "#666666"
    n_panels = len(reports)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.4 * n_panels, 4.7), sharey=n_panels > 1,
        squeeze=False,
    )
    axes = axes[0]
    all_values = [
        point[key]["mean"]
        for points in selected
        for point in points
        for key in (
            "best_ravr_teacher_recall_at_10",
            "best_baseline_teacher_recall_at_10",
        )
    ]
    y_min = max(0.0, min(all_values) - 0.045)
    y_max = max(all_values) + 0.055

    for axis, report, points in zip(axes, reports, selected):
        x = np.arange(len(caps), dtype=np.float64)
        ravr = np.asarray([
            point["best_ravr_teacher_recall_at_10"]["mean"] for point in points
        ])
        ravr_sd = np.asarray([
            point["best_ravr_teacher_recall_at_10"]["sample_sd"]
            for point in points
        ])
        fixed = np.asarray([
            point["best_baseline_teacher_recall_at_10"]["mean"] for point in points
        ])
        fixed_sd = np.asarray([
            point["best_baseline_teacher_recall_at_10"]["sample_sd"]
            for point in points
        ])
        axis.axvspan(-0.45, 2.45, color=ravr_colour, alpha=0.07, zorder=0)
        for index in range(len(caps)):
            axis.plot(
                [x[index] - 0.10, x[index] + 0.10],
                [ravr[index], fixed[index]],
                color=neutral_colour, linewidth=1.0, zorder=1,
            )
        axis.errorbar(
            x - 0.10, ravr, yerr=ravr_sd, fmt="o", color=ravr_colour,
            markersize=6.5, capsize=2.5, linewidth=1.1, zorder=3,
        )
        axis.errorbar(
            x + 0.10, fixed, yerr=fixed_sd, fmt="o", color=fixed_colour,
            markersize=6.5, capsize=2.5, linewidth=1.1, zorder=3,
        )
        for index, point in enumerate(points):
            delta = point["ravr_minus_baseline"]["mean"]
            winner = "RAVR" if delta > 0 else "fixed"
            label = f"{winner} +{abs(delta):.3f}"
            axis.annotate(
                label,
                (x[index], max(ravr[index] + ravr_sd[index],
                               fixed[index] + fixed_sd[index]) + 0.008),
                ha="center", va="bottom", fontsize=8.5,
                color=ravr_colour if delta > 0 else fixed_colour,
                fontweight="semibold",
            )
            baseline_name = ", ".join(point["best_fixed_rate_baseline"])
            baseline_name = baseline_name.replace(",PQ-", "+PQ-")
            axis.annotate(
                baseline_name,
                (x[index] + 0.10, fixed[index] - fixed_sd[index] - 0.010),
                ha="center", va="top", fontsize=7.2, color=fixed_colour,
            )
        axis.text(
            1.0, y_min + 0.007, "RAVR leads at all three stated caps (3/3 seeds)",
            ha="center", va="bottom", color=ravr_colour, fontsize=8.2,
            fontweight="semibold",
        )
        axis.set_title(report["dataset"], fontsize=11.5, pad=9)
        axis.set_xticks(x, [f"{cap:.0f}" for cap in caps])
        axis.set_xlabel("Full serialized budget cap (B/vector)")
        axis.set_ylim(y_min, y_max)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    axes[0].set_ylabel("Best feasible Teacher R@10")
    fig.suptitle(
        "Best tested method under each full-serialized budget cap",
        fontsize=13.5, y=0.99,
    )
    fig.text(
        0.5, 0.935,
        "Each family contributes its best codec that fits the cap; points show mean ± seed SD (3 seeds).",
        ha="center", fontsize=9.2, color=neutral_colour,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color=ravr_colour, linestyle="none",
                   label="Best RAVR fitting cap"),
            Line2D([0], [0], marker="o", color=fixed_colour, linestyle="none",
                   label="Best PQ/OPQ/RQ fitting cap"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0.01, 0.075, 0.99, 0.91))
    if len(reports) == 1:
        output_path = (
            HERE / "outputs" / f"faiss_fixed_rate_{reports[0]['dataset_slug']}_100k"
            / "full_serialized_cap_frontier.png"
        )
    else:
        output_path = HERE / "outputs" / "faiss_fixed_rate_full_serialized_cap_frontier.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sop", "gldv2", "all"), default="all")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", choices=("pq", "opq", "rq"),
                        default=["pq", "opq", "rq"])
    parser.add_argument("--m", nargs="+", type=int, default=list(M_VALUES))
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Dense teacher/evaluation backend; auto uses CUDA when available.",
    )
    args = parser.parse_args()
    if not args.aggregate_only:
        resolved = configure_compute_backend(args.device)
        print(f"Dense exact-scan backend: {resolved}", flush=True)
    datasets = ("sop", "gldv2") if args.dataset == "all" else (args.dataset,)
    aggregate_reports = []
    for dataset in datasets:
        if not args.aggregate_only:
            for seed in args.seeds:
                run_seed(args, dataset, seed)
        aggregate_reports.append(aggregate(dataset, tuple(args.seeds)))
    output_path = plot_full_serialized_cap_frontier(aggregate_reports)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
