"""Capacity ablation for the frozen Ordinal-KL RAVR objective.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmark_faiss_fixed_rate import (
    HERE,
    _evaluate_reconstruction,
    _load_dataset,
    _ravr_reconstruction,
    _ravr_state_path,
    _sha256,
    _summary,
    _teacher,
)
from ravr_rq import (
    OBJECTIVE_ORDINAL_KL,
    RAVRRQConfig,
    RAVRRQIndex,
    RAVRRQTrainer,
)


DEFAULT_DATA = Path.home() / "Desktop" / "AJigma_data"
DEFAULT_DEPTHS = (8, 12, 16)
DEFAULT_SEEDS = (17, 42, 73)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _roles(data: dict) -> dict:
    manifest = np.load(data["manifest_path"], allow_pickle=False)
    calibration_ids = np.asarray(manifest["calibration_ids"], dtype=np.int64)
    validation_ids = np.asarray(manifest["validation_ids"], dtype=np.int64)
    if data["slug"] == "sop":
        return {
            "calibration": data["gallery"][calibration_ids],
            "calibration_exclude": calibration_ids,
            "validation": data["gallery"][validation_ids],
            "validation_exclude": validation_ids,
            "codec_train_ids": np.asarray(manifest["codec_train_ids"], dtype=np.int64),
        }
    all_queries = np.load(
        data["root"] / "gldv2_test_queries_dinov2_vits14.npy", mmap_mode="r"
    )
    return {
        "calibration": np.asarray(all_queries[calibration_ids], dtype=np.float32),
        "calibration_exclude": None,
        "validation": np.asarray(all_queries[validation_ids], dtype=np.float32),
        "validation_exclude": None,
        "codec_train_ids": None,
    }


def _config(dataset: str, seed: int, depth: int) -> RAVRRQConfig:
    if depth < 8:
        raise ValueError("the capacity ablation is defined for depth >= 8")
    return RAVRRQConfig(
        seed=seed,
        num_centroids=64,
        nprobe=64,
        stages=depth,
        codebook_size=64,
        allowed_lengths=tuple(range(4, depth + 1)),
        base_stages=4,
        rerank_candidates=10000 if dataset == "sop" else 60000,
        retrieval_k=10,
        hard_negative_depth=50,
        objective=OBJECTIVE_ORDINAL_KL,
        kmeans_iterations=5,
        train_samples=12000,
        norm_mode="gram_tables",
    )


def _state_path(data: dict, depth: int, seed: int) -> Path:
    if depth == 8:
        return _ravr_state_path(data, seed)
    return data["root"] / "depth_ablation_states" / f"ravr_t{depth}_seed{seed}.npz"


def _state_matches(trainer: RAVRRQTrainer, expected: RAVRRQConfig) -> bool:
    actual = trainer.config
    # Validation may lower rerank_candidates after fitting. It does not affect
    # exhaustive decoded-gallery evaluation, and its selected value is shared
    # by every depth because the first four stages are nested.
    fields = (
        "seed", "num_centroids", "nprobe", "stages", "codebook_size",
        "allowed_lengths", "base_stages", "retrieval_k", "hard_negative_depth",
        "kmeans_iterations", "train_samples", "norm_mode", "item_id_dtype",
        "allocation_search_steps",
    )
    return all(getattr(actual, name) == getattr(expected, name) for name in fields)


def _trainer(
    data: dict,
    roles: dict,
    depth: int,
    seed: int,
    force_train: bool,
) -> tuple[RAVRRQTrainer, Path, bool]:
    expected = _config(data["slug"], seed, depth)
    state_path = _state_path(data, depth, seed)
    loaded = state_path.exists() and not force_train
    if loaded:
        trainer = RAVRRQTrainer.load_state(state_path, data["gallery"])
        if not _state_matches(trainer, expected):
            raise ValueError(f"state configuration mismatch: {state_path}")
        if trainer.config.objective != OBJECTIVE_ORDINAL_KL:
            trainer.recalibrate(
                roles["calibration"], roles["calibration_exclude"],
                objective=OBJECTIVE_ORDINAL_KL,
            )
            # The canonical T=8 states are historical shared artifacts. Keep the
            # objective conversion local; newly trained depth states are frozen.
            if depth != 8:
                trainer.save_state(state_path)
        return trainer, state_path, True

    if depth == 8:
        raise FileNotFoundError(
            f"canonical T=8 state is required for a fixed-condition ablation: {state_path}"
        )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trainer = RAVRRQTrainer(expected).fit(
        data["gallery"],
        roles["calibration"],
        calibration_gallery_ids=roles["calibration_exclude"],
        validation_queries=roles["validation"],
        validation_gallery_ids=roles["validation_exclude"],
        codec_train_ids=roles["codec_train_ids"],
    )
    trainer.save_state(state_path)
    print(
        f"trained {data['slug']} seed {seed} T={depth} in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return trainer, state_path, False


def _external_rows(dataset: str, seed: int) -> tuple[Path, dict, list[dict]]:
    path = HERE / "outputs" / f"faiss_fixed_rate_{dataset}_100k" / f"seed{seed}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in report["rows"] if row["method"] in {"PQ", "OPQ", "RQ"}]
    if len(rows) != 9:
        raise AssertionError(f"expected nine external points, found {len(rows)}")
    for row in rows:
        if os.path.getsize(row["bundle"]) != int(row["persistent_bytes"]):
            raise AssertionError(f"external bundle-size mismatch: {row['bundle']}")
    return path, report, rows


def _assert_nested(shallow: RAVRRQTrainer, deep: RAVRRQTrainer) -> None:
    stages = shallow.config.stages
    if deep.config.stages <= stages:
        raise ValueError("nested-codec check requires increasing depth")
    checks = (
        np.array_equal(shallow.centroids, deep.centroids),
        np.array_equal(shallow.cluster_ids, deep.cluster_ids),
        np.array_equal(shallow.codebooks, deep.codebooks[:stages]),
        np.array_equal(shallow.full_codes, deep.full_codes[:, :stages]),
    )
    if not all(checks):
        raise AssertionError(
            f"T={stages} is not an exact prefix of T={deep.config.stages}"
        )


def run_seed(
    dataset: str,
    seed: int,
    depths: tuple[int, ...],
    data_root: Path,
    force_train: bool = False,
) -> dict:
    data = _load_dataset(dataset, data_root)
    roles = _roles(data)
    source_path, source, external = _external_rows(dataset, seed)
    if source["manifest_sha256"] != _sha256(data["manifest_path"]):
        raise AssertionError("external report and current manifest do not match")
    teacher_ids, teacher_scores = _teacher(
        data["gallery"], data["queries"], data["exclude_ids"]
    )
    caps = sorted({int(row["persistent_bytes"]) for row in external})
    baseline_by_cap = {int(row["persistent_bytes"]): row for row in external}
    bundle_dir = data["root"] / "depth_ablation_ordinal_bundles"
    bundle_dir.mkdir(exist_ok=True)
    depth_reports = []
    trainers = []

    for depth in depths:
        print(f"{dataset} seed {seed}: preparing T={depth}", flush=True)
        trainer, state_path, loaded = _trainer(
            data, roles, depth, seed, force_train=force_train
        )
        if trainer.config.objective != OBJECTIVE_ORDINAL_KL:
            raise AssertionError("depth ablation did not use Ordinal-KL")
        if trainers:
            _assert_nested(trainers[-1], trainer)
        trainers.append(trainer)
        minimum = trainer.minimum_persistent_bytes()
        fixed_full = trainer.fixed_length_budget(depth)
        metric_cache: dict[str, tuple[dict, dict]] = {}
        points = []
        for cap in caps:
            baseline = baseline_by_cap[cap]
            if cap < minimum:
                points.append({
                    "depth": depth,
                    "target_cap_bytes": cap,
                    "external_display_name": baseline["display_name"],
                    "external_method": baseline["method"],
                    "feasible": False,
                    "minimum_persistent_bytes": minimum,
                    "minimum_full_bytes_per_item": minimum / trainer.n_items,
                })
                continue
            index = trainer.build(cap, "boundary")
            if index.persistent_bytes > cap:
                raise AssertionError("RAVR exceeded its external full-size cap")
            path = bundle_dir / f"ravr_ordinal_t{depth}_seed{seed}_cap{cap}.rqix"
            index.save(path)
            loaded_index = RAVRRQIndex.load(path)
            if loaded_index.persistent_bytes != index.persistent_bytes:
                raise AssertionError("reloaded RAVR ledger changed")
            lengths = index.assigned_lengths_by_id()
            allocation_hash = hashlib.sha256(
                np.ascontiguousarray(lengths).view(np.uint8)
            ).hexdigest()
            if allocation_hash not in metric_cache:
                reconstruction = _ravr_reconstruction(trainer, index)
                metric_cache[allocation_hash] = _evaluate_reconstruction(
                    reconstruction, data, teacher_ids, teacher_scores
                )
                del reconstruction
            metrics, per_query = metric_cache[allocation_hash]
            points.append({
                "depth": depth,
                "target_cap_bytes": cap,
                "external_display_name": baseline["display_name"],
                "external_method": baseline["method"],
                "external_full_bytes_per_item": baseline["full_serialized_bytes_per_item"],
                "external_teacher_recall_at_10": baseline["teacher_recall_at_10"],
                "external_teacher_ndcg_at_10": baseline["teacher_ndcg_at_10"],
                "feasible": True,
                "persistent_bytes": index.persistent_bytes,
                "full_bytes_per_item": index.persistent_bytes / trainer.n_items,
                "size_slack_bytes": cap - index.persistent_bytes,
                "physical_fixed_full_depth_bytes": fixed_full,
                "physical_fixed_full_depth_bytes_per_item": fixed_full / trainer.n_items,
                "code_only_bytes_per_item": index.code_only_bytes_per_item,
                "mean_assigned_length": float(np.mean(lengths)),
                "deepest_stage_items": int(np.sum(lengths == depth)),
                "allocation_histogram": index.allocation.histogram,
                "allocation_sha256": allocation_hash,
                "bundle": str(path),
                "bundle_sha256": _sha256(path),
                "storage": index.storage_breakdown(),
                **metrics,
                "per_query": per_query,
            })
        depth_reports.append({
            "depth": depth,
            "allowed_lengths": list(trainer.config.allowed_lengths),
            "config": asdict(trainer.config),
            "state": str(state_path),
            "state_sha256": _sha256(state_path),
            "state_loaded": loaded,
            "fit_seconds": trainer.fit_seconds,
            "minimum_persistent_bytes": minimum,
            "minimum_full_bytes_per_item": minimum / trainer.n_items,
            "physical_fixed_full_depth_bytes": fixed_full,
            "physical_fixed_full_depth_bytes_per_item": fixed_full / trainer.n_items,
            "points": points,
        })

    report = {
        "dataset": data["dataset"],
        "dataset_slug": dataset,
        "seed": seed,
        "gallery_items": len(data["gallery"]),
        "test_queries": len(data["queries"]),
        "objective": OBJECTIVE_ORDINAL_KL,
        "controlled_variables": {
            "codebook_size": 64,
            "base_stages": 4,
            "codebook_storage": "float16",
            "dimension": int(data["gallery"].shape[1]),
            "changed_only": ["stages", "allowed_lengths"],
        },
        "comparison_axis": "complete non-query-side serialized bytes",
        "manifest_sha256": _sha256(data["manifest_path"]),
        "external_source_report": str(source_path),
        "external_source_report_sha256": _sha256(source_path),
        "external_rows": [{
            "method": row["method"],
            "display_name": row["display_name"],
            "persistent_bytes": row["persistent_bytes"],
            "full_bytes_per_item": row["full_serialized_bytes_per_item"],
            "teacher_recall_at_10": row["teacher_recall_at_10"],
            "teacher_ndcg_at_10": row["teacher_ndcg_at_10"],
            "bundle": row["bundle"],
        } for row in external],
        "depths": depth_reports,
        "nested_prefix_check": "passed",
    }
    output = HERE / "outputs" / f"ravr_depth_ablation_{dataset}_100k"
    _atomic_json(output / f"seed{seed}.json", report)
    return report


def aggregate(dataset: str, seeds: tuple[int, ...], depths: tuple[int, ...]) -> dict:
    output = HERE / "outputs" / f"ravr_depth_ablation_{dataset}_100k"
    reports = [json.loads((output / f"seed{seed}.json").read_text()) for seed in seeds]
    external_names = [row["display_name"] for row in reports[0]["external_rows"]]
    depth_rows = []
    for depth in depths:
        depth_seed_reports = [
            next(row for row in report["depths"] if int(row["depth"]) == depth)
            for report in reports
        ]
        for name in external_names:
            matched = [
                next(point for point in row["points"] if point["external_display_name"] == name)
                for row in depth_seed_reports
            ]
            feasible = all(point["feasible"] for point in matched)
            entry = {
                "depth": depth,
                "external_display_name": name,
                "external_method": matched[0]["external_method"],
                "feasible_all_seeds": feasible,
                "minimum_full_bytes_per_item": _summary([
                    row["minimum_full_bytes_per_item"] for row in depth_seed_reports
                ]),
                "physical_fixed_full_depth_bytes_per_item": _summary([
                    row["physical_fixed_full_depth_bytes_per_item"]
                    for row in depth_seed_reports
                ]),
            }
            if feasible:
                if not all(
                    point["persistent_bytes"] <= point["target_cap_bytes"]
                    for point in matched
                ):
                    raise AssertionError("seed-level full-size constraint failed")
                entry.update({
                    "external_full_bytes_per_item": _summary([
                        point["external_full_bytes_per_item"] for point in matched
                    ]),
                    "external_teacher_recall_at_10": _summary([
                        point["external_teacher_recall_at_10"] for point in matched
                    ]),
                    "external_teacher_ndcg_at_10": _summary([
                        point["external_teacher_ndcg_at_10"] for point in matched
                    ]),
                    "ravr_full_bytes_per_item": _summary([
                        point["full_bytes_per_item"] for point in matched
                    ]),
                    "ravr_teacher_recall_at_10": _summary([
                        point["teacher_recall_at_10"] for point in matched
                    ]),
                    "ravr_teacher_ndcg_at_10": _summary([
                        point["teacher_ndcg_at_10"] for point in matched
                    ]),
                    "ravr_mean_assigned_length": _summary([
                        point["mean_assigned_length"] for point in matched
                    ]),
                    "ravr_deepest_stage_items": _summary([
                        point["deepest_stage_items"] for point in matched
                    ]),
                    "recall_delta_vs_external": _summary([
                        point["teacher_recall_at_10"]
                        - point["external_teacher_recall_at_10"]
                        for point in matched
                    ]),
                    "ndcg_delta_vs_external": _summary([
                        point["teacher_ndcg_at_10"]
                        - point["external_teacher_ndcg_at_10"]
                        for point in matched
                    ]),
                    "size_constraint_satisfied_all_seeds": True,
                })
            depth_rows.append(entry)

    # Deployment rule chosen without looking at retrieval performance: use the
    # shallowest codec whose exact all-items/full-depth serialized ceiling can
    # cover the requested external byte cap. This avoids test-set selection of T.
    budget_selected = []
    for name in external_names:
        candidates = [
            row for row in depth_rows
            if row["external_display_name"] == name and row["feasible_all_seeds"]
        ]
        if not candidates:
            raise AssertionError(f"no feasible depth for {name}")
        cap = candidates[0]["external_full_bytes_per_item"]["mean"]
        covering = [
            row for row in candidates
            if row["physical_fixed_full_depth_bytes_per_item"]["mean"] >= cap
        ]
        selected = min(covering, key=lambda row: row["depth"]) if covering else max(
            candidates, key=lambda row: row["depth"]
        )
        budget_selected.append({
            "external_display_name": name,
            "external_method": selected["external_method"],
            "selected_depth": selected["depth"],
            "selection_used_performance": False,
            "external_full_bytes_per_item": selected["external_full_bytes_per_item"],
            "external_teacher_recall_at_10": selected["external_teacher_recall_at_10"],
            "external_teacher_ndcg_at_10": selected["external_teacher_ndcg_at_10"],
            "ravr_full_bytes_per_item": selected["ravr_full_bytes_per_item"],
            "ravr_teacher_recall_at_10": selected["ravr_teacher_recall_at_10"],
            "ravr_teacher_ndcg_at_10": selected["ravr_teacher_ndcg_at_10"],
            "recall_delta_vs_external": selected["recall_delta_vs_external"],
            "ndcg_delta_vs_external": selected["ndcg_delta_vs_external"],
            "size_constraint_satisfied_all_seeds": True,
        })

    aggregate_report = {
        "dataset": reports[0]["dataset"],
        "dataset_slug": dataset,
        "seeds": list(seeds),
        "depths": list(depths),
        "objective": OBJECTIVE_ORDINAL_KL,
        "comparison_axis": "complete non-query-side serialized bytes",
        "fairness_rule": "RAVR persistent_bytes <= external persistent_bytes per seed and point",
        "nested_prefix_check_all_seeds": all(
            report["nested_prefix_check"] == "passed" for report in reports
        ),
        "rows": depth_rows,
        "budget_only_depth_rule": (
            "choose the smallest T whose exact all-items/full-depth serialized "
            "ceiling is at least the target byte cap"
        ),
        "budget_selected_rows": budget_selected,
        "sources": [
            {"seed": seed, "sha256": _sha256(output / f"seed{seed}.json")}
            for seed in seeds
        ],
    }
    lines = [
        f"# {aggregate_report['dataset']} RAVR depth ablation",
        "",
        "Means over three codec seeds. Every feasible point uses an actually serialized",
        "RAVR bundle no larger than the named external bundle; JSON retains sample SD",
        "and every seed-level result.",
        "",
        "| T | External cap | Feasible | RAVR full B/vector | Mean depth | RAVR R@10 | External R@10 | Delta |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in depth_rows:
        if row["feasible_all_seeds"]:
            lines.append(
                f"| {row['depth']} | {row['external_display_name']} | yes | "
                f"{row['ravr_full_bytes_per_item']['mean']:.4f} | "
                f"{row['ravr_mean_assigned_length']['mean']:.3f} | "
                f"{row['ravr_teacher_recall_at_10']['mean']:.4f} | "
                f"{row['external_teacher_recall_at_10']['mean']:.4f} | "
                f"{row['recall_delta_vs_external']['mean']:+.4f} |"
            )
        else:
            lines.append(
                f"| {row['depth']} | {row['external_display_name']} | no | "
                f"minimum {row['minimum_full_bytes_per_item']['mean']:.4f} | — | — | — | — |"
            )
    lines.extend([
        "",
        "## Size-only depth rule",
        "",
        "The selected depth is the shallowest codec whose exact physical full-depth",
        "ceiling covers the target cap; retrieval performance is not used for selection.",
        "",
        "| External cap | Selected T | RAVR full B/vector | RAVR R@10 | External R@10 | Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in budget_selected:
        lines.append(
            f"| {row['external_display_name']} | {row['selected_depth']} | "
            f"{row['ravr_full_bytes_per_item']['mean']:.4f} | "
            f"{row['ravr_teacher_recall_at_10']['mean']:.4f} | "
            f"{row['external_teacher_recall_at_10']['mean']:.4f} | "
            f"{row['recall_delta_vs_external']['mean']:+.4f} |"
        )
    _atomic_json(output / "evaluation.json", aggregate_report)
    (output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate_report


def plot(reports: list[dict]) -> Path:
    import matplotlib.pyplot as plt

    external_colours = {"PQ": "#D55E00", "OPQ": "#CC79A7", "RQ": "#009E73"}
    depth_colours = {8: "#0072B2", 12: "#56B4E9", 16: "#6A3D9A", 24: "#E69F00", 32: "#000000"}
    fig, axes = plt.subplots(1, len(reports), figsize=(6.4 * len(reports), 4.9))
    if len(reports) == 1:
        axes = [axes]
    for axis, report in zip(axes, reports):
        for method, colour in external_colours.items():
            rows = [
                row for row in report["rows"]
                if row["depth"] == report["depths"][0]
                and row["external_method"] == method
            ]
            axis.plot(
                [row["external_full_bytes_per_item"]["mean"] for row in rows],
                [row["external_teacher_recall_at_10"]["mean"] for row in rows],
                "o--", color=colour, alpha=0.72, label=f"Faiss {method}",
            )
        for depth in report["depths"]:
            rows = [
                row for row in report["rows"]
                if row["depth"] == depth and row["feasible_all_seeds"]
            ]
            rows.sort(key=lambda row: row["ravr_full_bytes_per_item"]["mean"])
            unique = {}
            for row in rows:
                key = round(row["ravr_full_bytes_per_item"]["mean"], 6)
                unique[key] = row
            rows = list(unique.values())
            axis.plot(
                [row["ravr_full_bytes_per_item"]["mean"] for row in rows],
                [row["ravr_teacher_recall_at_10"]["mean"] for row in rows],
                "o-", linewidth=2.1, color=depth_colours.get(depth), label=f"RAVR T={depth}",
            )
        selected = sorted(
            report["budget_selected_rows"],
            key=lambda row: row["ravr_full_bytes_per_item"]["mean"],
        )
        axis.plot(
            [row["ravr_full_bytes_per_item"]["mean"] for row in selected],
            [row["ravr_teacher_recall_at_10"]["mean"] for row in selected],
            "*:", color="#333333", linewidth=1.5, markersize=7,
            label="RAVR size-only T rule",
        )
        axis.set_title(report["dataset"])
        axis.set_xlabel("Complete serialized non-query-side B/vector")
        axis.grid(alpha=0.28)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Teacher R@10")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = HERE / "outputs" / "ravr_depth_ablation_frontier.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sop", "gldv2", "all"), default="all")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--depths", nargs="+", type=int, default=list(DEFAULT_DEPTHS))
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    args = parser.parse_args()
    depths = tuple(sorted(set(args.depths)))
    if not depths or depths[0] != 8:
        raise ValueError("include T=8 first so every deeper codec is prefix-checked")
    seeds = tuple(args.seeds)
    datasets = ("sop", "gldv2") if args.dataset == "all" else (args.dataset,)
    data_root = Path(args.data_root).expanduser().resolve()
    if not args.aggregate_only:
        for dataset in datasets:
            for seed in seeds:
                run_seed(
                    dataset, seed, depths, data_root, force_train=args.force_train
                )
    if not args.no_aggregate:
        reports = [aggregate(dataset, seeds, depths) for dataset in datasets]
        print(f"Wrote {plot(reports)}", flush=True)


if __name__ == "__main__":
    main()
