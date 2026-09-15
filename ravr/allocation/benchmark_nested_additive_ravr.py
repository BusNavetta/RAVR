"""Matched-codec experiment: uniform prefixes versus RAVR allocation.

Every strategy shares one trained Faiss additive codec and its full item codes.
Only the number of leading symbols retained per item changes.  Comparisons use
actual reloadable files at the same packed-code and persistent byte cap.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmark_faiss_fixed_rate import (
    _evaluate_reconstruction,
    _load_dataset,
    _sha256,
    _teacher,
)
from compute_backend import backend_report, configure_compute_backend
from faiss_fixed_rate import FaissCodecBundle
from faiss_fp16_additive import FP16AdditiveBundle, from_faiss_additive_bundle
from nested_additive_ravr import (
    NestedAdditiveIndex,
    calibrate_frozen_additive_codec,
    codec_sha256,
    query_ip_distortion,
)
from ravr_depth_ablation import _roles
from ravr_rq import solve_lagrangian


DEFAULT_DATA = Path.home() / "Desktop" / "AJigma_data"
# Existing corpora keep their recorded directory names so stored artifacts and
# the aggregator stay valid; any new dataset gets outputs/nested_additive_ravr_<slug>.
PANEL_OUTPUT_DIRS = {
    "sop": "nested_additive_ravr_sop_100k",
    "gldv2": "nested_additive_ravr_gldv2_100k",
}


def panel_output_dir(dataset: str) -> str:
    return PANEL_OUTPUT_DIRS.get(dataset, f"nested_additive_ravr_{dataset}")



RERANK_CANDIDATES = {"sop": 10000, "gldv2": 60000}
DEFAULT_RERANK_CANDIDATES = 10000


def rerank_candidates_for(dataset: str) -> int:
    return RERANK_CANDIDATES.get(dataset, DEFAULT_RERANK_CANDIDATES)



_REPO = Path(__file__).resolve().parents[2]
PANEL_RESULTS_ROOTS = {
    "sop": _REPO / "experiments" / "table1_sop" / "outputs",
    "gldv2": _REPO / "experiments" / "table2_corpora" / "gldv2_100k" / "outputs",
    "gldv2_full": _REPO / "experiments" / "table2_corpora" / "gldv2_762k" / "outputs",
    "deep1m": _REPO / "experiments" / "table2_corpora" / "deep1m" / "outputs",
}


def panel_results_root(dataset: str) -> Path:
    """Folder holding every `nested_additive_ravr_<slug>*` directory for `dataset`.

    The `_hnd<depth>` siblings of a corpus live here too, which is what the
    depth-sweep plots glob for.
    """
    override = os.environ.get("RAVR_RESULTS_ROOT")
    if override:
        return Path(override)
    return PANEL_RESULTS_ROOTS.get(dataset, Path.cwd() / "outputs")


def panel_results_dir(dataset: str) -> Path:
    """The per-seed report directory for `dataset` at the default mining depth."""
    return panel_results_root(dataset) / panel_output_dir(dataset)


DEFAULT_SEEDS = (17, 42, 73, 3067, 4294, 4996, 5423, 7520, 7937, 9794)
ALLOWED_LENGTHS = (8, 10, 12, 14, 16)
DEFAULT_TARGET_LENGTHS = (10, 12, 14)
STRATEGIES = (
    "uniform",
    "ravr",
    "ravr_unseen_rd",
    "ordinal_per_exposure",
    "candidate_frequency",
    "query_ip_mse",
    "random_histogram",
    "reconstruction",
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _state_path(
    data: dict, factory: str, seed: int, calibration_queries: int, precision: str
) -> Path:
    precision_tag = "" if precision == "fp32" else f"_{precision}"
    return data["root"] / "nested_additive_states" / (
        f"{factory.lower()}_seed{seed}_ordinal{precision_tag}_cal{calibration_queries}.npz"
    )


def _bundle_path(data: dict, factory: str, seed: int, precision: str = "fp32") -> Path:
    if precision == "fp16":
        return data["root"] / "fp16_additive_bundles" / (
            f"{factory.lower()}_seed{seed}_fp16.fa16"
        )
    return data["root"] / "advanced_codec_bundles" / (
        f"{factory.lower()}_seed{seed}.fcix"
    )


def _load_bundle(data: dict, factory: str, seed: int, precision: str):
    if precision == "fp32":
        return FaissCodecBundle.load(_bundle_path(data, factory, seed, precision))
    if precision != "fp16":
        raise ValueError("precision must be fp32 or fp16")
    path = _bundle_path(data, factory, seed, precision)
    if not path.exists():
        source = FaissCodecBundle.load(_bundle_path(data, factory, seed, "fp32"))
        converted = from_faiss_additive_bundle(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        converted.save(path)
        converted.close()
    return FP16AdditiveBundle.load(path)


def _save_state(
    path: Path, state: dict, calibration_queries: int, rerank_candidates: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rerank_candidates=np.array(rerank_candidates, dtype=np.int64),
        distortion=state["distortion"],
        prior_distortion=state["prior_distortion"],
        query_ip_distortion=state["query_ip_distortion"],
        prefix_norms=state["prefix_norms"],
        coverage=state["coverage"],
        calibration_report_json=np.array(json.dumps(state["calibration_report"])),
        codec_hash=np.array(state["codec_hash"]),
        allowed_lengths=np.asarray(ALLOWED_LENGTHS, dtype=np.int16),
        calibration_queries=np.array(calibration_queries, dtype=np.int64),
    )


def _load_or_calibrate(
    data: dict,
    roles: dict,
    bundle,
    factory: str,
    seed: int,
    calibration_queries: int,
    precision: str,
    force: bool,
    rerank_candidates: int | None = None,
) -> tuple[dict, Path, bool, float]:
    if rerank_candidates is None:
        rerank_candidates = rerank_candidates_for(data["slug"])
    path = _state_path(data, factory, seed, calibration_queries, precision)
    if path.exists() and not force:
        state = np.load(path, allow_pickle=False)
        if str(state["codec_hash"]) != codec_sha256(bundle):
            raise ValueError(f"calibration state codec mismatch: {path}")
        if tuple(int(x) for x in state["allowed_lengths"]) != ALLOWED_LENGTHS:
            raise ValueError(f"calibration state prefix mismatch: {path}")
        # States written before the candidate depth became dataset-aware carry no
        # field; every one of those was mined at the SOP default.
        cached_candidates = (
            int(state["rerank_candidates"])
            if "rerank_candidates" in state.files else DEFAULT_RERANK_CANDIDATES
        )
        if cached_candidates != rerank_candidates:
            raise ValueError(
                f"calibration state was mined with {cached_candidates} rerank "
                f"candidates, this run needs {rerank_candidates}: {path}"
            )
        values = {name: np.array(state[name]) for name in (
            "distortion", "prior_distortion", "prefix_norms", "coverage"
        )} | {
            "calibration_report": json.loads(str(state["calibration_report_json"])),
            "codec_hash": str(state["codec_hash"]),
        }
        if "query_ip_distortion" in state.files:
            values["query_ip_distortion"] = np.array(state["query_ip_distortion"])
        else:
            print(f"seed {seed}: computing QAQ-style query-IP loss", flush=True)
            values["query_ip_distortion"] = query_ip_distortion(
                bundle,
                data["gallery"],
                roles["calibration"][:calibration_queries],
                values["prefix_norms"],
                ALLOWED_LENGTHS,
            )
            state.close()
            _save_state(path, values, calibration_queries, rerank_candidates)
        return values, path, True, 0.0

    started = time.perf_counter()
    trainer = calibrate_frozen_additive_codec(
        bundle,
        data["gallery"],
        roles["calibration"],
        roles["calibration_exclude"],
        allowed_lengths=ALLOWED_LENGTHS,
        rerank_candidates=rerank_candidates,
        retrieval_k=10,
        hard_negative_depth=50,
        calibration_limit=calibration_queries,
    )
    query_loss = query_ip_distortion(
        bundle,
        data["gallery"],
        roles["calibration"][:calibration_queries],
        trainer.prefix_norms,
        ALLOWED_LENGTHS,
    )
    elapsed = time.perf_counter() - started
    result = {
        "distortion": trainer.distortion,
        "prior_distortion": trainer.prior_distortion,
        "query_ip_distortion": query_loss,
        "prefix_norms": trainer.prefix_norms,
        "coverage": trainer.coverage,
        "calibration_report": trainer.calibration_report,
        "codec_hash": trainer.codec_hash,
    }
    _save_state(path, result, calibration_queries, rerank_candidates)
    return result, path, False, elapsed


def _selections(
    state: dict, n_items: int, seed: int, target_length: int, nbits: int
) -> dict[str, np.ndarray]:
    if target_length not in ALLOWED_LENGTHS:
        raise ValueError("target_length must occur in ALLOWED_LENGTHS")
    base_bytes = ALLOWED_LENGTHS[0] * nbits // 8
    option_costs = np.array([
        length * nbits // 8 - base_bytes for length in ALLOWED_LENGTHS
    ], dtype=np.int64)
    target_variable_bytes = n_items * (
        target_length * nbits // 8 - base_bytes
    )
    ravr = solve_lagrangian(
        state["distortion"], option_costs, target_variable_bytes
    )
    reconstruction = solve_lagrangian(
        state["prior_distortion"], option_costs, target_variable_bytes
    )
    query_ip = solve_lagrangian(
        state["query_ip_distortion"], option_costs, target_variable_bytes
    )
    coverage = np.asarray(state["coverage"], dtype=np.float64)
    ordinal_per_exposure = solve_lagrangian(
        np.asarray(state["distortion"], dtype=np.float64)
        / np.maximum(coverage[:, None], 1.0),
        option_costs,
        target_variable_bytes,
    )
    # A deliberately simple popularity control.  Each extra stored symbol has
    # value proportional only to how often the item entered RAVR's calibrated
    # hard-candidate set; it ignores boundary margins and prefix score errors.
    remaining_symbols = (
        ALLOWED_LENGTHS[-1] - np.asarray(ALLOWED_LENGTHS, dtype=np.int64)
    )
    candidate_frequency = solve_lagrangian(
        coverage[:, None] * remaining_symbols[None, :],
        option_costs,
        target_variable_bytes,
    )

    ravr_unseen_rd = ravr.copy()
    unseen = np.flatnonzero(coverage == 0)
    if len(unseen):
        seen = np.flatnonzero(coverage > 0)
        seen_cost = int(option_costs[ravr[seen]].sum())
        unseen_budget = target_variable_bytes - seen_cost
        if unseen_budget < 0:
            raise AssertionError("covered RAVR allocation exceeded the cap")
        ravr_unseen_rd[unseen] = solve_lagrangian(
            state["prior_distortion"][unseen], option_costs, unseen_budget
        )
    rng = np.random.default_rng(20260825 + int(seed))
    return {
        "uniform": np.full(
            n_items, ALLOWED_LENGTHS.index(target_length), dtype=np.int64
        ),
        "ravr": ravr,
        "ravr_unseen_rd": ravr_unseen_rd,
        "ordinal_per_exposure": ordinal_per_exposure,
        "candidate_frequency": candidate_frequency,
        "query_ip_mse": query_ip,
        "random_histogram": rng.permutation(ravr),
        "reconstruction": reconstruction,
    }


def _evaluation_data(data: dict, roles: dict, split: str) -> dict:
    if split == "test":
        return data
    result = dict(data)
    result["queries"] = roles["validation"]
    result["exclude_ids"] = roles["validation_exclude"]
    if result["exclude_ids"] is None:
        if data["exclude_ids"] is not None:
            raise ValueError(
                "in-gallery corpora must supply validation exclusion ids"
            )

        result["query_labels"] = np.full(
            len(result["queries"]), -1, dtype=np.int64
        )
        return result
    result["query_labels"] = result["gallery_labels"][result["exclude_ids"]]
    return result


def run_seed(
    data_root: Path,
    seed: int,
    *,
    split: str,
    calibration_queries: int,
    target_length: int = 12,
    factory: str = "LSQ16x4",
    precision: str = "fp32",
    force_calibration: bool = False,
    strategies: tuple[str, ...] = STRATEGIES,
    dataset: str = "sop",
    output_dir: Path | None = None,
) -> dict:
    precision_tag = "" if precision == "fp32" else f"_{precision}"
    data = _load_dataset(dataset, data_root)
    roles = _roles(data)
    bundle_path = _bundle_path(data, factory, seed, precision)
    bundle = _load_bundle(data, factory, seed, precision)
    if bundle.metadata.get("factory") != factory:
        raise ValueError(f"unexpected frozen codec: {bundle.metadata.get('factory')}")
    config = bundle.metadata["config"]
    nbits = int(config["nbits"])
    stages = int(config.get("stages", config.get("m")))
    if stages != 16 or nbits != 4 or config.get("family", config.get("method")) not in {"rq", "lsq"}:
        raise ValueError("this matched-prefix panel requires an RQ/LSQ 16x4 codec")
    state, state_path, state_loaded, calibration_seconds = _load_or_calibrate(
        data, roles, bundle, factory, seed, calibration_queries, precision,
        force_calibration
    )
    selections = _selections(state, bundle.n_items, seed, target_length, nbits)
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}")
    if "uniform" not in strategies:
        raise ValueError("uniform must be included to define delta_vs_uniform")
    evaluation_data = _evaluation_data(data, roles, split)
    teacher_ids, teacher_scores = _teacher(
        data["gallery"], evaluation_data["queries"], evaluation_data["exclude_ids"]
    )

    output_dir = output_dir or panel_results_dir(dataset)
    physical_dir = data["root"] / (
        "nested_additive_bundles" if precision == "fp32"
        else f"nested_additive_bundles_{precision}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    expected_code_bytes = bundle.n_items * target_length * nbits // 8
    reference_persistent = None
    for strategy in strategies:
        selected = selections[strategy]
        index = NestedAdditiveIndex.from_allocation(
            bundle,
            state["prefix_norms"],
            ALLOWED_LENGTHS,
            selected,
            strategy=strategy,
            metadata={
                "dataset": data["dataset"],
                "manifest_sha256": _sha256(data["manifest_path"]),
                "calibration_state": str(state_path),
                "calibration_state_sha256": _sha256(state_path),
                "evaluation_split": split,
                "target_uniform_length": target_length,
            },
        )
        path = physical_dir / (
            f"{factory.lower()}{precision_tag}_seed{seed}_cal{calibration_queries}_"
            f"cap_u{target_length}_{strategy}.navx"
        )
        index.save(path)
        loaded = NestedAdditiveIndex.load(path)
        if loaded.metadata["codec_sha256"] != state["codec_hash"]:
            raise AssertionError("physical index and calibration codec differ")
        if loaded.code_only_bytes > expected_code_bytes:
            raise AssertionError(
                f"{strategy} uses {loaded.code_only_bytes} code bytes; "
                f"cap is {expected_code_bytes}"
            )
        if reference_persistent is None:
            if strategy != "uniform" or loaded.code_only_bytes != expected_code_bytes:
                raise AssertionError("uniform must establish and exactly use the cap")
            reference_persistent = loaded.persistent_bytes
        elif loaded.persistent_bytes > reference_persistent:
            raise AssertionError("an allocation exceeded the uniform persistent cap")
        reconstruction = loaded.decode_normalized()
        metrics, per_query = _evaluate_reconstruction(
            reconstruction, evaluation_data, teacher_ids, teacher_scores
        )
        # Audit the compact serving representation on deterministic queries.
        direct_count = min(4, len(evaluation_data["queries"]))
        direct = loaded.score_queries(evaluation_data["queries"][:direct_count])
        dense = evaluation_data["queries"][:direct_count] @ reconstruction.T
        rows[strategy] = {
            "strategy": strategy,
            "allocation_histogram": loaded.metadata["allocation_histogram"],
            "mean_assigned_length": float(np.mean(loaded.assigned_lengths_by_id())),
            "code_only_bytes": int(loaded.code_only_bytes),
            "code_only_bytes_per_item": float(loaded.code_only_bytes_per_item),
            "persistent_bytes": int(loaded.persistent_bytes),
            "persistent_bytes_per_item": float(loaded.persistent_bytes_per_item),
            "code_cap_slack_bytes": int(expected_code_bytes - loaded.code_only_bytes),
            "persistent_cap_slack_bytes": int(
                reference_persistent - loaded.persistent_bytes
            ),
            "storage": loaded.storage_breakdown(),
            "bundle": str(path),
            "bundle_sha256": _sha256(path),
            "codec_sha256": loaded.metadata["codec_sha256"],
            "direct_dense_max_abs_error": float(np.max(np.abs(direct - dense))),
            **metrics,
            "per_query": per_query,
        }
        print(
            f"seed {seed} {split} u{target_length} {strategy}: "
            f"R@10={metrics['teacher_recall_at_10']:.4f}, "
            f"NDCG={metrics['teacher_ndcg_at_10']:.4f}, "
            f"B/item={loaded.persistent_bytes_per_item:.5f}",
            flush=True,
        )
        loaded.close()
        del direct, dense, reconstruction, loaded, index

    baseline = rows["uniform"]
    for strategy, row in rows.items():
        row["delta_vs_uniform"] = {
            metric: float(row[metric] - baseline[metric])
            for metric in (
                "teacher_recall_at_1", "teacher_recall_at_10",
                "teacher_ndcg_at_10", "semantic_recall_at_1_on_eligible_queries",
            )
            # The semantic metric is None when no evaluated query carries a
            # gallery-representable label, which is the normal case for an
            # out-of-domain split.  Report no delta rather than crashing.
            if row[metric] is not None and baseline[metric] is not None
        }
    report = {
        "dataset": data["dataset"],
        "seed": int(seed),
        "split": split,
        "frozen_codec": factory,
        "codebook_storage_precision": precision,
        "source_bundle": str(bundle_path),
        "source_bundle_sha256": _sha256(bundle_path),
        "codec_sha256": codec_sha256(bundle),
        "allowed_lengths": list(ALLOWED_LENGTHS),
        "strategies": list(strategies),
        "target_uniform_length": target_length,
        "calibration_queries": int(calibration_queries),
        "calibration_seconds": float(calibration_seconds),
        "calibration_state": str(state_path),
        "calibration_state_loaded": state_loaded,
        "calibration_report": state["calibration_report"],
        "coverage_fraction": float(np.mean(state["coverage"] > 0)),
        "uniform_cap": {
            "code_only_bytes": int(expected_code_bytes),
            "persistent_bytes": int(reference_persistent),
        },
        "representation": (
            f"same frozen {precision} shared codebooks; byte-aligned 4-bit prefix "
            "codes grouped by assigned length; explicit uint32 IDs and float16 "
            "prefix norms; FP32 scoring after reload"
        ),
        "compute_backend": backend_report(),
        "rows": rows,
    }
    _atomic_json(
        output_dir / (
            f"{factory.lower()}{precision_tag}_seed{seed}_{split}_u{target_length}_"
            f"cal{calibration_queries}.json"
        ),
        report,
    )
    close_bundle = getattr(bundle, "close", None)
    if close_bundle is not None:
        close_bundle()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--dataset", default="sop",
        help="Corpus slug under --data-root; 'sop', 'gldv2', or any panel-layout "
             "dataset (see synthetic_panel_dataset.py).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument(
        "--target-lengths", nargs="+", type=int,
        default=list(DEFAULT_TARGET_LENGTHS),
    )
    parser.add_argument("--force-calibration", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Dense teacher/evaluation backend; auto uses CUDA when available.",
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES),
        help="Subset to materialize; uniform is required for matched deltas.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where the per-seed reports go; defaults to the results folder of "
             "the experiment that owns --dataset.",
    )
    args = parser.parse_args()
    resolved = configure_compute_backend(args.device)
    print(f"compute backend: {resolved} ({backend_report()})", flush=True)
    for seed in args.seeds:
        for target_length in args.target_lengths:
            run_seed(
                Path(args.data_root).resolve(),
                seed,
                split=args.split,
                calibration_queries=args.calibration_queries,
                target_length=target_length,
                factory=args.factory,
                precision=args.precision,
                force_calibration=args.force_calibration,
                strategies=tuple(args.strategies),
                dataset=args.dataset,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
