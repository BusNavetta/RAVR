"""Apply the frozen Ordinal-KL RAVR allocator to a QUIP-cov(z) PQ codec."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from benchmark_faiss_fixed_rate import _evaluate_reconstruction, _load_dataset, _sha256
from benchmark_qaq_ravr import _load_state, _paired_bootstrap, _save_state
from compute_backend import backend_report, configure_compute_backend
from qaq_pq import QAQPQIndex
from qaq_ravr import QAQNestedIndex, calibrate_qaq_codec, ordered_subspaces
from ravr_rq import solve_lagrangian


HERE = Path(__file__).resolve().parent

DEFAULT_PROTOCOL = Path(r"D:\Research\Data\AJigma\qaq_official\sop\seed17")
DEFAULT_SOURCE = DEFAULT_PROTOCOL / "bundles" / "quip_covz_m6_k256_seed17_it50_randomperm.quipix"
DEFAULT_ALLOWED = (3, 4, 5, 6)
DEFAULT_CAPS = {"fp32": 1_497_824, "fp16": 1_300_896}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--allowed-lengths", nargs="+", type=int, default=list(DEFAULT_ALLOWED))
    parser.add_argument("--persistent-cap", type=int, default=None)
    parser.add_argument("--calibration-queries", type=int, default=5000)
    parser.add_argument("--compute-backend", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--force-calibration", action="store_true")
    args = parser.parse_args()
    configure_compute_backend(args.compute_backend)

    protocol_dir = args.protocol.resolve()
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    base = _load_dataset("sop", Path(protocol["source_manifest"]).parent.parent)
    if args.source is None:
        base_name = f"quip_covz_m6_k256_seed{protocol['codec_seed']}_it50_randomperm"
        suffix = ".quipix" if args.precision == "fp32" else "_fp16.quipix"
        source_path = (protocol_dir / "bundles" / f"{base_name}{suffix}").resolve()
    else:
        source_path = args.source.resolve()
    persistent_cap = (
        DEFAULT_CAPS[args.precision]
        if args.persistent_cap is None else int(args.persistent_cap)
    )
    source_hash = _sha256(source_path)
    source = QAQPQIndex.load(source_path)
    source_precision = source.metadata.get(
        "codebook_storage_precision",
        "fp16" if source.codebooks.dtype == np.float16 else "fp32",
    )
    if source_precision != args.precision:
        raise ValueError(
            f"source stores {source_precision} codebooks, not requested {args.precision}"
        )
    if source.metadata.get("algorithm") != "QUIP-cov(z)":
        raise ValueError("the QUIP adapter requires a QUIP-cov(z) source codec")
    allowed = tuple(args.allowed_lengths)
    if allowed[-1] != source.stages:
        raise ValueError("the final QUIP option must retain every subspace")
    train_ids = np.load(protocol_dir / "train_ids.npy")
    calibration_ids = np.load(protocol_dir / "calibration_ids.npy")
    calibration = np.load(protocol_dir / "calibration.npy", mmap_mode="r")
    order = ordered_subspaces(source, base["gallery"], train_ids)
    state_path = source_path.with_name(
        f"{source_path.stem}_ravr_cal{args.calibration_queries}.npz"
    )
    started = time.perf_counter()
    if state_path.exists() and not args.force_calibration:
        state = _load_state(state_path)
        if state["source_sha256"] != source_hash:
            raise ValueError("cached RAVR state belongs to a different QUIP codec")
        if state["allowed_lengths"] != allowed or not np.array_equal(
            state["subspace_order"], order
        ):
            raise ValueError("cached RAVR state uses a different prefix definition")
        calibration_seconds = 0.0
        state_loaded = True
    else:
        print(
            f"[{time.strftime('%H:%M:%S')}] calibrating QUIP prefixes on "
            f"{args.calibration_queries} queries",
            flush=True,
        )
        trainer = calibrate_qaq_codec(
            source,
            base["gallery"],
            calibration[:args.calibration_queries],
            calibration_ids[:args.calibration_queries],
            order,
            allowed_lengths=allowed,
        )
        state = {
            "distortion": trainer.distortion,
            "prior_distortion": trainer.prior_distortion,
            "prefix_norms": trainer.prefix_norms,
            "coverage": trainer.coverage,
            "subspace_order": order,
            "allowed_lengths": allowed,
            "source_sha256": source_hash,
            "calibration_report": trainer.calibration_report,
        }
        _save_state(state_path, state)
        calibration_seconds = time.perf_counter() - started
        state_loaded = False

    ordered_codebooks = source.codebooks[order]
    code_offset = QAQNestedIndex.code_offset_for(
        ordered_codebooks, order, len(allowed), source.n_items
    )
    code_budget = persistent_cap - code_offset
    row_costs = np.asarray(
        [math.ceil(length * source.nbits / 8) for length in allowed], dtype=np.int64
    )
    if code_budget < source.n_items * row_costs[0]:
        raise ValueError("persistent cap cannot hold the shortest QUIP prefix")
    variable_costs = row_costs - row_costs[0]
    variable_budget = code_budget - source.n_items * int(row_costs[0])
    ravr = solve_lagrangian(state["distortion"], variable_costs, variable_budget)
    reconstruction = solve_lagrangian(
        state["prior_distortion"], variable_costs, variable_budget
    )
    reconstruction_matched_ravr = solve_lagrangian(
        state["prior_distortion"], variable_costs, int(variable_costs[ravr].sum())
    )
    ravr_unseen_rd = ravr.copy()
    coverage = np.asarray(state["coverage"])
    unseen = np.flatnonzero(coverage == 0)
    if len(unseen):
        seen = np.flatnonzero(coverage > 0)
        unseen_budget = variable_budget - int(variable_costs[ravr[seen]].sum())
        if unseen_budget < 0:
            raise AssertionError("covered QUIP RAVR allocation exceeded the cap")
        ravr_unseen_rd[unseen] = solve_lagrangian(
            state["prior_distortion"][unseen], variable_costs, unseen_budget
        )
    seed = int(source.metadata["config"]["seed"])
    rng = np.random.default_rng(20260825 + seed)
    uniform_option = max(
        option for option, cost in enumerate(row_costs)
        if code_offset + source.n_items * int(cost) <= persistent_cap
    )
    selections = {
        "uniform_under_cap": np.full(source.n_items, uniform_option, dtype=np.int64),
        "ravr": ravr,
        "ravr_unseen_rd": ravr_unseen_rd,
        "random_histogram": rng.permutation(ravr),
        "reconstruction_matched_ravr": reconstruction_matched_ravr,
        "random_fallback_histogram": rng.permutation(ravr_unseen_rd),
        "reconstruction": reconstruction,
    }
    teacher_ids = np.load(protocol_dir / "test_teacher_ids.npy")
    teacher_scores = np.load(protocol_dir / "test_teacher_scores.npy")
    output_dir = HERE / "outputs" / (
        "quip_ravr_sop_100k" if args.precision == "fp32"
        else "quip_ravr_fp16_sop_100k"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy, selected in selections.items():
        index = QAQNestedIndex.from_allocation(
            source,
            order,
            state["prefix_norms"],
            allowed,
            selected,
            strategy=strategy,
            metadata={
                "source_quip_bundle_sha256": source_hash,
                "persistent_cap": persistent_cap,
                "calibration_state_sha256": _sha256(state_path),
            },
        )
        index_path = protocol_dir / "bundles" / (
            f"{source_path.stem}_ravr_cap{persistent_cap}_{strategy}.quiprvix"
        )
        index.save(index_path)
        if index.persistent_bytes > persistent_cap:
            raise AssertionError(f"{strategy} exceeded the physical byte cap")
        loaded = QAQNestedIndex.load(index_path)
        metrics, per_query = _evaluate_reconstruction(
            loaded.decode_normalized(), base, teacher_ids, teacher_scores
        )
        row = {
            "strategy": strategy,
            "bundle": str(index_path),
            "bundle_sha256": _sha256(index_path),
            "persistent_cap": persistent_cap,
            "persistent_bytes": loaded.persistent_bytes,
            "full_serialized_bytes_per_item": loaded.persistent_bytes_per_item,
            "code_only_bytes_per_item": loaded.code_only_bytes_per_item,
            "mean_assigned_length": float(np.mean(np.asarray(allowed)[selected])),
            "allocation_histogram": loaded.metadata["allocation_histogram"],
            "storage": loaded.storage_breakdown(),
            **metrics,
            "per_query": per_query,
        }
        rows.append(row)
        loaded.close()
        print(
            f"{strategy}: {row['full_serialized_bytes_per_item']:.5f} B/item, "
            f"Teacher R@10={row['teacher_recall_at_10']:.5f}",
            flush=True,
        )
    by_strategy = {row["strategy"]: row for row in rows}
    comparisons = [
        _paired_bootstrap(by_strategy["ravr"], by_strategy[control], seed=20260825 + offset)
        for offset, control in enumerate((
            "random_histogram", "reconstruction_matched_ravr", "uniform_under_cap"
        ))
    ]
    report = {
        "method": "RAVR allocation over QUIP-cov(z)-trained PQ codec",
        "source_quip_bundle": str(source_path),
        "source_quip_bundle_sha256": source_hash,
        "codebook_storage_precision": args.precision,
        "protocol_sha256": _sha256(protocol_dir / "protocol.json"),
        "calibration_queries": args.calibration_queries,
        "calibration_seconds": calibration_seconds,
        "calibration_state_loaded": state_loaded,
        "calibration_state": str(state_path),
        "calibration_report": state["calibration_report"],
        "subspace_order": order.tolist(),
        "subspace_order_selection": "mean reconstruction gain on locked codec-training IDs only",
        "allowed_lengths": list(allowed),
        "row_bytes": row_costs.tolist(),
        "persistent_cap": persistent_cap,
        "fixed_code_offset": code_offset,
        "code_payload_budget": code_budget,
        "variable_code_payload_budget": variable_budget,
        "compute_backend": backend_report(),
        "rows": rows,
        "paired_comparisons": comparisons,
    }
    stem = f"seed{seed}_cap{persistent_cap}_cal{args.calibration_queries}"
    result_path = output_dir / f"{stem}.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"result: {result_path}", flush=True)
    source.close()


if __name__ == "__main__":
    main()
