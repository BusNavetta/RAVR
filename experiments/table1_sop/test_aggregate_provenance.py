from __future__ import annotations

from copy import deepcopy

import pytest

from aggregate_nested_additive_ravr import METRICS, _validate_reports
from benchmark_nested_additive_ravr import STRATEGIES


def _report(seed: int, target: int, backend: str = "cuda") -> dict:
    rows = {
        strategy: {
            "codec_sha256": "frozen-codec",
            "compute_backend": {"resolved": backend},
            "per_query": {metric: [0.0, 1.0] for metric in METRICS},
        }
        for strategy in STRATEGIES
    }
    return {
        "seed": seed,
        "split": "test",
        "frozen_codec": "LSQ16x4",
        "codebook_storage_precision": "fp32",
        "target_uniform_length": target,
        "calibration_queries": 5000,
        "compute_backend": {"resolved": backend},
        "codec_sha256": "frozen-codec",
        "strategies": list(STRATEGIES),
        "rows": rows,
    }


def test_provenance_validation_accepts_one_consistent_backend() -> None:
    reports = {(17, 10): _report(17, 10)}
    assert _validate_reports(
        reports, "LSQ16x4", "fp32", (17,), (10,), "test", 5000
    ) == "cuda"


def test_provenance_validation_rejects_mixed_backends() -> None:
    reports = {
        (17, 10): _report(17, 10, "cuda"),
        (42, 10): _report(42, 10, "cpu"),
    }
    with pytest.raises(ValueError, match="mixed compute backends"):
        _validate_reports(
            reports, "LSQ16x4", "fp32", (17, 42), (10,), "test", 5000
        )


def test_provenance_validation_rejects_stale_precision() -> None:
    report = deepcopy(_report(17, 10))
    report["codebook_storage_precision"] = "fp16"
    with pytest.raises(ValueError, match="codebook_storage_precision"):
        _validate_reports(
            {(17, 10): report},
            "LSQ16x4",
            "fp32",
            (17,),
            (10,),
            "test",
            5000,
        )


def test_provenance_validation_rejects_row_backend_mismatch() -> None:
    report = _report(17, 10)
    report["rows"]["uniform"]["compute_backend"]["resolved"] = "cpu"
    with pytest.raises(ValueError, match="backend mismatch in uniform"):
        _validate_reports(
            {(17, 10): report},
            "LSQ16x4",
            "fp32",
            (17,),
            (10,),
            "test",
            5000,
        )
