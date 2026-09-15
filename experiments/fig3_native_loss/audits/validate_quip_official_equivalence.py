from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from benchmark_faiss_fixed_rate import _sha256
from quip_pq import QUIPConfig, train_quip_covariance


# Results live in the experiment folder, one level up from audits/.
HERE = Path(__file__).resolve().parents[1]

DEFAULT_RELEASED = Path(
    r"D:\Research\Data\AJigma\external\Query-Aware-Quantization-sparse"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--released-baseline", type=Path, default=DEFAULT_RELEASED)
    args = parser.parse_args()
    released = args.released_baseline.resolve()
    sys.path.insert(0, str(released / "src" / "packages"))
    module = importlib.import_module("QUIP.cov")

    rng = np.random.default_rng(20260825)
    data = rng.normal(size=(96, 24)).astype(np.float32)
    queries = rng.normal(size=(41, 24)).astype(np.float32)
    m, k = 4, 8
    initial_rows = rng.choice(len(data), k, replace=False)
    initial = np.transpose(
        data[initial_rows].reshape(k, m, data.shape[1] // m), (1, 0, 2)
    ).copy()

    official_covariance = module.get_CovMA(queries, m)
    official = module.QUIP_cov(m, k, data.shape[1])
    official.pq_codebook = initial.astype(np.float64, copy=True)
    official_codes = official.encode(data, official_covariance)
    official_initial_loss = float(
        official.object_loss(data, official_covariance, official_codes)
    )
    official_losses = []
    for _ in range(4):
        official.codebook_updata(data, official_codes)
        official_codes = official.encode(data, official_covariance)
        official_losses.append(float(
            official.object_loss(data, official_covariance, official_codes)
        ))

    config = QUIPConfig(
        m=m, k=k, seed=17, max_iterations=4, relative_tolerance=0.0,
        encode_batch_size=32,
    )
    codebooks, cuda_codes, cuda_covariance, permutation, report = train_quip_covariance(
        data,
        queries,
        config,
        permutation=np.arange(data.shape[1]),
        initial_codebooks=initial,
    )
    result = {
        "passed": bool(
            np.array_equal(cuda_codes, official_codes)
            and np.max(np.abs(codebooks - official.pq_codebook)) < 2e-5
            and np.max(np.abs(cuda_covariance - official_covariance)) < 2e-6
        ),
        "codes_identical": bool(np.array_equal(cuda_codes, official_codes)),
        "covariance_max_abs_error": float(
            np.max(np.abs(cuda_covariance - official_covariance))
        ),
        "codebook_max_abs_error": float(
            np.max(np.abs(codebooks - official.pq_codebook))
        ),
        "official_initial_loss": official_initial_loss,
        "official_losses": official_losses,
        "cuda_losses": [row["loss"] for row in report["iterations"]],
        "final_loss_relative_error": float(
            abs(report["final_loss"] - official_losses[-1])
            / max(abs(official_losses[-1]), 1e-12)
        ),
        "identity_permutation": permutation.astype(int).tolist(),
        "released_baseline": str(released),
        "released_model_sha256": _sha256(
            released / "src" / "packages" / "QUIP" / "cov.py"
        ),
        "explicit_input_sha256": hashlib.sha256(
            data.tobytes() + queries.tobytes() + initial.tobytes()
        ).hexdigest(),
        "comparison": (
            "released NumPy covariance-weighted Lloyd updates versus algebraically "
            "batched CUDA implementation on identical explicit inputs"
        ),
    }
    output = HERE / "outputs" / "quip_official_sop_100k"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "official_cuda_equivalence.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("released/CUDA QUIP equivalence check failed")


if __name__ == "__main__":
    main()
