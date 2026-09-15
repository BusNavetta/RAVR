from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from qaq_pq import encode_cuda, update_codebook_cuda


DEFAULT_JULIA = Path(r"D:\Research\Tools\julia-1.10.12\bin\julia.exe")
DEFAULT_DEPOT = Path(r"D:\Research\Tools\julia-depot-qaq")
DEFAULT_OFFICIAL = Path(
    r"D:\Research\Data\AJigma\external\Query-Aware-Quantization-sparse"
)
DEFAULT_IO = Path(r"D:\Research\Data\AJigma\qaq_official\official_smoke")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--julia", type=Path, default=DEFAULT_JULIA)
    parser.add_argument("--depot", type=Path, default=DEFAULT_DEPOT)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--io-directory", type=Path, default=DEFAULT_IO)
    args = parser.parse_args()
    io = args.io_directory.resolve()
    io.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260825)
    n, d, m, k, kv, iterations = 17, 8, 2, 4, 3, 3
    points = rng.normal(size=(n, d)).astype(np.float64)
    codebook = rng.normal(size=(m, k, d // m)).astype(np.float64)
    roots = rng.normal(size=(kv, d, d)).astype(np.float64)
    matrices = np.einsum("cdi,cei->cde", roots, roots)
    labels_zero = rng.integers(0, kv, size=n, dtype=np.int64)
    np.savetxt(io / "meta.csv", np.array([[n, d, m, k, kv, iterations]]), delimiter=",", fmt="%d")
    np.savetxt(io / "points.csv", points, delimiter=",", fmt="%.17g")
    np.savetxt(io / "labels.csv", labels_zero[:, None] + 1, delimiter=",", fmt="%d")
    np.savetxt(io / "codebook.csv", codebook.reshape(m * k, d // m), delimiter=",", fmt="%.17g")
    np.savetxt(io / "matrices.csv", matrices.reshape(kv * d, d), delimiter=",", fmt="%.17g")

    project = Path(__file__).resolve().parent / "qaq_julia_env"
    command = [
        str(args.julia.resolve()),
        f"--project={project}",
        str(Path(__file__).resolve().parent / "qaq_official_smoke.jl"),
        str(args.official_repo.resolve()),
        str(io),
    ]
    environment = dict(__import__("os").environ)
    environment["JULIA_DEPOT_PATH"] = str(args.depot.resolve())
    completed = subprocess.run(command, check=True, text=True, capture_output=True, env=environment)

    official_codes = np.loadtxt(io / "official_codes.csv", delimiter=",").astype(np.int64) - 1
    official_loss = float(np.loadtxt(io / "official_loss.csv", delimiter=","))
    official_updated = np.loadtxt(
        io / "official_updated_codebook.csv", delimiter=","
    ).reshape(m, k, d // m)
    cuda_codes, cuda_loss, encode_report = encode_cuda(
        points.astype(np.float32), codebook.astype(np.float32), matrices.astype(np.float32),
        labels_zero.astype(np.int32), iterations=iterations,
    )
    cuda_updated, update_report = update_codebook_cuda(
        points.astype(np.float32), cuda_codes, matrices.astype(np.float32),
        labels_zero.astype(np.int32), k, 1e-3,
    )
    report = {
        "official_stdout": completed.stdout,
        "codes_identical": bool(np.array_equal(cuda_codes, official_codes)),
        "official_loss": official_loss,
        "cuda_loss": cuda_loss,
        "loss_relative_error": abs(cuda_loss - official_loss) / max(abs(official_loss), 1e-12),
        "codebook_max_abs_error": float(np.max(np.abs(cuda_updated - official_updated))),
        "codebook_mean_abs_error": float(np.mean(np.abs(cuda_updated - official_updated))),
        "encode_report": encode_report,
        "update_report": update_report,
        "julia": str(args.julia.resolve()),
        "official_repo": str(args.official_repo.resolve()),
        "official_patch": "remove unused `using CUDA` and `CA = CuArray` only",
    }
    report["passed"] = bool(
        report["codes_identical"]
        and report["loss_relative_error"] < 5e-5
        and report["codebook_max_abs_error"] < 5e-3
    )
    (io / "equivalence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    tracked_output = (
        Path(__file__).resolve().parents[1] / "outputs" / "qaq_official_sop_100k"
        / "official_cuda_equivalence.json"
    )
    tracked_output.parent.mkdir(parents=True, exist_ok=True)
    tracked_output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("official/CUDA QAQ equivalence check failed")


if __name__ == "__main__":
    main()
