"""CUDA implementation of the complete QUIP-cov(z) product-quantizer codec.

"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np


EPS = 1e-12


def _torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("QUIP acceleration requires a CUDA-enabled PyTorch")
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


@dataclass(frozen=True)
class QUIPConfig:
    m: int = 6
    k: int = 256
    seed: int = 17
    init_seed: int = 15
    max_iterations: int = 50
    relative_tolerance: float = 0.01
    encode_batch_size: int = 4096
    random_permutation: bool = True

    def validate(self, dimension: int, train_items: int, calibration_queries: int) -> None:
        if self.m <= 0 or dimension % self.m:
            raise ValueError("M must be a positive divisor of the embedding dimension")
        if self.k <= 1 or self.k > train_items or self.k & (self.k - 1):
            raise ValueError("K must be a power of two no larger than the training set")
        if calibration_queries <= 0:
            raise ValueError("QUIP-cov(z) requires held-out calibration queries")
        if self.max_iterations <= 0 or self.encode_batch_size <= 0:
            raise ValueError("iteration and batch counts must be positive")
        if self.relative_tolerance < 0:
            raise ValueError("relative tolerance must be non-negative")


def compute_covariance(queries: np.ndarray, m: int) -> np.ndarray:
    """Compute the paper's non-centred per-subspace query covariance."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] % m:
        raise ValueError("queries are incompatible with M")
    blocks = q.reshape(len(q), m, q.shape[1] // m)
    covariance = np.einsum("nmd,nme->mde", blocks, blocks, optimize=True)
    covariance /= float(len(q))
    return np.ascontiguousarray(covariance, dtype=np.float32)


def _initial_codebooks(data: np.ndarray, config: QUIPConfig) -> np.ndarray:
    """Mirror the released QUIP baseline's shared random-point initialization."""
    rng = np.random.RandomState(config.init_seed)
    rows = rng.choice(len(data), config.k, replace=False)
    blocks = np.asarray(data[rows], dtype=np.float32).reshape(
        config.k, config.m, data.shape[1] // config.m
    )
    return np.ascontiguousarray(np.transpose(blocks, (1, 0, 2)))


def encode_mahalanobis_cuda(
    data: np.ndarray,
    covariance: np.ndarray,
    codebooks: np.ndarray,
    *,
    batch_size: int = 4096,
    device: str = "cuda:0",
) -> tuple[np.ndarray, float, dict]:
    """Equation-equivalent batched Mahalanobis assignment on CUDA."""
    torch = _torch_cuda()
    started = time.perf_counter()
    x = np.ascontiguousarray(data, dtype=np.float32)
    cb = torch.as_tensor(np.ascontiguousarray(codebooks, dtype=np.float32), device=device)
    cov = torch.as_tensor(np.ascontiguousarray(covariance, dtype=np.float32), device=device)
    n, d = x.shape
    m, k, ds = cb.shape
    if d != m * ds or cov.shape != (m, ds, ds):
        raise ValueError("data, covariance, and codebooks are incompatible")
    quadratic = torch.einsum("mkd,mde,mke->mk", cb, cov, cb)
    codes = np.empty((n, m), dtype=np.uint8 if k <= 256 else np.uint16)
    loss = 0.0
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            block = torch.as_tensor(x[start:stop], device=device).reshape(-1, m, ds)
            projected = torch.einsum("bmd,mde->bme", block, cov)
            distance = -2.0 * torch.einsum("bme,mke->bmk", projected, cb)
            distance += quadratic[None, :, :]
            selected = torch.argmin(distance, dim=2)
            codes[start:stop] = selected.cpu().numpy().astype(codes.dtype, copy=False)
            reconstruction = torch.gather(
                cb[None, :, :, :].expand(len(block), -1, -1, -1),
                2,
                selected[:, :, None, None].expand(-1, -1, 1, ds),
            )[:, :, 0, :]
            residual = block - reconstruction
            loss += float(torch.einsum("bmd,mde,bme->", residual, cov, residual).item())
        torch.cuda.synchronize()
    report = {
        "implementation": "CUDA algebraic Mahalanobis assignment",
        "points": int(n),
        "batch_size": int(batch_size),
        "seconds": time.perf_counter() - started,
        "loss": float(loss),
    }
    return codes, float(loss), report


def update_centroids_cuda(
    data: np.ndarray,
    codes: np.ndarray,
    previous: np.ndarray,
    *,
    device: str = "cuda:0",
) -> tuple[np.ndarray, dict]:
    """Apply QUIP Eq. (3): the Euclidean mean of every assigned partition."""
    torch = _torch_cuda()
    started = time.perf_counter()
    old = np.ascontiguousarray(previous, dtype=np.float32)
    m, k, ds = old.shape
    x = torch.as_tensor(np.ascontiguousarray(data, dtype=np.float32), device=device)
    x = x.reshape(len(x), m, ds)
    assignment = torch.as_tensor(np.asarray(codes, dtype=np.int64), device=device)
    updated = torch.as_tensor(old.copy(), device=device)
    empty = []
    with torch.inference_mode():
        for stage in range(m):
            sums = torch.zeros((k, ds), dtype=torch.float32, device=device)
            counts = torch.zeros(k, dtype=torch.float32, device=device)
            sums.index_add_(0, assignment[:, stage], x[:, stage])
            counts.index_add_(
                0, assignment[:, stage], torch.ones(len(x), dtype=torch.float32, device=device)
            )
            present = counts > 0
            updated[stage, present] = sums[present] / counts[present, None]
            empty.append(int(torch.sum(~present).item()))
        torch.cuda.synchronize()
    result = updated.cpu().numpy().astype(np.float32, copy=False)
    return result, {
        "implementation": "CUDA partition sums implementing Euclidean centroid means",
        "empty_clusters_per_subspace": empty,
        "empty_cluster_policy": "retain previous centroid",
        "seconds": time.perf_counter() - started,
    }


def train_quip_covariance(
    data: np.ndarray,
    calibration_queries: np.ndarray,
    config: QUIPConfig,
    *,
    permutation: np.ndarray | None = None,
    initial_codebooks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Train a QUIP-cov(z) codec on explicit training and held-out query roles."""
    x = np.ascontiguousarray(data, dtype=np.float32)
    q = np.ascontiguousarray(calibration_queries, dtype=np.float32)
    config.validate(x.shape[1], len(x), len(q))
    if permutation is None:
        permutation = (
            np.random.default_rng(config.seed).permutation(x.shape[1])
            if config.random_permutation else np.arange(x.shape[1])
        )
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (x.shape[1],) or not np.array_equal(
        np.sort(permutation), np.arange(x.shape[1])
    ):
        raise ValueError("permutation must contain every input dimension exactly once")
    x_codec = np.ascontiguousarray(x[:, permutation])
    q_codec = np.ascontiguousarray(q[:, permutation])
    covariance = compute_covariance(q_codec, config.m)
    codebooks = (
        _initial_codebooks(x_codec, config)
        if initial_codebooks is None
        else np.ascontiguousarray(initial_codebooks, dtype=np.float32)
    )
    started = time.perf_counter()
    codes, loss, initial_encoding = encode_mahalanobis_cuda(
        x_codec, covariance, codebooks, batch_size=config.encode_batch_size
    )
    history = []
    for iteration in range(1, config.max_iterations + 1):
        codebooks, update = update_centroids_cuda(x_codec, codes, codebooks)
        new_codes, new_loss, encoding = encode_mahalanobis_cuda(
            x_codec, covariance, codebooks, batch_size=config.encode_batch_size
        )
        relative_change = (loss - new_loss) / max(abs(new_loss), EPS)
        history.append({
            "iteration": iteration,
            "loss": float(new_loss),
            "relative_loss_reduction": float(relative_change),
            "encoding": encoding,
            "update": update,
        })
        codes, loss = new_codes, new_loss
        if abs(relative_change) <= config.relative_tolerance:
            break
    report = {
        "algorithm": "QUIP-cov(z)",
        "config": asdict(config),
        "initial_encoding": initial_encoding,
        "iterations": history,
        "converged": len(history) < config.max_iterations,
        "final_loss": float(loss),
        "total_seconds": time.perf_counter() - started,
        "reference_equations": {
            "covariance": "Sigma_z = mean_z z z^T independently by subspace",
            "assignment": "argmin_c (x-u_c)^T Sigma_z (x-u_c)",
            "centroid": "Euclidean mean of the assigned database partition",
        },
    }
    return codebooks, codes, covariance, permutation.astype(np.uint16), report


def train_quip(
    gallery: np.ndarray,
    train: np.ndarray,
    calibration_queries: np.ndarray,
    config: QUIPConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Train on the codec role and encode the complete gallery."""
    codebooks, _, covariance, permutation, report = train_quip_covariance(
        train, calibration_queries, config
    )
    gallery_codec = np.ascontiguousarray(
        np.asarray(gallery, dtype=np.float32)[:, permutation.astype(np.int64)]
    )
    codes, loss, encoding = encode_mahalanobis_cuda(
        gallery_codec,
        covariance,
        codebooks,
        batch_size=config.encode_batch_size,
    )
    report["gallery_encoding"] = encoding
    report["gallery_loss"] = float(loss)
    return codebooks, codes, covariance, permutation, report


__all__ = [
    "QUIPConfig", "compute_covariance", "encode_mahalanobis_cuda", "train_quip",
    "train_quip_covariance", "update_centroids_cuda",
]
