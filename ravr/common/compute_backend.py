"""Deterministic CPU/CUDA inner-product top-k backend for AJigma audits."""
from __future__ import annotations

import os
import time

import numpy as np


VALID_BACKENDS = ("auto", "cpu", "cuda")
_REQUESTED_BACKEND = os.environ.get("AJIGMA_COMPUTE_BACKEND", "auto").lower()


def configure_compute_backend(backend: str) -> str:
    """Select ``auto``, ``cpu``, or ``cuda`` and return the resolved backend."""
    global _REQUESTED_BACKEND
    backend = str(backend).lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}")
    _REQUESTED_BACKEND = backend
    return resolved_compute_backend()


def _cuda_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def resolved_compute_backend() -> str:
    if _REQUESTED_BACKEND == "cpu":
        return "cpu"
    torch = _cuda_torch()
    if torch is not None:
        return "cuda"
    if _REQUESTED_BACKEND == "cuda":
        raise RuntimeError(
            "CUDA was requested but this Python environment has no CUDA-enabled PyTorch"
        )
    return "cpu"


def backend_report() -> dict:
    resolved = resolved_compute_backend()
    report = {"requested": _REQUESTED_BACKEND, "resolved": resolved}
    if resolved == "cuda":
        torch = _cuda_torch()
        assert torch is not None
        report.update({
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "device": str(torch.cuda.get_device_name(0)),
            "capability": list(torch.cuda.get_device_capability(0)),
            "tf32": False,
        })
    return report


def _stable_rows(ids: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort each top-k row by decreasing score, then increasing item ID."""
    for row in range(len(ids)):
        order = np.lexsort((ids[row], -scores[row]))
        ids[row] = ids[row, order]
        scores[row] = scores[row, order]
    return ids, scores


def _cpu_topk(
    queries: np.ndarray,
    gallery: np.ndarray,
    k: int,
    exclude_ids: np.ndarray | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_queries = len(queries)
    ids = np.empty((n_queries, k), dtype=np.int64)
    values = np.empty((n_queries, k), dtype=np.float32)
    for start in range(0, n_queries, batch_size):
        stop = min(start + batch_size, n_queries)
        scores = np.asarray(queries[start:stop] @ gallery.T, dtype=np.float32)
        if exclude_ids is not None:
            scores[np.arange(stop - start), exclude_ids[start:stop]] = -np.inf
        for local, row in enumerate(scores):
            candidates = np.argpartition(row, len(row) - k)[-k:]
            order = np.lexsort((candidates, -row[candidates]))
            keep = candidates[order]
            ids[start + local] = keep
            values[start + local] = row[keep]
    return ids, values


def _cuda_topk(
    queries: np.ndarray,
    gallery: np.ndarray,
    k: int,
    exclude_ids: np.ndarray | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch = _cuda_torch()
    if torch is None:
        raise RuntimeError("CUDA backend is unavailable")
    # Ranking audits need FP32 behavior, not the faster TF32 approximation.
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    # ``gallery`` is often a read-only memmap.  Make an owned host copy before
    # handing it to PyTorch so tensor construction never aliases read-only data.
    right_host = np.array(gallery, dtype=np.float32, copy=True, order="C")
    right = torch.from_numpy(right_host).to(device=device)
    ids, values = [], []
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            stop = min(start + batch_size, len(queries))
            left_host = np.array(
                queries[start:stop], dtype=np.float32, copy=True, order="C"
            )
            left = torch.from_numpy(left_host).to(device=device)
            scores = left @ right.T
            if exclude_ids is not None:
                rows = torch.arange(stop - start, device=device)
                columns = torch.as_tensor(
                    np.ascontiguousarray(exclude_ids[start:stop], dtype=np.int64),
                    device=device,
                )
                scores[rows, columns] = -torch.inf
            value, index = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            ids.append(index.cpu().numpy().astype(np.int64, copy=False))
            values.append(value.cpu().numpy().astype(np.float32, copy=False))
    return _stable_rows(np.concatenate(ids), np.concatenate(values))


def topk_inner_products(
    queries: np.ndarray,
    gallery: np.ndarray,
    k: int,
    exclude_ids: np.ndarray | None = None,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return exact-scan top-k IDs/scores and backend timing metadata."""
    queries = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    gallery = np.ascontiguousarray(np.asarray(gallery, dtype=np.float32))
    k = min(int(k), len(gallery))
    if queries.ndim != 2 or gallery.ndim != 2 or queries.shape[1] != gallery.shape[1]:
        raise ValueError("queries and gallery must be compatible matrices")
    if exclude_ids is not None:
        exclude_ids = np.ascontiguousarray(np.asarray(exclude_ids, dtype=np.int64))
        if exclude_ids.shape != (len(queries),):
            raise ValueError("exclude_ids has the wrong shape")
    backend = resolved_compute_backend()
    started = time.perf_counter()
    if backend == "cuda":
        ids, scores = _cuda_topk(queries, gallery, k, exclude_ids, batch_size)
    else:
        ids, scores = _cpu_topk(queries, gallery, k, exclude_ids, batch_size)
    report = backend_report() | {
        "queries": int(len(queries)),
        "gallery_items": int(len(gallery)),
        "dimension": int(gallery.shape[1]),
        "batch_size": int(batch_size),
        "seconds": float(time.perf_counter() - started),
    }
    return ids, scores, report
