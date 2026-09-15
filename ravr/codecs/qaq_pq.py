"""CUDA-accelerated, equation-equivalent implementation of official QAQ PQ.

"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import struct
import time
from typing import Iterable

import faiss
import numpy as np


MAGIC = b"QAQPQIX1"
FORMAT_VERSION = 1
HEADER_BYTES = 4096
ALIGN_BYTES = 64
EPS = 1e-9


def _align(value: int) -> int:
    return (int(value) + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES


def _torch_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("QAQ acceleration requires a CUDA-enabled PyTorch")
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch


def _power_of_two_bits(k: int) -> int:
    bits = int(round(math.log2(k)))
    if k != 1 << bits:
        raise ValueError("QAQ physical packing requires a power-of-two K")
    return bits


def dimension_permutation(metadata: dict, dimension: int) -> np.ndarray | None:
    """Return a validated stored dimension permutation, when one is present."""
    value = metadata.get("dimension_permutation")
    if value is None:
        return None
    permutation = np.asarray(value, dtype=np.int64)
    if permutation.shape != (dimension,) or not np.array_equal(
        np.sort(permutation), np.arange(dimension)
    ):
        raise ValueError("stored dimension_permutation is invalid")
    return permutation


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack each PQ row using little-endian symbols and per-row byte padding."""
    codes = np.asarray(codes)
    if codes.ndim != 2 or np.any(codes < 0) or np.any(codes >= (1 << bits)):
        raise ValueError("codes are incompatible with the symbol width")
    shifts = np.arange(bits, dtype=np.uint16)
    unpacked = ((codes.astype(np.uint16)[..., None] >> shifts) & 1).astype(np.uint8)
    unpacked = unpacked.reshape(len(codes), codes.shape[1] * bits)
    return np.ascontiguousarray(np.packbits(unpacked, axis=1, bitorder="little"))


def unpack_codes(packed: np.ndarray, stages: int, bits: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError("packed codes must be a matrix")
    raw = np.unpackbits(
        packed, axis=1, count=int(stages) * int(bits), bitorder="little"
    ).reshape(len(packed), stages, bits)
    weights = (1 << np.arange(bits, dtype=np.uint16))[None, None, :]
    dtype = np.uint8 if bits <= 8 else np.uint16
    return np.asarray(np.sum(raw * weights, axis=2), dtype=dtype)


@dataclass(frozen=True)
class QAQConfig:
    m: int
    k: int
    kv: int = 16
    seed: int = 17
    kmeans_iterations: int = 20
    train_iterations: int = 2
    encode_iterations: int = 3
    query_batch_size: int = 500
    sample_count: int = 1
    regularization: float = 1e-3

    def validate(self, dimension: int, train_items: int, calibration_queries: int) -> None:
        if self.m <= 0 or dimension % self.m:
            raise ValueError("M must be a positive divisor of the embedding dimension")
        _power_of_two_bits(self.k)
        if self.k * self.m > train_items:
            raise ValueError("official point initialization needs at least M*K training rows")
        if not (1 <= self.query_batch_size <= calibration_queries):
            raise ValueError("query_batch_size is incompatible with calibration")
        if min(self.kv, self.train_iterations, self.encode_iterations, self.sample_count) <= 0:
            raise ValueError("QAQ iteration and cluster counts must be positive")


def compute_query_matrices(
    samples: np.ndarray, centroids: np.ndarray, *, device: str = "cuda:0"
) -> np.ndarray:
    """Official QAQ Eq. (9) softmax query matrices, one per item cluster."""
    torch = _torch_cuda()
    q = torch.as_tensor(np.array(samples, dtype=np.float32, copy=True, order="C"), device=device)
    c = torch.as_tensor(np.array(centroids, dtype=np.float32, copy=True, order="C"), device=device)
    weights = torch.softmax(q @ c.T, dim=0)
    matrices = []
    with torch.inference_mode():
        for cluster in range(len(c)):
            matrices.append((q.T * weights[:, cluster]) @ q)
    return torch.stack(matrices).cpu().numpy().astype(np.float32, copy=False)


def _initial_codebook(train: np.ndarray, config: QAQConfig) -> np.ndarray:
    """Mirror the official point initialization with an explicit fixed seed."""
    rng = np.random.default_rng(1234)
    order = rng.permutation(len(train))
    ds = train.shape[1] // config.m
    codebook = np.empty((config.m, config.k, ds), dtype=np.float32)
    for stage in range(config.m):
        rows = order[stage * config.k:(stage + 1) * config.k]
        codebook[stage] = train[rows, stage * ds:(stage + 1) * ds]
    return codebook


def cluster_gallery(
    gallery: np.ndarray, config: QAQConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Euclidean k-means matching the clustering role in official QAQ."""
    x = np.ascontiguousarray(gallery, dtype=np.float32)
    started = time.perf_counter()
    kmeans = faiss.Kmeans(
        x.shape[1], config.kv, niter=config.kmeans_iterations,
        nredo=1, seed=config.seed, verbose=False, spherical=False,
    )
    kmeans.train(x)
    _, labels = kmeans.index.search(x, 1)
    report = {
        "implementation": "faiss.Kmeans CPU",
        "clusters": config.kv,
        "iterations": config.kmeans_iterations,
        "seconds": time.perf_counter() - started,
        "objective": float(kmeans.obj[-1]),
    }
    return (
        np.ascontiguousarray(kmeans.centroids, dtype=np.float32),
        np.ascontiguousarray(labels[:, 0], dtype=np.int32),
        report,
    )


def _decode_raw_torch(torch, codes, codebook):
    n, m = codes.shape
    ds = codebook.shape[2]
    output = torch.empty((n, m, ds), dtype=codebook.dtype, device=codebook.device)
    for stage in range(m):
        output[:, stage] = codebook[stage, codes[:, stage]]
    return output.reshape(n, m * ds)


def encode_cuda(
    points: np.ndarray,
    codebook: np.ndarray,
    matrices: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    iterations: int = 3,
    device: str = "cuda:0",
) -> tuple[np.ndarray, float, dict]:
    """Exact coordinate descent from official ``single_encode``, batched on CUDA."""
    torch = _torch_cuda()
    started = time.perf_counter()
    x = torch.as_tensor(np.array(points, dtype=np.float32, copy=True, order="C"), device=device)
    cb = torch.as_tensor(np.array(codebook, dtype=np.float32, copy=True, order="C"), device=device)
    ma = torch.as_tensor(np.array(matrices, dtype=np.float32, copy=True, order="C"), device=device)
    labels = torch.as_tensor(np.asarray(cluster_ids, dtype=np.int64), device=device)
    n, d = x.shape
    m, k, ds = cb.shape
    if d != m * ds or len(labels) != n:
        raise ValueError("points, codebook, and cluster IDs are incompatible")
    codes = torch.zeros((n, m), dtype=torch.long, device=device)
    reconstruction = cb[:, 0, :].reshape(1, d).expand(n, d).clone()
    cluster_rows = [torch.nonzero(labels == c, as_tuple=False)[:, 0] for c in range(len(ma))]
    with torch.inference_mode():
        for _ in range(iterations):
            for stage in range(m):
                start, stop = stage * ds, (stage + 1) * ds
                local_cb = cb[stage]
                for cluster, rows in enumerate(cluster_rows):
                    if not len(rows):
                        continue
                    current = reconstruction[rows]
                    z = x[rows] - current
                    z[:, start:stop] += current[:, start:stop]
                    projected = z @ ma[cluster, :, start:stop]
                    quad = torch.einsum(
                        "ku,uv,kv->k", local_cb, ma[cluster, start:stop, start:stop], local_cb
                    )
                    candidate_loss = -2.0 * (projected @ local_cb.T) + quad[None, :]
                    selected = torch.argmin(candidate_loss, dim=1)
                    codes[rows, stage] = selected
                    reconstruction[rows, start:stop] = local_cb[selected]
        residual = x - reconstruction
        loss = torch.zeros((), dtype=torch.float32, device=device)
        for cluster, rows in enumerate(cluster_rows):
            if len(rows):
                local = residual[rows]
                loss += torch.einsum("nd,df,nf->", local, ma[cluster], local)
        torch.cuda.synchronize()
    result = codes.cpu().numpy().astype(np.uint8 if k <= 256 else np.uint16, copy=False)
    report = {
        "implementation": "CUDA algebraic expansion of official single_encode",
        "points": int(n),
        "iterations": int(iterations),
        "seconds": time.perf_counter() - started,
        "loss": float(loss.item()),
    }
    return result, float(loss.item()), report


def update_codebook_cuda(
    points: np.ndarray,
    codes: np.ndarray,
    matrices: np.ndarray,
    cluster_ids: np.ndarray,
    codebook_size: int,
    regularization: float,
    *,
    device: str = "cuda:0",
) -> tuple[np.ndarray, dict]:
    """Solve the same ``(T + 1e-3 I) C = R`` system as official QAQ."""
    torch = _torch_cuda()
    started = time.perf_counter()
    x = torch.as_tensor(np.array(points, dtype=np.float32, copy=True, order="C"), device=device)
    code = torch.as_tensor(np.asarray(codes, dtype=np.int64), device=device)
    ma = torch.as_tensor(np.array(matrices, dtype=np.float32, copy=True, order="C"), device=device)
    labels = torch.as_tensor(np.asarray(cluster_ids, dtype=np.int64), device=device)
    n, d = x.shape
    m = code.shape[1]
    ds = d // m
    k = int(codebook_size)
    if torch.any((code < 0) | (code >= k)):
        raise ValueError("encoded symbol lies outside the configured codebook")
    onehot_by_cluster = []
    construct = torch.empty_like(x)
    for cluster in range(len(ma)):
        rows = torch.nonzero(labels == cluster, as_tuple=False)[:, 0]
        if len(rows):
            construct[rows] = x[rows] @ ma[cluster]
            onehot = torch.nn.functional.one_hot(code[rows], num_classes=k).float()
            onehot_by_cluster.append(
                torch.einsum("nia,njb->iajb", onehot, onehot)[None, ...]
            )
        else:
            onehot_by_cluster.append(torch.zeros((1, m, k, m, k), device=device))
    counts = torch.cat(onehot_by_cluster, dim=0)
    ma_blocks = ma.reshape(len(ma), m, ds, m, ds)
    normal = torch.einsum("ciajb,ciujv->iaujbv", counts, ma_blocks)
    normal = normal.reshape(k * d, k * d)
    rhs = torch.zeros((m, k, ds), dtype=torch.float32, device=device)
    construct_blocks = construct.reshape(n, m, ds)
    for cluster in range(len(ma)):
        rows = torch.nonzero(labels == cluster, as_tuple=False)[:, 0]
        if len(rows):
            onehot = torch.nn.functional.one_hot(code[rows], num_classes=k).float()
            rhs += torch.einsum("nmk,nmu->mku", onehot, construct_blocks[rows])
    flat_rhs = rhs.reshape(k * d)
    normal.diagonal().add_(float(regularization))
    solution = torch.linalg.solve(normal, flat_rhs)
    torch.cuda.synchronize()
    codebook = solution.reshape(m, k, ds).cpu().numpy().astype(np.float32, copy=False)
    report = {
        "implementation": "cluster/code co-occurrence form of official normal equations",
        "system_dimension": int(k * d),
        "seconds": time.perf_counter() - started,
        "regularization": float(regularization),
    }
    return codebook, report


def train_qaq(
    gallery: np.ndarray,
    train: np.ndarray,
    train_ids: np.ndarray,
    calibration: np.ndarray,
    config: QAQConfig,
    validation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Train QAQ and encode the full gallery using the locked protocol roles."""
    config.validate(gallery.shape[1], len(train), len(calibration))
    started = time.perf_counter()
    centroids, gallery_clusters, clustering = cluster_gallery(gallery, config)
    train_clusters = gallery_clusters[np.asarray(train_ids, dtype=np.int64)]
    codebook = _initial_codebook(train, config)
    rng = np.random.default_rng(config.seed)
    cycles = []
    best_codebook = None
    best_matrices = None
    best_recall = -math.inf
    best_sample_id = None
    validation_teacher = None
    if validation is not None:
        torch = _torch_cuda()
        with torch.inference_mode():
            q = torch.as_tensor(
                np.array(validation, dtype=np.float32, copy=True, order="C"),
                device="cuda:0",
            )
            x = torch.as_tensor(
                np.array(train, dtype=np.float32, copy=True, order="C"),
                device="cuda:0",
            )
            validation_teacher = torch.argmax(q @ x.T, dim=1)
    for sample_id in range(config.sample_count):
        sample_ids = rng.choice(len(calibration), config.query_batch_size, replace=False)
        matrices = compute_query_matrices(calibration[sample_ids], centroids)
        iteration_reports = []
        for iteration in range(config.train_iterations):
            codes, loss, encoding = encode_cuda(
                train, codebook, matrices, train_clusters,
                iterations=config.encode_iterations,
            )
            codebook, update = update_codebook_cuda(
                train, codes, matrices, train_clusters, config.k, config.regularization
            )
            iteration_reports.append({
                "iteration": iteration + 1,
                "pre_update_loss": loss,
                "encoding": encoding,
                "update": update,
            })
        selection = None
        if validation is not None:

            torch = _torch_cuda()
            with torch.inference_mode():
                q = torch.as_tensor(
                    np.array(validation, dtype=np.float32, copy=True, order="C"),
                    device="cuda:0",
                )
                cb = torch.as_tensor(codebook, dtype=torch.float32, device="cuda:0")
                code = torch.as_tensor(codes.astype(np.int64), device="cuda:0")
                reconstructed = _decode_raw_torch(torch, code, cb)
                returned = torch.topk(q @ reconstructed.T, k=min(100, len(train)), dim=1).indices
                recall100 = torch.mean(
                    torch.any(returned == validation_teacher[:, None], dim=1).float()
                ).item()
                exact_ip = torch.sum(q * torch.as_tensor(
                    np.array(train, dtype=np.float32, copy=True, order="C"),
                    device="cuda:0",
                )[validation_teacher], dim=1)
                approximate_ip = torch.sum(q * reconstructed[validation_teacher], dim=1)
                relative_loss = torch.mean(
                    torch.abs(exact_ip - approximate_ip) / torch.clamp(exact_ip, min=EPS)
                ).item()
            selection = {
                "official_recall_1_at_100_on_training_set": float(recall100),
                "official_relative_top1_ip_loss": float(relative_loss),
            }
            if recall100 > best_recall:
                best_recall = float(recall100)
                best_codebook = codebook.copy()
                best_matrices = matrices.copy()
                best_sample_id = sample_id + 1
        cycles.append({
            "sample_id": sample_id + 1,
            "calibration_ids": sample_ids.tolist(),
            "iterations": iteration_reports,
            "selection": selection,
        })
    if best_codebook is not None:
        codebook = best_codebook
        matrices = best_matrices
    else:
        best_sample_id = config.sample_count
    # Official QAQ uses the selected sample's cluster matrices for final encode.
    final_codes, final_loss, final_encoding = encode_cuda(
        gallery, codebook, matrices, gallery_clusters,
        iterations=config.encode_iterations,
    )
    report = {
        "algorithm": "QAQ / SSR_V2",
        "config": asdict(config),
        "clustering": clustering,
        "cycles": cycles,
        "selection": {
            "metric": (
                "official Recall 1@100 over codec-training set"
                if validation is not None else "last sample (no validation supplied)"
            ),
            "best_sample_id": best_sample_id,
            "best_recall": None if not math.isfinite(best_recall) else best_recall,
        },
        "final_encoding": final_encoding,
        "final_loss": final_loss,
        "total_seconds": time.perf_counter() - started,
        "reference_equations": {
            "query_matrix": "sum_q softmax(q^T centroid) q q^T",
            "encoding": "coordinate descent on (x-xhat)^T A_cluster (x-xhat)",
            "codebook": "(T + regularization I)^-1 R",
        },
    }
    return codebook, final_codes, gallery_clusters, report


class QAQPQIndex:
    """Reloadable physical QAQ PQ index with a complete byte ledger."""

    def __init__(
        self,
        codebooks: np.ndarray,
        packed_codes: np.ndarray,
        item_ids: np.ndarray,
        inverse_norms: np.ndarray,
        metadata: dict,
        *,
        path: str | None = None,
        layout: dict | None = None,
    ):
        stored = np.asarray(codebooks)
        if stored.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("QAQ codebooks must be stored as float16 or float32")
        self.codebooks = stored
        self.packed_codes = np.asarray(packed_codes, dtype=np.uint8)
        self.item_ids = np.asarray(item_ids, dtype=np.uint32)
        self.inverse_norms = np.asarray(inverse_norms, dtype=np.float16)
        self.metadata = metadata
        self.path = path
        if self.codebooks.ndim != 3 or self.packed_codes.ndim != 2:
            raise ValueError("invalid QAQ codebook or packed-code shape")
        if len(self.item_ids) != len(self.packed_codes) or len(self.inverse_norms) != len(self.item_ids):
            raise ValueError("QAQ item arrays differ in length")
        expected = (self.stages * self.nbits + 7) // 8
        if self.packed_codes.shape[1] != expected:
            raise ValueError("packed QAQ row width is inconsistent")
        if layout is None:
            self._layout, self.persistent_bytes = self._compute_layout()
        else:
            self._layout = layout
            self.persistent_bytes = os.path.getsize(path)

    @property
    def stages(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def k(self) -> int:
        return int(self.codebooks.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.codebooks.shape[0] * self.codebooks.shape[2])

    @property
    def nbits(self) -> int:
        return _power_of_two_bits(self.k)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def code_only_bytes(self) -> int:
        return int(self.packed_codes.nbytes)

    @property
    def code_only_bytes_per_item(self) -> float:
        return self.code_only_bytes / self.n_items

    @property
    def persistent_bytes_per_item(self) -> float:
        return self.persistent_bytes / self.n_items

    @classmethod
    def from_codes(
        cls,
        codebooks: np.ndarray,
        codes: np.ndarray,
        metadata: dict | None = None,
        *,
        precision: str = "fp32",
    ) -> "QAQPQIndex":
        if precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        dtype = np.float16 if precision == "fp16" else np.float32
        codebooks = np.ascontiguousarray(codebooks, dtype=dtype)
        codes = np.asarray(codes)
        bits = _power_of_two_bits(codebooks.shape[1])
        packed = pack_codes(codes, bits)
        raw = np.empty((len(codes), codebooks.shape[0], codebooks.shape[2]), dtype=np.float32)
        rows = np.arange(len(codes))
        for stage in range(codebooks.shape[0]):
            raw[:, stage] = codebooks[stage, codes[:, stage].astype(np.int64)]
        raw = raw.reshape(len(codes), -1)
        norms = np.linalg.norm(raw, axis=1)
        inverse = (1.0 / np.maximum(norms, EPS)).astype(np.float16)
        merged = {
            "format": "QAQ fixed-rate product quantizer",
            "items": len(codes),
            "dimension": raw.shape[1],
            "M": codebooks.shape[0],
            "K": codebooks.shape[1],
            "bits_per_symbol": bits,
            "nominal_code_bits_per_item": codebooks.shape[0] * bits,
            "representation": "row-wise bit-packed PQ codes",
            "ids": "explicit uint32 row IDs",
            "normalisation": "float16 inverse reconstruction norm",
            "codebook_storage_precision": precision,
            "scoring_compute": "FP32 after persisted-codebook reload",
        }
        if metadata:
            merged.update(metadata)
        return cls(
            codebooks, packed, np.arange(len(codes), dtype=np.uint32), inverse, merged
        )

    def to_precision(self, precision: str) -> "QAQPQIndex":
        """Keep symbols fixed, round shared codebooks, and recompute norms."""
        if precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        dtype = np.float16 if precision == "fp16" else np.float32
        rounded = np.ascontiguousarray(self.codebooks, dtype=dtype)
        codes = unpack_codes(self.packed_codes, self.stages, self.nbits)
        raw = np.empty(
            (self.n_items, self.stages, rounded.shape[2]), dtype=np.float32
        )
        for stage in range(self.stages):
            raw[:, stage] = rounded[
                stage, codes[:, stage].astype(np.int64)
            ]
        inverse = (
            1.0 / np.maximum(np.linalg.norm(raw.reshape(self.n_items, -1), axis=1), EPS)
        ).astype(np.float16)
        metadata = dict(self.metadata)
        metadata.update({
            "codebook_storage_precision": precision,
            "source_codebook_sha256": codebook_sha256(self.codebooks),
            "normalisation": (
                f"float16 inverse norm recomputed from {precision} codebooks"
            ),
            "scoring_compute": "FP32 after persisted-codebook reload",
        })
        return type(self)(
            rounded,
            np.asarray(self.packed_codes, dtype=np.uint8),
            np.asarray(self.item_ids, dtype=np.uint32),
            inverse,
            metadata,
        )

    def _compute_layout(self) -> tuple[dict, int]:
        arrays = {
            "codebooks": self.codebooks,
            "ids": self.item_ids,
            "norms": self.inverse_norms,
            "codes": self.packed_codes,
        }
        layout = {}
        offset = HEADER_BYTES
        for name, value in arrays.items():
            offset = _align(offset)
            layout[name] = {
                "offset": offset,
                "nbytes": int(value.nbytes),
                "dtype": np.dtype(value.dtype).str,
                "shape": list(value.shape),
            }
            offset += int(value.nbytes)
        return layout, offset

    def storage_breakdown(self) -> dict:
        fields = {
            "codebooks": int(self.codebooks.nbytes),
            "codes": int(self.packed_codes.nbytes),
            "ids": int(self.item_ids.nbytes),
            "norms": int(self.inverse_norms.nbytes),
        }
        payload = sum(fields.values())
        return fields | {
            "header_and_alignment": int(self.persistent_bytes - payload),
            "persistent_bytes": int(self.persistent_bytes),
            "code_only_bytes": int(self.code_only_bytes),
        }

    def decode_normalized(self) -> np.ndarray:
        raw = self.decode_raw()
        inverse_by_id = np.empty(self.n_items, dtype=np.float32)
        inverse_by_id[np.asarray(self.item_ids, dtype=np.int64)] = np.asarray(
            self.inverse_norms, dtype=np.float32
        )
        raw *= inverse_by_id[:, None]
        return np.ascontiguousarray(raw)

    def decode_raw(self) -> np.ndarray:
        codes = unpack_codes(self.packed_codes, self.stages, self.nbits)
        raw = np.empty((self.n_items, self.stages, self.codebooks.shape[2]), dtype=np.float32)
        for stage in range(self.stages):
            raw[:, stage] = self.codebooks[stage, codes[:, stage].astype(np.int64)]
        raw = raw.reshape(self.n_items, self.dimension)
        output = np.empty_like(raw)
        output[np.asarray(self.item_ids, dtype=np.int64)] = raw
        permutation = dimension_permutation(self.metadata, self.dimension)
        if permutation is not None:
            unpermuted = np.empty_like(output)
            unpermuted[:, permutation] = output
            output = unpermuted
        return np.ascontiguousarray(output)

    def save(self, path: str | os.PathLike) -> str:
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {
            "format_version": FORMAT_VERSION,
            "layout": self._layout,
            "metadata": self.metadata,
            "persistent_bytes": int(self.persistent_bytes),
        }
        payload = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) + 16 > HEADER_BYTES:
            raise ValueError("QAQ metadata exceeds the fixed header")
        arrays = {
            "codebooks": self.codebooks,
            "ids": self.item_ids,
            "norms": self.inverse_norms,
            "codes": self.packed_codes,
        }
        with open(path, "wb") as handle:
            handle.write(MAGIC)
            handle.write(struct.pack("<Q", len(payload)))
            handle.write(payload)
            handle.write(b"\0" * (HEADER_BYTES - 16 - len(payload)))
            for name, value in arrays.items():
                target = int(self._layout[name]["offset"])
                if handle.tell() < target:
                    handle.write(b"\0" * (target - handle.tell()))
                handle.write(np.ascontiguousarray(value).tobytes())
        if os.path.getsize(path) != self.persistent_bytes:
            raise AssertionError("QAQ serialized size differs from byte ledger")
        self.path = path
        return path

    def close(self) -> None:
        """Release loaded memory maps before replacing/deleting a file on Windows."""
        for value in (
            self.codebooks, self.packed_codes, self.item_ids, self.inverse_norms
        ):
            current = value
            visited = set()
            while current is not None and id(current) not in visited:
                visited.add(id(current))
                mapping = getattr(current, "_mmap", None)
                if mapping is not None:
                    mapping.close()
                    break
                current = getattr(current, "base", None)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "QAQPQIndex":
        path = os.path.abspath(path)
        with open(path, "rb") as handle:
            if handle.read(8) != MAGIC:
                raise ValueError("not a QAQ PQ index")
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        if int(header["format_version"]) != FORMAT_VERSION:
            raise ValueError("unsupported QAQ PQ index version")
        arrays = {}
        for name, spec in header["layout"].items():
            arrays[name] = np.memmap(
                path, mode="r", dtype=np.dtype(spec["dtype"]),
                offset=int(spec["offset"]), shape=tuple(spec["shape"]),
            )
        index = cls(
            arrays["codebooks"], arrays["codes"], arrays["ids"], arrays["norms"],
            header["metadata"], path=path, layout=header["layout"],
        )
        if index.persistent_bytes != int(header["persistent_bytes"]):
            raise ValueError("QAQ file size differs from its persisted ledger")
        return index


def codebook_sha256(codebook: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(codebook).view(np.uint8)).hexdigest()


__all__ = [
    "QAQConfig", "QAQPQIndex", "cluster_gallery", "codebook_sha256",
    "compute_query_matrices", "dimension_permutation", "encode_cuda", "pack_codes", "train_qaq",
    "unpack_codes", "update_codebook_cuda",
]
