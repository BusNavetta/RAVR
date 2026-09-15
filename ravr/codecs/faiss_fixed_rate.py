"""Auditable fixed-rate Faiss PQ, OPQ, and RQ codec bundles.

The bundle is intentionally simple: a trained, empty Faiss codec plus the raw
standalone codes, uint32 item IDs, and float16 inverse reconstruction norms.
Its on-disk size is therefore the complete persistent cost used by the benchmark,
while ``codes.nbytes`` is the separate code-only budget.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import struct
import time

import faiss
import numpy as np


MAGIC = b"FCODEC1\0"
FORMAT_VERSION = 1
HEADER_BYTES = 4096
ALIGN_BYTES = 64
EPS = 1e-9


def _align(value: int) -> int:
    return (int(value) + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES


@dataclass(frozen=True)
class FaissCodecConfig:
    method: str
    m: int
    nbits: int = 8
    seed: int = 42
    kmeans_iterations: int = 5
    opq_iterations: int = 5
    rq_beam_size: int = 5

    def validate(self, dimension: int) -> None:
        if self.method not in {"pq", "opq", "rq"}:
            raise ValueError("method must be pq, opq, or rq")
        if self.m <= 0 or self.nbits <= 0:
            raise ValueError("m and nbits must be positive")
        if self.method in {"pq", "opq"} and dimension % self.m:
            raise ValueError("PQ/OPQ require the dimension to be divisible by m")
        if self.nbits != 8:
            raise ValueError("the fixed-rate reviewer benchmark is locked to 8-bit codes")

    @property
    def factory(self) -> str:
        if self.method == "pq":
            return f"PQ{self.m}x{self.nbits}"
        if self.method == "opq":
            return f"OPQ{self.m},PQ{self.m}x{self.nbits}"
        return f"RQ{self.m}x{self.nbits}"

    @property
    def display_name(self) -> str:
        if self.method == "pq":
            return f"PQ-{self.m}x8"
        if self.method == "opq":
            return f"OPQ-{self.m},PQ-{self.m}x8"
        return f"RQ-{self.m}x8"


def _configure_codec(codec, config: FaissCodecConfig) -> None:
    if config.method == "pq":
        codec.pq.cp.niter = config.kmeans_iterations
        codec.pq.cp.seed = config.seed
    elif config.method == "opq":
        transform = faiss.downcast_VectorTransform(codec.chain.at(0))
        transform.niter = config.opq_iterations
        transform.niter_pq = config.kmeans_iterations
        transform.niter_pq_0 = config.kmeans_iterations
        inner = faiss.downcast_index(codec.index)
        inner.pq.cp.niter = config.kmeans_iterations
        inner.pq.cp.seed = config.seed
    else:
        codec.rq.cp.niter = config.kmeans_iterations
        codec.rq.cp.seed = config.seed
        codec.rq.niter_codebook_refine = config.kmeans_iterations
        codec.rq.max_beam_size = config.rq_beam_size


class FaissCodecBundle:
    def __init__(
        self,
        codec,
        codes: np.ndarray,
        item_ids: np.ndarray,
        inverse_norms: np.ndarray,
        metadata: dict,
        path: str | None = None,
        layout: dict | None = None,
    ):
        self.codec = codec
        self.codes = np.asarray(codes, dtype=np.uint8)
        self.item_ids = np.asarray(item_ids, dtype=np.uint32)
        self.inverse_norms = np.asarray(inverse_norms, dtype=np.float16)
        self.metadata = metadata
        self.path = path
        self._codec_blob = np.array(faiss.serialize_index(codec), dtype=np.uint8, copy=True)
        self._layout, self.persistent_bytes = (
            self._compute_layout() if layout is None else (layout, os.path.getsize(path))
        )

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def code_only_bytes(self) -> int:
        return int(self.codes.nbytes)

    @property
    def code_only_bytes_per_item(self) -> float:
        return self.code_only_bytes / self.n_items

    @property
    def persistent_bytes_per_item(self) -> float:
        return self.persistent_bytes / self.n_items

    def _compute_layout(self) -> tuple[dict, int]:
        arrays = {
            "codec": self._codec_blob,
            "codes": self.codes,
            "ids": self.item_ids,
            "norms": self.inverse_norms,
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

    def storage_breakdown(self) -> dict[str, int | float]:
        codec_bytes = int(self._layout["codec"]["nbytes"])
        code_bytes = int(self._layout["codes"]["nbytes"])
        id_bytes = int(self._layout["ids"]["nbytes"])
        norm_bytes = int(self._layout["norms"]["nbytes"])
        payload = codec_bytes + code_bytes + id_bytes + norm_bytes
        return {
            "header_and_alignment": int(self.persistent_bytes - payload),
            "codec_and_rotation": codec_bytes,
            "ivf_centroids": 0,
            "codes": code_bytes,
            "ids": id_bytes,
            "norms": norm_bytes,
            "length_metadata": 0,
            "persistent_bytes": int(self.persistent_bytes),
            "code_only_bytes": code_bytes,
            "rotation_matrix_nominal_bytes": int(
                self.metadata.get("rotation_matrix_nominal_bytes", 0)
            ),
            "codebooks_nominal_bytes": int(
                self.metadata.get("codebooks_nominal_bytes", 0)
            ),
        }

    def decode_normalized(self) -> np.ndarray:
        reconstruction = np.asarray(self.codec.sa_decode(self.codes), dtype=np.float32)
        reconstruction *= self.inverse_norms.astype(np.float32)[:, None]
        return np.ascontiguousarray(reconstruction)

    def save(self, path: str) -> str:
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = {
            "format_version": FORMAT_VERSION,
            "layout": self._layout,
            "metadata": self.metadata,
            "persistent_bytes": self.persistent_bytes,
        }
        payload = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) + 16 > HEADER_BYTES:
            raise ValueError("bundle metadata exceeds fixed header reservation")
        arrays = {
            "codec": self._codec_blob,
            "codes": self.codes,
            "ids": self.item_ids,
            "norms": self.inverse_norms,
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
            raise AssertionError("serialized bundle size does not match its layout")
        self.path = path
        return path

    def close(self) -> None:
        """Release memory maps so loaded bundles can be replaced on Windows."""
        for value in (self._codec_blob, self.codes, self.item_ids, self.inverse_norms):
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
    def load(cls, path: str) -> "FaissCodecBundle":
        path = os.path.abspath(path)
        with open(path, "rb") as handle:
            if handle.read(8) != MAGIC:
                raise ValueError("not a Faiss fixed-rate codec bundle")
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        if header["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported Faiss codec bundle version")
        layout = header["layout"]
        arrays = {}
        for name in ("codec", "codes", "ids", "norms"):
            meta = layout[name]
            arrays[name] = np.memmap(
                path,
                mode="r",
                dtype=np.dtype(meta["dtype"]),
                offset=int(meta["offset"]),
                shape=tuple(meta["shape"]),
            )
        codec = faiss.deserialize_index(np.array(arrays["codec"], dtype=np.uint8, copy=True))
        bundle = cls(
            codec=codec,
            codes=arrays["codes"],
            item_ids=arrays["ids"],
            inverse_norms=arrays["norms"],
            metadata=header["metadata"],
            path=path,
            layout=layout,
        )
        if bundle.persistent_bytes != int(header["persistent_bytes"]):
            raise ValueError("bundle size does not match its header")
        return bundle


def train_bundle(
    gallery: np.ndarray,
    train_ids: np.ndarray,
    config: FaissCodecConfig,
) -> FaissCodecBundle:
    gallery = np.ascontiguousarray(np.asarray(gallery, dtype=np.float32))
    train_ids = np.asarray(train_ids, dtype=np.int64)
    config.validate(gallery.shape[1])
    codec = faiss.index_factory(gallery.shape[1], config.factory, faiss.METRIC_L2)
    _configure_codec(codec, config)
    started = time.perf_counter()
    codec.train(np.ascontiguousarray(gallery[train_ids]))
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    codes = np.asarray(codec.sa_encode(gallery), dtype=np.uint8)
    encode_seconds = time.perf_counter() - started
    reconstruction = np.asarray(codec.sa_decode(codes), dtype=np.float32)
    inverse_norms = (
        1.0 / np.maximum(np.linalg.norm(reconstruction, axis=1), EPS)
    ).astype(np.float16)
    d = gallery.shape[1]
    k = 1 << config.nbits
    rotation_bytes = d * d * 4 if config.method == "opq" else 0
    codebook_bytes = d * k * 4 if config.method in {"pq", "opq"} else config.m * k * d * 4
    metadata = {
        "config": asdict(config),
        "factory": config.factory,
        "display_name": config.display_name,
        "dimension": d,
        "items": len(gallery),
        "training_items": len(train_ids),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "train_seconds": train_seconds,
        "encode_seconds": encode_seconds,
        "rotation_matrix_nominal_bytes": rotation_bytes,
        "codebooks_nominal_bytes": codebook_bytes,
        "normalisation": "float16 inverse decoded-vector norm",
        "ids": "uint32 explicit row IDs",
    }
    return FaissCodecBundle(
        codec=codec,
        codes=codes,
        item_ids=np.arange(len(gallery), dtype=np.uint32),
        inverse_norms=inverse_norms,
        metadata=metadata,
    )


__all__ = ["FaissCodecBundle", "FaissCodecConfig", "train_bundle"]
