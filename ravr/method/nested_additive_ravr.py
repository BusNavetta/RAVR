"""Workload-aware allocation over a frozen nested additive codec.

The physical index keeps the exact serialized Faiss codec and only a variable
number of leading packed symbols for each item.  It never stores decoded
``N x D`` vectors or one-hot code matrices.  Items are grouped by prefix length,
so the small bucket table is also the complete length metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict

import faiss
import numpy as np

from additive_coded_search import faiss_additive_codebooks, score_packed_additive_codes, unpack_stage
from faiss_fixed_rate import EPS
from ravr_rq import OBJECTIVE_ORDINAL_KL, RAVRRQConfig, RAVRRQTrainer


MAGIC = b"NAVRLSQ1"
FORMAT_VERSION = 1
HEADER_BYTES = 4096
ALIGN_BYTES = 64


def _align(value: int) -> int:
    return (int(value) + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES


def codec_sha256(value) -> str:
    """Hash the exact persisted decoder values for Faiss or raw codebooks."""
    stored = getattr(value, "codebooks", None)
    if stored is not None:
        return hashlib.sha256(
            np.ascontiguousarray(stored).view(np.uint8)
        ).hexdigest()
    if isinstance(value, np.ndarray):
        return hashlib.sha256(
            np.ascontiguousarray(value).view(np.uint8)
        ).hexdigest()
    codec = getattr(value, "codec", value)
    blob = np.asarray(faiss.serialize_index(codec), dtype=np.uint8)
    return hashlib.sha256(blob.tobytes()).hexdigest()


def unpack_additive_codes(packed_codes: np.ndarray, stages: int, nbits: int) -> np.ndarray:
    """Return the compact integer stage IDs used only during calibration."""
    packed = np.asarray(packed_codes, dtype=np.uint8)
    dtype = np.uint8 if nbits <= 8 else np.uint16
    result = np.empty((len(packed), stages), dtype=dtype)
    for stage in range(stages):
        result[:, stage] = unpack_stage(packed, stage, nbits)
    return result


def calibrate_frozen_additive_codec(
    bundle,
    gallery: np.ndarray,
    calibration_queries: np.ndarray,
    calibration_exclude_ids: np.ndarray | None,
    *,
    allowed_lengths: tuple[int, ...],
    rerank_candidates: int = 10000,
    retrieval_k: int = 10,
    hard_negative_depth: int = 50,
    calibration_limit: int | None = None,
) -> RAVRRQTrainer:
    """Reuse Ordinal-KL calibration without training or changing the codec."""
    codebooks, nbits = faiss_additive_codebooks(bundle)
    stages = codebooks.shape[0]
    if tuple(sorted(set(allowed_lengths))) != tuple(allowed_lengths):
        raise ValueError("allowed_lengths must be strictly increasing")
    if allowed_lengths[-1] > stages:
        raise ValueError("a prefix exceeds the frozen codec depth")
    if any((length * nbits) % 8 for length in allowed_lengths):
        raise ValueError("physical prefix rows must end on byte boundaries")
    queries = np.asarray(calibration_queries, dtype=np.float32)
    exclude = None if calibration_exclude_ids is None else np.asarray(
        calibration_exclude_ids, dtype=np.int64
    )
    if calibration_limit is not None:
        queries = queries[: int(calibration_limit)]
        if exclude is not None:
            exclude = exclude[: len(queries)]
    config = RAVRRQConfig(
        seed=int(bundle.metadata["config"]["seed"]),
        num_centroids=1,
        nprobe=1,
        stages=stages,
        codebook_size=1 << nbits,
        allowed_lengths=tuple(int(x) for x in allowed_lengths),
        base_stages=int(allowed_lengths[0]),
        rerank_candidates=int(rerank_candidates),
        retrieval_k=int(retrieval_k),
        hard_negative_depth=int(hard_negative_depth),
        objective=OBJECTIVE_ORDINAL_KL,
        norm_mode="stored_final_norm",
    )
    config.validate(len(gallery))
    trainer = RAVRRQTrainer(config)
    trainer.x = np.ascontiguousarray(np.asarray(gallery, dtype=np.float32))
    norms = np.linalg.norm(trainer.x, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4):
        trainer.x = trainer.x / np.maximum(norms[:, None], EPS)
    trainer.n_items, trainer.dimension = trainer.x.shape
    trainer.centroids = np.zeros((1, trainer.dimension), dtype=np.float32)
    trainer.cluster_ids = np.zeros(trainer.n_items, dtype=np.int32)
    trainer.codebooks = np.ascontiguousarray(codebooks, dtype=np.float32)
    trainer.full_codes = unpack_additive_codes(bundle.codes, stages, nbits)
    trainer.validation_report = {
        "queries": 0,
        "candidate_sweep": {},
        "selected_rerank_candidates": int(rerank_candidates),
        "selection_target": None,
        "selection_note": "frozen before allocator evaluation",
    }
    trainer._compute_prefix_statistics()
    trainer._calibrate(queries, exclude)
    trainer.gallery_hash = hashlib.sha256(
        np.ascontiguousarray(trainer.x).view(np.uint8)
    ).hexdigest()
    trainer.codec_hash = codec_sha256(bundle)
    trainer.config_hash = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return trainer


def query_ip_distortion(
    bundle,
    gallery: np.ndarray,
    calibration_queries: np.ndarray,
    prefix_norms: np.ndarray,
    allowed_lengths: tuple[int, ...],
    *,
    block_size: int = 8192,
) -> np.ndarray:
    """QUIP/QAQ-style expected inner-product error for frozen prefixes.

    For every item and prefix this computes
    ``E_q[(q^T x - q^T x_hat_prefix)^2]`` using the empirical non-centred
    calibration-query covariance.  The original methods use this loss to learn
    a quantizer; here it is deliberately transferred only to rate allocation so
    the codec and codes remain identical across methods.
    """
    x = np.array(gallery, dtype=np.float32, order="C", copy=True)
    q = np.array(calibration_queries, dtype=np.float32, order="C", copy=True)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), EPS)
    covariance = np.ascontiguousarray((q.T @ q) / len(q), dtype=np.float32)
    codebooks, nbits = faiss_additive_codebooks(bundle)
    codes = unpack_additive_codes(bundle.codes, codebooks.shape[0], nbits)
    norms = np.asarray(prefix_norms, dtype=np.float32)
    if norms.shape != (len(x), len(allowed_lengths)):
        raise ValueError("prefix_norms do not match the requested prefixes")
    option_for_length = {
        int(length): option for option, length in enumerate(allowed_lengths)
    }
    result = np.empty_like(norms, dtype=np.float32)
    reconstruction = np.zeros_like(x)
    for stage in range(codebooks.shape[0]):
        reconstruction += codebooks[stage, codes[:, stage].astype(np.int64)]
        length = stage + 1
        if length not in option_for_length:
            continue
        option = option_for_length[length]
        for start in range(0, len(x), block_size):
            stop = min(len(x), start + block_size)
            approximate = reconstruction[start:stop] / np.maximum(
                norms[start:stop, option, None], EPS
            )
            error = x[start:stop] - approximate
            result[start:stop, option] = np.einsum(
                "ij,ij->i", error @ covariance, error, optimize=True
            )
    return result


class NestedAdditiveIndex:
    """Reloadable variable-prefix index over one frozen RQ/LSQ codec."""

    def __init__(
        self,
        codec_or_codebooks,
        code_payload: np.ndarray,
        item_ids: np.ndarray,
        inverse_norms: np.ndarray,
        bucket_spans: np.ndarray,
        metadata: dict,
        *,
        path: str | None = None,
        layout: dict | None = None,
    ):
        decoder_format = metadata.get("decoder_format", "faiss-index")
        self.codec = None
        self.codebooks = None
        if decoder_format == "raw-additive-codebooks":
            stored = np.asarray(codec_or_codebooks)
            if stored.ndim != 3 or stored.dtype not in (
                np.dtype(np.float16), np.dtype(np.float32)
            ):
                raise ValueError("raw additive codebooks must be float16 or float32")
            self.codebooks = stored
            self._codec_blob = self.codebooks
        elif decoder_format == "faiss-index":
            self.codec = codec_or_codebooks
            self._codec_blob = np.array(
                faiss.serialize_index(self.codec), dtype=np.uint8, copy=True
            )
        else:
            raise ValueError(f"unsupported decoder format: {decoder_format}")
        self.code_payload = np.asarray(code_payload, dtype=np.uint8).reshape(-1)
        self.item_ids = np.asarray(item_ids, dtype=np.uint32).reshape(-1)
        self.inverse_norms = np.asarray(inverse_norms, dtype=np.float16).reshape(-1)
        self.bucket_spans = np.asarray(bucket_spans, dtype=np.uint64)
        self.metadata = metadata
        self.path = path
        if self.inverse_norms.shape != self.item_ids.shape:
            raise ValueError("item IDs and inverse norms differ in length")
        if self.bucket_spans.shape != (len(self.allowed_lengths), 4):
            raise ValueError("bucket_spans must have one four-field row per prefix")
        if layout is None:
            self._layout, self.persistent_bytes = self._compute_layout()
        else:
            self._layout = layout
            self.persistent_bytes = os.path.getsize(path)
        self._validate_spans()

    @property
    def allowed_lengths(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.metadata["allowed_lengths"])

    @property
    def nbits(self) -> int:
        return int(self.metadata["nbits"])

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def code_only_bytes(self) -> int:
        return int(self.code_payload.nbytes)

    @property
    def code_only_bytes_per_item(self) -> float:
        return self.code_only_bytes / self.n_items

    @property
    def persistent_bytes_per_item(self) -> float:
        return self.persistent_bytes / self.n_items

    @classmethod
    def from_allocation(
        cls,
        bundle,
        prefix_norms: np.ndarray,
        allowed_lengths: tuple[int, ...],
        selected_options: np.ndarray,
        *,
        strategy: str,
        metadata: dict | None = None,
    ) -> "NestedAdditiveIndex":
        codebooks, nbits = faiss_additive_codebooks(bundle)
        lengths = tuple(int(x) for x in allowed_lengths)
        if any((length * nbits) % 8 for length in lengths):
            raise ValueError("every stored prefix must end on a byte boundary")
        selected = np.asarray(selected_options, dtype=np.int64)
        prefix_norms = np.asarray(prefix_norms, dtype=np.float32)
        if selected.shape != (bundle.n_items,):
            raise ValueError("selected_options must have one entry per item")
        if prefix_norms.shape != (bundle.n_items, len(lengths)):
            raise ValueError("prefix_norms do not match items and options")
        if np.any((selected < 0) | (selected >= len(lengths))):
            raise ValueError("selected option is out of range")

        ids, norms, parts, spans = [], [], [], []
        item_offset = 0
        code_offset = 0
        for option, length in enumerate(lengths):
            members = np.flatnonzero(selected == option)
            row_bytes = length * nbits // 8
            prefix = np.ascontiguousarray(bundle.codes[members, :row_bytes])
            ids.append(np.asarray(bundle.item_ids[members], dtype=np.uint32))
            norms.append((1.0 / np.maximum(prefix_norms[members, option], EPS)).astype(np.float16))
            parts.append(prefix.reshape(-1))
            spans.append((item_offset, len(members), code_offset, row_bytes))
            item_offset += len(members)
            code_offset += int(prefix.nbytes)
        allocation_histogram = {
            str(length): int(np.sum(selected == option))
            for option, length in enumerate(lengths)
        }
        raw_codebooks = getattr(bundle, "codebooks", None)
        decoder_format = (
            "raw-additive-codebooks" if raw_codebooks is not None else "faiss-index"
        )
        decoder = raw_codebooks if raw_codebooks is not None else bundle.codec
        merged_metadata = {
            "format": "nested-additive-variable-prefix",
            "config": bundle.metadata["config"],
            "family": bundle.metadata["config"].get("family", bundle.metadata["config"].get("method")),
            "factory": bundle.metadata.get("factory"),
            "dimension": int(bundle.metadata["dimension"]),
            "items": int(bundle.n_items),
            "stages": int(codebooks.shape[0]),
            "nbits": int(nbits),
            "allowed_lengths": list(lengths),
            "strategy": strategy,
            "allocation_histogram": allocation_histogram,
            "codec_sha256": codec_sha256(bundle),
            "decoder_format": decoder_format,
            "decoder_storage_precision": (
                str(np.asarray(raw_codebooks).dtype)
                if raw_codebooks is not None else "native Faiss float32"
            ),
            "source_codec_metadata": bundle.metadata,
            "representation": "byte-aligned packed prefix codes grouped by length",
            "normalisation": "float16 inverse norm for the assigned prefix",
            "ids": "uint32 explicit row IDs",
        }
        if metadata:
            merged_metadata.update(metadata)
        return cls(
            codec_or_codebooks=decoder,
            code_payload=np.concatenate(parts) if parts else np.empty(0, np.uint8),
            item_ids=np.concatenate(ids) if ids else np.empty(0, np.uint32),
            inverse_norms=np.concatenate(norms) if norms else np.empty(0, np.float16),
            bucket_spans=np.asarray(spans, dtype=np.uint64),
            metadata=merged_metadata,
        )

    def _compute_layout(self) -> tuple[dict, int]:
        # Codes are last, so the exact file cap depends only on actual code bytes.
        arrays = {
            "codec": self._codec_blob,
            "ids": self.item_ids,
            "norms": self.inverse_norms,
            "spans": self.bucket_spans,
            "codes": self.code_payload,
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

    def _validate_spans(self) -> None:
        seen_items = 0
        seen_codes = 0
        for length, row in zip(self.allowed_lengths, self.bucket_spans):
            item_start, count, code_start, row_bytes = (int(x) for x in row)
            if item_start != seen_items or code_start != seen_codes:
                raise ValueError("bucket spans are not contiguous")
            if row_bytes != length * self.nbits // 8:
                raise ValueError("bucket row width and prefix length disagree")
            seen_items += count
            seen_codes += count * row_bytes
        if seen_items != self.n_items or seen_codes != self.code_only_bytes:
            raise ValueError("bucket spans do not cover the physical arrays")

    def _bucket_codes(self, option: int) -> np.ndarray:
        _, count, code_start, row_bytes = (int(x) for x in self.bucket_spans[option])
        stop = code_start + count * row_bytes
        return np.asarray(self.code_payload[code_start:stop]).reshape(count, row_bytes)

    def assigned_lengths_by_id(self) -> np.ndarray:
        result = np.empty(self.n_items, dtype=np.uint16)
        for length, row in zip(self.allowed_lengths, self.bucket_spans):
            item_start, count = int(row[0]), int(row[1])
            ids = np.asarray(self.item_ids[item_start:item_start + count], dtype=np.int64)
            result[ids] = length
        return result

    def decode_normalized(self) -> np.ndarray:
        codebooks, nbits = faiss_additive_codebooks(self)
        output = np.empty((self.n_items, codebooks.shape[2]), dtype=np.float32)
        for option, length in enumerate(self.allowed_lengths):
            item_start, count = (int(x) for x in self.bucket_spans[option, :2])
            if not count:
                continue
            codes = self._bucket_codes(option)
            reconstruction = np.zeros((count, codebooks.shape[2]), dtype=np.float32)
            rows = np.arange(count)
            for stage in range(length):
                reconstruction += codebooks[stage, unpack_stage(codes, stage, nbits)]
            reconstruction *= np.asarray(
                self.inverse_norms[item_start:item_start + count], dtype=np.float32
            )[:, None]
            ids = np.asarray(self.item_ids[item_start:item_start + count], dtype=np.int64)
            output[ids] = reconstruction
        return np.ascontiguousarray(output)

    def score_queries(self, queries: np.ndarray) -> np.ndarray:
        """Exact LUT scoring from codes; no decoded gallery is materialized."""
        codebooks, nbits = faiss_additive_codebooks(self)
        result = np.empty((len(queries), self.n_items), dtype=np.float32)
        for option, length in enumerate(self.allowed_lengths):
            item_start, count = (int(x) for x in self.bucket_spans[option, :2])
            if not count:
                continue
            local = score_packed_additive_codes(
                queries,
                codebooks[:length],
                self._bucket_codes(option),
                self.inverse_norms[item_start:item_start + count],
                nbits,
            )
            ids = np.asarray(self.item_ids[item_start:item_start + count], dtype=np.int64)
            result[:, ids] = local
        return result

    def storage_breakdown(self) -> dict[str, int | float]:
        fields = {
            "codec_and_codebooks": int(self._layout["codec"]["nbytes"]),
            "codes": int(self._layout["codes"]["nbytes"]),
            "ids": int(self._layout["ids"]["nbytes"]),
            "norms": int(self._layout["norms"]["nbytes"]),
            "length_metadata": int(self._layout["spans"]["nbytes"]),
        }
        payload = sum(fields.values())
        fields["header_and_alignment"] = int(self.persistent_bytes - payload)
        fields["persistent_bytes"] = int(self.persistent_bytes)
        fields["code_only_bytes"] = int(self.code_only_bytes)
        return fields

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
            raise ValueError("bundle metadata exceeds the fixed header")
        arrays = {
            "codec": self._codec_blob,
            "ids": self.item_ids,
            "norms": self.inverse_norms,
            "spans": self.bucket_spans,
            "codes": self.code_payload,
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
            raise AssertionError("serialized size differs from the byte ledger")
        self.path = path
        return path

    def close(self) -> None:
        """Release memory maps promptly (required before replacing files on Windows)."""
        for value in (
            self._codec_blob, self.code_payload, self.item_ids,
            self.inverse_norms, self.bucket_spans,
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
    def load(cls, path: str | os.PathLike) -> "NestedAdditiveIndex":
        path = os.path.abspath(path)
        with open(path, "rb") as handle:
            if handle.read(8) != MAGIC:
                raise ValueError("not a nested additive index")
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        if int(header["format_version"]) != FORMAT_VERSION:
            raise ValueError("unsupported nested additive index version")
        arrays = {}
        for name, spec in header["layout"].items():
            arrays[name] = np.memmap(
                path,
                mode="r",
                dtype=np.dtype(spec["dtype"]),
                offset=int(spec["offset"]),
                shape=tuple(spec["shape"]),
            )
        decoder_format = header["metadata"].get("decoder_format", "faiss-index")
        if decoder_format == "raw-additive-codebooks":
            decoder = arrays["codec"]
        else:
            decoder = faiss.deserialize_index(
                np.array(arrays["codec"], dtype=np.uint8, copy=True)
            )
        index = cls(
            decoder,
            arrays["codes"],
            arrays["ids"],
            arrays["norms"],
            arrays["spans"],
            header["metadata"],
            path=path,
            layout=header["layout"],
        )
        if index.persistent_bytes != int(header["persistent_bytes"]):
            raise ValueError("file size differs from the persisted ledger")
        return index


__all__ = [
    "NestedAdditiveIndex",
    "calibrate_frozen_additive_codec",
    "codec_sha256",
    "query_ip_distortion",
    "unpack_additive_codes",
]
