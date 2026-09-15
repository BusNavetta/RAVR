"""RAVR allocation adapter for a frozen QAQ product quantizer."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
import struct

import numpy as np

from qaq_pq import (
    ALIGN_BYTES, EPS, HEADER_BYTES, QAQPQIndex, dimension_permutation,
    pack_codes, unpack_codes,
)
from ravr_rq import OBJECTIVE_ORDINAL_KL, RAVRRQConfig, RAVRRQTrainer


MAGIC = b"QAQRAVR1"
FORMAT_VERSION = 1


def _align(value: int) -> int:
    return (int(value) + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES


def _codec_vectors(index: QAQPQIndex, vectors: np.ndarray) -> np.ndarray:
    """Map vectors into the coordinate order used by the stored codebooks."""
    permutation = dimension_permutation(index.metadata, index.dimension)
    if permutation is None:
        return np.asarray(vectors, dtype=np.float32)
    return np.asarray(vectors[:, permutation], dtype=np.float32)


def ordered_subspaces(index: QAQPQIndex, gallery: np.ndarray, train_ids: np.ndarray) -> np.ndarray:
    """Choose a global nested order using only codec-training reconstruction gain."""
    codes = unpack_codes(index.packed_codes, index.stages, index.nbits)
    codec_gallery = _codec_vectors(index, gallery)
    train_ids = np.asarray(train_ids, dtype=np.int64)
    ds = index.codebooks.shape[2]
    gain = np.empty(index.stages, dtype=np.float64)
    for stage in range(index.stages):
        original = np.asarray(
            codec_gallery[train_ids, stage * ds:(stage + 1) * ds], dtype=np.float32
        )
        approximate = index.codebooks[stage, codes[train_ids, stage].astype(np.int64)]
        gain[stage] = np.mean(
            2.0 * np.einsum("nd,nd->n", original, approximate)
            - np.einsum("nd,nd->n", approximate, approximate)
        )
    return np.lexsort((np.arange(index.stages), -gain)).astype(np.uint16)


def calibrate_qaq_codec(
    index: QAQPQIndex,
    gallery: np.ndarray,
    calibration_queries: np.ndarray,
    calibration_exclude_ids: np.ndarray,
    subspace_order: np.ndarray,
    *,
    allowed_lengths: tuple[int, ...],
    rerank_candidates: int = 10_000,
    hard_negative_depth: int = 50,
) -> RAVRRQTrainer:
    """Run the unchanged Ordinal-KL RAVR calibration on QAQ PQ contributions."""
    order = np.asarray(subspace_order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(index.stages)):
        raise ValueError("subspace_order must be a permutation")
    if tuple(sorted(set(allowed_lengths))) != tuple(allowed_lengths):
        raise ValueError("allowed_lengths must be strictly increasing")
    if allowed_lengths[-1] > index.stages:
        raise ValueError("a QAQ prefix exceeds M")
    x = np.array(_codec_vectors(index, gallery), dtype=np.float32, copy=True, order="C")
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    codes = unpack_codes(index.packed_codes, index.stages, index.nbits)
    ds = index.codebooks.shape[2]
    embedded = np.zeros((index.stages, index.k, index.dimension), dtype=np.float32)
    for position, original_stage in enumerate(order):
        start, stop = original_stage * ds, (original_stage + 1) * ds
        embedded[position, :, start:stop] = index.codebooks[original_stage]
    ordered_codes = np.ascontiguousarray(codes[:, order])
    seed = int(index.metadata.get("config", {}).get("seed", 17))
    config = RAVRRQConfig(
        seed=seed,
        num_centroids=1,
        nprobe=1,
        stages=index.stages,
        codebook_size=index.k,
        allowed_lengths=tuple(int(x) for x in allowed_lengths),
        base_stages=int(allowed_lengths[0]),
        rerank_candidates=int(rerank_candidates),
        retrieval_k=10,
        hard_negative_depth=int(hard_negative_depth),
        objective=OBJECTIVE_ORDINAL_KL,
        norm_mode="stored_final_norm",
    )
    config.validate(len(x))
    trainer = RAVRRQTrainer(config)
    trainer.x = x
    trainer.n_items, trainer.dimension = x.shape
    trainer.centroids = np.zeros((1, trainer.dimension), dtype=np.float32)
    trainer.cluster_ids = np.zeros(trainer.n_items, dtype=np.int32)
    trainer.codebooks = embedded
    trainer.full_codes = ordered_codes
    trainer.validation_report = {
        "queries": 0,
        "candidate_sweep": {},
        "selected_rerank_candidates": int(rerank_candidates),
        "selection_target": None,
        "selection_note": "QAQ codec and subspace order frozen before allocation",
    }
    trainer._compute_prefix_statistics()
    trainer._calibrate(
        np.asarray(_codec_vectors(index, calibration_queries), dtype=np.float32),
        np.asarray(calibration_exclude_ids, dtype=np.int64),
    )
    trainer.gallery_hash = hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()
    trainer.codec_hash = hashlib.sha256(
        index.codebooks.tobytes() + order.tobytes()
    ).hexdigest()
    trainer.config_hash = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return trainer


class QAQNestedIndex:
    """Reloadable, grouped variable-prefix index over a frozen QAQ codec."""

    def __init__(
        self,
        codebooks: np.ndarray,
        subspace_ids: np.ndarray,
        code_payload: np.ndarray,
        item_ids: np.ndarray,
        inverse_norms: np.ndarray,
        bucket_spans: np.ndarray,
        metadata: dict,
        *,
        path: str | None = None,
        layout: dict | None = None,
    ):
        stored = np.asarray(codebooks)
        if stored.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError("nested QAQ codebooks must be float16 or float32")
        self.codebooks = stored
        self.subspace_ids = np.asarray(subspace_ids, dtype=np.uint16)
        self.code_payload = np.asarray(code_payload, dtype=np.uint8).reshape(-1)
        self.item_ids = np.asarray(item_ids, dtype=np.uint32).reshape(-1)
        self.inverse_norms = np.asarray(inverse_norms, dtype=np.float16).reshape(-1)
        self.bucket_spans = np.asarray(bucket_spans, dtype=np.uint64)
        self.metadata = metadata
        self.path = path
        if self.codebooks.ndim != 3 or len(self.subspace_ids) != len(self.codebooks):
            raise ValueError("invalid QAQ nested codec arrays")
        if len(self.item_ids) != len(self.inverse_norms):
            raise ValueError("QAQ nested IDs and norms differ")
        if self.bucket_spans.shape != (len(self.allowed_lengths), 4):
            raise ValueError("QAQ bucket spans have the wrong shape")
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
        return int(self.metadata["bits_per_symbol"])

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def dimension(self) -> int:
        return len(self.codebooks) * self.codebooks.shape[2]

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
        source: QAQPQIndex,
        subspace_order: np.ndarray,
        prefix_norms: np.ndarray,
        allowed_lengths: tuple[int, ...],
        selected_options: np.ndarray,
        *,
        strategy: str,
        metadata: dict | None = None,
    ) -> "QAQNestedIndex":
        order = np.asarray(subspace_order, dtype=np.int64)
        codes = unpack_codes(source.packed_codes, source.stages, source.nbits)[:, order]
        ordered_codebooks = np.ascontiguousarray(source.codebooks[order])
        selected = np.asarray(selected_options, dtype=np.int64)
        prefix_norms = np.asarray(prefix_norms, dtype=np.float32)
        if selected.shape != (source.n_items,):
            raise ValueError("allocation needs one option per QAQ item")
        if prefix_norms.shape != (source.n_items, len(allowed_lengths)):
            raise ValueError("QAQ prefix norms have the wrong shape")
        ids, norms, payloads, spans = [], [], [], []
        item_offset = code_offset = 0
        for option, length in enumerate(allowed_lengths):
            members = np.flatnonzero(selected == option)
            packed = pack_codes(codes[members, :length], source.nbits)
            row_bytes = packed.shape[1]
            ids.append(np.asarray(source.item_ids[members], dtype=np.uint32))
            norms.append((1.0 / np.maximum(prefix_norms[members, option], EPS)).astype(np.float16))
            payloads.append(packed.reshape(-1))
            spans.append((item_offset, len(members), code_offset, row_bytes))
            item_offset += len(members)
            code_offset += int(packed.nbytes)
        merged = {
            "format": "variable product-subspace prefix",
            "items": source.n_items,
            "dimension": source.dimension,
            "M": source.stages,
            "K": source.k,
            "bits_per_symbol": source.nbits,
            "allowed_lengths": list(allowed_lengths),
            "strategy": strategy,
            "allocation_histogram": {
                str(length): int(np.sum(selected == option))
                for option, length in enumerate(allowed_lengths)
            },
            "representation": "items grouped by prefix length; row-wise bit-packed PQ symbols",
            "subspace_order": "uint16 codec-training reconstruction-gain order",
            "ids": "explicit uint32 row IDs",
            "normalisation": "float16 inverse assigned-prefix norm",
            "source_codec_metadata": source.metadata,
            "codebook_storage_precision": source.metadata.get(
                "codebook_storage_precision", str(source.codebooks.dtype)
            ),
            "scoring_compute": "FP32 after persisted-codebook reload",
        }
        if metadata:
            merged.update(metadata)
        return cls(
            ordered_codebooks,
            order.astype(np.uint16),
            np.concatenate(payloads),
            np.concatenate(ids),
            np.concatenate(norms),
            np.asarray(spans, dtype=np.uint64),
            merged,
        )

    @classmethod
    def code_offset_for(
        cls, codebooks: np.ndarray, subspace_ids: np.ndarray, option_count: int, n_items: int
    ) -> int:
        arrays = (
            np.asarray(codebooks),
            np.asarray(subspace_ids, dtype=np.uint16),
            np.empty(n_items, dtype=np.uint32),
            np.empty(n_items, dtype=np.float16),
            np.empty((option_count, 4), dtype=np.uint64),
        )
        offset = HEADER_BYTES
        for value in arrays:
            offset = _align(offset)
            offset += value.nbytes
        return _align(offset)

    def _compute_layout(self) -> tuple[dict, int]:
        arrays = {
            "codebooks": self.codebooks,
            "subspace_ids": self.subspace_ids,
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
            offset += value.nbytes
        return layout, int(offset)

    def _validate_spans(self) -> None:
        items = codes = 0
        for length, row in zip(self.allowed_lengths, self.bucket_spans):
            item_start, count, code_start, row_bytes = map(int, row)
            if item_start != items or code_start != codes:
                raise ValueError("QAQ bucket spans are not contiguous")
            if row_bytes != math.ceil(length * self.nbits / 8):
                raise ValueError("QAQ prefix row width is inconsistent")
            items += count
            codes += count * row_bytes
        if items != self.n_items or codes != self.code_only_bytes:
            raise ValueError("QAQ bucket spans do not cover the payload")

    def _bucket_codes(self, option: int) -> np.ndarray:
        _, count, start, width = map(int, self.bucket_spans[option])
        return self.code_payload[start:start + count * width].reshape(count, width)

    def decode_normalized(self) -> np.ndarray:
        output = self.decode_raw()
        inverse_by_id = np.empty(self.n_items, dtype=np.float32)
        inverse_by_id[np.asarray(self.item_ids, dtype=np.int64)] = np.asarray(
            self.inverse_norms, dtype=np.float32
        )
        output *= inverse_by_id[:, None]
        return np.ascontiguousarray(output)

    def decode_raw(self) -> np.ndarray:
        output = np.empty((self.n_items, self.dimension), dtype=np.float32)
        ds = self.codebooks.shape[2]
        for option, length in enumerate(self.allowed_lengths):
            item_start, count = map(int, self.bucket_spans[option, :2])
            if not count:
                continue
            symbols = unpack_codes(self._bucket_codes(option), length, self.nbits)
            reconstruction = np.zeros((count, self.dimension), dtype=np.float32)
            for position in range(length):
                original = int(self.subspace_ids[position])
                start, stop = original * ds, (original + 1) * ds
                reconstruction[:, start:stop] = self.codebooks[
                    position, symbols[:, position].astype(np.int64)
                ]
            ids = np.asarray(self.item_ids[item_start:item_start + count], dtype=np.int64)
            output[ids] = reconstruction
        source_metadata = self.metadata.get(
            "source_codec_metadata", self.metadata.get("source_qaq_metadata", {})
        )
        permutation = dimension_permutation(source_metadata, self.dimension)
        if permutation is not None:
            unpermuted = np.empty_like(output)
            unpermuted[:, permutation] = output
            output = unpermuted
        return np.ascontiguousarray(output)

    def storage_breakdown(self) -> dict:
        fields = {
            "codebooks": int(self.codebooks.nbytes),
            "subspace_order": int(self.subspace_ids.nbytes),
            "codes": int(self.code_payload.nbytes),
            "ids": int(self.item_ids.nbytes),
            "norms": int(self.inverse_norms.nbytes),
            "length_metadata": int(self.bucket_spans.nbytes),
        }
        payload = sum(fields.values())
        return fields | {
            "header_and_alignment": int(self.persistent_bytes - payload),
            "persistent_bytes": int(self.persistent_bytes),
            "code_only_bytes": int(self.code_only_bytes),
        }

    def save(self, path: str | os.PathLike) -> str:
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
            raise ValueError("QAQ RAVR metadata exceeds fixed header")
        arrays = {
            "codebooks": self.codebooks,
            "subspace_ids": self.subspace_ids,
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
            raise AssertionError("QAQ RAVR file size differs from ledger")
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "QAQNestedIndex":
        path = os.path.abspath(path)
        with open(path, "rb") as handle:
            if handle.read(8) != MAGIC:
                raise ValueError("not a QAQ RAVR index")
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        arrays = {}
        for name, spec in header["layout"].items():
            arrays[name] = np.memmap(
                path, mode="r", dtype=np.dtype(spec["dtype"]),
                offset=int(spec["offset"]), shape=tuple(spec["shape"]),
            )
        return cls(
            arrays["codebooks"], arrays["subspace_ids"], arrays["codes"],
            arrays["ids"], arrays["norms"], arrays["spans"], header["metadata"],
            path=path, layout=header["layout"],
        )

    def close(self) -> None:
        for value in (
            self.codebooks, self.subspace_ids, self.code_payload, self.item_ids,
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


__all__ = [
    "QAQNestedIndex", "calibrate_qaq_codec", "ordered_subspaces",
]
