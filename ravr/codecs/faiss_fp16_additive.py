"""Reloadable FP16 storage for trained Faiss RQ/LSQ additive codecs.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np

from additive_coded_search import faiss_additive_codebooks, unpack_stage
from faiss_fixed_rate import ALIGN_BYTES, EPS, HEADER_BYTES, _align
from nested_additive_ravr import codec_sha256


MAGIC = b"FADD16V1"
FORMAT_VERSION = 1


class FP16AdditiveBundle:
    def __init__(
        self,
        codebooks: np.ndarray,
        codes: np.ndarray,
        item_ids: np.ndarray,
        inverse_norms: np.ndarray,
        metadata: dict,
        *,
        path: str | None = None,
        layout: dict | None = None,
    ):
        self.codebooks = np.asarray(codebooks, dtype=np.float16)
        self.codes = np.asarray(codes, dtype=np.uint8)
        self.item_ids = np.asarray(item_ids, dtype=np.uint32)
        self.inverse_norms = np.asarray(inverse_norms, dtype=np.float16)
        self.metadata = metadata
        self.path = path
        if self.codebooks.ndim != 3 or self.codes.ndim != 2:
            raise ValueError("invalid FP16 additive codebook or packed-code shape")
        if len(self.codes) != len(self.item_ids) or len(self.item_ids) != len(self.inverse_norms):
            raise ValueError("FP16 additive item arrays differ in length")
        expected_width = (
            self.codebooks.shape[0] * int(self.metadata["config"]["nbits"]) + 7
        ) // 8
        if self.codes.shape[1] != expected_width:
            raise ValueError("packed codes do not match the FP16 additive codec")
        if layout is None:
            self._layout, self.persistent_bytes = self._compute_layout()
        else:
            self._layout = layout
            self.persistent_bytes = os.path.getsize(path)

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
            "codebooks": self.codebooks,
            "codes": self.codes,
            "ids": self.item_ids,
            "norms": self.inverse_norms,
        }
        layout = {}
        offset = HEADER_BYTES
        for name, value in arrays.items():
            offset = _align(offset)
            layout[name] = {
                "offset": int(offset),
                "nbytes": int(value.nbytes),
                "dtype": np.dtype(value.dtype).str,
                "shape": list(value.shape),
            }
            offset += int(value.nbytes)
        return layout, int(offset)

    def storage_breakdown(self) -> dict[str, int]:
        fields = {
            "codebooks": int(self._layout["codebooks"]["nbytes"]),
            "codes": int(self._layout["codes"]["nbytes"]),
            "ids": int(self._layout["ids"]["nbytes"]),
            "norms": int(self._layout["norms"]["nbytes"]),
        }
        payload = sum(fields.values())
        return fields | {
            "header_and_alignment": int(self.persistent_bytes - payload),
            "persistent_bytes": int(self.persistent_bytes),
            "code_only_bytes": int(self.code_only_bytes),
        }

    def decode_raw(self) -> np.ndarray:
        codebooks = self.codebooks.astype(np.float32)
        nbits = int(self.metadata["config"]["nbits"])
        reconstruction = np.zeros(
            (self.n_items, self.codebooks.shape[2]), dtype=np.float32
        )
        for stage in range(self.codebooks.shape[0]):
            reconstruction += codebooks[
                stage, unpack_stage(self.codes, stage, nbits).astype(np.int64)
            ]
        return np.ascontiguousarray(reconstruction)

    def decode_normalized(self) -> np.ndarray:
        reconstruction = self.decode_raw()
        reconstruction *= self.inverse_norms.astype(np.float32)[:, None]
        return np.ascontiguousarray(reconstruction)

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
            raise ValueError("FP16 additive metadata exceeds the fixed header")
        arrays = {
            "codebooks": self.codebooks,
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
            raise AssertionError("FP16 additive file size differs from its ledger")
        self.path = path
        return path

    def close(self) -> None:
        for value in (self.codebooks, self.codes, self.item_ids, self.inverse_norms):
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
    def load(cls, path: str | os.PathLike) -> "FP16AdditiveBundle":
        path = os.path.abspath(path)
        with open(path, "rb") as handle:
            if handle.read(8) != MAGIC:
                raise ValueError("not an FP16 additive bundle")
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        if int(header["format_version"]) != FORMAT_VERSION:
            raise ValueError("unsupported FP16 additive bundle version")
        arrays = {}
        for name, spec in header["layout"].items():
            arrays[name] = np.memmap(
                path,
                mode="r",
                dtype=np.dtype(spec["dtype"]),
                offset=int(spec["offset"]),
                shape=tuple(spec["shape"]),
            )
        bundle = cls(
            arrays["codebooks"], arrays["codes"], arrays["ids"], arrays["norms"],
            header["metadata"], path=path, layout=header["layout"],
        )
        if bundle.persistent_bytes != int(header["persistent_bytes"]):
            raise ValueError("FP16 additive file size differs from its header")
        return bundle


def from_faiss_additive_bundle(bundle) -> FP16AdditiveBundle:
    """Round a frozen RQ/LSQ decoder while preserving every item symbol."""
    codebooks, nbits = faiss_additive_codebooks(bundle)
    rounded = codebooks.astype(np.float16)
    reconstruction = np.zeros(
        (bundle.n_items, rounded.shape[2]), dtype=np.float32
    )
    compute_codebooks = rounded.astype(np.float32)
    for stage in range(rounded.shape[0]):
        reconstruction += compute_codebooks[
            stage, unpack_stage(bundle.codes, stage, nbits).astype(np.int64)
        ]
    inverse = (
        1.0 / np.maximum(np.linalg.norm(reconstruction, axis=1), EPS)
    ).astype(np.float16)
    metadata = dict(bundle.metadata)
    metadata.update({
        "storage_precision": "IEEE float16 shared codebooks",
        "decoder_format": "raw-additive-codebooks",
        "source_faiss_codec_sha256": codec_sha256(bundle),
        "source_bundle_path": str(bundle.path) if bundle.path else None,
        "codebooks_sha256": hashlib.sha256(
            np.ascontiguousarray(rounded).view(np.uint8)
        ).hexdigest(),
        "normalisation": "float16 inverse norm recomputed from FP16 codebooks",
        "scoring_compute": "FP32 lookup and accumulation after FP16 reload",
    })
    return FP16AdditiveBundle(
        rounded,
        np.asarray(bundle.codes, dtype=np.uint8),
        np.asarray(bundle.item_ids, dtype=np.uint32),
        inverse,
        metadata,
    )


__all__ = ["FP16AdditiveBundle", "from_faiss_additive_bundle"]
